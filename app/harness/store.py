"""事务存储与统一私有范围。"""

import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from sqlalchemy import event, inspect, select, text, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import aliased

from .errors import HarnessError
from .models import NOTIFICATION_TYPES, Base, Event, Notification, Run, Thread, ThreadRun, ToolCall
from .schedule_models import Occurrence, Schedule

ACTIVE_STATUSES = ("queued", "running", "waiting_approval")
TERMINAL_STATUSES = ("completed", "failed", "cancelled", "interrupted")
RUN_PUBLIC_FIELDS = (
    "id",
    "session_id",
    "message",
    "status",
    "answer",
    "error",
    "model",
    "trace_id",
    "step",
    "approval",
    "parent_run_id",
    "created",
)


def uid() -> str:
    return str(uuid4())


def scope(model, principal):
    return (model.tenant_id == principal.tenant_id, model.owner_id == principal.user_id)


def run_dict(run):
    return {
        **{key: getattr(run, key) for key in RUN_PUBLIC_FIELDS},
        "collaboration_mode": run.config.get("collaboration_mode"),
        "project_mode": run.config.get("project_mode"),
        "thread_id": run.config.get("thread_id"),
        "document_references": run.config.get("document_references", []),
        "model_parameters": run.config.get("model_parameters", {}),
    }


def run_view_query(principal):
    """只投影公开状态字段，避免轮询反复下载和解码执行上下文。"""
    return select(*(getattr(Run, key) for key in RUN_PUBLIC_FIELDS), Run.config).where(
        *scope(Run, principal)
    )


def asset_dict(asset):
    fields = ("id", "kind", "name", "content", "status", "version")
    return {**{key: getattr(asset, key) for key in fields}, "metadata": asset.attributes}


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.engine = create_async_engine(settings.resolved_database_url(), pool_pre_ping=True)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.write_lock = asyncio.Lock()
        self.threads_ready = False
        self.notifications_ready = False
        self.schedules_ready = False
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine.sync_engine, "connect")
            def configure_sqlite(connection, _):
                cursor = connection.cursor()
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

    async def initialize(self):
        async with self.engine.begin() as connection:
            tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            if self.settings.auto_create_schema:
                deferred = {
                    Thread.__tablename__,
                    ThreadRun.__tablename__,
                    Notification.__tablename__,
                    Schedule.__tablename__,
                    Occurrence.__tablename__,
                }
                selected = [
                    table
                    for table in Base.metadata.sorted_tables
                    if "harness_runs" not in tables or table.name not in deferred
                ]
                await connection.run_sync(
                    lambda sync: Base.metadata.create_all(sync, tables=selected)
                )
                tables = await connection.run_sync(
                    lambda sync: set(inspect(sync).get_table_names())
                )
            self.threads_ready = {Thread.__tablename__, ThreadRun.__tablename__} <= tables
            self.notifications_ready = Notification.__tablename__ in tables
            self.schedules_ready = {Schedule.__tablename__, Occurrence.__tablename__} <= tables

    @asynccontextmanager
    async def transaction(self, existing=None):
        if existing is not None:
            yield existing
            return
        async with self.write_lock, self.sessions.begin() as session:
            if self.engine.dialect.name == "sqlite":
                # 跨 Store 在读候选之前取得写锁，避免延迟事务的读写升级竞争。
                await session.execute(text("BEGIN IMMEDIATE"))
            yield session

    async def owned(self, session, model, identifier, principal):
        obj = await session.scalar(
            select(model).where(model.id == identifier, *scope(model, principal))
        )
        if obj is None:
            raise HarnessError(404, "资源不存在或无权访问")
        return obj

    async def run_view(self, session, identifier, principal):
        row = (
            await session.execute(run_view_query(principal).where(Run.id == identifier))
        ).one_or_none()
        if row is None:
            raise HarnessError(404, "资源不存在或无权访问")
        return row

    async def require_run_owner(self, session, identifier, principal):
        identifier = await session.scalar(
            select(Run.id).where(Run.id == identifier, *scope(Run, principal))
        )
        if identifier is None:
            raise HarnessError(404, "资源不存在或无权访问")

    def emit(self, session, run, event_type, data):
        event_record = Event(
            tenant_id=run.tenant_id,
            owner_id=run.owner_id,
            run_id=run.id,
            type=event_type,
            data=data,
        )
        session.add(event_record)
        if (
            self.notifications_ready
            and run.parent_run_id is None
            and event_type in NOTIFICATION_TYPES
        ):
            session.add(
                Notification(
                    event=event_record,
                    tenant_id=run.tenant_id,
                    owner_id=run.owner_id,
                    run_id=run.id,
                    type=event_type,
                    title=run.message[:160],
                    created=time.time(),
                )
            )

    async def unknown_tool(self, session, run):
        return await session.scalar(
            select(ToolCall.id).where(
                ToolCall.run_id == run.id,
                ToolCall.tenant_id == run.tenant_id,
                ToolCall.owner_id == run.owner_id,
                ToolCall.status == "started",
            )
        )

    async def stop_child(self, session, run):
        """父行已锁定时停止活跃子运行，保留尚未确认的工具执行。"""
        locked = await session.execute(
            update(Run)
            .where(
                Run.id == run.id,
                Run.tenant_id == run.tenant_id,
                Run.owner_id == run.owner_id,
                Run.status.in_(ACTIVE_STATUSES),
            )
            .values(status=Run.status)
        )
        if not locked.rowcount:
            return
        unknown = await self.unknown_tool(session, run)
        status = "interrupted" if unknown else "cancelled"
        reason = "父运行失效，子工具结果未知，禁止自动重放" if unknown else "父运行失效，停止子运行"
        await session.execute(
            update(Run)
            .where(Run.id == run.id)
            .values(status=status, error=reason, lease_owner=None, lease_until=None)
        )
        self.emit(session, run, status, {"parent_run_id": run.parent_run_id, "reason": reason})

    async def stop_children(self, session, parent):
        children = (
            await session.scalars(
                select(Run)
                .where(
                    Run.parent_run_id == parent.id,
                    Run.tenant_id == parent.tenant_id,
                    Run.owner_id == parent.owner_id,
                    Run.status.in_(ACTIVE_STATUSES),
                )
                .order_by(Run.id)
            )
        ).all()
        for child in children:
            await self.stop_child(session, child)

    async def parent_active(self, session, run, stop_invalid=False):
        """先锁同私有范围父行，再判断租约；与父终态更新按行互斥。"""
        if run.parent_run_id is None:
            return True
        locked = await session.execute(
            update(Run)
            .where(
                Run.id == run.parent_run_id,
                Run.tenant_id == run.tenant_id,
                Run.owner_id == run.owner_id,
            )
            .values(status=Run.status)
        )
        if not locked.rowcount:
            return False
        parent = await session.get(Run, run.parent_run_id, populate_existing=True)
        active = (
            parent.status == "running"
            and parent.lease_owner is not None
            and parent.lease_until is not None
            and parent.lease_until > time.time()
        )
        if not active and stop_invalid:
            await self.stop_child(session, run)
        return active

    async def running(self, session, run_id, worker_id):
        """执行边界统一验证父子；失效子终态在事务提交后由调用方停止执行。"""
        run = await session.get(Run, run_id)
        if run is None or not await self.parent_active(session, run, stop_invalid=True):
            return None
        locked = await session.execute(
            update(Run)
            .where(Run.id == run_id, Run.status == "running", Run.lease_owner == worker_id)
            .values(status=Run.status)
        )
        if not locked.rowcount:
            return None
        await session.refresh(run)
        return run

    async def claim(self, worker_id, child_only=False):
        """先恢复父子生命周期，再在父有效租约内条件领取候选。"""
        async with self.transaction() as session:
            now = time.time()
            stale = (
                await session.scalars(
                    select(Run)
                    .where(Run.status == "running", Run.lease_until < now)
                    .order_by(Run.parent_run_id.nullsfirst(), Run.id)
                )
            ).all()
            for run in stale:
                if not await self.parent_active(session, run, stop_invalid=True):
                    continue
                locked = await session.execute(
                    update(Run)
                    .where(Run.id == run.id, Run.status == "running", Run.lease_until < now)
                    .values(status=Run.status)
                )
                if not locked.rowcount:
                    continue
                unknown = await self.unknown_tool(session, run)
                recovered_status = "interrupted" if unknown else "queued"
                recovered = await session.execute(
                    update(Run)
                    .where(Run.id == run.id, Run.status == "running", Run.lease_until < now)
                    .values(
                        status=recovered_status,
                        lease_owner=None,
                        lease_until=None,
                        error="工具结果未知，禁止自动重放" if unknown else None,
                    )
                )
                if recovered.rowcount and unknown:
                    self.emit(session, run, "interrupted", {"reason": "工具结果未知"})
                    await self.stop_children(session, run)

            child = aliased(Run)
            parents = (
                await session.scalars(
                    select(Run)
                    .where(
                        Run.status.in_(TERMINAL_STATUSES),
                        select(child.id)
                        .where(
                            child.parent_run_id == Run.id,
                            child.tenant_id == Run.tenant_id,
                            child.owner_id == Run.owner_id,
                            child.status.in_(ACTIVE_STATUSES),
                        )
                        .exists(),
                    )
                    .order_by(Run.id)
                )
            ).all()
            for parent in parents:
                locked = await session.execute(
                    update(Run)
                    .where(Run.id == parent.id, Run.status.in_(TERMINAL_STATUSES))
                    .values(status=Run.status)
                )
                if locked.rowcount:
                    await self.stop_children(session, parent)
            run_kind = Run.parent_run_id.is_not(None) if child_only else Run.parent_run_id.is_(None)
            eligible = select(Run).where(Run.status == "queued", run_kind)
            if child_only:
                parent = aliased(Run)
                eligible = eligible.where(
                    select(parent.id)
                    .where(
                        parent.id == Run.parent_run_id,
                        parent.tenant_id == Run.tenant_id,
                        parent.owner_id == Run.owner_id,
                        parent.status == "running",
                        parent.lease_owner.is_not(None),
                        parent.lease_until > time.time(),
                    )
                    .exists()
                )
            candidates = (
                await session.scalars(
                    eligible.order_by(Run.created, Run.id).limit(32 if child_only else 1)
                )
            ).all()
            for run in candidates:
                if not await self.parent_active(session, run):
                    continue
                claimed_at = time.time()
                result = await session.execute(
                    update(Run)
                    .where(Run.id == run.id, Run.status == "queued")
                    .values(
                        status="running",
                        lease_owner=worker_id,
                        lease_until=claimed_at + self.settings.lease_seconds,
                    )
                )
                if result.rowcount:
                    self.emit(
                        session,
                        run,
                        "running",
                        {"claimed_at": claimed_at, "trace_id": run.trace_id},
                    )
                    return run.id
            return None

    async def close(self):
        await self.engine.dispose()
