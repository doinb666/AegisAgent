"""事务存储与统一私有范围。"""

import asyncio
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from .errors import HarnessError
from .models import Base, Event, Run, ToolCall


def uid() -> str:
    return str(uuid4())


def scope(model, principal):
    return (model.tenant_id == principal.tenant_id, model.owner_id == principal.user_id)


def run_dict(run):
    fields = (
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
    )
    return {key: getattr(run, key) for key in fields}


def asset_dict(asset):
    fields = ("id", "kind", "name", "content", "status", "version")
    return {**{key: getattr(asset, key) for key in fields}, "metadata": asset.attributes}


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.engine = create_async_engine(settings.resolved_database_url())
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.write_lock = asyncio.Lock()
        if self.engine.dialect.name == "sqlite":

            @event.listens_for(self.engine.sync_engine, "connect")
            def configure_sqlite(connection, _):
                cursor = connection.cursor()
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

    async def initialize(self):
        if self.settings.auto_create_schema:
            async with self.engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def transaction(self, existing=None):
        if existing is not None:
            yield existing
            return
        async with self.write_lock, self.sessions.begin() as session:
            yield session

    async def owned(self, session, model, identifier, principal):
        obj = await session.scalar(
            select(model).where(model.id == identifier, *scope(model, principal))
        )
        if obj is None:
            raise HarnessError(404, "资源不存在或无权访问")
        return obj

    def emit(self, session, run, event_type, data):
        session.add(
            Event(
                tenant_id=run.tenant_id,
                owner_id=run.owner_id,
                run_id=run.id,
                type=event_type,
                data=data,
            )
        )

    async def claim(self, worker_id, child_only=False):
        """候选读取后条件更新；跨进程也只有一个胜者。"""
        async with self.write_lock, self.sessions.begin() as session:
            now = time.time()
            stale = (
                await session.scalars(
                    select(Run).where(Run.status == "running", Run.lease_until < now)
                )
            ).all()
            for run in stale:
                unknown = await session.scalar(
                    select(ToolCall.id).where(
                        ToolCall.run_id == run.id,
                        ToolCall.tenant_id == run.tenant_id,
                        ToolCall.owner_id == run.owner_id,
                        ToolCall.status == "started",
                    )
                )
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
            run_kind = Run.parent_run_id.is_not(None) if child_only else Run.parent_run_id.is_(None)
            candidate = await session.scalar(
                select(Run.id)
                .where(Run.status == "queued", run_kind)
                .order_by(Run.created)
                .limit(1)
            )
            if candidate is None:
                return None
            result = await session.execute(
                update(Run)
                .where(Run.id == candidate, Run.status == "queued")
                .values(
                    status="running",
                    lease_owner=worker_id,
                    lease_until=now + self.settings.lease_seconds,
                )
            )
            if result.rowcount:
                run = await session.get(Run, candidate)
                self.emit(session, run, "running", {"claimed_at": now, "trace_id": run.trace_id})
            return candidate if result.rowcount else None

    async def close(self):
        await self.engine.dispose()
