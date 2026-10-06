"""项目会话组织只管理本人范围；不授予仓库路径或工具权限。"""

import time

from sqlalchemy import and_, or_, select, update

from .errors import HarnessError
from .models import Asset, Run, Thread, ThreadRun, User
from .security import canonical, digest
from .store import scope, uid


def thread_dict(thread, latest=None, project_name=None):
    return {
        "id": thread.id,
        "title": thread.title,
        "session_id": thread.session_id,
        "project_id": thread.project_asset_id,
        "project_name": project_name,
        "archived": thread.archived,
        "created": thread.created,
        "latest_run_id": latest,
    }


class ThreadService:
    def __init__(self, store):
        self.store = store

    def require_ready(self):
        if not self.store.threads_ready:
            raise HarnessError(
                503, "会话组织尚未迁移，请管理员先运行 migrate_threads 预览与备份迁移"
            )

    @staticmethod
    async def lock_owner(session, principal):
        await session.execute(
            update(User)
            .where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
            .values(role=User.role)
        )

    async def project(self, session, principal, project_id, active=True):
        project = await self.store.owned(session, Asset, project_id, principal)
        if project.kind != "project":
            raise HarnessError(404, "项目不存在或无权访问")
        if active and project.status != "active":
            raise HarnessError(409, "项目须审核启用后才能开始新会话或任务")
        return project

    @staticmethod
    def title(value):
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise HarnessError(422, "会话标题须为1至200字符")
        return value.strip()

    async def create(self, principal, title, project_id=None, request_key=None):
        self.require_ready()
        title = self.title(title)
        if principal.role == "viewer":
            raise HarnessError(403, "只读成员不能管理会话")
        if project_id is not None and (
            not isinstance(project_id, str) or not project_id or len(project_id) > 128
        ):
            raise HarnessError(422, "项目标识无效")
        if request_key is not None and (
            not isinstance(request_key, str) or not request_key or len(request_key) > 256
        ):
            raise HarnessError(422, "会话幂等键须为1至256字符")
        request_hash = digest(canonical({"title": title, "project_id": project_id}))
        async with self.store.transaction() as session:
            await self.lock_owner(session, principal)
            if request_key:
                existing = await session.scalar(
                    select(Thread).where(
                        *scope(Thread, principal), Thread.request_key == request_key
                    )
                )
                if existing:
                    if existing.request_hash != request_hash:
                        raise HarnessError(409, "同一会话幂等键不能用于不同请求")
                    return thread_dict(existing)
            project = await self.project(session, principal, project_id) if project_id else None
            identifier = uid()
            thread = Thread(
                id=identifier,
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                session_id=identifier,
                project_asset_id=project_id,
                title=title,
                archived=False,
                created=time.time(),
                request_key=request_key,
                request_hash=request_hash,
            )
            session.add(thread)
            return thread_dict(thread, project_name=project.name if project else None)

    async def resolve_run(self, session, principal, message, session_id=None, thread_id=None):
        """在create_run事务和用户行锁中调用，保持归档与任务创建串行。"""
        if not self.store.threads_ready:
            if thread_id is not None:
                self.require_ready()
            return None, None
        thread = None
        if thread_id is not None:
            thread = await self.store.owned(session, Thread, thread_id, principal)
            if session_id is not None and session_id != thread.session_id:
                raise HarnessError(422, "thread_id与session_id指向不同会话")
        elif session_id:
            thread = await session.scalar(
                select(Thread).where(*scope(Thread, principal), Thread.session_id == session_id)
            )
            foreign = await session.scalar(
                select(Thread.id).where(Thread.session_id == session_id).limit(1)
            )
            if thread is None and foreign is not None:
                raise HarnessError(404, "会话不存在或无权访问")
        if thread is None:
            identifier = uid()
            thread = Thread(
                id=identifier,
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                session_id=session_id or identifier,
                title=self.title(message[:200]),
                archived=False,
                created=time.time(),
            )
            session.add(thread)
        if thread.archived:
            raise HarnessError(409, "会话已归档，请先恢复再继续")
        project = (
            await self.project(session, principal, thread.project_asset_id)
            if thread.project_asset_id
            else None
        )
        return thread, project

    def query(self, principal):
        latest = (
            select(Run.id)
            .join(ThreadRun, ThreadRun.run_id == Run.id)
            .where(
                *scope(Run, principal),
                *scope(ThreadRun, principal),
                ThreadRun.thread_id == Thread.id,
            )
            .order_by(Run.created.desc(), Run.id.desc())
            .limit(1)
            .correlate(Thread)
            .scalar_subquery()
        )
        project_name = (
            select(Asset.name)
            .where(
                Asset.id == Thread.project_asset_id,
                *scope(Asset, principal),
                Asset.kind == "project",
            )
            .correlate(Thread)
            .scalar_subquery()
        )
        return select(Thread, latest, project_name).where(*scope(Thread, principal))

    async def get(self, principal, thread_id):
        self.require_ready()
        async with self.store.sessions() as session:
            row = (
                await session.execute(self.query(principal).where(Thread.id == thread_id))
            ).one_or_none()
            if row is None:
                raise HarnessError(404, "会话不存在或无权访问")
            return thread_dict(*row)

    async def list(self, principal, project_id=None, archived=False, limit=20, before=None):
        self.require_ready()
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 50
            or not isinstance(archived, bool)
            or (before is not None and (not isinstance(before, str) or len(before) > 128))
        ):
            raise HarnessError(422, "会话查询参数无效")
        async with self.store.sessions() as session:
            if project_id:
                await self.project(session, principal, project_id, active=False)
            statement = self.query(principal).where(Thread.archived == archived)
            if project_id is not None:
                statement = statement.where(Thread.project_asset_id == project_id)
            if before:
                cursor = await self.store.owned(session, Thread, before, principal)
                if cursor.archived != archived or (
                    project_id is not None and cursor.project_asset_id != project_id
                ):
                    raise HarnessError(404, "会话游标不属于当前列表")
                statement = statement.where(
                    or_(
                        Thread.created < cursor.created,
                        and_(Thread.created == cursor.created, Thread.id < cursor.id),
                    )
                )
            rows = (
                await session.execute(
                    statement.order_by(Thread.created.desc(), Thread.id.desc()).limit(limit)
                )
            ).all()
            return [thread_dict(*row) for row in rows]

    async def update(self, principal, thread_id, title=None, archived=None):
        self.require_ready()
        if principal.role == "viewer":
            raise HarnessError(403, "只读成员不能管理会话")
        if title is not None:
            title = self.title(title)
        if archived is not None and not isinstance(archived, bool):
            raise HarnessError(422, "归档状态无效")
        async with self.store.transaction() as session:
            await self.lock_owner(session, principal)
            thread = await self.store.owned(session, Thread, thread_id, principal)
            if title is not None:
                thread.title = title
            if archived is not None:
                thread.archived = archived
        return await self.get(principal, thread_id)

    async def projects(self, principal, limit=20, before=None):
        self.require_ready()
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise HarnessError(422, "项目分页参数无效")
        async with self.store.sessions() as session:
            query = select(Asset.id, Asset.name, Asset.status).where(
                *scope(Asset, principal), Asset.kind == "project", Asset.status == "active"
            )
            if before:
                cursor = await self.project(session, principal, before)
                query = query.where(
                    or_(
                        Asset.name > cursor.name,
                        and_(Asset.name == cursor.name, Asset.id > cursor.id),
                    )
                )
            rows = (await session.execute(query.order_by(Asset.name, Asset.id).limit(limit))).all()
            return [dict(row._mapping) for row in rows]
