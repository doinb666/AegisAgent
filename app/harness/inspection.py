"""仅本人可访问的工作台查询；不执行工具、审批或修改文件。"""

from sqlalchemy import and_, func, or_, select

from .errors import HarnessError
from .models import Asset, Run
from .store import run_dict, run_view_query, scope

RUN_STATUSES = (
    "queued",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
)


class InspectionService:
    def __init__(self, store, workspace):
        self.store = store
        self.workspace = workspace

    async def thread(self, principal, run_id, limit=20, before=None):
        """以选中的运行为时间边界，只读取本人顶层会话的公开问答。"""
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 50
            or (before is not None and (not isinstance(before, str) or len(before) > 128))
        ):
            raise HarnessError(422, "会话分页参数无效")
        async with self.store.sessions() as session:
            selected = await self.store.run_view(session, run_id, principal)
            statement = select(
                Run.id, Run.message, Run.answer, Run.error, Run.status, Run.created
            ).where(*scope(Run, principal))
            if selected.parent_run_id:
                # 子任务不混入主线程，也不展示其他子任务的独立上下文。
                statement = statement.where(Run.id == selected.id)
            else:
                statement = statement.where(
                    Run.session_id == selected.session_id,
                    Run.parent_run_id.is_(None),
                    or_(
                        Run.created < selected.created,
                        and_(Run.created == selected.created, Run.id <= selected.id),
                    ),
                )
            if before is not None:
                cursor = await self.store.run_view(session, before, principal)
                if (
                    cursor.session_id != selected.session_id
                    or cursor.parent_run_id != selected.parent_run_id
                    or (cursor.created, cursor.id) > (selected.created, selected.id)
                    or (selected.parent_run_id and cursor.id != selected.id)
                ):
                    raise HarnessError(404, "会话游标不存在或无权访问")
                statement = statement.where(
                    or_(
                        Run.created < cursor.created,
                        and_(Run.created == cursor.created, Run.id < cursor.id),
                    )
                )
            rows = (
                await session.execute(
                    statement.order_by(Run.created.desc(), Run.id.desc()).limit(limit + 1)
                )
            ).all()
            has_more = len(rows) > limit
            items = [dict(row._mapping) for row in reversed(rows[:limit])]
            return {
                "items": items,
                "has_more": has_more,
                "next_before": items[0]["id"] if has_more else None,
            }

    async def list_runs(self, principal, query=None, status=None, limit=100, before=None):
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 100
            or (query is not None and (not isinstance(query, str) or len(query) > 200))
            or (status is not None and status not in RUN_STATUSES)
            or (before is not None and (not isinstance(before, str) or len(before) > 128))
        ):
            raise HarnessError(422, "任务查询参数无效")
        async with self.store.sessions() as session:
            statement = run_view_query(principal)
            if before is not None:
                cursor = await self.store.run_view(session, before, principal)
                statement = statement.where(
                    or_(
                        Run.created < cursor.created,
                        and_(Run.created == cursor.created, Run.id < cursor.id),
                    )
                )
            if query:
                statement = statement.where(Run.message.contains(query, autoescape=True))
            if status is not None:
                statement = statement.where(Run.status == status)
            runs = await session.execute(
                statement.order_by(Run.created.desc(), Run.id.desc()).limit(limit)
            )
            return [run_dict(run) for run in runs]

    async def overview(self, principal):
        async with self.store.sessions() as session:

            async def counts(model, column):
                rows = await session.execute(
                    select(column, func.count()).where(*scope(model, principal)).group_by(column)
                )
                return dict(rows.all())

            run_status = await counts(Run, Run.status)
            asset_kind = await counts(Asset, Asset.kind)
            asset_status = await counts(Asset, Asset.status)
            return {
                "runs": {"total": sum(run_status.values()), "by_status": run_status},
                "assets": {
                    "total": sum(asset_kind.values()),
                    "by_kind": asset_kind,
                    "by_status": asset_status,
                },
            }

    async def _files(self, principal, run_id, path=None):
        # 先核对数据库所有权，再触碰文件系统，避免利用错误探测他人的目录。
        async with self.store.sessions() as session:
            await self.store.require_run_owner(session, run_id, principal)
        try:
            if path is None:
                return await self.workspace.list_files(principal, run_id)
            return await self.workspace.preview_file(principal, run_id, path)
        except ValueError as exc:
            raise HarnessError(422, str(exc)) from exc
        except OSError as exc:
            raise HarnessError(404, "文件或工作区不存在或无法读取") from exc

    async def list_files(self, principal, run_id):
        return await self._files(principal, run_id)

    async def preview_file(self, principal, run_id, path):
        return await self._files(principal, run_id, path)
