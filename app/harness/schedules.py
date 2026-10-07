"""用户私有定时计划及运行创建时的事务权限栅栏。"""

import time

from pydantic import ValidationError
from sqlalchemy import and_, func, or_, select, update

from .errors import HarnessError, Principal
from .models import Run, User
from .schedule_models import Occurrence, Schedule
from .schedule_schema import ScheduleInput, ScheduleTransition, utc_iso
from .security import canonical, digest
from .store import ACTIVE_STATUSES, scope, uid

# 明确列举已经核对过的内核只读工具；新增工具不会自动获得后台执行权限。
SCHEDULE_TOOLS = frozenset({"calculator", "knowledge_search", "skill_read", "artifact_read"})
MAX_LIVE_SCHEDULES = 100


def schedule_dict(schedule):
    keys = (
        "id",
        "title",
        "message",
        "status",
        "version",
        "interval_seconds",
        "mode",
        "model",
        "created",
    )
    return {**{key: getattr(schedule, key) for key in keys}, "next_at": utc_iso(schedule.next_at)}


def occurrence_dict(item):
    return {
        "id": item.id,
        "scheduled_at": utc_iso(item.scheduled_at),
        "status": item.status,
        "reason": item.reason,
        "run_id": item.run_id,
    }


def occurrence_key(item):
    return "schedule:" + item.id


class ScheduleService:
    def __init__(self, harness, clock=None):
        from .schedule_runtime import ScheduleDispatcher

        self.harness = harness
        self.store = harness.store
        self.clock = clock or time.time
        self.dispatcher = ScheduleDispatcher(self)

    def require_ready(self):
        if not self.store.schedules_ready:
            raise HarnessError(409, "定时任务尚未迁移，请显式执行 --migrate-schedules --apply")

    async def writer(self, session, principal):
        # 与创建运行使用相同的身份行锁顺序；跨 Store 的创建配额也由数据库串行化。
        await session.execute(
            update(User)
            .where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
            .values(role=User.role)
        )
        user = await session.scalar(
            select(User).where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
        )
        if user is None or user.role not in {"admin", "operator"}:
            raise HarnessError(403, "只读成员不能创建或管理定时任务")
        return Principal(user.id, user.tenant_id, user.role)

    async def create(self, principal, idempotency_key, **values):
        self.require_ready()
        try:
            body = ScheduleInput.model_validate(values)
        except ValidationError as exc:
            raise HarnessError(422, "定时任务字段无效，请检查时间、间隔和内容") from exc
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 256
        ):
            raise HarnessError(422, "幂等键须为1至256个字符")
        routes = getattr(self.harness.model_router, "public_routes", lambda: [])()
        if (
            body.model
            and routes
            and body.model
            not in {value for route in routes for value in (route["id"], route["model"])}
        ):
            raise HarnessError(422, "模型未配置，请从可用来源中选择")
        payload_hash = digest(canonical(body.model_dump(mode="json")))
        async with self.store.transaction() as session:
            await self.writer(session, principal)
            existing = await session.scalar(
                select(Schedule).where(
                    *scope(Schedule, principal), Schedule.idempotency_key == idempotency_key
                )
            )
            if existing:
                if existing.payload_hash != payload_hash:
                    raise HarnessError(409, "同一幂等键不能用于不同的计划")
                return schedule_dict(existing)
            live = await session.scalar(
                select(func.count())
                .select_from(Schedule)
                .where(*scope(Schedule, principal), Schedule.status.in_(("active", "paused")))
            )
            if live >= MAX_LIVE_SCHEDULES:
                raise HarnessError(429, "每位用户最多保留100个活动或暂停计划")
            schedule = Schedule(
                id=uid(),
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
                title=body.title,
                message=body.message,
                status="active",
                version=1,
                next_at=body.scheduled_at.timestamp(),
                interval_seconds=body.interval_seconds,
                mode=body.mode,
                model=body.model,
                created=time.time(),
            )
            session.add(schedule)
            return schedule_dict(schedule)

    async def list(self, principal, limit=20, before=None, status=None):
        self.require_ready()
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise HarnessError(422, "分页数量须为1至50")
        if status not in {None, "active", "paused", "cancelled", "completed"}:
            raise HarnessError(422, "计划状态无效")
        query = select(Schedule).where(*scope(Schedule, principal))
        if status:
            query = query.where(Schedule.status == status)
        async with self.store.sessions() as session:
            if before:
                cursor = await session.scalar(query.where(Schedule.id == before))
                if cursor is None:
                    raise HarnessError(404, "分页游标不属于当前私有查询范围")
                query = query.where(
                    or_(
                        Schedule.created < cursor.created,
                        and_(Schedule.created == cursor.created, Schedule.id < cursor.id),
                    )
                )
            items = await session.scalars(
                query.order_by(Schedule.created.desc(), Schedule.id.desc()).limit(limit)
            )
            return [schedule_dict(item) for item in items]

    async def get(self, principal, identifier):
        self.require_ready()
        async with self.store.sessions() as session:
            schedule = await self.store.owned(session, Schedule, identifier, principal)
            items = await session.scalars(
                select(Occurrence)
                .where(Occurrence.schedule_id == identifier, *scope(Occurrence, principal))
                .order_by(Occurrence.scheduled_at.desc(), Occurrence.id.desc())
                .limit(20)
            )
            return {
                **schedule_dict(schedule),
                "occurrences": [occurrence_dict(item) for item in items],
            }

    async def transition(self, principal, identifier, action, expected_version):
        self.require_ready()
        try:
            ScheduleTransition(action=action, expected_version=expected_version)
        except ValidationError as exc:
            raise HarnessError(422, "计划操作或版本无效") from exc
        async with self.store.transaction() as session:
            await self.writer(session, principal)
            schedule = await self.store.owned(session, Schedule, identifier, principal)
            if schedule.version != expected_version:
                raise HarnessError(409, "计划版本已变更，请刷新后重试")
            if schedule.status in {"cancelled", "completed"}:
                raise HarnessError(409, "终态计划不能恢复或修改")
            target = {"pause": "paused", "resume": "active", "cancel": "cancelled"}[action]
            if target == schedule.status:
                raise HarnessError(409, "计划已处于该状态")
            values = {"status": target, "version": expected_version + 1}
            if action == "resume" and schedule.next_at is None:
                submitted = await session.scalar(
                    select(Run.id)
                    .where(
                        *scope(Run, principal), Run.config["schedule_id"].as_string() == identifier
                    )
                    .limit(1)
                )
                pending = await session.scalar(
                    select(Occurrence.id)
                    .where(
                        Occurrence.schedule_id == identifier,
                        Occurrence.status.in_(("pending", "retry", "leased")),
                    )
                    .limit(1)
                )
                if submitted:
                    # 一次性计划已产生运行时，恢复操作不得重复触发第二个运行。
                    values["status"] = "completed"
                elif not pending:
                    last_at = await session.scalar(
                        select(func.max(Occurrence.scheduled_at)).where(
                            Occurrence.schedule_id == identifier
                        )
                    )
                    # 冻结时钟或时钟回拨时也不能复用已终结触发的唯一时间键。
                    values["next_at"] = max(
                        self.clock(), last_at + 0.000001 if last_at is not None else self.clock()
                    )
            changed = await session.execute(
                update(Schedule)
                .where(
                    Schedule.id == identifier,
                    *scope(Schedule, principal),
                    Schedule.version == expected_version,
                    Schedule.status.in_(("active", "paused")),
                )
                .values(**values)
            )
            if not changed.rowcount:
                raise HarnessError(409, "计划版本已变更，请刷新后重试")
            await session.refresh(schedule)
            return schedule_dict(schedule)

    async def gate(self, session, principal, claim):
        """运行创建事务内锁计划、触发和身份；任何预检均不能代替此栅栏。"""
        occurrence_id, lease_owner = claim
        item = await self.store.owned(session, Occurrence, occurrence_id, principal)
        await session.execute(
            update(Schedule)
            .where(Schedule.id == item.schedule_id, *scope(Schedule, principal))
            .values(version=Schedule.version)
        )
        schedule = await self.store.owned(session, Schedule, item.schedule_id, principal)
        await self.require_lease(session, claim)
        principal = await self.writer(session, principal)
        if schedule.status != "active":
            raise HarnessError(409, "schedule_inactive")
        newer = await session.scalar(
            select(Occurrence.id)
            .where(
                Occurrence.schedule_id == item.schedule_id,
                Occurrence.scheduled_at > item.scheduled_at,
            )
            .limit(1)
        )
        next_at = schedule.next_at
        newer_due = (
            schedule.interval_seconds is not None
            and next_at is not None
            and next_at <= self.clock()
            and item.scheduled_at < next_at
        )
        if newer or newer_due:
            raise HarnessError(409, "superseded_by_latest")
        previous = await session.scalar(
            select(Run.id)
            .where(
                *scope(Run, principal),
                Run.status.in_(ACTIVE_STATUSES),
                Run.config["schedule_id"].as_string() == schedule.id,
            )
            .limit(1)
        )
        if previous:
            raise HarnessError(409, "previous_run_active")
        tools = sorted(SCHEDULE_TOOLS & set(self.harness.accessible_tool_names(principal)))
        return principal, tools, schedule.id

    async def require_lease(self, session, claim):
        """必须在取得事务锁之后读取时钟；提交前也复核，不缓存轮询开始时间。"""
        occurrence_id, lease_owner = claim
        locked = await session.execute(
            update(Occurrence)
            .where(
                Occurrence.id == occurrence_id,
                Occurrence.status == "leased",
                Occurrence.lease_owner == lease_owner,
                Occurrence.lease_until > self.clock(),
            )
            .values(lease_owner=Occurrence.lease_owner)
        )
        if not locked.rowcount:
            raise HarnessError(409, "occurrence_lease_lost")

    async def require_recovery(self, session, principal, run, claim):
        """只恢复由同一计划产生的运行，不把手动幂等键碰撞当作调度成果。"""
        item = await self.store.owned(session, Occurrence, claim[0], principal)
        same_schedule = run.config.get("schedule_id") == item.schedule_id
        if not same_schedule or run.idempotency_key != occurrence_key(item):
            raise HarnessError(409, "schedule_run_idempotency_collision")

    async def tick(self, now=None):
        return await self.dispatcher.tick(self.clock() if now is None else now)

    async def claim(self, now):
        await self.dispatcher.materialize(now)
        return await self.dispatcher.claim(now)

    async def dispatch(self, claim, now):
        return await self.dispatcher.dispatch(claim, now)

    def start(self):
        if self.store.schedules_ready:
            self.dispatcher.start()

    async def close(self):
        await self.dispatcher.close()
