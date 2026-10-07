"""有界调度轮询、数据库租约、确定幂等键及退避账本。"""

import asyncio
import logging

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import SQLAlchemyError

from .errors import HarnessError, Principal
from .schedule_models import Occurrence, Schedule
from .store import uid

logger = logging.getLogger(__name__)
BATCH_SIZE = 20
MAX_ATTEMPTS = 5
LEASE_SECONDS = 60


class ScheduleDispatcher:
    def __init__(self, schedules):
        self.schedules = schedules
        self.store = schedules.store
        self.harness = schedules.harness
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.loop())

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None

    async def loop(self):
        failures = 0
        while not self.harness.closing:
            try:
                await self.tick(self.schedules.clock())
                failures = 0
            except Exception as exc:
                failures += 1
                # 未知异常保留 leased 账本等租约恢复；不打印异常原文或潜在凭据。
                logger.warning("定时任务轮询暂时失败：%s", type(exc).__name__)
            await self.pause(min(30, 1 + failures * 2))

    async def pause(self, seconds):
        await asyncio.sleep(seconds)

    async def materialize(self, now):
        """停机只追赶每个计划最近一个时间点；不为错过的历史逐条造任务。"""
        if not self.store.schedules_ready:
            return
        async with self.store.transaction() as session:
            due = (
                await session.scalars(
                    select(Schedule)
                    .where(
                        Schedule.status == "active",
                        Schedule.next_at <= now,
                    )
                    .order_by(Schedule.next_at, Schedule.id)
                    .limit(BATCH_SIZE)
                )
            ).all()
            for schedule in due:
                locked = await session.execute(
                    update(Schedule)
                    .where(
                        Schedule.id == schedule.id,
                        Schedule.status == "active",
                        Schedule.next_at <= now,
                    )
                    .values(version=Schedule.version)
                )
                if not locked.rowcount:
                    continue
                await session.refresh(schedule)
                scheduled_at = schedule.next_at
                if schedule.interval_seconds:
                    scheduled_at += (
                        (now - scheduled_at) // schedule.interval_seconds
                    ) * schedule.interval_seconds
                exists = await session.scalar(
                    select(Occurrence.id).where(
                        Occurrence.schedule_id == schedule.id,
                        Occurrence.scheduled_at == scheduled_at,
                    )
                )
                if not exists:
                    session.add(
                        Occurrence(
                            id=uid(),
                            schedule_id=schedule.id,
                            tenant_id=schedule.tenant_id,
                            owner_id=schedule.owner_id,
                            scheduled_at=scheduled_at,
                            status="pending",
                            attempts=0,
                            retry_at=now,
                        )
                    )
                schedule.next_at = (
                    scheduled_at + schedule.interval_seconds if schedule.interval_seconds else None
                )

    async def claim(self, now):
        """CAS 可被不同 Store 争抢；只领取非终态，失联租约使用原账本恢复。"""
        async with self.store.transaction() as session:
            now = self.schedules.clock()
            eligible = or_(
                and_(Occurrence.status.in_(("pending", "retry")), Occurrence.retry_at <= now),
                and_(Occurrence.status == "leased", Occurrence.lease_until <= now),
            )
            candidates = (
                await session.scalars(
                    select(Occurrence.id)
                    .where(eligible)
                    .order_by(Occurrence.scheduled_at, Occurrence.id)
                    .limit(BATCH_SIZE)
                )
            ).all()
            for identifier in candidates:
                owner = uid()
                updated = await session.execute(
                    update(Occurrence)
                    .where(
                        Occurrence.id == identifier,
                        eligible,
                    )
                    .values(status="leased", lease_owner=owner, lease_until=now + LEASE_SECONDS)
                )
                if updated.rowcount:
                    return identifier, owner
        return None

    async def tick(self, now):
        if not self.store.schedules_ready:
            return 0
        await self.materialize(now)
        count = 0
        for _ in range(BATCH_SIZE):
            claim = await self.claim(now)
            if claim is None:
                break
            await self.dispatch(claim, now)
            count += 1
        return count

    async def dispatch(self, claim, now):
        from .schedules import occurrence_key

        identifier, lease_owner = claim
        async with self.store.sessions() as session:
            item = await session.get(Occurrence, identifier)
            if item is None or item.status != "leased" or item.lease_owner != lease_owner:
                return
            schedule = await session.get(Schedule, item.schedule_id)
        principal = Principal(item.owner_id, item.tenant_id, "operator")
        try:
            run = await self.harness.create_run(
                principal,
                schedule.message,
                occurrence_key(item),
                mode=schedule.mode,
                model=schedule.model,
                allowed_tools=[],
                max_steps=self.harness.settings.max_steps,
                _schedule_gate=claim,
            )
        except HarnessError as exc:
            reason = exc.detail
            if exc.status_code == 429:
                reason = "user_run_quota"
            elif exc.status_code == 403:
                reason = "identity_permission_revoked"
            await self.ack(claim, now, reason=reason, retry=exc.status_code == 429)
        except SQLAlchemyError:
            # 若提交已成功但客户端收到错误，恢复时先用确定幂等键关联原运行。
            await self.ack(claim, now, reason="database_temporarily_unavailable", retry=True)
        else:
            await self.ack(claim, now, run_id=run["id"])

    async def ack(self, claim, now, run_id=None, reason=None, retry=False):
        identifier, lease_owner = claim
        async with self.store.transaction() as session:
            item = await session.get(Occurrence, identifier)
            if item is None:
                return
            # ACK 与创建栅栏保持 Schedule → Occurrence 锁顺序，避免 PostgreSQL 互锁。
            await session.execute(
                update(Schedule)
                .where(Schedule.id == item.schedule_id)
                .values(version=Schedule.version)
            )
            await session.refresh(item)
            now = self.schedules.clock()
            # 租约令牌变化时旧调度者不得覆盖另一调度者的终态。
            if (
                item.status != "leased"
                or item.lease_owner != lease_owner
                or item.lease_until <= now
            ):
                return
            attempts = item.attempts + 1
            status = "submitted" if run_id else "skipped"
            if retry:
                status = "retry" if attempts < MAX_ATTEMPTS else "failed"
            changed = await session.execute(
                update(Occurrence)
                .where(
                    Occurrence.id == identifier,
                    Occurrence.status == "leased",
                    Occurrence.lease_owner == lease_owner,
                    Occurrence.lease_until > now,
                )
                .values(
                    status=status,
                    reason=reason,
                    run_id=run_id,
                    attempts=attempts,
                    retry_at=now + min(300, 2**attempts),
                    lease_owner=None,
                    lease_until=None,
                )
            )
            if changed.rowcount and status in {"submitted", "skipped", "failed"}:
                # 不覆盖并发暂停或取消，也不更改已有运行的状态。
                await session.execute(
                    update(Schedule)
                    .where(
                        Schedule.id == item.schedule_id,
                        Schedule.status == "active",
                        Schedule.interval_seconds.is_(None),
                        Schedule.next_at.is_(None),
                    )
                    .values(status="completed", version=Schedule.version + 1)
                )
