"""真实数据库上的定时触发、私有范围、租约与事务栅栏。"""

import asyncio
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.models import Run, User
from app.harness.schedule_models import Occurrence, Schedule


def timestamp(value):
    return datetime.fromtimestamp(value, UTC).isoformat()


class ControlledClock:
    def __init__(self, now=1000):
        self.now = now

    def __call__(self):
        return self.now


async def tick(service, now):
    service.schedules.clock.now = now
    return await service.schedules.tick()


async def setup_service(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, database_url="", max_user_runs=32))
    service.schedules.clock = ControlledClock()
    await service.store.initialize()
    user = await service.register("schedule-owner", "password123")
    return service, Principal(user["user_id"], user["tenant_id"], user["role"])


async def create(service, owner, key="schedule", **values):
    return await service.schedules.create(
        owner, key, title="只读计算", message="计算 1+1", scheduled_at=timestamp(1000), **values
    )


@pytest.mark.asyncio
async def test_private_idempotency_and_version(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        assert (await create(service, owner))["id"] == schedule["id"]
        with pytest.raises(HarnessError) as conflict:
            await create(service, owner, interval_seconds=60)
        assert conflict.value.status_code == 409
        member = await service.create_user(owner, "schedule-peer", "password123", "operator")
        peer = Principal(member["user_id"], member["tenant_id"], "operator")
        assert await service.schedules.list(peer) == []
        with pytest.raises(HarnessError) as missing:
            await service.schedules.get(peer, schedule["id"])
        assert missing.value.status_code == 404
        paused = await service.schedules.transition(owner, schedule["id"], "pause", 1)
        assert paused["status"] == "paused" and paused["version"] == 2
        with pytest.raises(HarnessError):
            await service.schedules.transition(owner, schedule["id"], "resume", 1)
        cancelled = await service.schedules.transition(owner, schedule["id"], "cancel", 2)
        with pytest.raises(HarnessError):
            await service.schedules.transition(
                owner, schedule["id"], "resume", cancelled["version"]
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_two_instances_latest_only_overlap_and_restart(tmp_path):
    first, owner = await setup_service(tmp_path)
    second = HarnessService(first.settings)
    second.schedules.clock = first.schedules.clock
    await second.store.initialize()
    try:
        schedule = await create(first, owner, interval_seconds=60)
        await asyncio.gather(tick(first, 1601), tick(second, 1601))
        detail = await first.schedules.get(owner, schedule["id"])
        assert len(detail["occurrences"]) == 1
        occurrence = detail["occurrences"][0]
        assert occurrence["scheduled_at"] == timestamp(1600)
        assert occurrence["run_id"]
        async with first.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 1
        await tick(second, 1660)
        detail = await first.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "skipped"
        assert detail["occurrences"][0]["reason"] == "previous_run_active"
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_current_identity_and_paused_creation_gate(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        claim = await service.schedules.claim(1000)
        assert claim
        await service.schedules.transition(owner, schedule["id"], "pause", 1)
        await service.schedules.dispatch(claim, 1000)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["run_id"] is None
        assert detail["occurrences"][0]["reason"] == "schedule_inactive"
        service.schedules.clock.now = 1060
        await service.schedules.transition(owner, schedule["id"], "resume", 2)
        async with service.store.transaction() as session:
            await session.execute(
                update(User).where(User.id == owner.user_id).values(role="viewer")
            )
        await tick(service, 1060)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["reason"] == "identity_permission_revoked"
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_commit_before_ack_recovers_same_run_even_after_cancel(tmp_path):
    service, owner = await setup_service(tmp_path)
    recovered = HarnessService(service.settings)
    recovered.schedules.clock = service.schedules.clock
    await recovered.store.initialize()
    try:
        schedule = await create(service, owner)
        claim = await service.schedules.claim(1000)
        original_ack = service.schedules.dispatcher.ack

        async def disconnected_ack(*args, **kwargs):
            raise OperationalError("ack", {}, RuntimeError("模拟确认连接断开"))

        service.schedules.dispatcher.ack = disconnected_ack
        with pytest.raises(OperationalError):
            await service.schedules.dispatch(claim, 1000)
        service.schedules.dispatcher.ack = original_ack
        await service.schedules.transition(owner, schedule["id"], "cancel", 1)
        await tick(recovered, 1061)
        detail = await recovered.schedules.get(owner, schedule["id"])
        assert detail["status"] == "cancelled"
        assert detail["occurrences"][0]["status"] == "submitted"
        async with recovered.store.sessions() as session:
            runs = (await session.scalars(select(Run))).all()
            assert len(runs) == 1
            assert runs[0].id == detail["occurrences"][0]["run_id"]
            assert runs[0].status == "queued"
        # 终态触发在过期租约轮询中不会再次被领取。
        assert await tick(recovered, 1200) == 0
    finally:
        await service.close()
        await recovered.close()


@pytest.mark.asyncio
async def test_quota_retry_backoff_is_bounded_and_audited(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        service.settings.max_user_runs = 1
        await service.create_run(owner, "占用活动配额", "occupy")
        schedule = await create(service, owner)
        assert await tick(service, 1000) == 1
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "retry"
        assert detail["occurrences"][0]["reason"] == "user_run_quota"
        assert await tick(service, 1001) == 0
        for now in (1002, 1006, 1014, 1030):
            assert await tick(service, now) == 1
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "failed"
        assert await tick(service, 2000) == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_database_error_retry_uses_same_occurrence(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        original = service.create_run

        async def unavailable(*args, **kwargs):
            raise OperationalError("create", {}, RuntimeError("模拟瞬时断开"))

        service.create_run = unavailable
        await tick(service, 1000)
        assert (await service.schedules.get(owner, schedule["id"]))["occurrences"][0][
            "reason"
        ] == "database_temporarily_unavailable"
        service.create_run = original
        await tick(service, 1002)
        detail = await service.schedules.get(owner, schedule["id"])
        assert len(detail["occurrences"]) == 1
        assert detail["occurrences"][0]["run_id"]
        assert detail["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_management_wins_before_transaction_gate(tmp_path, action):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        entered, release = asyncio.Event(), asyncio.Event()
        original = service.create_run

        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        service.create_run = delayed
        task = asyncio.create_task(tick(service, 1000))
        await asyncio.wait_for(entered.wait(), timeout=2)
        await service.schedules.transition(owner, schedule["id"], action, 1)
        release.set()
        await asyncio.wait_for(task, timeout=2)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["reason"] == "schedule_inactive"
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_creation_wins_gate_then_cancel_preserves_run(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        entered, release = asyncio.Event(), asyncio.Event()
        original = service.recall

        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        service.recall = delayed
        task = asyncio.create_task(tick(service, 1000))
        await asyncio.wait_for(entered.wait(), timeout=2)
        cancel = asyncio.create_task(
            service.schedules.transition(owner, schedule["id"], "cancel", 1)
        )
        assert service.store.write_lock.locked()
        release.set()
        await asyncio.wait_for(asyncio.gather(task, cancel), timeout=3)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["status"] == "cancelled"
        async with service.store.sessions() as session:
            run = await session.get(Run, detail["occurrences"][0]["run_id"])
            assert run.status == "queued"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stable_pagination_cursor_and_live_schedule_limit(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedules = [await create(service, owner, key=str(index)) for index in range(100)]
        async with service.store.transaction() as session:
            await session.execute(update(Schedule).values(created=500))
        items, cursor = [], None
        while True:
            page = await service.schedules.list(owner, limit=20, before=cursor)
            if not page:
                break
            items.extend(item["id"] for item in page)
            cursor = page[-1]["id"]
        assert len(items) == len(set(items)) == 100
        assert items == sorted(items, reverse=True)
        with pytest.raises(HarnessError) as full:
            await create(service, owner, key="over-limit")
        assert full.value.status_code == 429
        paused = await service.schedules.transition(owner, schedules[0]["id"], "pause", 1)
        with pytest.raises(HarnessError) as wrong_scope:
            await service.schedules.list(owner, before=paused["id"], status="active")
        assert wrong_scope.value.status_code == 404
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_once_resume_after_committed_run_never_creates_second_run(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        claim = await service.schedules.claim(1000)
        from app.harness.schedules import occurrence_key

        async with service.store.sessions() as session:
            item = await session.get(Occurrence, claim[0])
        await service.create_run(
            owner,
            "计算 1+1",
            occurrence_key(item),
            allowed_tools=[],
            max_steps=service.settings.max_steps,
            _schedule_gate=claim,
        )
        await service.schedules.transition(owner, schedule["id"], "pause", 1)
        resumed = await service.schedules.transition(owner, schedule["id"], "resume", 2)
        assert resumed["status"] == "completed" and resumed["next_at"] is None
        await tick(service, 1061)
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 1
        assert (await service.schedules.get(owner, schedule["id"]))["occurrences"][0]["run_id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_manual_run_cannot_impersonate_schedule_recovery(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        claim = await service.schedules.claim(1000)
        from app.harness.schedules import occurrence_key

        async with service.store.sessions() as session:
            item = await session.get(Occurrence, claim[0])
        await service.create_run(owner, "恶意占用触发键", occurrence_key(item))
        await service.schedules.dispatch(claim, 1000)
        occurrence = (await service.schedules.get(owner, schedule["id"]))["occurrences"][0]
        assert occurrence["run_id"] is None
        assert occurrence["reason"] == "schedule_run_idempotency_collision"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_expired_lease_after_waiting_for_store_lock_cannot_create_or_ack(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        claim = await service.schedules.claim(1000)
        entered = asyncio.Event()
        original = service.create_run

        async def observed(*args, **kwargs):
            entered.set()
            return await original(*args, **kwargs)

        service.create_run = observed
        async with service.store.write_lock:
            task = asyncio.create_task(service.schedules.dispatch(claim, 1000))
            await asyncio.wait_for(entered.wait(), timeout=2)
            service.schedules.clock.now = 1061
        await asyncio.wait_for(task, timeout=2)
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 0
            item = await session.get(Occurrence, claim[0])
            assert item.status == "leased" and item.run_id is None and item.attempts == 0
        # 重领新租约后，旧令牌既不能覆盖新令牌，也不能把账本写成终态。
        new_claim = await service.schedules.dispatcher.claim(1061)
        assert new_claim[0] == claim[0] and new_claim[1] != claim[1]
        await service.schedules.dispatcher.ack(claim, 1000, run_id="旧令牌伪造的运行")
        async with service.store.sessions() as session:
            item = await session.get(Occurrence, claim[0])
            assert item.status == "leased" and item.lease_owner == new_claim[1]
        await service.schedules.dispatch(new_claim, 1061)
        assert (await service.schedules.get(owner, schedule["id"]))["occurrences"][0]["run_id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_lease_expiry_during_recall_rolls_back_run_before_commit(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        entered, release = asyncio.Event(), asyncio.Event()
        original = service.recall

        async def stalled(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        service.recall = stalled
        claim = await service.schedules.claim(1000)
        task = asyncio.create_task(service.schedules.dispatch(claim, 1000))
        await asyncio.wait_for(entered.wait(), timeout=2)
        service.schedules.clock.now = 1061
        release.set()
        await asyncio.wait_for(task, timeout=2)
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 0
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "leased"
        assert detail["occurrences"][0]["run_id"] is None
        service.recall = original
        await tick(service, 1061)
        assert (await service.schedules.get(owner, schedule["id"]))["occurrences"][0]["run_id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_expired_ack_after_run_commit_waits_for_reclaim_and_links_original(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        original = service.create_run

        async def elapsed_after_commit(*args, **kwargs):
            run = await original(*args, **kwargs)
            service.schedules.clock.now = 1061
            return run

        service.create_run = elapsed_after_commit
        claim = await service.schedules.claim(1000)
        await service.schedules.dispatch(claim, 1000)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "leased"
        assert detail["occurrences"][0]["run_id"] is None
        service.create_run = original
        await tick(service, 1061)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "submitted"
        async with service.store.sessions() as session:
            runs = (await session.scalars(select(Run))).all()
            assert len(runs) == 1 and runs[0].id == detail["occurrences"][0]["run_id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_restart_supersedes_uncommitted_old_trigger_and_runs_latest_only(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner, interval_seconds=60)
        old_claim = await service.schedules.claim(1000)
        assert old_claim
        await tick(service, 1601)
        detail = await service.schedules.get(owner, schedule["id"])
        latest, old = detail["occurrences"]
        assert latest["scheduled_at"] == timestamp(1600) and latest["run_id"]
        assert old["status"] == "skipped" and old["reason"] == "superseded_by_latest"
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_restart_preserves_already_committed_old_run_and_skips_overlap(tmp_path):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner, interval_seconds=60)
        old_claim = await service.schedules.claim(1000)
        original_ack = service.schedules.dispatcher.ack

        async def crashed_ack(*args, **kwargs):
            raise OperationalError("ack", {}, RuntimeError("模拟确认前崩溃"))

        service.schedules.dispatcher.ack = crashed_ack
        with pytest.raises(OperationalError):
            await service.schedules.dispatch(old_claim, 1000)
        service.schedules.dispatcher.ack = original_ack
        await tick(service, 1601)
        latest, old = (await service.schedules.get(owner, schedule["id"]))["occurrences"]
        assert old["status"] == "submitted" and old["run_id"]
        assert latest["status"] == "skipped" and latest["reason"] == "previous_run_active"
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_real_dispatcher_loop_survives_unknown_failure_and_closes_promptly(tmp_path, caplog):
    service, owner = await setup_service(tmp_path)
    try:
        schedule = await create(service, owner)
        dispatcher = service.schedules.dispatcher
        original = service.recall
        attempts = 0
        first_pause, release, recovered_pause = asyncio.Event(), asyncio.Event(), asyncio.Event()
        forever = asyncio.Event()
        delays = []

        async def flaky_recall(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("不得出现在日志的敏感异常原文")
            return await original(*args, **kwargs)

        async def controlled_pause(seconds):
            delays.append(seconds)
            if len(delays) == 1:
                first_pause.set()
                await release.wait()
            else:
                recovered_pause.set()
                await forever.wait()

        service.recall = flaky_recall
        dispatcher.pause = controlled_pause
        dispatcher.start()
        await asyncio.wait_for(first_pause.wait(), timeout=2)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "leased"
        assert detail["occurrences"][0]["run_id"] is None
        assert not dispatcher.task.done()
        assert "RuntimeError" in caplog.text
        assert "敏感异常原文" not in caplog.text
        assert 1 <= delays[0] <= 30
        service.schedules.clock.now = 1061
        release.set()
        await asyncio.wait_for(recovered_pause.wait(), timeout=2)
        detail = await service.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "submitted"
        assert detail["occurrences"][0]["run_id"]
        assert len(detail["occurrences"]) == 1 and attempts == 2
        assert delays[1] == 1
        await asyncio.wait_for(dispatcher.close(), timeout=0.5)
        assert dispatcher.task is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_recovery_keeps_original_run_when_default_budget_changes(tmp_path):
    service, owner = await setup_service(tmp_path)
    recovered = None
    try:
        schedule = await create(service, owner)
        original_ack = service.schedules.dispatcher.ack

        async def crashed_ack(*args, **kwargs):
            raise OperationalError("ack", {}, RuntimeError("模拟确认前崩溃"))

        service.schedules.dispatcher.ack = crashed_ack
        with pytest.raises(OperationalError):
            await tick(service, 1000)
        service.schedules.dispatcher.ack = original_ack
        service.settings.max_steps = 11
        recovered = HarnessService(service.settings)
        recovered.schedules.clock = service.schedules.clock
        await recovered.store.initialize()
        await tick(recovered, 1061)
        detail = await recovered.schedules.get(owner, schedule["id"])
        assert detail["occurrences"][0]["status"] == "submitted"
        async with recovered.store.sessions() as session:
            runs = (await session.scalars(select(Run))).all()
            assert len(runs) == 1 and runs[0].config["max_steps"] == 10
            assert runs[0].id == detail["occurrences"][0]["run_id"]
    finally:
        await service.close()
        if recovered:
            await recovered.close()
