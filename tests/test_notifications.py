"""站内通知的事务性、所有者、幂等与分页负例。"""

import pytest

from app.harness import HarnessError
from app.harness.models import Run
from tests.test_harness_store import identity
from tests.test_threads import service_at


@pytest.mark.asyncio
async def test_notifications_commit_with_event_and_private_read(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "notice-owner")
        member = await service.create_user(owner, "notice-member", "test-password", "operator")
        from app.harness.errors import Principal

        other = Principal(member["user_id"], owner.tenant_id, "operator")
        run = await service.create_run(owner, "检查任务", "notice-run")
        async with service.store.transaction() as session:
            stored = await session.get(Run, run["id"])
            service.store.emit(session, stored, "waiting_approval", {"secret": "不应进入通知"})
        result = await service.notifications.list(owner)
        assert len(result) == 1
        assert result[0]["type"] == "waiting_approval" and result[0]["run_id"] == run["id"]
        assert "secret" not in str(result)
        assert await service.notifications.unread(owner) == 1
        assert await service.notifications.list(other) == []
        with pytest.raises(HarnessError) as error:
            await service.notifications.read(other, result[0]["id"])
        assert error.value.status_code == 404
        first = await service.notifications.read(owner, result[0]["id"])
        second = await service.notifications.read(owner, result[0]["id"])
        assert first["read_at"] == second["read_at"]
        assert await service.notifications.unread(owner) == 0
        with pytest.raises(RuntimeError):
            async with service.store.transaction() as session:
                stored = await session.get(Run, run["id"])
                service.store.emit(session, stored, "failed", {})
                raise RuntimeError("回滚")
        assert len(await service.notifications.list(owner)) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_notification_pagination_and_cursor_boundary(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "notice-page-owner")
        other = await identity(service, "notice-page-other")
        for principal in (owner, other):
            run = await service.create_run(principal, "任务通知", "run")
            async with service.store.transaction() as session:
                stored = await session.get(Run, run["id"])
                for _ in range(4):
                    service.store.emit(session, stored, "completed", {})
        first = await service.notifications.list(owner, limit=2)
        second = await service.notifications.list(owner, limit=2, before=first[-1]["id"])
        assert len(first) == len(second) == 2
        assert not set(item["id"] for item in first) & set(item["id"] for item in second)
        foreign = (await service.notifications.list(other))[0]["id"]
        with pytest.raises(HarnessError) as error:
            await service.notifications.list(owner, before=foreign)
        assert error.value.status_code == 404
        for limit in (0, 51, True):
            with pytest.raises(HarnessError) as error:
                await service.notifications.list(owner, limit=limit)
            assert error.value.status_code == 422
    finally:
        await service.close()
