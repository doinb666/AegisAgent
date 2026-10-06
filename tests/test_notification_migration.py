"""通知旧事件补齐只增不改，幂等且保留来源事件。"""

import pytest
from sqlalchemy import inspect, select

from app.harness import HarnessService, HarnessSettings
from app.harness.models import Base, Notification, Run
from app.harness.notification_migration import migrate_notifications
from tests.test_harness_store import identity


@pytest.mark.asyncio
async def test_notice_migration_preview_backup_and_repeat(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, database_url=""))
    async with service.store.engine.begin() as connection:
        await connection.run_sync(
            lambda sync: Base.metadata.create_all(
                sync,
                tables=[
                    table
                    for table in Base.metadata.sorted_tables
                    if table.name != Notification.__tablename__
                ],
            )
        )
    await service.store.initialize()
    try:
        assert not service.store.notifications_ready
        owner = await identity(service, "notice-legacy-owner")
        run = await service.create_run(owner, "旧任务", "legacy")
        async with service.store.transaction() as session:
            stored = await session.get(Run, run["id"])
            service.store.emit(session, stored, "waiting_approval", {"sensitive": "不复制"})
            service.store.emit(session, stored, "completed", {})
        preview = await migrate_notifications(service.settings)
        assert preview["pending_notifications"] == 2
        async with service.store.engine.connect() as connection:
            assert Notification.__tablename__ not in await connection.run_sync(
                lambda sync: inspect(sync).get_table_names()
            )
        with pytest.raises(ValueError):
            await migrate_notifications(service.settings, True)
        applied = await migrate_notifications(service.settings, True, tmp_path / "notices.sqlite")
        assert applied["created_notifications"] == 2
        await service.store.initialize()
        notifications = await service.notifications.list(owner)
        assert len(notifications) == 2 and "sensitive" not in str(notifications)
        await service.notifications.read(owner, notifications[0]["id"])
        repeated = await migrate_notifications(service.settings, True, tmp_path / "repeat.sqlite")
        assert repeated["created_notifications"] == repeated["pending_notifications"] == 0
        assert await service.notifications.unread(owner) == 1
        async with service.store.sessions() as session:
            assert len((await session.scalars(select(Notification))).all()) == 2
            assert (await session.get(Run, run["id"])).message == "旧任务"
    finally:
        await service.close()
