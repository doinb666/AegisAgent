"""旧库只通过明确迁移建表，备份和重复执行不损坏旧数据。"""

import pytest
from sqlalchemy import inspect

from app.harness import HarnessError, HarnessService, HarnessSettings
from app.harness.models import Base
from app.harness.schedule_migration import migrate_schedules
from app.harness.schedule_models import Occurrence, Schedule
from tests.test_harness_store import identity


def test_launcher_routes_schedule_migration_arguments(monkeypatch):
    import app.launcher
    from app.harness import schedule_migration

    received = []
    monkeypatch.setattr(
        "sys.argv", ["aegiscode", "--migrate-schedules", "--data-dir", "example", "--apply"]
    )
    monkeypatch.setattr(schedule_migration, "main", lambda arguments: received.extend(arguments))
    app.launcher.main()
    assert received == ["--data-dir", "example", "--apply"]


@pytest.mark.asyncio
async def test_schedule_migration_preview_backup_repeat_and_old_data(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, database_url=""))
    async with service.store.engine.begin() as connection:
        await connection.run_sync(
            lambda sync: Base.metadata.create_all(
                sync,
                tables=[
                    table
                    for table in Base.metadata.sorted_tables
                    if table.name not in {Schedule.__tablename__, Occurrence.__tablename__}
                ],
            )
        )
    await service.store.initialize()
    try:
        owner = await identity(service, "schedule-legacy")
        run = await service.create_run(owner, "保留旧任务", "old-run")
        asset = await service.put_asset(owner, "memory", "保留旧资产", "真实内容", status="active")
        assert not service.store.schedules_ready
        with pytest.raises(HarnessError) as unavailable:
            await service.schedules.list(owner)
        assert unavailable.value.status_code == 409
        preview = await migrate_schedules(service.settings)
        assert len(preview["pending_tables"]) == 2 and preview["created_tables"] == []
        async with service.store.engine.connect() as connection:
            assert Schedule.__tablename__ not in await connection.run_sync(
                lambda sync: inspect(sync).get_table_names()
            )
        with pytest.raises(ValueError):
            await migrate_schedules(service.settings, True)
        applied = await migrate_schedules(service.settings, True, tmp_path / "schedules.sqlite")
        assert len(applied["created_tables"]) == 2
        assert (tmp_path / "schedules.sqlite").is_file()
        repeated = await migrate_schedules(service.settings, True, tmp_path / "repeat.sqlite")
        assert repeated["pending_tables"] == repeated["created_tables"] == []
        await service.store.initialize()
        assert service.store.schedules_ready
        assert (await service.get_run(owner, run["id"]))["message"] == "保留旧任务"
        assert (await service.get_asset(owner, asset["id"]))["content"] == "真实内容"
    finally:
        await service.close()
