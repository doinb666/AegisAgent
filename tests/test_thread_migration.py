"""旧数据库显式迁移、备份、重复执行和私有范围。"""

import sqlite3

import pytest
from sqlalchemy import inspect, select

from app.harness import HarnessService, HarnessSettings
from app.harness.models import Base, Run, Thread, ThreadRun
from app.harness.thread_migration import migrate_threads
from tests.test_harness_store import identity


async def legacy_service(tmp_path):
    settings = HarnessSettings(data_dir=tmp_path, database_url="", max_user_runs=32)
    service = HarnessService(settings)
    async with service.store.engine.begin() as connection:
        tables = [
            table
            for table in Base.metadata.sorted_tables
            if table.name not in {Thread.__tablename__, ThreadRun.__tablename__}
        ]
        await connection.run_sync(lambda sync: Base.metadata.create_all(sync, tables=tables))
    await service.store.initialize()
    assert not service.store.threads_ready
    return service


@pytest.mark.asyncio
async def test_explicit_backup_and_idempotent_legacy_migration(tmp_path):
    service = await legacy_service(tmp_path)
    try:
        owner = await identity(service, "legacy-owner")
        other = await identity(service, "legacy-other")
        project = await service.put_asset(owner, "project", "旧项目", "约束", status="active")
        first = await service.create_run(owner, "历史1", "one", session_id=project["id"])
        second = await service.create_run(owner, "历史2", "two", session_id=project["id"])
        other_run = await service.create_run(other, "其他账号", "other", session_id="shared-legacy")
        before = await migrate_threads(service.settings)
        assert before["top_runs"] == 3 and before["session_groups"] == 2
        assert before["pending_threads"] == 2 and before["pending_links"] == 3
        async with service.store.engine.connect() as connection:
            assert Thread.__tablename__ not in await connection.run_sync(
                lambda sync: inspect(sync).get_table_names()
            )
        backup = tmp_path / "before-migration.sqlite"
        result = await migrate_threads(service.settings, apply=True, backup_path=backup)
        assert result["created_threads"] == 2 and result["linked_runs"] == 3
        assert backup.is_file()
        with sqlite3.connect(backup) as database:
            assert database.execute("SELECT count(*) FROM harness_runs").fetchone()[0] == 3
            assert (
                database.execute(
                    "SELECT count(*) FROM sqlite_master WHERE name='harness_threads'"
                ).fetchone()[0]
                == 0
            )
        await service.store.initialize()
        assert service.store.threads_ready
        owner_threads = await service.threads.list(owner)
        assert len(owner_threads) == 1 and owner_threads[0]["project_id"] == project["id"]
        assert owner_threads[0]["latest_run_id"] == second["id"]
        async with service.store.sessions() as session:
            assert len((await session.scalars(select(Run))).all()) == 3
            relation = await session.get(ThreadRun, other_run["id"])
            assert relation.owner_id == other.user_id
            run = await session.get(Run, first["id"])
            assert run.message == "历史1" and run.session_id == project["id"]
            assert run.config["thread_id"] == owner_threads[0]["id"]
        repeated = await migrate_threads(
            service.settings, apply=True, backup_path=tmp_path / "second.sqlite"
        )
        assert repeated["created_threads"] == repeated["linked_runs"] == 0
        assert repeated["pending_threads"] == repeated["pending_links"] == 0
        assert [row["id"] for row in await service.threads.list(owner)] == [
            row["id"] for row in owner_threads
        ]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_migration_excludes_children_and_same_session_other_owner(tmp_path):
    service = await legacy_service(tmp_path)
    try:
        owner = await identity(service, "split-owner")
        member = await service.create_user(owner, "split-member", "test-password", "operator")
        from app.harness.errors import Principal

        other = Principal(member["user_id"], owner.tenant_id, "operator")
        first = await service.create_run(owner, "本人", "one", session_id="shared")
        second = await service.create_run(other, "成员", "two", session_id="member-session")
        child = await service.create_run(owner, "子任务", "child", session_id="child")
        async with service.store.sessions.begin() as session:
            # 直接构造历史导入数据；当前API本来就拒绝借用别人的session。
            (await session.get(Run, second["id"])).session_id = "shared"
            run = await session.get(Run, child["id"])
            run.parent_run_id = first["id"]
        result = await migrate_threads(service.settings, True, tmp_path / "scoped.sqlite")
        assert result["created_threads"] == result["linked_runs"] == 2
        async with service.store.sessions() as session:
            links = {
                item.run_id: item.thread_id
                for item in (await session.scalars(select(ThreadRun))).all()
            }
            assert links[first["id"]] != links[second["id"]]
            assert child["id"] not in links
            assert (await session.get(Run, child["id"])).config.get("thread_id") is None
        async with service.store.engine.connect() as connection:
            indexes = await connection.run_sync(
                lambda sync: inspect(sync).get_indexes(Run.__tablename__)
            )
            assert any(index["name"] == "ix_run_owner_session_page" for index in indexes)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_conflicting_link_rolls_back_run_config(tmp_path):
    service = await legacy_service(tmp_path)
    try:
        owner = await identity(service, "rollback-owner")
        run = await service.create_run(owner, "原内容", "run", session_id="session")
        await migrate_threads(service.settings, True, tmp_path / "first.sqlite")
        async with service.store.sessions.begin() as session:
            link = await session.get(ThreadRun, run["id"])
            link.thread_id = "invalid-thread"
            stored = await session.get(Run, run["id"])
            stored.config = {"sentinel": "不修改"}
        with pytest.raises(ValueError, match="冲突"):
            await migrate_threads(service.settings, True, tmp_path / "rollback.sqlite")
        async with service.store.sessions() as session:
            assert (await session.get(Run, run["id"])).config == {"sentinel": "不修改"}
            assert (await session.get(Run, run["id"])).message == "原内容"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_migration_requires_new_backup_target(tmp_path):
    service = await legacy_service(tmp_path)
    try:
        with pytest.raises(ValueError):
            await migrate_threads(service.settings, apply=True)
        backup = tmp_path / "existing.sqlite"
        backup.touch()
        with pytest.raises(FileExistsError):
            await migrate_threads(service.settings, apply=True, backup_path=backup)
        assert not service.store.threads_ready
    finally:
        await service.close()
