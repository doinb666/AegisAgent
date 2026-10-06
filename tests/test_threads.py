"""项目与会话真实事务：上下文、所有者、归档和分页边界。"""

import asyncio

import pytest
from sqlalchemy import select

from app.harness import HarnessError, HarnessService, HarnessSettings
from app.harness.errors import Principal
from app.harness.models import Run
from tests.test_harness_store import identity


async def service_at(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, database_url="", max_user_runs=32))
    await service.store.initialize()
    return service


@pytest.mark.asyncio
async def test_two_threads_in_one_project_have_separate_context(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "threads-owner")
        project = await service.put_asset(
            owner, "project", "项目A", "禁止改动生产配置", status="active"
        )
        first = await service.threads.create(owner, "检查边界", project["id"])
        second = await service.threads.create(owner, "检查性能", project["id"])
        assert first["id"] != second["id"]
        run = await service.create_run(owner, "原会话任务", "first", thread_id=first["id"])
        async with service.store.sessions.begin() as session:
            stored = await session.get(Run, run["id"])
            stored.status = "completed"
            stored.answer = "仅第一会话可见的结论"
        resumed = await service.create_run(owner, "继续第一会话", "resume", thread_id=first["id"])
        separate = await service.create_run(owner, "第二会话", "second", thread_id=second["id"])
        assert run["thread_id"] == first["id"]
        assert resumed["session_id"] == first["id"]
        async with service.store.sessions() as session:
            resumed_context = str((await session.get(Run, resumed["id"])).messages)
            separate_context = str((await session.get(Run, separate["id"])).messages)
        assert "仅第一会话可见的结论" in resumed_context
        assert "仅第一会话可见的结论" not in separate_context
        assert "禁止改动生产配置" in separate_context
        assert (await service.threads.get(owner, first["id"]))["latest_run_id"] == resumed["id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_thread_private_scope_archival_and_idempotent_retry(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "thread-owner")
        other = await identity(service, "thread-other")
        thread = await service.threads.create(owner, "私有会话")
        run = await service.create_run(owner, "测试", "stable", thread_id=thread["id"])
        for action in (
            lambda: service.threads.get(other, thread["id"]),
            lambda: service.threads.update(other, thread["id"], "侵入", False),
            lambda: service.create_run(other, "侵入", "foreign", thread_id=thread["id"]),
        ):
            with pytest.raises(HarnessError) as error:
                await action()
            assert error.value.status_code == 404
        assert await service.threads.list(other) == []
        updated = await service.threads.update(owner, thread["id"], "已整理", True)
        assert updated["archived"] and updated["title"] == "已整理"
        assert await service.threads.list(owner) == []
        assert len(await service.threads.list(owner, archived=True)) == 1
        assert (await service.create_run(owner, "测试", "stable", thread_id=thread["id"]))[
            "id"
        ] == run["id"]
        with pytest.raises(HarnessError) as error:
            await service.create_run(owner, "新任务", "archived", thread_id=thread["id"])
        assert error.value.status_code == 409
        await service.threads.update(owner, thread["id"], archived=False)
        assert (await service.get_run(owner, run["id"]))["message"] == "测试"
        with pytest.raises(HarnessError) as error:
            await service.create_run(
                owner, "无效", "mismatch", thread_id=thread["id"], session_id="wrong"
            )
        assert error.value.status_code == 422
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_project_state_and_bounded_query(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "query-owner")
        draft = await service.put_asset(owner, "project", "草稿", "未审核")
        with pytest.raises(HarnessError) as error:
            await service.threads.create(owner, "新会话", draft["id"])
        assert error.value.status_code == 409
        project = await service.put_asset(owner, "project", "启用项目", "约束", status="active")
        for index in range(8):
            await service.threads.create(owner, f"会话{index}", project["id"])
        separate = await service.threads.create(owner, "独立会话")
        first = await service.threads.list(owner, project_id=project["id"], limit=3)
        second = await service.threads.list(
            owner, project_id=project["id"], limit=3, before=first[-1]["id"]
        )
        assert len(first) == len(second) == 3
        assert not {row["id"] for row in first} & {row["id"] for row in second}
        for limit in (0, 51, True):
            with pytest.raises(HarnessError) as error:
                await service.threads.list(owner, limit=limit)
            assert error.value.status_code == 422
        with pytest.raises(HarnessError) as error:
            await service.threads.list(owner, project_id=project["id"], before=separate["id"])
        assert error.value.status_code == 404
        with pytest.raises(HarnessError):
            await service.threads.create(owner, " " * 2)
        async with service.store.sessions() as session:
            assert (await session.scalars(select(Run))).all() == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_same_tenant_is_private_and_viewer_cannot_manage(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "scope-owner")
        member = await service.create_user(owner, "scope-member", "test-password", "operator")
        other = Principal(member["user_id"], owner.tenant_id, "operator")
        viewer = Principal(owner.user_id, owner.tenant_id, "viewer")
        project = await service.put_asset(
            owner, "project", "共享名字私有项目", "约束", status="active"
        )
        thread = await service.threads.create(owner, "空会话", project["id"])
        assert await service.threads.projects(other) == []
        for action in (
            lambda: service.threads.create(other, "侵入", project["id"]),
            lambda: service.threads.get(other, thread["id"]),
            lambda: service.create_run(other, "侵入", "foreign-session", session_id=thread["id"]),
        ):
            with pytest.raises(HarnessError) as error:
                await action()
            assert error.value.status_code == 404
        assert (await service.threads.get(viewer, thread["id"]))["id"] == thread["id"]
        for action in (
            lambda: service.threads.create(viewer, "新会话"),
            lambda: service.threads.update(viewer, thread["id"], archived=True),
        ):
            with pytest.raises(HarnessError) as error:
                await action()
            assert error.value.status_code == 403
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cross_store_thread_idempotency_and_archive_creation_race(tmp_path):
    first = await service_at(tmp_path)
    second = await service_at(tmp_path)
    try:
        owner = await identity(first, "race-owner")
        results = await asyncio.gather(
            first.threads.create(owner, "重试会话", request_key="stable"),
            second.threads.create(owner, "重试会话", request_key="stable"),
        )
        thread_id = results[0]["id"]
        assert results[1]["id"] == thread_id
        assert len(await first.threads.list(owner)) == 1
        with pytest.raises(HarnessError) as error:
            await second.threads.create(owner, "不同请求", request_key="stable")
        assert error.value.status_code == 409
        race = await asyncio.gather(
            first.threads.update(owner, thread_id, archived=True),
            second.create_run(owner, "竞态任务", "race", thread_id=thread_id),
            return_exceptions=True,
        )
        assert not isinstance(race[0], Exception)
        if isinstance(race[1], Exception):
            assert isinstance(race[1], HarnessError) and race[1].status_code == 409
        else:
            assert race[1]["thread_id"] == thread_id
        assert (await first.threads.get(owner, thread_id))["archived"]
        with pytest.raises(HarnessError) as error:
            await second.create_run(owner, "归档后任务", "after", thread_id=thread_id)
        assert error.value.status_code == 409
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_project_projection_paging_and_retirement(tmp_path):
    service = await service_at(tmp_path)
    try:
        owner = await identity(service, "projects-owner")
        for index in range(4):
            await service.put_asset(owner, "project", f"项目{index}", "长内容", status="active")
        await service.put_asset(owner, "project", "未启用", "内容")
        page = await service.threads.projects(owner, limit=2)
        next_page = await service.threads.projects(owner, limit=2, before=page[-1]["id"])
        assert len(page) == len(next_page) == 2
        assert not set(row["id"] for row in page) & set(row["id"] for row in next_page)
        assert set(page[0]) == {"id", "name", "status"}
        thread = await service.threads.create(owner, "项目会话", page[0]["id"])
        async with service.store.sessions.begin() as session:
            from app.harness.models import Asset

            project = await session.get(Asset, page[0]["id"])
            project.status = "retired"
        assert (await service.threads.get(owner, thread["id"]))["project_name"] == page[0]["name"]
        with pytest.raises(HarnessError) as error:
            await service.create_run(owner, "继续已退役项目", "retired", thread_id=thread["id"])
        assert error.value.status_code == 409
    finally:
        await service.close()
