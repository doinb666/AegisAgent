"""任务资料入口使用真实 SQLite，校验私有范围、快照及旧幂等兼容。"""

import json

import pytest
from sqlalchemy import event, func, select

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.models import Run
from app.harness.security import canonical, digest
from tests.test_harness_store import identity


async def initialized(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, database_url="", max_user_runs=32))
    await service.store.initialize()
    return service, await identity(service, "task-input-owner")


@pytest.mark.asyncio
async def test_document_directory_is_private_projected_and_stably_paged(tmp_path):
    service, owner = await initialized(tmp_path)
    try:
        other = await identity(service, "task-input-outsider")
        peer_data = await service.create_user(owner, "task-input-peer", "password123", "operator")
        peer = Principal(peer_data["user_id"], peer_data["tenant_id"], "operator")
        documents = [
            await service.put_asset(owner, "document", str(i), "正文", status="active")
            for i in range(5)
        ]
        retired = await service.put_asset(owner, "document", "退役", "旧内容", status="retired")
        await service.put_asset(owner, "document", "空白", " \n\t", status="active")
        await service.put_asset(owner, "memory", "非文档", "内容", status="active")
        statements = []

        def record(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        event.listen(service.store.engine.sync_engine, "before_cursor_execute", record)
        try:
            page = await service.task_inputs.document_references(owner, limit=2)
            next_page = await service.task_inputs.document_references(
                owner, limit=3, before=page[-1]["id"]
            )
        finally:
            event.remove(service.store.engine.sync_engine, "before_cursor_execute", record)
        assert [item["id"] for item in page + next_page] == sorted(
            (item["id"] for item in documents), reverse=True
        )
        assert all(set(item) == {"id", "name", "version"} for item in page)
        for statement in statements:
            projection = statement.split("FROM")[0]
            assert (
                "harness_assets.content" not in projection
                and "harness_assets.attributes" not in projection
            )
        for outsider in (other, peer):
            assert await service.task_inputs.document_references(outsider) == []
            with pytest.raises(HarnessError) as missing:
                await service.task_inputs.document_references(outsider, before=documents[0]["id"])
            assert missing.value.status_code == 404
        with pytest.raises(HarnessError) as retired_cursor:
            await service.task_inputs.document_references(owner, before=retired["id"])
        assert retired_cursor.value.status_code == 404
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_explicit_documents_snapshot_boundaries_and_idempotent_original(tmp_path):
    service, owner = await initialized(tmp_path)
    try:
        documents = [
            await service.put_asset(owner, "document", str(i), "资料" * 2000, status="active")
            for i in range(3)
        ]
        ids = [item["id"] for item in documents]
        result = await service.create_run(owner, "最终用户任务", "with-docs", document_ids=ids)
        async with service.store.sessions() as session:
            run = await session.get(Run, result["id"])
            snapshots = json.loads(
                next(
                    item["content"].split("：", 1)[1]
                    for item in run.messages
                    if item["content"].startswith("不可信显式资料快照")
                )
            )
            assert run.messages[-1] == {"role": "user", "content": "最终用户任务"}
            assert all(
                item["role"] == "user"
                for item in run.messages
                if "不可信显式资料快照" in item["content"]
            )
            assert sum(len(item["preview"]) for item in snapshots) == 9000
            assert all(item["truncated"] and len(item["content_hash"]) == 64 for item in snapshots)
            assert run.config["document_references"] == result["document_references"]
            assert all(
                "preview" not in item and "content" not in item
                for item in run.config["document_references"]
            )
            assert run.config["allowed_tools"] is None
            saved_messages = run.messages
        await service.put_asset(
            owner, "document", "已修改并退役", "新内容", status="retired", asset_id=ids[0]
        )
        repeated = await service.create_run(owner, "最终用户任务", "with-docs", document_ids=ids)
        assert repeated == result
        async with service.store.sessions() as session:
            assert (await session.get(Run, result["id"])).messages == saved_messages
            assert await session.scalar(select(func.count()).select_from(Run)) == 1
        with pytest.raises(HarnessError) as conflict:
            await service.create_run(owner, "最终用户任务", "with-docs", document_ids=ids[1:])
        assert conflict.value.status_code == 409
        plain = await service.create_run(owner, "旧任务", "legacy")
        assert (await service.create_run(owner, "旧任务", "legacy", document_ids=[]))[
            "id"
        ] == plain["id"]
        async with service.store.sessions() as session:
            run = await session.get(Run, plain["id"])
            assert run.payload_hash == digest(
                canonical(
                    dict(
                        message="旧任务",
                        session_id=None,
                        mode="react",
                        model=None,
                        parent_run_id=None,
                        allowed_tools=None,
                        max_steps=None,
                    )
                )
            )
        assert plain["document_references"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ids",
    [
        ["a"] * 2,
        ["a", "b", "c", "d"],
        [True],
        [123],
        [None],
        [""],
        ["  "],
        ["a" * 129],
        "a",
        {"id": "a"},
    ],
)
async def test_document_ids_reject_invalid_shapes_before_creating_run(tmp_path, ids):
    service, owner = await initialized(tmp_path)
    try:
        with pytest.raises(HarnessError) as error:
            await service.create_run(owner, "任务", "invalid", document_ids=ids)
        assert error.value.status_code == 422
        assert await service.list_runs(owner) == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_document_negative_states_no_implicit_inheritance_or_child_schedule_input(tmp_path):
    service, owner = await initialized(tmp_path)
    try:
        other = await identity(service, "other-doc-owner")
        peer_data = await service.create_user(owner, "peer-doc-owner", "password123", "operator")
        peer = Principal(peer_data["user_id"], peer_data["tenant_id"], "operator")
        invalid_documents = [
            await service.put_asset(other, "document", "他人", "正文", status="active"),
            await service.put_asset(peer, "document", "同租户他人", "正文", status="active"),
            await service.put_asset(owner, "document", "已退役", "正文", status="retired"),
            await service.put_asset(owner, "document", "空内容", "", status="active"),
            await service.put_asset(owner, "memory", "其他类型", "正文", status="active"),
        ]
        for item in [*invalid_documents, {"id": "不存在"}]:
            with pytest.raises(HarnessError) as error:
                await service.create_run(
                    owner, "任务", "invalid-" + item["id"], document_ids=[item["id"]]
                )
            assert error.value.status_code == 404
        document = await service.put_asset(
            owner, "document", "显式资料", "禁止默认带给子任务的正文", status="active"
        )
        parent = await service.create_run(owner, "父任务", "parent", document_ids=[document["id"]])
        await service.store.claim("parent-worker")
        for values in (
            {"parent_run_id": parent["id"]},
            {"_schedule_gate": ("occurrence", "lease-owner")},
        ):
            with pytest.raises(HarnessError) as denied:
                await service.create_run(
                    owner, "禁止显式资料", "blocked", document_ids=[document["id"]], **values
                )
            assert denied.value.status_code == 403
        child = await service.create_run(
            owner,
            "子任务",
            "child",
            parent_run_id=parent["id"],
            session_id=parent["session_id"],
            allowed_tools=[],
            max_steps=2,
        )
        plain = await service.create_run(owner, "无显式资料的新任务", "plain")
        async with service.store.sessions() as session:
            for result in (child, plain):
                stored = await session.get(Run, result["id"])
                assert stored.config.get("document_references", []) == []
                assert not any(
                    "禁止默认带给子任务的正文" in str(item["content"]) for item in stored.messages
                )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_insufficient_document_context_is_422_without_run_and_plain_task_unchanged(tmp_path):
    service, owner = await initialized(tmp_path)
    try:
        service.settings.context_chars = 4000
        documents = [
            await service.put_asset(owner, "document", str(index), "正文" * 3000, status="active")
            for index in range(3)
        ]
        message = "任务" * 1750
        with pytest.raises(HarnessError) as insufficient:
            await service.create_run(
                owner,
                message,
                "insufficient",
                document_ids=[document["id"] for document in documents],
            )
        assert insufficient.value.status_code == 422
        assert "上下文预算不足" in insufficient.value.detail
        assert await service.list_runs(owner) == []
        # 失败请求不占用幂等键；不选资料的旧路径仍可以创建同一长度任务。
        plain = await service.create_run(owner, message, "insufficient")
        assert plain["document_references"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_document_retry_keeps_snapshot_when_current_budget_is_too_small(tmp_path):
    service, owner = await initialized(tmp_path)
    try:
        document = await service.put_asset(
            owner, "document", "资料", "证据" * 3000, status="active"
        )
        message = "任务" * 8000
        result = await service.create_run(
            owner, message, "budget-retry", document_ids=[document["id"]]
        )
        service.settings.context_chars = 4000
        repeated = await service.create_run(
            owner, message, "budget-retry", document_ids=[document["id"]]
        )
        assert repeated == result
        async with service.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 1
    finally:
        await service.close()
