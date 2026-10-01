"""私有范围、令牌、幂等与资产生命周期的安全回归。"""

import asyncio

import pytest
from sqlalchemy import select

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.models import Token, User


async def identity(service, name):
    user = await service.register(name, "password123")
    return Principal(user["user_id"], user["tenant_id"], user["role"])


def make_service(tmp_path):
    settings = HarnessSettings(
        data_dir=tmp_path, database_url="", max_user_runs=32, max_concurrent_runs=2
    )
    return HarnessService(settings)


@pytest.mark.asyncio
async def test_private_tenants_and_same_tenant_owners(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        owner = await identity(service, "alice")
        other = await identity(service, "bob")
        member = await service.create_user(owner, "charlie", "password123", "operator")
        colleague = Principal(member["user_id"], member["tenant_id"], member["role"])
        assert owner.tenant_id != other.tenant_id
        assert owner.tenant_id == colleague.tenant_id
        asset = await service.put_asset(owner, "memory", "私有笔记", "机密", status="active")
        run = await service.create_run(owner, "任务", "key")
        for principal in (other, colleague):
            assert await service.list_assets(principal) == []
            assert await service.list_runs(principal) == []
            for operation in (
                service.get_asset(principal, asset["id"]),
                service.get_run(principal, run["id"]),
                service.events(principal, run["id"]),
                service.approve(principal, run["id"], True),
                service.feedback(principal, run["id"], True),
            ):
                with pytest.raises(HarnessError) as error:
                    await operation
                assert error.value.status_code == 404
        with pytest.raises(HarnessError):
            await service.create_user(colleague, "unwanted", "password123", "viewer")
        with pytest.raises(HarnessError):
            await service.register("alice", "password123")
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_tokens_are_hashed_expire_and_logout(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        login = await service.login("alice", "password123")
        assert await service.authenticate(login["token"]) == principal
        async with service.store.sessions() as session:
            token = await session.scalar(select(Token))
            user = await session.scalar(select(User))
            assert token.hash != login["token"] and len(token.hash) == 64
            assert "password123" not in user.password
        await service.logout(login["token"])
        with pytest.raises(HarnessError) as error:
            await service.authenticate(login["token"])
        assert error.value.status_code == 401
        login = await service.login("alice", "password123")
        async with service.store.sessions.begin() as session:
            token = await session.scalar(select(Token))
            token.expires = 0
        with pytest.raises(HarnessError):
            await service.authenticate(login["token"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_twenty_concurrent_idempotent_and_conflict(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        runs = await asyncio.gather(
            *(service.create_run(principal, "同一任务", "same") for _ in range(20))
        )
        assert len({run["id"] for run in runs}) == 1
        assert len(await service.list_runs(principal)) == 1
        with pytest.raises(HarnessError) as error:
            await service.create_run(principal, "不同任务", "same")
        assert error.value.status_code == 409
        claims = await asyncio.gather(
            service.store.claim("worker-a"), service.store.claim("worker-b")
        )
        assert sum(claim is not None for claim in claims) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_asset_versions_retire_restore_and_recall(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        asset = await service.put_asset(
            principal, "skill", "读取数据库报告", "使用只读查询", {"tools": ["read"]}
        )
        assert await service.recall(principal, "读取数据库") == []
        active = await service.transition_asset(principal, asset["id"], "active")
        assert active["version"] == 2
        assert len(await service.recall(principal, "读取数据库", ["read"])) == 1
        assert await service.recall(principal, "读取数据库", []) == []
        await service.delete_asset(principal, asset["id"])
        assert await service.recall(principal, "读取数据库") == []
        restored = await service.transition_asset(principal, asset["id"], "active")
        assert restored["version"] == 4
        history = await service.asset_history(principal, asset["id"])
        assert [version["status"] for version in history] == [
            "draft",
            "active",
            "retired",
            "active",
        ]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_database_serializes_two_services_idempotency_and_claim(tmp_path):
    first = make_service(tmp_path)
    second = make_service(tmp_path)
    await first.store.initialize()
    try:
        principal = await identity(first, "alice")
        results = await asyncio.gather(
            *(
                service.create_run(principal, "跨实例任务", "same")
                for service in [first, second] * 10
            )
        )
        assert len({run["id"] for run in results}) == 1
        claims = await asyncio.gather(first.store.claim("first"), second.store.claim("second"))
        assert sum(claim is not None for claim in claims) == 1
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_sessions_and_child_budgets_cannot_cross_private_scope(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        owner = await identity(service, "alice")
        other = await identity(service, "bob")
        parent = await service.create_run(owner, "父任务", "parent", max_steps=4)
        await service.store.claim("parent-worker")
        with pytest.raises(HarnessError):
            await service.create_run(other, "侵入会话", "intrude", session_id=parent["session_id"])
        with pytest.raises(HarnessError):
            await service.create_run(
                other, "跨用户子任务", "child", parent_run_id=parent["id"], allowed_tools=["read"]
            )
        child = await service.create_run(
            owner,
            "子任务",
            "child",
            parent_run_id=parent["id"],
            allowed_tools=["read"],
            max_steps=2,
        )
        with pytest.raises(HarnessError):
            await service.create_run(
                owner, "递归子任务", "recursive", parent_run_id=child["id"], allowed_tools=["read"]
            )
        with pytest.raises(HarnessError):
            await service.create_run(
                owner,
                "超预算子任务",
                "overflow",
                parent_run_id=parent["id"],
                allowed_tools=["read"],
                max_steps=3,
            )
        with pytest.raises(HarnessError):
            await service.create_run(
                owner,
                "危险子任务",
                "risk",
                parent_run_id=parent["id"],
                allowed_tools=["shell_exec"],
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_document_byte_limit_and_profile_bootstrap(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, max_upload_bytes=50000))
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        document = await service.put_asset(
            principal, "document", "长文档", "a" * 30000, status="active"
        )
        assert len(document["content"]) == 30000
        with pytest.raises(HarnessError) as error:
            await service.put_asset(principal, "document", "超限中文文档", "中" * 20000)
        assert error.value.status_code == 413
        await service.put_asset(principal, "profile", "用户画像", "我偏好中文", status="active")
        await service.put_asset(
            principal, "procedure", "只读过程", "步骤", {"tools": ["unavailable"]}, "active"
        )
        boot = await service.bootstrap(principal)
        assert boot["memories"][0]["kind"] == "profile"
        assert boot["skills"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_viewer_cannot_mutate_and_parent_cancel_cascades(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        user = await service.create_user(principal, "viewer", "password123", "viewer")
        viewer = Principal(user["user_id"], user["tenant_id"], "viewer")
        with pytest.raises(HarnessError) as error:
            await service.put_asset(viewer, "memory", "禁止写入", "内容")
        assert error.value.status_code == 403
        run = await service.create_run(viewer, "只读任务", "viewer")
        with pytest.raises(HarnessError) as error:
            await service.feedback(viewer, run["id"], True)
        assert error.value.status_code == 403
        parent = await service.create_run(principal, "父任务", "parent", max_steps=6)
        from app.harness.models import Run

        async with service.store.sessions.begin() as session:
            active = await session.get(Run, parent["id"])
            active.status, active.lease_owner, active.lease_until = (
                "running",
                "parent-worker",
                99999999999,
            )
        children = [
            await service.create_run(
                principal,
                "子任务",
                "child-" + str(index),
                parent_run_id=parent["id"],
                allowed_tools=["read"],
                max_steps=2,
            )
            for index in range(2)
        ]
        assert await service.store.claim("child-worker", child_only=True) in {
            child["id"] for child in children
        }
        await service.cancel(principal, parent["id"])
        statuses = [(await service.get_run(principal, child["id"]))["status"] for child in children]
        assert statuses == ["cancelled", "cancelled"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_client_cannot_forge_verified_asset_metadata(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        for key in ("source_run_id", "tool_evidence", "manual_verified", "user_verified"):
            with pytest.raises(HarnessError) as error:
                await service.put_asset(principal, "skill", "伪造来源", "技能", {key: "forged"})
            assert error.value.status_code == 422
        assert await service.list_assets(principal) == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_restore_version_and_cancelled_parent_cannot_create_children(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    try:
        principal = await identity(service, "alice")
        asset = await service.put_asset(principal, "memory", "原名", "原文", {"tags": ["原标签"]})
        await service.put_asset(
            principal, "profile", "新名", "新文", {"tags": ["新标签"]}, "active", asset["id"]
        )
        restored = await service.restore_asset(principal, asset["id"], 1)
        assert restored["version"] == 3
        assert restored["kind"] == "memory" and restored["content"] == "原文"
        assert restored["metadata"] == {"tags": ["原标签"]} and restored["status"] == "draft"
        assert len(await service.asset_history(principal, asset["id"])) == 3
        parent = await service.create_run(principal, "父任务", "parent")
        await service.store.claim("parent-worker")
        await service.cancel(principal, parent["id"])
        with pytest.raises(HarnessError) as error:
            await service.create_run(
                principal,
                "孤儿子任务",
                "orphan",
                parent_run_id=parent["id"],
                allowed_tools=["read"],
            )
        assert error.value.status_code == 409
    finally:
        await service.close()
