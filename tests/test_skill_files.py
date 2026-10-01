"""Skill 文件生态的真实 SQLite、HTTP 与工具隔离验收。"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import update

from app.harness.errors import HarnessError
from app.harness.models import Asset, Run, User
from app.harness.security import canonical
from app.harness.service import HarnessService
from app.harness.settings import HarnessSettings
from app.harness_tools.catalog import HarnessTools
from app.main import create_app

DOCUMENT = (
    "---\nname: report-reader\ndescription: 检查文献证据\ntags: [research]\n"
    "allowed-tools: [calculator]\n---\n阅读说明，按 references/checklist.md 执行。\n"
)
RESOURCES = {"references/checklist.md": "核对真实来源", "scripts/example.py": "print('示例文本')"}


class IdleModel:
    async def chat(self, *args, **kwargs):
        return SimpleNamespace(content="测试", raw={}, usage={}, model_id="test")


async def login(client, username):
    response = await client.post(
        "/api/v1/auth/login", json={"username": username, "password": "test-password"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": "Bearer " + response.json()["token"]}


@pytest_asyncio.fixture
async def api(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path, evolution_enabled=False), IdleModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client,
    ):
        response = await client.post(
            "/api/v1/auth/register", json={"username": "skill-owner", "password": "test-password"}
        )
        assert response.status_code == 201
        headers = await login(client, "skill-owner")
        service = app.state.harness
        principal = await service.authenticate(headers["Authorization"][7:])
        yield client, headers, service, principal


async def imported(api, **changes):
    client, headers, _, _ = api
    response = await client.post(
        "/api/v1/skills/import",
        headers=headers,
        json={
            "document": DOCUMENT,
            "directory": "research/reading",
            "resources": RESOURCES,
            **changes,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.asyncio
async def test_import_export_roundtrip_and_owner_draft_resources(api):
    client, headers, service, principal = api
    asset = await imported(api)
    assert asset["kind"] == "skill" and asset["status"] == "draft"
    assert asset["name"] == "report-reader"
    assert asset["metadata"]["description"] == "检查文献证据"
    assert asset["metadata"]["tools"] == ["calculator"]
    assert not await service.recall(principal, "reading 文献证据", ["calculator", "skill_read"])
    exported = await client.get(f"/api/v1/skills/{asset['id']}/export", headers=headers)
    assert exported.status_code == 200, exported.text
    assert exported.json()["directory"] == "research/reading"
    assert exported.json()["resources"] == RESOURCES
    copied = await imported(api, **exported.json())
    assert copied["content"] == asset["content"]
    resource = await client.get(
        f"/api/v1/skills/{asset['id']}/resources",
        params={"path": "scripts/example.py"},
        headers=headers,
    )
    assert (
        resource.status_code == 200
        and resource.json()["content"] == RESOURCES["scripts/example.py"]
    )
    with pytest.raises(HarnessError) as error:
        await service.tool_executor.execute(
            "skill_read", {"asset_id": asset["id"]}, principal, "run"
        )
    assert error.value.status_code == 404


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: test\nname: second\ndescription: duplicated",
        "name: &anchor test\ndescription: *anchor",
        "name: !!python/object:os.system {}\ndescription: exploit",
        "name: test\ndescription: [nested, [list]]",
        "name: test\ndescription: safe\nsource_run_id: forged",
        "name: test\ndescription: safe\nmanual_verified: true",
        "name: test\ndescription: safe\nunknown: x",
        "name: Test\ndescription: uppercase",
        "name: -test\ndescription: invalid",
        "name: test-\ndescription: invalid",
        "name: test\ndescription: ''",
        "name: test\ndescription: " + "x" * 1025,
        "name: test\ndescription: safe\ntags: [true]",
        "name: test\ndescription: safe\nallowed-tools: {calculator: true}",
        "name: test\ndescription: safe\ntags: [" + ",".join("tag" for _ in range(33)) + "]",
        "name: test\ndescription: safe\ntags: " + "[" * 1000 + "x" + "]" * 1000,
    ],
)
@pytest.mark.asyncio
async def test_reject_unsafe_or_invalid_frontmatter(api, frontmatter):
    client, headers, _, _ = api
    result = await client.post(
        "/api/v1/skills/import",
        headers=headers,
        json={"document": f"---\n{frontmatter}\n---\n正文"},
    )
    assert result.status_code == 422, result.text


@pytest.mark.parametrize(
    "changes",
    [
        {"document": "没有 frontmatter"},
        {"document": DOCUMENT + "x" * 16001},
        {"document": "---\nname: valid\ndescription: safe\n---\n" + "中" * 12000},
        {"document": DOCUMENT + "\x00"},
        *(
            {"directory": path}
            for path in [
                "../x",
                "/root",
                "C:/root",
                "a\\b",
                ".hidden",
                "a/./b",
                "a//b",
                "a/b/c/d/e",
                "a" * 129,
            ]
        ),
        *(
            {"resources": {path: "文本"}}
            for path in [
                "../x",
                "/root",
                "C:/root",
                "a\\b",
                ".hidden",
                "a/./b",
                "SKILL.md",
                "skill.MD",
                "a/b/c/d/e",
                "x" * 129,
            ]
        ),
        {"resources": {"a.txt": "\x00"}},
        {"resources": {"a.txt": "中" * 5334}},
        {"resources": {f"r{i}.txt": "x" for i in range(17)}},
        {"resources": {f"r{i}.txt": "x" * 16000 for i in range(5)}},
        {"resources": {"notes.md": "one", "NOTES.md": "two"}},
        {"resources": {"notes.md": 12}},
        {"expected_version": 1},
        {"asset_id": "missing"},
        {"expected_version": 0, "asset_id": "missing"},
        {"status": "active"},
        {"metadata": {"source_run_id": "forged"}},
    ],
)
@pytest.mark.asyncio
async def test_input_paths_and_resource_budgets_are_bounded(api, changes):
    client, headers, _, _ = api
    result = await client.post(
        "/api/v1/skills/import", headers=headers, json={"document": DOCUMENT, **changes}
    )
    assert result.status_code == 422, result.text


@pytest.mark.asyncio
async def test_scope_viewer_and_non_skill_rejection(api):
    client, headers, service, principal = api
    asset = await imported(api)
    for username, role in [("peer", "operator"), ("viewer", "viewer")]:
        response = await client.post(
            "/api/v1/auth/members",
            headers=headers,
            json={"username": username, "password": "test-password", "role": role},
        )
        assert response.status_code == 201
    await client.post(
        "/api/v1/auth/register", json={"username": "outsider", "password": "test-password"}
    )
    for username in ["peer", "viewer", "outsider"]:
        other = await login(client, username)
        for suffix in ["export", "resources?path=references/checklist.md"]:
            assert (
                await client.get(f"/api/v1/skills/{asset['id']}/{suffix}", headers=other)
            ).status_code == 404
        update = await client.post(
            "/api/v1/skills/import",
            headers=other,
            json={"document": DOCUMENT, "asset_id": asset["id"], "expected_version": 1},
        )
        assert update.status_code == (403 if username == "viewer" else 404)
    viewer_headers = await login(client, "viewer")
    assert (
        await client.post(
            "/api/v1/skills/import", headers=viewer_headers, json={"document": DOCUMENT}
        )
    ).status_code == 403
    non_skill = await service.put_asset(principal, "memory", "note", "正文", status="active")
    for suffix in ["export", "resources?path=notes.md"]:
        assert (
            await client.get(f"/api/v1/skills/{non_skill['id']}/{suffix}", headers=headers)
        ).status_code == 404
    with pytest.raises(HarnessError) as error:
        await service.tool_executor.execute(
            "skill_read", {"asset_id": non_skill["id"]}, principal, "run"
        )
    assert error.value.status_code == 404
    result = await client.post(
        "/api/v1/skills/import",
        headers=headers,
        json={"document": DOCUMENT, "asset_id": non_skill["id"], "expected_version": 1},
    )
    assert result.status_code == 422


@pytest.mark.asyncio
async def test_directory_description_recall_and_unknown_tool_filter(api):
    _, _, service, principal = api
    asset = await imported(api)
    active = await service.transition_asset(principal, asset["id"], "active")
    for query in ["reading", "文献证据"]:
        assert [
            a["id"] for a in await service.recall(principal, query, ["calculator", "skill_read"])
        ] == [asset["id"]]
    read = await service.tool_executor.execute(
        "skill_read", {"asset_id": asset["id"]}, principal, "run"
    )
    assert read["content"] == active["content"]
    resource = await service.tool_executor.execute(
        "skill_read", {"asset_id": asset["id"], "path": "references/checklist.md"}, principal, "run"
    )
    assert resource["content"] == "核对真实来源"
    for path in ["../x", "/root", "scripts/missing.py"]:
        with pytest.raises(HarnessError):
            await service.tool_executor.execute(
                "skill_read", {"asset_id": asset["id"], "path": path}, principal, "run"
            )
    unknown = await imported(api, document=DOCUMENT.replace("calculator", "root_shell"))
    await service.transition_asset(principal, unknown["id"], "active")
    assert unknown["id"] not in [
        a["id"] for a in await service.recall(principal, "reading", ["calculator", "skill_read"])
    ]
    assert "root_shell" not in [
        s["function"]["name"] for s in service.tool_executor.catalog(principal)
    ]
    await service.transition_asset(principal, asset["id"], "retired")
    with pytest.raises(HarnessError) as error:
        await service.tool_executor.execute(
            "skill_read", {"asset_id": asset["id"]}, principal, "run"
        )
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_update_cas_retains_provenance_and_revokes_verification(api):
    client, headers, service, principal = api
    asset = await imported(api)
    payload = {"document": DOCUMENT, "asset_id": asset["id"], "expected_version": 1}
    results = await asyncio.gather(
        *(client.post("/api/v1/skills/import", headers=headers, json=payload) for _ in range(2))
    )
    assert sorted(result.status_code for result in results) == [201, 409]
    # 通过真实数据库模拟服务端提炼的有来源 Skill，客户端不能写保留字段。
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(Asset)
            .where(Asset.id == asset["id"])
            .values(
                status="active",
                attributes={
                    **asset["metadata"],
                    "source_run_id": "server-run",
                    "manual_verified": True,
                    "user_verified": True,
                },
            )
        )
    changed = await client.post(
        "/api/v1/skills/import",
        headers=headers,
        json={
            **payload,
            "expected_version": 2,
            "document": DOCUMENT.replace("calculator", "unknown_tool"),
        },
    )
    assert changed.status_code == 201, changed.text
    revised = changed.json()
    assert revised["status"] == "draft" and revised["metadata"]["source_run_id"] == "server-run"
    assert not revised["metadata"]["user_verified"] and not revised["metadata"]["manual_verified"]
    await service.transition_asset(principal, revised["id"], "active")
    assert not await service.recall(principal, "reading", ["calculator", "skill_read"])


@pytest.mark.asyncio
async def test_generic_metadata_edits_cannot_bypass_read_validation(api):
    client, headers, service, principal = api
    asset = await imported(api)
    bad_metadata = {**asset["metadata"], "resources": {"../escape": "secret"}}
    with pytest.raises(HarnessError) as error:
        await service.put_asset(
            principal,
            "skill",
            asset["name"],
            asset["content"],
            bad_metadata,
            status="active",
            asset_id=asset["id"],
        )
    assert error.value.status_code == 422
    # 即使历史数据库已有异常记录，读取与导出仍需再次阻断。
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(Asset)
            .where(Asset.id == asset["id"])
            .values(status="active", attributes=bad_metadata)
        )
    for suffix in ["export", "resources?path=../escape"]:
        assert (
            await client.get(f"/api/v1/skills/{asset['id']}/{suffix}", headers=headers)
        ).status_code == 422
    with pytest.raises(HarnessError) as error:
        await service.tool_executor.execute(
            "skill_read", {"asset_id": asset["id"]}, principal, "run"
        )
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_legacy_generated_skill_has_stable_export_slug(api):
    client, headers, service, principal = api
    asset = await service.put_asset(
        principal,
        "skill",
        "文献核查经验",
        "核验每条证据",
        {"tools": ["calculator"]},
        status="active",
    )
    read = await service.tool_executor.execute(
        "skill_read", {"asset_id": asset["id"]}, principal, "run"
    )
    assert read["content"] == asset["content"]
    exported = await client.get(f"/api/v1/skills/{asset['id']}/export", headers=headers)
    assert exported.status_code == 200, exported.text
    repeated = await client.get(f"/api/v1/skills/{asset['id']}/export", headers=headers)
    assert repeated.json() == exported.json()
    copied = await imported(api, **exported.json())
    assert copied["name"].startswith("skill-")
    assert copied["metadata"]["description"] == "文献核查经验"
    assert copied["content"] == asset["content"] and copied["status"] == "draft"


@pytest.mark.asyncio
async def test_active_skill_tool_scope_and_owner_viewer_read_only(api):
    client, headers, service, principal = api
    asset = await imported(api)
    await service.transition_asset(principal, asset["id"], "active")
    for username, same_tenant in [("tool-peer", True), ("tool-outsider", False)]:
        path = "/api/v1/auth/members" if same_tenant else "/api/v1/auth/register"
        result = await client.post(
            path,
            headers=headers,
            json={"username": username, "password": "test-password"},
        )
        assert result.status_code == 201
        other_headers = await login(client, username)
        other = await service.authenticate(other_headers["Authorization"][7:])
        with pytest.raises(HarnessError) as error:
            await service.tool_executor.execute(
                "skill_read", {"asset_id": asset["id"]}, other, "run"
            )
        assert error.value.status_code == 404
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(User).where(User.id == principal.user_id).values(role="viewer")
        )
    viewer = await service.authenticate(headers["Authorization"][7:])
    assert viewer.role == "viewer"
    assert (
        await client.get(f"/api/v1/skills/{asset['id']}/export", headers=headers)
    ).status_code == 200
    read = await service.tool_executor.execute(
        "skill_read", {"asset_id": asset["id"]}, viewer, "run"
    )
    assert read["content"] == asset["content"]
    assert (
        await client.post("/api/v1/skills/import", headers=headers, json={"document": DOCUMENT})
    ).status_code == 403


@pytest.mark.asyncio
async def test_allowed_tools_space_separated_compatibility(api):
    asset = await imported(api, document=DOCUMENT.replace("[calculator]", "calculator skill_read"))
    assert asset["metadata"]["tools"] == ["calculator", "skill_read"]


@pytest.mark.asyncio
async def test_export_serialization_and_merged_tool_budget_checked_before_import(api):
    client, headers, service, _ = api
    prefix = "---\nname: a\ndescription: b\n---\n"
    remaining = 32768 - len(prefix.encode())
    document = prefix + "中" * (remaining // 3) + "x" * (remaining % 3)
    result = await client.post(
        "/api/v1/skills/import", headers=headers, json={"document": document}
    )
    assert result.status_code == 422, "不能存入无法再次导入的导出文档"
    asset = await imported(api)
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(Asset)
            .where(Asset.id == asset["id"])
            .values(
                attributes={
                    **asset["metadata"],
                    "source_run_id": "server-run",
                    "tools": [f"tool{i}" for i in range(32)],
                }
            )
        )
    result = await client.post(
        "/api/v1/skills/import",
        headers=headers,
        json={"document": DOCUMENT, "asset_id": asset["id"], "expected_version": 1},
    )
    assert result.status_code == 422, "保留的来源工具加导入工具也必须满足总上限"


@pytest.mark.asyncio
async def test_skill_read_is_allowed_in_read_only_child_run(tmp_path):
    settings = HarnessSettings(data_dir=tmp_path)
    tools = HarnessTools(settings)
    service = HarnessService(settings, tool_executor=tools)
    tools.service = service
    await service.store.initialize()
    try:
        await service.register("child-skill-owner", "test-password")
        token = await service.login("child-skill-owner", "test-password")
        principal = await service.authenticate(token["token"])
        parent = await service.create_run(principal, "阅读 Skill", "parent", max_steps=6)
        assert await service.store.claim("parent-worker") == parent["id"]
        child = await service.create_run(
            principal,
            "只读 Skill",
            "child",
            parent_run_id=parent["id"],
            allowed_tools=["skill_read"],
            max_steps=2,
        )
        assert child["parent_run_id"] == parent["id"]
        assert not tools.requires_approval("skill_read", {})
        asset = await service.import_skill(principal, DOCUMENT, resources=RESOURCES)
        await service.transition_asset(principal, asset["id"], "active")
        read = await tools.execute("skill_read", {"asset_id": asset["id"]}, principal, child["id"])
        assert read["content"] == asset["content"]
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_recall_keeps_resource_paths_and_reads_large_text_on_demand(api):
    _, _, service, principal = api
    resource_text = "RESOURCE_PRIVATE_TEXT_" + "x" * 12000
    asset = await imported(api, resources={"references/checklist.md": resource_text})
    await service.transition_asset(principal, asset["id"], "active")
    run = await service.create_run(principal, "reading 文献证据", "progressive-reading")
    async with service.store.sessions() as session:
        stored = await service.store.owned(session, Run, run["id"], principal)
        candidate_message = next(
            item["content"]
            for item in stored.messages
            if "不可信记忆候选" in str(item.get("content"))
        )
    assert "references/checklist.md" in candidate_message
    assert "RESOURCE_PRIVATE_TEXT_" not in candidate_message
    recalled = await service.recall(principal, "reading", ["calculator", "skill_read"])
    assert recalled[0]["metadata"]["resource_paths"] == ["references/checklist.md"]
    assert "resources" not in recalled[0]["metadata"]
    assert "RESOURCE_PRIVATE_TEXT_" not in canonical(recalled)
    read = await service.tool_executor.execute(
        "skill_read",
        {"asset_id": asset["id"], "path": "references/checklist.md"},
        principal,
        run["id"],
    )
    assert read["content"] == resource_text


@pytest.mark.parametrize("tools", [7, None, [{}], [f"new{i}" for i in range(32)]])
@pytest.mark.asyncio
async def test_generic_source_skill_update_rejects_invalid_or_oversized_tools(api, tools):
    client, headers, service, _ = api
    asset = await imported(api)
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(Asset)
            .where(Asset.id == asset["id"])
            .values(attributes={**asset["metadata"], "source_run_id": "server-run"})
        )
    response = await client.put(
        f"/api/v1/assets/{asset['id']}",
        headers=headers,
        json={
            "kind": "skill",
            "name": asset["name"],
            "content": asset["content"],
            "metadata": {**asset["metadata"], "tools": tools},
        },
    )
    assert response.status_code == 422, response.text
    unchanged = (await client.get(f"/api/v1/assets/{asset['id']}", headers=headers)).json()
    assert unchanged["version"] == asset["version"]
    assert unchanged["metadata"]["tools"] == ["calculator"]


@pytest.mark.parametrize(
    "changes",
    [
        {"description": "文献证据" + "x" * 200008},
        {"directory": "a/b/c/d/e"},
        {"extra": "UNTRUSTED_METADATA_SENTINEL" + "x" * 9000},
        {"tools": 7},
        {"tools": None},
        {"tools": [{}]},
    ],
)
@pytest.mark.asyncio
async def test_generic_skill_rejects_invalid_metadata_and_skips_legacy_bad_rows(api, changes):
    client, headers, service, principal = api
    asset = await imported(api)
    bad_metadata = {**asset["metadata"], **changes}
    response = await client.put(
        f"/api/v1/assets/{asset['id']}",
        headers=headers,
        json={
            "kind": "skill",
            "name": asset["name"],
            "content": asset["content"],
            "metadata": bad_metadata,
            "status": "active",
        },
    )
    assert response.status_code == 422, response.text
    good = await imported(api)
    await service.transition_asset(principal, good["id"], "active")
    # 模拟旧版本或数据库写入的异常记录，召回必须隔离它而不影响有效候选。
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(Asset)
            .where(Asset.id == asset["id"])
            .values(status="active", attributes=bad_metadata)
        )
    recalled = await service.recall(principal, "reading 文献证据", ["calculator", "skill_read"])
    assert [item["id"] for item in recalled] == [good["id"]]
    run = await service.create_run(principal, "reading 文献证据", "invalid-metadata-context")
    async with service.store.sessions() as session:
        stored = await service.store.owned(session, Run, run["id"], principal)
        messages = canonical(stored.messages)
    assert good["id"] in messages and asset["id"] not in messages
    assert "UNTRUSTED_METADATA_SENTINEL" not in messages
    assert len(messages.encode()) < 24000


@pytest.mark.asyncio
async def test_metadata_json_budgets_support_escaped_resources_and_reject_bad_values(api):
    client, headers, service, principal = api
    response = await client.post(
        "/api/v1/assets",
        headers=headers,
        json={
            "kind": "memory",
            "name": "预算",
            "content": "说明",
            "metadata": {"extra": "x" * (512 * 1024)},
        },
    )
    assert response.status_code == 422, response.text
    for metadata in [{"bad": object()}, {"bad": {1: "one", "mixed": "two"}}]:
        with pytest.raises(HarnessError) as error:
            await service.put_asset(principal, "memory", "非法 JSON", "正文", metadata)
        assert error.value.status_code == 422
    nested = []
    nested.append(nested)
    with pytest.raises(HarnessError) as error:
        await service.put_asset(principal, "memory", "循环 JSON", "正文", {"bad": nested})
    assert error.value.status_code == 422
    escaped = await imported(api, resources={f"r{i}.txt": "\x01" * 16000 for i in range(4)})
    assert len(canonical(escaped["metadata"]).encode()) > 64 * 1024
    assert (
        await client.get(f"/api/v1/skills/{escaped['id']}/export", headers=headers)
    ).status_code == 200


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.asyncio
async def test_metadata_non_finite_json_numbers_return_422(api, number):
    client, headers, _, _ = api
    payload = (
        '{"kind":"memory","name":"numbers","content":"text","metadata":{"bad":' + number + "}}"
    )
    result = await client.post(
        "/api/v1/assets", headers={**headers, "Content-Type": "application/json"}, content=payload
    )
    assert result.status_code == 422, result.text[:300]
