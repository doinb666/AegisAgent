"""HTTP 模板/资料/MCP 目录及资料运行输入边界，无外部 MCP 调用。"""

import json
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import func, select

from app.harness import HarnessSettings, Principal
from app.harness.context import PLAN_PROMPT
from app.harness.models import Run
from app.harness.task_inputs import DOCUMENT_SNAPSHOT_PREFIX
from app.harness_tools.mcp import MCPGateway
from app.main import create_app
from tests.test_harness_api import ToolModel, account, wait_run


@pytest.mark.asyncio
async def test_http_templates_documents_and_run_input(tmp_path):
    app = create_app(
        HarnessSettings(data_dir=tmp_path, database_url="", evolution_enabled=False), ToolModel()
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        owner = await account(client, "task-input-http")
        harness = app.state.harness
        principal = await harness.authenticate(owner["Authorization"][7:])
        for path in ("/task-templates", "/documents/references", "/mcp/servers"):
            assert (await client.get("/api/v1" + path)).status_code == 401
        templates = (await client.get("/api/v1/task-templates", headers=owner)).json()
        assert len(templates) == 4
        assert {item["id"] for item in templates} == {
            "knowledge-answer",
            "readonly-code-review",
            "calculation-check",
            "retrospective",
        }
        assert all(
            set(item)
            == {
                "id",
                "title",
                "description",
                "message",
                "required_tools",
                "expected_output",
                "available",
            }
            for item in templates
        )
        async with harness.store.sessions() as session:
            assert await session.scalar(select(func.count()).select_from(Run)) == 0
        assert (await client.get("/api/v1/capabilities", headers=owner)).json()["task_inputs"] == {
            "enabled": True,
            "max_documents": 3,
        }
        document = await harness.put_asset(
            principal, "document", "参考资料", "计算依据 2+3", status="active"
        )
        assert (await client.get("/api/v1/documents/references", headers=owner)).json() == [
            {"id": document["id"], "name": "参考资料", "version": 1}
        ]
        for query in ("limit=0", "limit=51"):
            assert (
                await client.get("/api/v1/documents/references?" + query, headers=owner)
            ).status_code == 422
        for query in ("before=", "before=missing", "before=" + "a" * 129):
            assert (
                await client.get("/api/v1/documents/references?" + query, headers=owner)
            ).status_code == 404
        for ids in ([document["id"]] * 2, ["a"] * 4, [123], [" "], "a"):
            response = await client.post(
                "/api/v1/runs",
                headers={**owner, "Idempotency-Key": "invalid"},
                json={"message": "计算", "document_ids": ids},
            )
            assert response.status_code == 422
        response = await client.post(
            "/api/v1/runs",
            headers={**owner, "Idempotency-Key": "explicit-http"},
            json={"message": "计算", "document_ids": [document["id"]]},
        )
        assert response.status_code == 202
        references = response.json()["document_references"]
        assert references[0]["id"] == document["id"] and len(references[0]["content_hash"]) == 64
        completed = await wait_run(client, owner, response.json()["id"], {"completed"})
        assert completed["document_references"] == references
        assert (await client.get("/api/v1/runs", headers=owner)).json()[0][
            "document_references"
        ] == references


@pytest.mark.asyncio
async def test_mcp_authorized_config_only_never_calls_gateway_or_returns_credentials(
    tmp_path, monkeypatch
):
    app = create_app(
        HarnessSettings(data_dir=tmp_path, database_url="", evolution_enabled=False), ToolModel()
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        owner = await account(client, "task-mcp-owner")
        outsider = await account(client, "task-mcp-outsider")
        harness = app.state.harness
        principal = await harness.authenticate(owner["Authorization"][7:])
        configuration = [
            {
                "name": "ready",
                "url": "https://private.example/mcp?secret=query",
                "tools": ["search"],
                "user_ids": [principal.user_id],
                "token_env": "TASK_MCP_TEST_SECRET",
            },
            {
                "name": "missing",
                "url": "https://private.example/second",
                "tools": ["other"],
                "tenant_ids": [principal.tenant_id],
                "token_env": "TASK_MCP_TEST_MISSING",
            },
            {
                "name": "other-user",
                "url": "https://not-for-owner.example/mcp",
                "tools": ["forbidden"],
                "user_ids": ["another-owner"],
            },
        ]
        gateway = MCPGateway(json.dumps(configuration))

        def forbidden(*args, **kwargs):
            pytest.fail("配置目录不得创建 MCP session 或外部请求")

        monkeypatch.setattr(gateway, "session", forbidden)
        monkeypatch.setattr(gateway, "list_tools", forbidden)
        monkeypatch.setattr(gateway, "call", forbidden)
        monkeypatch.setenv("TASK_MCP_TEST_SECRET", "secret-value-never-return")
        monkeypatch.delenv("TASK_MCP_TEST_MISSING", raising=False)
        harness.tool_executor.mcp = gateway
        response = await client.get("/api/v1/mcp/servers", headers=owner)
        assert response.status_code == 200
        rows = response.json()
        assert rows == [
            {
                "name": "missing",
                "tools": ["other"],
                "credential_ready": False,
                "configuration_only": True,
            },
            {
                "name": "ready",
                "tools": ["search"],
                "credential_ready": True,
                "configuration_only": True,
            },
        ]
        for secret in (
            "private.example",
            "TASK_MCP_TEST_SECRET",
            "secret-value-never-return",
            "another-owner",
            "forbidden",
            "token_env",
            "url",
        ):
            assert secret not in response.text
        assert (await client.get("/api/v1/mcp/servers", headers=outsider)).json() == []
        assert (
            harness.task_inputs.mcp_servers(
                Principal(principal.user_id, principal.tenant_id, "viewer")
            )
            == []
        )
        harness.tool_executor.mcp = None
        assert harness.task_inputs.mcp_servers(principal) == []


def test_templates_mark_missing_tools_unavailable_without_writing(tmp_path):
    from app.harness.service import HarnessService

    harness = HarnessService(HarnessSettings(data_dir=tmp_path, database_url=""))
    principal = Principal("owner", "tenant", "operator")
    rows = harness.task_inputs.templates(principal)
    assert {row["id"] for row in rows if not row["available"]} == {
        "knowledge-answer",
        "calculation-check",
    }
    rows[0]["required_tools"].append("file_write")
    assert "file_write" not in harness.task_inputs.templates(principal)[0]["required_tools"]


def test_mcp_configuration_directory_caps_services_and_tools_without_external_calls(tmp_path):
    from app.harness.service import HarnessService

    servers = [
        {
            "name": f"service-{index:02d}",
            "url": "https://local.test/mcp",
            "tools": [f"tool-{tool}" for tool in range(100)],
            "user_ids": ["owner"],
        }
        for index in range(60)
    ]
    harness = HarnessService(
        HarnessSettings(data_dir=tmp_path, database_url=""),
        tool_executor=SimpleNamespace(mcp=MCPGateway(json.dumps(servers))),
    )
    rows = harness.task_inputs.mcp_servers(Principal("owner", "tenant", "operator"))
    assert len(rows) == 50 and all(len(row["tools"]) == 64 for row in rows)
    assert all(row["credential_ready"] and row["configuration_only"] for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize(("budget", "task_length"), [(4000, 1200), (24000, 16000)])
@pytest.mark.parametrize(
    ("mode", "collaboration_mode"), [("react", None), ("plan", "team"), ("reflection", None)]
)
async def test_first_real_model_input_keeps_all_selected_documents_with_long_task(
    tmp_path, budget, task_length, mode, collaboration_mode
):
    class RecordingModel:
        def __init__(self):
            self.inputs = []

        async def chat(self, messages, **kwargs):
            self.inputs.append(messages)
            return SimpleNamespace(content="已核对输入", model_id="test", usage={}, raw={})

    model = RecordingModel()
    app = create_app(
        HarnessSettings(
            data_dir=tmp_path, database_url="", evolution_enabled=False, context_chars=budget
        ),
        model,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "context-budget-owner")
        harness = app.state.harness
        principal = await harness.authenticate(headers["Authorization"][7:])
        documents = [
            await harness.put_asset(
                principal,
                "document",
                f"资料{index}",
                f"资料{index}依据" + "正文" * 3000,
                status="active",
            )
            for index in range(3)
        ]
        message = "任务" * (task_length // 2)
        response = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "budgeted-documents"},
            json={
                "message": message,
                "document_ids": [document["id"] for document in documents],
                "mode": mode,
                "collaboration_mode": collaboration_mode,
            },
        )
        assert response.status_code == 202, response.text
        await wait_run(client, headers, response.json()["id"], {"completed"})
        first_input = model.inputs[0]
        snapshot_messages = [
            item
            for item in first_input
            if item.get("content", "").startswith(DOCUMENT_SNAPSHOT_PREFIX)
        ]
        assert len(snapshot_messages) == 1, first_input
        snapshots = json.loads(snapshot_messages[0]["content"][len(DOCUMENT_SNAPSHOT_PREFIX) :])
        assert [snapshot["id"] for snapshot in snapshots] == [
            document["id"] for document in documents
        ]
        assert all(
            128 <= len(snapshot["preview"]) <= 3000 and snapshot["truncated"]
            for snapshot in snapshots
        )
        assert sum(len(snapshot["preview"]) for snapshot in snapshots) < 9000
        assert {"role": "user", "content": message} in first_input
        if mode != "plan":
            assert first_input[-1] == {"role": "user", "content": message}
        assert len(json.dumps(first_input, ensure_ascii=False)) <= budget
        async with harness.store.sessions() as session:
            stored = await session.get(Run, response.json()["id"])
            assert stored.config["document_references"] == response.json()["document_references"]
            assert all(reference["truncated"] for reference in stored.config["document_references"])
            assert stored.config["document_context_prepared"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "prepared_chars", "expected_status"),
    [
        ("react", 400, "completed"),
        ("plan", 400, "completed"),
        ("react", 1150, "completed"),
        ("plan", 1150, "failed"),
        ("react", 2500, "failed"),
        ("plan", 2500, "failed"),
    ],
)
async def test_first_task_model_context_refits_after_project_approval_or_fails_explicitly(
    tmp_path, monkeypatch, mode, prepared_chars, expected_status
):
    class ProjectModel:
        def __init__(self):
            self.risk_inputs, self.task_inputs = [], []

        async def chat(self, messages, **kwargs):
            if "独立工具风险分类器" in messages[0]["content"]:
                self.risk_inputs.append(messages)
                content = '{"risk":"review","reason":"等待绑定审批"}'
            else:
                self.task_inputs.append(messages)
                content = '{"steps":[]}' if messages[-1]["content"] == PLAN_PROMPT else "完成"
            return SimpleNamespace(content=content, model_id="test", usage={}, raw={})

    model = ProjectModel()
    app = create_app(
        HarnessSettings(
            data_dir=tmp_path / "data",
            database_url="",
            evolution_enabled=False,
            context_chars=4000,
            repository_root=tmp_path,
            reflection_enabled=False,
        ),
        model,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "project-document-budget")
        harness = app.state.harness
        principal = await harness.authenticate(headers["Authorization"][7:])
        documents = [
            await harness.put_asset(
                principal, "document", str(index), "资料证据" * 125, status="active"
            )
            for index in range(3)
        ]

        async def prepared(*args, **kwargs):
            return {"status": "prepared", "evidence": "证" * prepared_chars}

        monkeypatch.setattr(harness.tool_executor.projects, "prepare", prepared)
        response = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "project-document-budget"},
            json={
                "message": "任务" * 600,
                "document_ids": [document["id"] for document in documents],
                "mode": mode,
                "project_mode": "fork",
            },
        )
        assert response.status_code == 202, response.text
        assert not any(item["truncated"] for item in response.json()["document_references"])
        run_id = response.json()["id"]
        pending = await wait_run(client, headers, run_id, {"waiting_approval"})
        async with harness.store.sessions() as session:
            before = await session.get(Run, run_id)
            original_snapshot = next(
                item["content"]
                for item in before.messages
                if item["content"].startswith(DOCUMENT_SNAPSHOT_PREFIX)
            )
        approval = pending["approval"]
        approved = await client.post(
            f"/api/v1/runs/{run_id}/approval",
            headers=headers,
            json={"approved": True, "call_id": approval["call_id"], "args_hash": approval["hash"]},
        )
        assert approved.status_code == 200
        result = await wait_run(client, headers, run_id, {"completed", "failed"})
        events = await harness.events(principal, run_id)
        if expected_status == "failed":
            assert not any(event["type"] == "plan_fallback" for event in events)
        assert result["status"] == expected_status, result
        assert model.risk_inputs
        assert not any(
            DOCUMENT_SNAPSHOT_PREFIX in item["content"]
            for call in model.risk_inputs
            for item in call
        )
        async with harness.store.sessions() as session:
            stored = await session.get(Run, run_id)
            assert stored.config["project_prepared"] == {"mode": "fork"}
            assert not any(item.get("content") == PLAN_PROMPT for item in stored.messages)
            if expected_status == "completed":
                assert stored.config["document_context_prepared"] is True
                first_input = model.task_inputs[0]
                actual_snapshot = next(
                    item["content"]
                    for item in first_input
                    if item.get("content", "").startswith(DOCUMENT_SNAPSHOT_PREFIX)
                )
                assert len(actual_snapshot) < len(original_snapshot)
                assert len(json.dumps(first_input, ensure_ascii=False)) <= 4000
                snapshots = json.loads(actual_snapshot[len(DOCUMENT_SNAPSHOT_PREFIX) :])
                assert len(snapshots) == 3 and all(
                    len(item["preview"]) >= 128 for item in snapshots
                )
                assert (
                    next(
                        item["content"]
                        for item in stored.messages
                        if item.get("content", "").startswith(DOCUMENT_SNAPSHOT_PREFIX)
                    )
                    == actual_snapshot
                )
                assert all(item["truncated"] for item in stored.config["document_references"])
                assert stored.config["document_references"] == [
                    {
                        key: item[key]
                        for key in ("id", "name", "version", "content_hash", "truncated")
                    }
                    for item in snapshots
                ]
            else:
                assert "已有审批工具可能已完成" in stored.error
                assert not stored.config.get("document_context_prepared")
                assert model.task_inputs == []
