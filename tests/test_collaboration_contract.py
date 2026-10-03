"""真实账本、父工作区和有向协作图的安全合同。"""

import json
import time
from types import SimpleNamespace

import pytest
import pytest_asyncio

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.models import Run, ToolCall
from app.harness.store import uid
from app.harness_tools.catalog import HarnessTools
from tests.test_delegation_completion import child_rows, delegate_result, finished, make_service


@pytest_asyncio.fixture
async def backend(tmp_path):
    settings = HarnessSettings(data_dir=tmp_path, database_url="", max_steps=10)
    tools = HarnessTools(settings)
    service = HarnessService(settings, tool_executor=tools)
    tools.service = service
    await service.store.initialize()
    user = await service.register("合同用户", "password123")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    parent = await service.create_run(principal, "父任务", "parent")
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, parent["id"])
        row.status, row.lease_owner, row.lease_until = "running", "worker", time.time() + 60
    try:
        yield service, tools, principal, parent["id"]
    finally:
        await service.close()


async def child_for(backend, tools=None):
    service, _, principal, parent_id = backend
    return await service.create_run(
        principal,
        "子任务",
        uid(),
        parent_run_id=parent_id,
        allowed_tools=tools if tools is not None else ["file_read", "calculator"],
        max_steps=2,
    )


async def mark_prepared(backend):
    service, tools, principal, parent_id = backend
    await tools.workspace.write(principal, parent_id, "目录/中文.py", "中文实时代码")
    await service.mark_project_prepared(principal, parent_id, "fork")


@pytest.mark.asyncio
async def test_child_reads_live_parent_workspace_after_preparation(backend):
    service, tools, principal, parent_id = backend
    await mark_prepared(backend)
    child = await child_for(backend)
    assert (
        await tools.execute("file_read", {"path": "目录/中文.py"}, principal, child["id"])
        == "中文实时代码"
    )
    await tools.workspace.write(principal, parent_id, "目录/中文.py", "已更新")
    assert (
        await tools.execute("file_read", {"path": "目录/中文.py"}, principal, child["id"])
        == "已更新"
    )
    assert tools.workspace._read_only_directory(principal, child["id"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["unprepared", "expired", "stopped", "parent_permission", "child_permission", "foreign"],
)
async def test_parent_workspace_requires_prepared_live_owned_permission_intersection(
    backend, failure
):
    service, tools, principal, parent_id = backend
    if failure != "unprepared":
        await mark_prepared(backend)
    child = await child_for(backend, [] if failure == "child_permission" else None)
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, parent_id)
        if failure == "expired":
            row.lease_until = time.time() - 1
        if failure == "stopped":
            row.status = "completed"
        if failure == "parent_permission":
            row.config = {**row.config, "allowed_tools": ["calculator"]}
        if failure == "foreign":
            row.owner_id = "其他所有者"
    with pytest.raises(HarnessError):
        await tools.execute("file_read", {"path": "目录/中文.py"}, principal, child["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["../outside", "C:/host", ".git/config", "目录/../中文.py"])
async def test_child_parent_read_keeps_strict_paths(backend, path):
    _, tools, principal, _ = backend
    await mark_prepared(backend)
    child = await child_for(backend)
    with pytest.raises(ValueError):
        await tools.execute("file_read", {"path": path}, principal, child["id"])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["done", "failed", "started", "unknown"])
async def test_independent_acceptance_checks_actual_ledger(backend, status):
    service, _, principal, parent_id = backend
    child = await child_for(backend)
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, child["id"])
        row.status, row.answer = "completed", "声称已验证"
        session.add(
            ToolCall(
                id=uid(),
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                run_id=row.id,
                call_id="evidence",
                name="calculator",
                arguments={},
                arguments_hash="hash",
                status=status,
                result={"value": 2} if status == "done" else None,
            )
        )
    result = await service.verify_collaboration_child(
        principal, parent_id, child["id"], ["calculator"]
    )
    assert result["status"] == ("verified" if status == "done" else "rejected")
    assert result["run_id"] == child["id"]
    assert bool(result["reasons"]) == (status != "done")


@pytest.mark.asyncio
async def test_acceptance_rejects_missing_required_evidence_but_allows_plain_answer(backend):
    service, _, principal, parent_id = backend
    child = await child_for(backend)
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, child["id"])
        row.status, row.answer = "completed", "已有答案"
    assert (
        await service.verify_collaboration_child(principal, parent_id, child["id"], ["calculator"])
    )["status"] == "rejected"
    assert (await service.verify_collaboration_child(principal, parent_id, child["id"], []))[
        "status"
    ] == "verified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tasks",
    [
        ["   "],
        [{"id": "a", "message": "甲"}, {"id": "a", "message": "乙"}],
        [{"id": "a", "message": "甲", "depends_on": ["a"]}],
        [{"id": "a", "message": "甲", "depends_on": ["missing"]}],
        [
            {"id": "a", "message": "甲", "depends_on": ["b"]},
            {"id": "b", "message": "乙", "depends_on": ["a"]},
        ],
        [{"id": "a", "message": "甲", "acceptance": {"invented": True}}],
        [{"id": "a", "message": "甲", "acceptance": {"required_tools": ["file_write"]}}],
    ],
)
async def test_graph_preflight_has_zero_creation(backend, tasks):
    service, tools, principal, parent_id = backend
    from app.harness_tools.delegation import delegate

    result = await delegate(service, tools.settings, principal, parent_id, {"tasks": tasks})
    assert result["status"] == "failed" and result["children"] == []
    assert await child_rows(service, parent_id) == []


class GraphModel:
    def __init__(self, required=True, fail_root=False):
        self.required = required
        self.fail_root = fail_root
        self.successor_input = None

    async def chat(self, messages, **kwargs):
        task = next(item["content"] for item in reversed(messages) if item["role"] == "user")
        result_exists = any(item["role"] == "tool" for item in messages)
        calls = []
        answer = "已完成"
        if task == "图主任务" and not result_exists:
            tasks = [
                {"id": "a", "message": "根任务", "acceptance": {"required_tools": ["calculator"]}},
                {"id": "b", "message": "后继任务", "depends_on": ["a"]},
            ]
            calls = [
                {
                    "id": "graph",
                    "type": "function",
                    "function": {"name": "delegate", "arguments": json.dumps({"tasks": tasks})},
                }
            ]
        elif task == "根任务" and self.required and not result_exists:
            if self.fail_root:
                raise RuntimeError("根任务执行失败")
            calls = [
                {
                    "id": "calc",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
                }
            ]
        elif "后继任务" in task:
            self.successor_input = task
        return SimpleNamespace(
            content="" if calls else answer,
            model_id="合同模型",
            usage={},
            raw={"choices": [{"message": {"tool_calls": calls}}]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("has_evidence", [True, False])
async def test_dependency_created_only_after_independent_acceptance(tmp_path, has_evidence):
    model = GraphModel(has_evidence)
    service, _, principal = await make_service(tmp_path, model, max_steps=10)
    try:
        parent = await service.create_run(principal, "图主任务", "graph")
        assert (await finished(service, principal, parent["id"]))["status"] == "completed"
        result = await delegate_result(service, parent["id"])
        assert len(await child_rows(service, parent["id"])) == (2 if has_evidence else 1)
        assert result[1]["status"] == ("completed" if has_evidence else "blocked")
        if has_evidence:
            assert "不可信" in model.successor_input and "已完成" in model.successor_input
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_collaboration_config_is_top_level_operator_and_idempotent(backend):
    service, _, principal, _ = backend
    run = await service.create_run(principal, "配置任务", "config", collaboration_mode="team")
    assert run["collaboration_mode"] == "team" and run["project_mode"] is None
    assert (await service.create_run(principal, "配置任务", "config", collaboration_mode="team"))[
        "id"
    ] == run["id"]
    with pytest.raises(HarnessError, match="幂等"):
        await service.create_run(principal, "配置任务", "config", collaboration_mode="fork")
    with pytest.raises(HarnessError):
        await service.create_run(principal, "配置任务", "project", project_mode="fork")
    with pytest.raises(HarnessError):
        await service.create_run(
            Principal(principal.user_id, principal.tenant_id, "viewer"),
            "配置任务",
            "viewer",
            collaboration_mode="team",
        )


@pytest.mark.asyncio
async def test_child_creation_rejects_unknown_read_only_looking_tool(backend):
    with pytest.raises(HarnessError, match="只读"):
        await child_for(backend, ["arbitrary_read"])


@pytest.mark.asyncio
async def test_project_prepare_refuses_child_before_mutating_workspace(backend, monkeypatch):
    service, tools, principal, _ = backend
    service.settings.repository_root = tools.settings.data_dir
    child = await child_for(backend)
    touched = []

    async def unsafe_prepare(*args):
        touched.append(True)
        return {"mode": "fork"}

    monkeypatch.setattr(tools.projects, "prepare", unsafe_prepare)
    with pytest.raises(HarnessError):
        await tools.execute("project_prepare", {"mode": "fork"}, principal, child["id"])
    assert touched == []


@pytest.mark.asyncio
async def test_workspace_read_rejects_caller_supplied_source(backend):
    from jsonschema import ValidationError

    _, tools, principal, parent_id = backend
    child = await child_for(backend)
    with pytest.raises(ValidationError):
        await tools.execute(
            "file_read", {"path": "中文.py", "source_run_id": parent_id}, principal, child["id"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["empty", "failed", "wrong_parent", "foreign_owner", "foreign_tenant"]
)
async def test_acceptance_rejects_invalid_child_identity_or_terminal_answer(backend, failure):
    service, _, principal, parent_id = backend
    child = await child_for(backend)
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, child["id"])
        row.status, row.answer = "completed", "答案"
        if failure == "empty":
            row.answer = "   "
        elif failure == "failed":
            row.status = "failed"
        elif failure == "wrong_parent":
            row.parent_run_id = "另一个父任务"
        elif failure == "foreign_owner":
            row.owner_id = "其他所有者"
        elif failure == "foreign_tenant":
            row.tenant_id = "其他租户"
    if failure.startswith("foreign"):
        with pytest.raises(HarnessError):
            await service.verify_collaboration_child(principal, parent_id, child["id"], [])
    else:
        result = await service.verify_collaboration_child(principal, parent_id, child["id"], [])
        assert result["status"] == "rejected" and result["reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fork", "worktree"])
async def test_successful_project_tool_marks_prepared_parent_view(backend, monkeypatch, mode):
    from tests.test_workspace_tool_boundary import archive_bytes

    service, tools, principal, parent_id = backend
    service.settings.repository_root = tools.settings.data_dir

    async def repository_archive(root, *arguments):
        if arguments[0] == "archive":
            return archive_bytes(("目录/中文.py", "准备后的代码"))
        return b""

    monkeypatch.setattr(tools.projects, "_git", repository_archive)
    result = await tools.execute("project_prepare", {"mode": mode}, principal, parent_id)
    assert result["mode"] == mode
    child = await child_for(backend)
    content = await tools.execute("file_read", {"path": "目录/中文.py"}, principal, child["id"])
    assert content == "准备后的代码"
    async with service.store.sessions() as session:
        parent = await session.get(Run, parent_id)
        assert parent.config["project_prepared"] == {"mode": mode}


@pytest.mark.asyncio
async def test_failed_project_prepare_does_not_publish_parent_view(backend, monkeypatch):
    service, tools, principal, parent_id = backend
    service.settings.repository_root = tools.settings.data_dir

    async def rejected_repository(*args):
        raise RuntimeError("仓库准备失败")

    monkeypatch.setattr(tools.projects, "_git", rejected_repository)
    with pytest.raises(RuntimeError):
        await tools.execute("project_prepare", {"mode": "fork"}, principal, parent_id)
    async with service.store.sessions() as session:
        parent = await session.get(Run, parent_id)
        assert "project_prepared" not in parent.config


@pytest.mark.asyncio
async def test_nested_collaboration_config_is_rejected_without_creation(backend):
    service, _, principal, parent_id = backend
    with pytest.raises(HarnessError):
        await service.create_run(
            principal,
            "子任务",
            "nested-config",
            parent_run_id=parent_id,
            allowed_tools=["calculator"],
            collaboration_mode="team",
        )
    assert await child_rows(service, parent_id) == []


@pytest.mark.asyncio
async def test_schema_invalid_delegate_returns_known_failure_without_children(backend):
    service, tools, principal, parent_id = backend
    result = await tools.execute(
        "delegate", {"tasks": [{"message": "缺少ID"}]}, principal, parent_id
    )
    assert result["status"] == "failed" and result["children"] == []
    assert await child_rows(service, parent_id) == []


@pytest.mark.asyncio
async def test_run_mutation_responses_keep_collaboration_config(backend):
    service, _, principal, _ = backend
    run = await service.create_run(
        principal, "配置输出", "mutation-config", collaboration_mode="fork"
    )
    assert (await service.approve(principal, run["id"], False))["collaboration_mode"] == "fork"
    cancelled = await service.cancel(principal, run["id"])
    assert cancelled["collaboration_mode"] == "fork" and cancelled["project_mode"] is None


@pytest.mark.asyncio
async def test_omitted_collaboration_options_preserve_legacy_idempotency_hash(backend):
    from app.harness.security import canonical, digest

    service, _, principal, _ = backend
    run = await service.create_run(principal, "旧版本请求", "legacy-key")
    legacy = {
        "message": "旧版本请求",
        "session_id": None,
        "mode": "react",
        "model": None,
        "parent_run_id": None,
        "allowed_tools": None,
        "max_steps": None,
    }
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, run["id"])
        row.payload_hash = digest(canonical(legacy))
    assert (await service.create_run(principal, "旧版本请求", "legacy-key"))["id"] == run["id"]


@pytest.mark.asyncio
async def test_failed_predecessor_blocks_successor_without_creation(tmp_path):
    service, _, principal = await make_service(tmp_path, GraphModel(fail_root=True), max_steps=10)
    try:
        parent = await service.create_run(principal, "图主任务", "failure-graph")
        assert (await finished(service, principal, parent["id"]))["status"] == "completed"
        result = await delegate_result(service, parent["id"])
        assert result[0]["status"] == "failed" and result[0]["acceptance"]["status"] == "rejected"
        assert result[1]["status"] == "blocked" and result[1]["run_id"] is None
        assert len(await child_rows(service, parent["id"])) == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_object_roots_run_in_parallel_and_preserve_graph_events(tmp_path):
    import asyncio

    class ParallelModel:
        def __init__(self):
            self.started = set()
            self.both_started = asyncio.Event()
            self.release = asyncio.Event()

        async def chat(self, messages, **kwargs):
            task = next(item["content"] for item in reversed(messages) if item["role"] == "user")
            calls = []
            if task == "并行主控" and not any(item["role"] == "tool" for item in messages):
                tasks = [{"id": "a", "message": "并行甲"}, {"id": "b", "message": "并行乙"}]
                calls = [
                    {
                        "id": "parallel",
                        "type": "function",
                        "function": {"name": "delegate", "arguments": json.dumps({"tasks": tasks})},
                    }
                ]
            elif task in {"并行甲", "并行乙"}:
                self.started.add(task)
                if len(self.started) == 2:
                    self.both_started.set()
                await self.release.wait()
            return SimpleNamespace(
                content="" if calls else "并行任务答案",
                model_id="合同模型",
                usage={},
                raw={"choices": [{"message": {"tool_calls": calls}}]},
            )

    model = ParallelModel()
    service, _, principal = await make_service(tmp_path, model, max_steps=10)
    try:
        parent = await service.create_run(principal, "并行主控", "parallel-graph")
        async with asyncio.timeout(5):
            await model.both_started.wait()
        assert len(await child_rows(service, parent["id"])) == 2
        model.release.set()
        assert (await finished(service, principal, parent["id"]))["status"] == "completed"
        result = await delegate_result(service, parent["id"])
        assert all(node["acceptance"]["status"] == "verified" for node in result)
        events = await service.events(principal, parent["id"])
        graph = [event for event in events if event["type"] == "collaboration_graph"][-1]
        assert [node["status"] for node in graph["data"]["nodes"]] == ["completed", "completed"]
        assert len([event for event in events if event["type"] == "collaboration_acceptance"]) == 2
    finally:
        model.release.set()
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_status", ["failed", "error", "warning", "degraded", "done"])
async def test_done_ledger_explicit_failure_payload_cannot_pass_acceptance(backend, payload_status):
    service, _, principal, parent_id = backend
    child = await child_for(backend)
    async with service.store.sessions.begin() as session:
        row = await session.get(Run, child["id"])
        row.status, row.answer = "completed", "声称已完成"
        session.add(
            ToolCall(
                id=uid(),
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                run_id=row.id,
                call_id="payload-evidence",
                name="calculator",
                arguments={},
                arguments_hash="hash",
                status="done",
                result={"status": payload_status, "error": "工具状态证据"},
            )
        )
    result = await service.verify_collaboration_child(
        principal, parent_id, child["id"], ["calculator"]
    )
    failed = payload_status in {"failed", "error"}
    assert result["status"] == ("rejected" if failed else "verified")
    assert bool(result["evidence"]) != failed
    if failed:
        assert any("必需工具" in reason for reason in result["reasons"])
        plain = await service.verify_collaboration_child(principal, parent_id, child["id"], [])
        assert plain["status"] == "rejected"


@pytest.mark.asyncio
async def test_cancel_after_child_commit_before_response_cleans_whole_parallel_batch(
    backend, monkeypatch
):
    import asyncio

    from app.harness_tools.delegation import delegate

    service, tools, principal, parent_id = backend
    original_create = service.create_run
    committed = []
    both_committed = asyncio.Event()
    never_return = asyncio.Event()

    async def committed_before_response(*args, **kwargs):
        child = await original_create(*args, **kwargs)
        committed.append(child["id"])
        if len(committed) == 2:
            both_committed.set()
        await never_return.wait()
        return child

    monkeypatch.setattr(service, "create_run", committed_before_response)
    task = asyncio.create_task(
        delegate(
            service,
            tools.settings,
            principal,
            parent_id,
            {"tasks": [{"id": "a", "message": "甲"}, {"id": "b", "message": "乙"}]},
        )
    )
    try:
        async with asyncio.timeout(3):
            await both_committed.wait()
        assert len(await child_rows(service, parent_id)) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
        rows = await child_rows(service, parent_id)
        assert {row.id for row in rows} == set(committed)
        assert all(row.status == "cancelled" for row in rows)
        assert all(row.config["max_steps"] == 3 for row in rows)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [4000, 24000])
async def test_long_successor_goal_keeps_bounded_summary_and_full_artifact(
    backend, monkeypatch, budget
):
    from app.harness.context import bounded_messages
    from app.harness_tools.delegation import delegate

    service, tools, principal, parent_id = backend
    tools.settings.context_chars = budget
    original_create = service.create_run
    created = []
    upstream_answer = "关键证据" + '"' * 500 + "全文结束"

    async def completed_child(*args, **kwargs):
        child = await original_create(*args, **kwargs)
        created.append(child["id"])
        async with service.store.sessions.begin() as session:
            row = await session.get(Run, child["id"])
            row.status, row.answer = (
                "completed",
                upstream_answer if len(created) == 1 else "后继完成",
            )
        return child

    monkeypatch.setattr(service, "create_run", completed_child)
    result = await delegate(
        service,
        tools.settings,
        principal,
        parent_id,
        {
            "tasks": [
                {"id": "a", "message": "上游任务"},
                {"id": "b", "message": "目标" + '"' * 11998, "depends_on": ["a"]},
            ]
        },
    )
    async with service.store.sessions() as session:
        successor = await session.get(Run, result[1]["run_id"])
        messages = bounded_messages(successor.messages, budget)
    content = "\n".join(message.get("content", "") for message in messages)
    assert "关键证据" in content and "不可信" in content
    assert result[0]["run_id"] in content and result[0]["artifact_id"] in content
    assert len(json.dumps(messages, ensure_ascii=False)) <= budget
    artifact = await service.get_asset(principal, result[0]["artifact_id"])
    assert artifact["content"] == upstream_answer
