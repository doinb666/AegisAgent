"""使用假模型验证真实数据库状态，不声称外部模型验收。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.context import bounded_messages
from app.harness.models import Run, ToolCall
from app.harness.store import uid


class FakeModel:
    def __init__(self, name=None, repeat=False, plan_invalid=False, delay=0):
        self.name = name
        self.repeat = repeat
        self.plan_invalid = plan_invalid
        self.delay = delay
        self.calls = 0
        self.messages = []

    async def chat(self, messages, model_preference=None, **kwargs):
        self.calls += 1
        self.messages = messages
        await asyncio.sleep(self.delay)
        if self.plan_invalid and not kwargs.get("tools"):
            return SimpleNamespace(content="无效JSON", model_id="fake", usage={}, raw={})
        tool_calls = []
        if self.name and (self.repeat or self.calls == 1):
            tool_calls = [
                {
                    "id": "call-" + str(self.calls),
                    "type": "function",
                    "function": {"name": self.name, "arguments": '{"value":"示例"}'},
                }
            ]
        return SimpleNamespace(
            content="完成" if not tool_calls else "",
            model_id="fake",
            usage={"total_tokens": 1},
            raw={"choices": [{"message": {"tool_calls": tool_calls}}]},
        )


class FakeTools:
    def __init__(self, name="read", output=None, unknown=False, delay=0):
        self.name = name
        self.output = output if output is not None else {"verified": True}
        self.unknown = unknown
        self.delay = delay
        self.executions = 0

    def catalog(self, principal):
        return [
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": "测试工具",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                },
            }
        ]

    def requires_approval(self, name, arguments):
        return name == "write"

    async def execute(self, name, arguments, principal, run_id):
        self.executions += 1
        await asyncio.sleep(self.delay)
        if self.unknown:
            raise RuntimeError("副作用后连接断开")
        return self.output


async def setup_service(tmp_path, model=None, tools=None, **settings):
    service = HarnessService(
        HarnessSettings(data_dir=tmp_path, database_url="", max_concurrent_runs=2, **settings),
        model,
        tools,
    )
    await service.initialize()
    user = await service.register("alice", "password123")
    return service, Principal(user["user_id"], user["tenant_id"], user["role"])


async def wait_status(service, principal, run_id, statuses, timeout=5):
    async with asyncio.timeout(timeout):
        while True:
            run = await service.get_run(principal, run_id)
            if run["status"] in statuses:
                return run
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_approval_has_zero_effects_and_executes_once(tmp_path):
    tools = FakeTools("write")
    service, principal = await setup_service(tmp_path, FakeModel("write"), tools)
    try:
        run = await service.create_run(principal, "写入", "key")
        waiting = await wait_status(service, principal, run["id"], {"waiting_approval"})
        assert tools.executions == 0
        assert waiting["approval"]["hash"] and waiting["approval"]["call_id"]
        await asyncio.gather(
            *(
                service.approve(
                    principal,
                    run["id"],
                    True,
                    waiting["approval"]["call_id"],
                    waiting["approval"]["hash"],
                )
                for _ in range(10)
            )
        )
        done = await wait_status(service, principal, run["id"], {"completed", "failed"})
        assert done["status"] == "completed", done
        await service.approve(principal, run["id"], True)
        assert tools.executions == 1
        async with service.store.sessions() as session:
            persisted = await session.get(Run, run["id"])
            assert persisted.messages[-2]["role"] == "tool"
            assert (
                persisted.messages[-3]["tool_calls"][0]["id"]
                == persisted.messages[-2]["tool_call_id"]
            )
        # 自评分不能激活；真实用户反馈和工具证据同时存在才激活。
        drafts = await service.list_assets(principal, "skill")
        assert drafts and all(asset["status"] == "draft" for asset in drafts)
        await service.feedback(principal, run["id"], True)
        assert (await service.list_assets(principal, "skill"))[0]["status"] == "active"
        await service.feedback(principal, run["id"], False, "结果不正确")
        assert (await service.list_assets(principal, "skill"))[0]["status"] == "draft"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_budget_and_missing_model_fail_explicitly(tmp_path):
    service, principal = await setup_service(tmp_path / "missing")
    try:
        run = await service.create_run(principal, "任务", "missing")
        failed = await wait_status(service, principal, run["id"], {"failed"})
        assert "模型未配置" in failed["error"]
    finally:
        await service.close()
    tools = FakeTools()
    service, principal = await setup_service(
        tmp_path / "budget", FakeModel("read", repeat=True), tools, max_steps=2
    )
    try:
        run = await service.create_run(principal, "循环任务", "budget")
        failed = await wait_status(service, principal, run["id"], {"failed"})
        assert "步数" in failed["error"] and failed["step"] == 2
        assert tools.executions == 2
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_cancel_during_model_and_unknown_tool_no_replay(tmp_path):
    service, principal = await setup_service(tmp_path / "cancel", FakeModel(delay=10))
    try:
        run = await service.create_run(principal, "慢任务", "cancel")
        await wait_status(service, principal, run["id"], {"running"})
        await service.cancel(principal, run["id"])
        assert (await service.get_run(principal, run["id"]))["status"] == "cancelled"
    finally:
        await service.close()
    tools = FakeTools(unknown=True)
    service, principal = await setup_service(tmp_path / "unknown", FakeModel("read"), tools)
    try:
        run = await service.create_run(principal, "读取", "unknown")
        await wait_status(service, principal, run["id"], {"interrupted"})
        assert tools.executions == 1
        assert await service.store.claim("other-worker") is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_expired_lease_with_unknown_effect_interrupted(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path))
    await service.store.initialize()
    try:
        user = await service.register("alice", "password123")
        principal = Principal(user["user_id"], user["tenant_id"], "admin")
        run = await service.create_run(principal, "任务", "lease")
        async with service.store.sessions.begin() as session:
            persisted = await session.get(Run, run["id"])
            persisted.status, persisted.lease_until, persisted.lease_owner = "running", 0, "dead"
            session.add(
                ToolCall(
                    id=uid(),
                    tenant_id=principal.tenant_id,
                    owner_id=principal.user_id,
                    run_id=run["id"],
                    call_id="unknown",
                    name="write",
                    arguments={},
                    arguments_hash="hash",
                    status="started",
                )
            )
        assert await service.store.claim("new") is None
        assert (await service.get_run(principal, run["id"]))["status"] == "interrupted"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_large_results_externalized_and_event_replay(tmp_path):
    output = {"data": "证据" * 10000}
    service, principal = await setup_service(tmp_path, FakeModel("read"), FakeTools(output=output))
    try:
        run = await service.create_run(principal, "读取", "big")
        await wait_status(service, principal, run["id"], {"completed"})
        artifacts = await service.list_assets(principal, "artifact")
        assert len(artifacts) == 1
        assert (
            json.loads((await service.get_asset(principal, artifacts[0]["id"]))["content"])
            == output
        )
        events = await service.events(principal, run["id"])
        assert events and await service.events(principal, run["id"], events[-1]["id"]) == []
        assert any(event["type"] == "tool_result" for event in events)
        async with service.store.sessions() as session:
            persisted = await session.get(Run, run["id"])
            result = json.loads(
                next(
                    message["content"]
                    for message in persisted.messages
                    if message["role"] == "tool"
                )
            )
            assert result["artifact_id"] == artifacts[0]["id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_viewer_and_allowed_tools_enforced(tmp_path):
    tools = FakeTools("write")
    service, principal = await setup_service(tmp_path, FakeModel("write"), tools)
    try:
        viewer = await service.create_user(principal, "viewer", "password123", "viewer")
        viewer = Principal(viewer["user_id"], viewer["tenant_id"], "viewer")
        run = await service.create_run(viewer, "写入", "viewer")
        failed = await wait_status(service, viewer, run["id"], {"failed"})
        assert "viewer" in failed["error"] and tools.executions == 0
        service.model_router = FakeModel("write")
        run = await service.create_run(principal, "禁止工具", "allowed", allowed_tools=[])
        failed = await wait_status(service, principal, run["id"], {"failed"})
        assert "清单" in failed["error"] and tools.executions == 0
    finally:
        await service.close()


def test_context_budget_preserves_tool_groups():
    messages = [
        {"role": "system", "content": "稳定前缀"},
        {"role": "user", "content": "旧消息" * 3000},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "证据"},
        {"role": "assistant", "content": "完成"},
    ]
    bounded = bounded_messages(messages, 4000)
    assert len(json.dumps(bounded, ensure_ascii=False)) <= 4000
    for index, message in enumerate(bounded):
        if message["role"] == "tool":
            assert bounded[index - 1]["tool_calls"][0]["id"] == message["tool_call_id"]


@pytest.mark.asyncio
async def test_cross_service_approval_recovery_executes_once(tmp_path):
    tools = FakeTools("write")
    service, principal = await setup_service(tmp_path, FakeModel("write"), tools)
    second = HarnessService(service.settings, service.model_router, tools)
    try:
        run = await service.create_run(principal, "写入", "approval-race")
        waiting = await wait_status(service, principal, run["id"], {"waiting_approval"})
        # 重启后依赖持久checkpoint恢复，不要求原进程保持连接。
        await service.close()
        await second.initialize()
        third = HarnessService(second.settings)
        try:
            await asyncio.gather(
                second.approve(
                    principal,
                    run["id"],
                    True,
                    waiting["approval"]["call_id"],
                    waiting["approval"]["hash"],
                ),
                third.approve(
                    principal,
                    run["id"],
                    True,
                    waiting["approval"]["call_id"],
                    waiting["approval"]["hash"],
                ),
            )
            done = await wait_status(second, principal, run["id"], {"completed", "failed"})
            assert done["status"] == "completed", done
            assert tools.executions == 1
        finally:
            await third.close()
    finally:
        await second.close()


@pytest.mark.asyncio
async def test_plan_fallback_and_session_window(tmp_path):
    model = FakeModel("read", plan_invalid=True)
    service, principal = await setup_service(tmp_path, model, FakeTools())
    try:
        run = await service.create_run(principal, "先前用户原文", "plan", mode="plan")
        await wait_status(service, principal, run["id"], {"completed"})
        events = await service.events(principal, run["id"])
        assert any(event["type"] == "plan_fallback" for event in events)
        next_run = await service.create_run(
            principal, "下一轮", "next", session_id=run["session_id"]
        )
        await wait_status(service, principal, next_run["id"], {"completed"})
        assert any("先前用户原文" in str(message.get("content")) for message in model.messages)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_running_tool_cancel_does_not_resurrect_run(tmp_path):
    tools = FakeTools(delay=10)
    service, principal = await setup_service(tmp_path, FakeModel("read"), tools)
    try:
        run = await service.create_run(principal, "慢工具", "cancel-tool")
        async with asyncio.timeout(5):
            while tools.executions == 0:
                await asyncio.sleep(0.02)
        await service.cancel(principal, run["id"])
        await asyncio.sleep(0.05)
        assert (await service.get_run(principal, run["id"]))["status"] == "cancelled"
        async with service.store.sessions() as session:
            tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run["id"]))
            assert tool.status == "started"
        assert await service.store.claim("recover") is None
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_completed_tool_checkpoint_reuses_result_after_restart(tmp_path):
    output = {"data": "证据" * 10000}
    tools = FakeTools(output=output)
    service = HarnessService(HarnessSettings(data_dir=tmp_path), FakeModel(), tools)
    await service.store.initialize()
    user = await service.register("alice", "password123")
    principal = Principal(user["user_id"], user["tenant_id"], "admin")
    run = await service.create_run(principal, "恢复已完成工具", "resume")
    artifact = await service.put_artifact(
        principal,
        "已有输出",
        json.dumps(output),
        run["id"],
        "saved",
    )
    from app.harness.security import canonical, digest

    async with service.store.sessions.begin() as session:
        persisted = await session.get(Run, run["id"])
        persisted.messages = persisted.messages + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "saved",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            }
        ]
        persisted.step = 1
        session.add(
            ToolCall(
                id=uid(),
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                run_id=run["id"],
                call_id="saved",
                name="read",
                arguments={},
                status="done",
                result=output,
                arguments_hash=digest(
                    canonical({"call_id": "saved", "name": "read", "arguments": {}})
                ),
            )
        )
    await service.initialize()
    try:
        done = await wait_status(service, principal, run["id"], {"completed", "failed"})
        assert done["status"] == "completed", done
        assert tools.executions == 0
        assert len(await service.list_assets(principal, "artifact")) == 1
        assert (await service.get_asset(principal, artifact["id"]))["id"] == artifact["id"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_stale_approval_cannot_approve_next_tool(tmp_path):
    tools = FakeTools("write")
    service, principal = await setup_service(tmp_path, FakeModel("write", repeat=True), tools)
    try:
        run = await service.create_run(principal, "连续写操作", "stale")
        first = await wait_status(service, principal, run["id"], {"waiting_approval"})
        original = first["approval"]
        await service.approve(principal, run["id"], True, original["call_id"], original["hash"])
        async with asyncio.timeout(5):
            while True:
                current = await service.get_run(principal, run["id"])
                if (
                    current["status"] == "waiting_approval"
                    and current["approval"]["call_id"] != original["call_id"]
                ):
                    break
                await asyncio.sleep(0.02)
        with pytest.raises(HarnessError) as error:
            await service.approve(principal, run["id"], True, original["call_id"], original["hash"])
        assert error.value.status_code == 409
        assert tools.executions == 1
        with pytest.raises(HarnessError) as error:
            await service.approve(principal, run["id"], True)
        assert error.value.status_code == 422
        await service.cancel(principal, run["id"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_child_pool_progresses_when_root_worker_is_waiting(tmp_path):
    class DelegatingTools(FakeTools):
        async def execute(self, name, arguments, principal, run_id):
            self.executions += 1
            child = await self.service.create_run(
                principal,
                "只读子任务",
                "child",
                parent_run_id=run_id,
                allowed_tools=["read"],
                max_steps=2,
            )
            finished = await wait_status(
                self.service, principal, child["id"], {"completed", "failed"}
            )
            return {"child": child["id"], "status": finished["status"]}

    tools = DelegatingTools("delegate")
    service = HarnessService(
        HarnessSettings(data_dir=tmp_path, max_concurrent_runs=1), FakeModel("delegate"), tools
    )
    tools.service = service
    await service.initialize()
    try:
        user = await service.register("alice", "password123")
        principal = Principal(user["user_id"], user["tenant_id"], "admin")
        run = await service.create_run(principal, "委派任务", "parent")
        done = await wait_status(service, principal, run["id"], {"completed", "failed"})
        assert done["status"] == "completed", done
        children = [
            item
            for item in await service.list_runs(principal)
            if item["parent_run_id"] == run["id"]
        ]
        assert children[0]["status"] == "completed"
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_viewer_large_read_output_is_saved_internally(tmp_path):
    service, admin = await setup_service(
        tmp_path, FakeModel("read"), FakeTools(output={"data": "x" * 10000})
    )
    try:
        user = await service.create_user(admin, "viewer", "password123", "viewer")
        viewer = Principal(user["user_id"], user["tenant_id"], "viewer")
        run = await service.create_run(viewer, "读取长证据", "viewer-read")
        done = await wait_status(service, viewer, run["id"], {"completed", "failed"})
        assert done["status"] == "completed", done
        assets = await service.list_assets(viewer, "artifact")
        assert assets
        with pytest.raises(HarnessError):
            await service.transition_asset(viewer, assets[0]["id"], "retired")
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "review", "deny"])
async def test_independent_risk_classification_keeps_approval_gate(tmp_path, decision):
    class RiskModel(FakeModel):
        async def chat(self, messages, model_preference=None, **kwargs):
            if not kwargs.get("tools"):
                self.review_calls += 1
                assert len(messages) == 2 and messages[0]["role"] == "system"
                assert "用户任务秘密" not in str(messages)
                return SimpleNamespace(
                    content=json.dumps({"risk": decision, "reason": "测试分类"}),
                    model_id="fake-risk",
                    usage={},
                    raw={},
                )
            return await super().chat(messages, model_preference, **kwargs)

    model = RiskModel("write")
    model.review_calls = 0
    tools = FakeTools("write")
    service, principal = await setup_service(tmp_path, model, tools)
    try:
        run = await service.create_run(principal, "用户任务秘密", "risk")
        current = await wait_status(service, principal, run["id"], {"waiting_approval", "failed"})
        assert tools.executions == 0
        reviews = [
            event
            for event in await service.events(principal, run["id"])
            if event["type"] == "risk_review"
        ]
        assert reviews[0]["data"]["risk"] == decision
        if decision == "deny":
            assert current["status"] == "failed"
        else:
            assert current["status"] == "waiting_approval"
            approval = current["approval"]
            await service.approve(principal, run["id"], True, approval["call_id"], approval["hash"])
            done = await wait_status(service, principal, run["id"], {"completed", "failed"})
            assert done["status"] == "completed", done
            assert tools.executions == 1
        assert model.review_calls == 1
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_risk_budget_and_transient_claim_failure(tmp_path):
    tools = FakeTools("write")
    service, principal = await setup_service(
        tmp_path / "budget", FakeModel("write"), tools, max_steps=1
    )
    try:
        run = await service.create_run(principal, "预算不足写操作", "budget")
        failed = await wait_status(service, principal, run["id"], {"failed"})
        assert "风险审查" in failed["error"] and tools.executions == 0
    finally:
        await service.close()
    service = HarnessService(
        HarnessSettings(data_dir=tmp_path / "claim", max_concurrent_runs=1, max_child_runs=0),
        FakeModel(),
    )
    original = service.store.claim
    attempts = 0

    async def flaky_claim(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("数据库短暂不可用")
        return await original(*args, **kwargs)

    service.store.claim = flaky_claim
    await service.initialize()
    try:
        user = await service.register("alice", "password123")
        principal = Principal(user["user_id"], user["tenant_id"], "admin")
        run = await service.create_run(principal, "自动恢复", "retry")
        await wait_status(service, principal, run["id"], {"completed"})
        assert attempts >= 2 and not service.workers[0].task.done()
    finally:
        await service.close()
