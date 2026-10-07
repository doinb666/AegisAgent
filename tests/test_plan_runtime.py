"""计划执行使用真实数据库与确定性模型验证事务、预算和恢复。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.errors import HarnessError
from app.harness.models import Event, Run, ToolCall, User
from app.harness.runtime import Worker
from tests.test_harness_runtime import FakeTools


def plan_step(objective="节点", kind="answer", required=None, dependencies=None):
    return {
        "objective": objective,
        "inputs": {"instruction": objective, "from_steps": dependencies or []},
        "acceptance": {"output_kind": kind, "required_tools": required or []},
    }


def tool_message(name="read", call_id="c1", arguments=None):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments or {"value": "示例"})},
    }


class ScriptModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        item = self.responses.pop(0)
        content, calls = item if isinstance(item, tuple) else (item, [])
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False)
        return SimpleNamespace(
            content=content,
            model_id="script",
            usage={},
            raw={"choices": [{"message": {"tool_calls": calls}}]},
        )


async def prepared(tmp_path, model, tools=None, **settings):
    service = HarnessService(
        HarnessSettings(
            data_dir=tmp_path,
            evolution_enabled=False,
            risk_review_enabled=settings.pop("risk_review_enabled", False),
            **settings,
        ),
        model,
        tools or FakeTools(),
    )
    await service.store.initialize()
    user = await service.register("plan-runtime", "test-password")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    created = await service.create_run(principal, "原始目标", "plan", mode="plan")
    worker = Worker(service)
    assert await service.store.claim(worker.id) == created["id"]
    return service, principal, worker, created["id"]


async def persisted(service, run_id):
    async with service.store.sessions() as session:
        return await session.get(Run, run_id)


async def drive(worker, run_id):
    try:
        await worker.execute(run_id)
    except Exception as exc:
        await worker.finish(run_id, "failed", error=str(exc))


@pytest.mark.asyncio
async def test_two_nodes_do_not_finish_at_first_answer_and_share_actions(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step("前序"), plan_step("后序", dependencies=[0])]}, "前序输出", "最终答复"
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    try:
        await drive(worker, run_id)
        run = await persisted(service, run_id)
        assert run.answer == "最终答复"
        assert run.status == "completed"
        assert run.step == 6  # 规划、两个模型、两个验收、最终封口。
        assert [node["status"] for node in run.config["plan"]["nodes"]] == ["completed"] * 2
        assert "前序输出" in json.dumps(model.calls[2][0], ensure_ascii=False)
        assert '"规划后可用动作":9' in model.calls[0][0][-1]["content"]
        async with service.store.sessions() as session:
            events = (await session.scalars(select(Event).where(Event.run_id == run_id))).all()
        assert [event.data["node_id"] for event in events if event.type == "node_started"] == [
            "n0-1",
            "n0-2",
        ]
        assert all(
            event.data["plan_revision"] == 0 for event in events if event.type == "node_accepted"
        )
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_tool_evidence_uses_real_ledger_then_bounded_summary(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()]),
        "交付摘要",
    )
    tools = FakeTools()
    service, _, worker, run_id = await prepared(tmp_path, model, tools)
    try:
        await drive(worker, run_id)
        run = await persisted(service, run_id)
        assert run.status == "completed", run.error
        node = run.config["plan"]["nodes"][0]
        assert node["acceptance_result"]["contract_satisfied"]
        assert node["acceptance_result"]["business_success_verified"] is False
        assert node["tool_call_ids"]
        assert tools.executions == 1
        assert run.step == 5
        assert not model.calls[-1][1].get("tools")
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_impossible_plan_rejected_before_tool_effect(tmp_path):
    model = ScriptModel({"steps": [plan_step(kind="tool_evidence", required=["read"])]}, "预算不足")
    tools = FakeTools()
    service, _, worker, run_id = await prepared(tmp_path, model, tools, max_steps=3)
    try:
        await drive(worker, run_id)
        run = await persisted(service, run_id)
        assert tools.executions == 0
        assert run.step <= 3
        async with service.store.sessions() as session:
            events = (await session.scalars(select(Event).where(Event.run_id == run_id))).all()
        assert any(event.type == "plan_fallback" for event in events)
        assert run.config["budget_policy"] == "shared_actions_v1"
    finally:
        await service.store.close()


async def reclaim(service, run_id):
    async with service.store.transaction() as session:
        run = await session.get(Run, run_id)
        run.lease_until = 0
    worker = Worker(service)
    assert await service.store.claim(worker.id) == run_id
    return worker


@pytest.mark.asyncio
async def test_known_node_answer_checkpoint_resumes_without_model_regeneration(
    tmp_path, monkeypatch
):
    model = ScriptModel({"steps": [plan_step()]}, "已知节点答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    checkpoint = worker.checkpoint

    async def crash_after_answer(*args, **kwargs):
        await checkpoint(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "node":
            raise RuntimeError("模拟完整答案检查点后进程崩溃")

    monkeypatch.setattr(worker, "checkpoint", crash_after_answer)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "running"
        assert saved.config["plan"]["nodes"][0]["status"] == "accepting"
        resumed = await reclaim(service, run_id)
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed"
        assert saved.answer == "已知节点答复"
        assert saved.step == 4
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_acceptance_failure_rolls_back_consumption_and_budget(tmp_path, monkeypatch):
    model = ScriptModel({"steps": [plan_step()]}, "已知答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = service.store.emit

    def crash_on_acceptance(session, run, event, data):
        if event == "node_accepted":
            raise RuntimeError("模拟验收事务提交前崩溃")
        original(session, run, event, data)

    monkeypatch.setattr(service.store, "emit", crash_on_acceptance)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        before = await persisted(service, run_id)
        assert before.step == 2
        assert before.config["model_result_pending"]["purpose"] == "node"
        monkeypatch.setattr(service.store, "emit", original)
        resumed = await reclaim(service, run_id)
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.step == 4
        assert saved.status == "completed"
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_replan_preserves_completed_prefix_and_recovers_its_known_response(
    tmp_path, monkeypatch
):
    model = ScriptModel(
        {"steps": [plan_step("前序"), plan_step("后序", required=["read"])]},
        "保留前序",
        "缺少工具",
        {"steps": [plan_step("修订后序")]},
        "修复答复",
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    checkpoint = worker.checkpoint

    async def crash_after_replan(*args, **kwargs):
        await checkpoint(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "replan":
            raise RuntimeError("模拟重规划检查点崩溃")

    monkeypatch.setattr(worker, "checkpoint", crash_after_replan)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        before = await persisted(service, run_id)
        assert before.config["plan"]["replans_used"] == 1
        resumed = await reclaim(service, run_id)
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed", saved.error
        plan = saved.config["plan"]
        assert plan["nodes"][0]["id"] == "n0-1"
        assert plan["nodes"][0]["output"] == "保留前序"
        assert plan["nodes"][1]["id"] == "n1-1"
        assert plan["nodes"][1]["inputs"]["from_nodes"] == ["n0-1"]
        assert plan["history"][0]["nodes"][0]["blocked_reason"]
        assert len(model.calls) == 5
        assert saved.step == 9
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_started_unknown_is_interrupted_and_never_replayed(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]}, ("", [tool_message()])
    )
    service, _, worker, run_id = await prepared(tmp_path, model, FakeTools(unknown=True))
    try:
        await worker.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "interrupted"
        assert saved.config["plan"]["nodes"][0]["status"] == "blocked"
        assert service.tool_executor.executions == 1
        assert await service.store.claim("another") is None
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"status": "rejected"}, {"exit_code": 2}, {"isError": True}])
async def test_done_ledger_with_explicit_failure_does_not_accept(tmp_path, payload):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()]),
        {"steps": [plan_step()]},
        "修复答复",
    )
    service, _, worker, run_id = await prepared(tmp_path, model, FakeTools(output=payload))
    try:
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        failed = saved.config["plan"]["history"][0]["nodes"][0]
        assert failed["status"] == "blocked"
        assert failed["acceptance_result"]["contract_satisfied"] is False
        assert failed["blocked_reason"]
        assert saved.config["plan"]["replans_used"] == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_cancelled_tool_preserves_done_without_node_advancement(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]}, ("", [tool_message()])
    )
    tools = FakeTools()
    service, principal, worker, run_id = await prepared(tmp_path, model, tools)

    async def cancel_inside_execute(*args, **kwargs):
        await service.cancel(principal, run_id)
        return {"value": "真实完成"}

    tools.execute = cancel_inside_execute
    try:
        with pytest.raises(asyncio.CancelledError):
            await worker.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "cancelled"
        node = saved.config["plan"]["nodes"][0]
        assert node["status"] == "blocked"
        assert node["acceptance_result"] is None
        async with service.store.sessions() as session:
            tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id))
        assert tool.status == "done"
        assert tool.result == {"value": "真实完成"}
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_node_pending_identity_mismatch_fails_closed(tmp_path, monkeypatch):
    model = ScriptModel({"steps": [plan_step()]}, "答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def corrupt(*args, **kwargs):
        await original(*args, **kwargs)
        async with service.store.transaction() as session:
            saved = await session.get(Run, run_id)
            pending = saved.config.get("model_result_pending")
            if pending and pending.get("purpose") == "node":
                saved.config = {
                    **saved.config,
                    "model_result_pending": {**pending, "node_id": "other"},
                }

    monkeypatch.setattr(worker, "checkpoint", corrupt)
    try:
        with pytest.raises(HarnessError) as failure:
            await worker.execute(run_id)
        assert failure.value.status_code == 409
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_risk_charge_keeps_node_tool_checkpoint_and_approval_across_restart(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["write"])]},
        ("", [tool_message("write")]),
        {"risk": "review", "reason": "需要人工批准"},
        "已知写入摘要",
    )
    tools = FakeTools("write")
    service, principal, worker, run_id = await prepared(
        tmp_path, model, tools, risk_review_enabled=True
    )
    try:
        await worker.execute(run_id)
        waiting = await persisted(service, run_id)
        assert waiting.status == "waiting_approval"
        assert waiting.step == 3
        assert waiting.config["model_result_pending"]["purpose"] == "node"
        assert waiting.config["plan"]["nodes"][0]["status"] == "running"
        approval = waiting.approval
        await service.approve(principal, run_id, True, approval["call_id"], approval["hash"])
        await service.approve(principal, run_id, True, approval["call_id"], approval["hash"])
        resumed = Worker(service)
        assert await service.store.claim(resumed.id) == run_id
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed", saved.error
        assert tools.executions == 1
        assert saved.step == 6
        assert len(model.calls) == 4
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_plan_rechecks_actual_role_inside_long_running_model_before_tool(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["write"])]},
        ("", [tool_message("write")]),
    )
    service, principal, worker, run_id = await prepared(tmp_path, model, FakeTools("write"))
    original = model.chat

    async def downgrade_before_tool(*args, **kwargs):
        response = await original(*args, **kwargs)
        if len(model.calls) == 2:
            async with service.store.transaction() as session:
                user = await session.get(User, principal.user_id)
                user.role = "viewer"
        return response

    model.chat = downgrade_before_tool
    try:
        with pytest.raises(HarnessError) as failure:
            await worker.execute(run_id)
        assert failure.value.status_code == 403
        assert (await persisted(service, run_id)).approval is None
        assert service.tool_executor.executions == 0
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_replan_reuses_known_side_effect_from_archived_suffix(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(required=["write"])]},
        ("", [tool_message("write", "original-write")]),
        "",
        {"steps": [plan_step(kind="tool_evidence", required=["write"])]},
        ("", [tool_message("write", "new-write")]),
        "历史调用已知摘要",
    )
    tools = FakeTools("write")
    service, principal, worker, run_id = await prepared(tmp_path, model, tools)
    try:
        await worker.execute(run_id)
        waiting = await persisted(service, run_id)
        assert waiting.status == "waiting_approval"
        approval = waiting.approval
        await service.approve(principal, run_id, True, approval["call_id"], approval["hash"])
        resumed = Worker(service)
        assert await service.store.claim(resumed.id) == run_id
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed", saved.error
        assert tools.executions == 1
        assert saved.step == 9
        plan = saved.config["plan"]
        assert plan["nodes"][0]["tool_call_ids"] == plan["history"][0]["nodes"][0]["tool_call_ids"]
        async with service.store.sessions() as session:
            tools_saved = (
                await session.scalars(select(ToolCall).where(ToolCall.run_id == run_id))
            ).all()
            reused = await session.scalar(
                select(Event).where(Event.run_id == run_id, Event.type == "tool_reused")
            )
        assert len(tools_saved) == 1
        assert reused.data["source_id"] == tools_saved[0].id
        assert reused.data["reported_known_result"] is True
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_answer_required_tools_are_in_minimum_plan_cost(tmp_path):
    model = ScriptModel({"steps": [plan_step(required=["read"])]}, "剩余预算答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=4)
    try:
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        assert saved.config.get("plan") is None
        assert saved.config.get("plan_fallback") is True
        assert saved.step == 3
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_model_marker_persists_node_instruction_before_request(tmp_path):
    model = ScriptModel({"steps": [plan_step("明确节点指令")]}, "答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = model.chat
    seen = []

    async def inspect_inflight(messages, **kwargs):
        if len(model.calls) == 1:
            saved = await persisted(service, run_id)
            seen.append(saved)
        return await original(messages, **kwargs)

    model.chat = inspect_inflight
    try:
        await worker.execute(run_id)
        assert seen[0].config["model_inflight"]["purpose"] == "node"
        assert "明确节点指令" in seen[0].messages[-1]["content"]
        assert seen[0].messages[-1]["role"] == "user"
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "",
        {"steps": []},
        {"steps": ["旧格式", plan_step()]},
        {"steps": [plan_step(required=["forbidden"])]},
        " " * 32769,
    ],
    ids=["empty", "no-steps", "mixed", "unauthorized", "raw-limit"],
)
async def test_invalid_plans_fall_back_once_and_keep_budget(tmp_path, raw):
    model = ScriptModel(raw, "有界答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=3)
    try:
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed", saved.error
        assert saved.step == 3
        assert saved.config["budget_policy"] == "shared_actions_v1"
        async with service.store.sessions() as session:
            events = (
                await session.scalars(
                    select(Event).where(Event.run_id == run_id, Event.type == "plan_fallback")
                )
            ).all()
        assert len(events) == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_planning_tool_calls_are_never_executed(tmp_path):
    model = ScriptModel(({"steps": ["节点"]}, [tool_message()]), "降级答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    try:
        await worker.execute(run_id)
        assert service.tool_executor.executions == 0
        assert (await persisted(service, run_id)).config["plan_fallback"]
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_legacy_planned_run_keeps_original_semantics_and_one_event(tmp_path):
    model = ScriptModel("原有答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    try:
        async with service.store.transaction() as session:
            saved = await session.get(Run, run_id)
            saved.config = {**saved.config, "planned": True}
        await worker.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed"
        assert saved.step == 1
        assert "plan" not in saved.config
        assert "budget_policy" not in saved.config
        async with service.store.sessions() as session:
            events = (
                await session.scalars(
                    select(Event).where(
                        Event.run_id == run_id, Event.type == "plan_legacy_compatibility"
                    )
                )
            ).all()
        assert len(events) == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_entire_new_tool_batch_budget_checked_before_first_effect(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message(call_id="one"), tool_message(call_id="two")]),
    )
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=5)
    try:
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "failed"
        assert saved.step == 2
        assert service.tool_executor.executions == 0
        assert "整批" in saved.error
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_done_tool_checkpoint_recovery_supplements_message_without_reexecution(
    tmp_path, monkeypatch
):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()]),
        "汇总",
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def crash_before_tool_message(*args, **kwargs):
        if args[3] == "checkpoint":
            raise RuntimeError("模拟已知工具结果后消息提交前崩溃")
        await original(*args, **kwargs)

    monkeypatch.setattr(worker, "checkpoint", crash_before_tool_message)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        before = await persisted(service, run_id)
        assert before.step == 3
        assert not any(item.get("role") == "tool" for item in before.messages)
        resumed = await reclaim(service, run_id)
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed"
        assert saved.step == 5
        assert service.tool_executor.executions == 1
        assert len(model.calls) == 3
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_invalid_replan_fallback_cannot_skip_required_tools(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(required=["read"])]}, "缺少必需工具", {"steps": []}, "仍未调用工具"
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    try:
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "failed"
        plan = saved.config["plan"]
        assert plan["fallback"] is True
        assert plan["nodes"][0]["acceptance"]["required_tools"] == ["read"]
        assert plan["nodes"][0]["acceptance_result"]["contract_satisfied"] is False
        assert plan["replans_used"] == 1
        assert saved.step == 6
        assert len(model.calls) == 4
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_existing_started_tool_is_interrupted_before_replanning(tmp_path, monkeypatch):
    from app.harness.security import canonical, digest

    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]}, ("", [tool_message()])
    )
    service, principal, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def seed_unknown(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "node":
            async with service.store.transaction() as session:
                session.add(
                    ToolCall(
                        id="unknown-ledger",
                        tenant_id=principal.tenant_id,
                        owner_id=principal.user_id,
                        run_id=run_id,
                        call_id="c1",
                        name="read",
                        arguments={"value": "示例"},
                        arguments_hash=digest(
                            canonical(
                                {"call_id": "c1", "name": "read", "arguments": {"value": "示例"}}
                            )
                        ),
                        status="started",
                    )
                )

    monkeypatch.setattr(worker, "checkpoint", seed_unknown)
    try:
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "interrupted"
        assert saved.config["plan"]["replans_used"] == 0
        assert service.tool_executor.executions == 0
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_tool_evidence_acceptance_rollback_keeps_pending_known_result(tmp_path, monkeypatch):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()]),
        "最终汇总",
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = service.store.emit

    def crash_on_acceptance(session, run, event, data):
        if event == "node_accepted":
            raise RuntimeError("模拟证据验收事务崩溃")
        original(session, run, event, data)

    monkeypatch.setattr(service.store, "emit", crash_on_acceptance)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        before = await persisted(service, run_id)
        assert before.config["model_result_pending"]["purpose"] == "node"
        assert before.step == 3
        monkeypatch.setattr(service.store, "emit", original)
        resumed = await reclaim(service, run_id)
        await resumed.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed"
        assert saved.step == 5
        assert len(model.calls) == 3
        assert service.tool_executor.executions == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_preflight_rejects_viewer_high_risk_even_for_saved_done(tmp_path):
    from app.harness.plan_execution import normalize_plan
    from app.harness.plan_runtime import POLICY, PlanRuntime
    from app.harness.security import canonical, digest

    tools = FakeTools("custom_exec")
    service, principal, worker, run_id = await prepared(tmp_path, ScriptModel(), tools)
    try:
        async with service.store.transaction() as session:
            user = await session.get(User, principal.user_id)
            user.role = "viewer"
            run = await session.get(Run, run_id)
            run.config = {
                **run.config,
                "budget_policy": POLICY,
                "planned": True,
                "plan": normalize_plan({"steps": [plan_step()]}, ["custom_exec"]),
            }
            session.add(
                ToolCall(
                    id="saved-exec",
                    tenant_id=principal.tenant_id,
                    owner_id=principal.user_id,
                    run_id=run_id,
                    call_id="c1",
                    name="custom_exec",
                    arguments={"value": "示例"},
                    arguments_hash=digest(
                        canonical(
                            {"call_id": "c1", "name": "custom_exec", "arguments": {"value": "示例"}}
                        )
                    ),
                    status="done",
                    result={"value": "历史结果"},
                )
            )
        run = await worker.load(run_id)
        planner = PlanRuntime(worker, run_id, principal, tools.catalog(principal), {"custom_exec"})
        with pytest.raises(HarnessError) as failure:
            await planner.preflight(run, [tool_message("custom_exec")])
        assert failure.value.status_code == 403
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_real_file_write_approval_restore_and_reuse_never_refreshes_baseline(
    tmp_path, monkeypatch
):
    from app.harness_tools.catalog import HarnessTools
    from app.harness_tools.file_write import FileWriter

    arguments = {"path": "result.txt", "content": "已批准写入"}
    model = ScriptModel(
        {"steps": [plan_step(required=["file_write"])]},
        ("", [tool_message("file_write", "original", arguments)]),
        "",
        {"steps": [plan_step(kind="tool_evidence", required=["file_write"])]},
        ("", [tool_message("file_write", "replacement", arguments)]),
        "已知历史文件结果",
    )
    service, principal, worker, run_id = await prepared(tmp_path, model)
    tools = HarnessTools(service.settings)
    tools.service = service
    service.tool_executor = tools
    try:
        await worker.execute(run_id)
        waiting = await persisted(service, run_id)
        approval = waiting.approval
        frozen = approval["file_write"]
        assert frozen["baseline"]["exists"] is False
        assert waiting.step == 2
        await service.store.close()
        restored = HarnessService(service.settings, model, tools)
        tools.service = restored
        await restored.store.initialize()
        service = restored

        async def forbid_freeze(*args, **kwargs):
            raise AssertionError("恢复及历史复用不得重新冻结文件基线")

        monkeypatch.setattr(FileWriter, "freeze", forbid_freeze)
        await restored.approve(
            principal, run_id, True, approval["call_id"], approval["hash"], frozen["baseline_hash"]
        )
        resumed = Worker(restored)
        assert await restored.store.claim(resumed.id) == run_id
        await resumed.execute(run_id)
        saved = await persisted(restored, run_id)
        assert saved.status == "completed", saved.error
        assert (
            tools.workspace.directory(principal, run_id)
            .joinpath("result.txt")
            .read_text(encoding="utf-8")
            == arguments["content"]
        )
        async with restored.store.sessions() as session:
            ledger = (
                await session.scalars(select(ToolCall).where(ToolCall.run_id == run_id))
            ).all()
        assert len(ledger) == 1
        assert saved.step == 9
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", ["tenant", "owner", "run"])
async def test_acceptance_loads_only_same_scope_and_run_ledger(tmp_path, monkeypatch, foreign):
    from copy import deepcopy

    model = ScriptModel(
        {"steps": [plan_step(required=["read"])]},
        "伪造证据答复",
        {"steps": [plan_step()]},
        "修复答复",
    )
    service, principal, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint
    injected = False

    async def inject_foreign(*args, **kwargs):
        nonlocal injected
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        if injected or (saved.config.get("model_result_pending") or {}).get("purpose") != "node":
            return
        injected = True
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            plan = deepcopy(run.config["plan"])
            plan["nodes"][0]["tool_call_ids"] = ["foreign-ledger"]
            run.config = {**run.config, "plan": plan}
            session.add(
                ToolCall(
                    id="foreign-ledger",
                    tenant_id="foreign" if foreign == "tenant" else principal.tenant_id,
                    owner_id="foreign" if foreign == "owner" else principal.user_id,
                    run_id="foreign" if foreign == "run" else run_id,
                    call_id="foreign-call",
                    name="read",
                    arguments={},
                    arguments_hash="x",
                    status="done",
                    result={"value": "已知结果"},
                )
            )

    monkeypatch.setattr(worker, "checkpoint", inject_foreign)
    try:
        await worker.execute(run_id)
        saved = await persisted(service, run_id)
        failed = saved.config["plan"]["history"][0]["nodes"][0]
        assert failed["acceptance_result"]["contract_satisfied"] is False
        assert failed["acceptance_result"]["ledger_sources"] == []
        assert saved.status == "completed"
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_project_prepare_is_charged_after_shared_policy_initialization(tmp_path):
    model = ScriptModel({"steps": [plan_step()]}, "项目答复")
    tools = FakeTools("project_prepare")
    service, _, worker, run_id = await prepared(tmp_path, model, tools)
    original = tools.execute
    seen = []

    async def inspect_policy(*args, **kwargs):
        run = await persisted(service, run_id)
        seen.append((run.step, run.config["budget_policy"]))
        return await original(*args, **kwargs)

    tools.execute = inspect_policy
    try:
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            run.config = {**run.config, "project_mode": "copy"}
        await worker.execute(run_id)
        run = await persisted(service, run_id)
        assert run.status == "completed"
        assert run.step == 5
        assert seen == [(1, "shared_actions_v1")]
        assert tools.executions == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_cancel_during_final_summary_does_not_revive_run(tmp_path):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()]),
        "最终摘要",
    )
    service, principal, worker, run_id = await prepared(tmp_path, model)
    original = model.chat

    async def cancel_final(*args, **kwargs):
        if len(model.calls) == 2:
            await service.cancel(principal, run_id)
        return await original(*args, **kwargs)

    model.chat = cancel_final
    try:
        with pytest.raises(asyncio.CancelledError):
            await worker.execute(run_id)
        run = await persisted(service, run_id)
        assert run.status == "cancelled"
        assert run.answer is None
        assert run.config["plan"]["nodes"][0]["status"] == "completed"
        assert run.step == 5
        assert service.tool_executor.executions == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_own_expired_lease_cannot_consume_or_charge_node_result(tmp_path):
    model = ScriptModel({"steps": [plan_step()]}, "节点答案")
    service, _, worker, run_id = await prepared(tmp_path, model)
    try:
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            run.lease_until = 0
        with pytest.raises(asyncio.CancelledError):
            await worker.load(run_id)
        run = await persisted(service, run_id)
        assert run.step == 0
        assert model.calls == []
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_delegated_steps_reduce_shared_plan_budget_without_reset(tmp_path):
    model = ScriptModel({"steps": [plan_step(), plan_step()]}, "剩余预算答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=8)
    try:
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            run.config = {**run.config, "delegated_steps": 4}
        await worker.execute(run_id)
        run = await persisted(service, run_id)
        assert run.status == "completed"
        assert run.step == 3
        assert run.config["delegated_steps"] == 4
        assert run.config["plan_fallback"] is True
        assert "plan" not in run.config
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_answer_required_tool_reserves_tool_round_and_followup_answer_model(tmp_path):
    model = ScriptModel({"steps": [plan_step(required=["read"])]}, "预算有界答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=5)
    try:
        await drive(worker, run_id)
        run = await persisted(service, run_id)
        assert run.status == "completed"
        assert run.config.get("plan") is None
        assert run.config["plan_fallback"] is True
        assert run.step == 3
        assert service.tool_executor.executions == 0
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [7, 10])
async def test_real_plan_delegate_preserves_parent_nodes_and_final_budget(tmp_path, budget):
    from app.harness_tools.catalog import HarnessTools

    arguments = {
        "tasks": [{"id": "child", "message": "子任务", "acceptance": {"required_tools": []}}]
    }
    model = ScriptModel(
        {
            "steps": [
                plan_step("协作", kind="tool_evidence", required=["delegate"]),
                plan_step("主控整合", dependencies=[0]),
            ]
        },
        ("", [tool_message("delegate", "delegate-call", arguments)]),
        "子任务已知答复",
        "父任务交付",
    )
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=budget)
    tools = HarnessTools(service.settings)
    tools.service = service
    service.tool_executor = tools
    child_worker = Worker(service, child_only=True)
    service.workers = [child_worker]
    child_worker.start()
    try:
        await drive(worker, run_id)
        run = await persisted(service, run_id)
        async with service.store.sessions() as session:
            children = (await session.scalars(select(Run).where(Run.parent_run_id == run_id))).all()
        if budget == 7:
            assert children == []
            assert run.status == "failed"
        else:
            assert run.status == "completed", run.error
            assert run.answer == "父任务交付"
            assert len(children) == 1
            assert children[0].status == "completed"
            assert run.step == 7
            assert run.config["delegated_steps"] == 3
            assert run.step + run.config["delegated_steps"] <= budget
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_planning_tool_calls_remain_forbidden_after_known_checkpoint_restart(
    tmp_path, monkeypatch
):
    model = ScriptModel(({"steps": ["节点"]}, [tool_message()]), "恢复后有界答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def crash_after_planning(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "plan_initial":
            raise RuntimeError("模拟规划工具结果检查点后崩溃")

    monkeypatch.setattr(worker, "checkpoint", crash_after_planning)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        resumed = await reclaim(service, run_id)
        await resumed.execute(run_id)
        run = await persisted(service, run_id)
        assert run.status == "completed"
        assert service.tool_executor.executions == 0
        assert len(model.calls) == 2
        assert run.step == 3
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_initial_known_result_with_wrong_action_step_is_rejected(tmp_path, monkeypatch):
    model = ScriptModel({"steps": [plan_step()]}, "答复")
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def corrupt_action_step(*args, **kwargs):
        await original(*args, **kwargs)
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            pending = run.config.get("model_result_pending")
            if pending and pending.get("purpose") == "plan_initial":
                run.config = {**run.config, "model_result_pending": {**pending, "step": 0}}

    monkeypatch.setattr(worker, "checkpoint", corrupt_action_step)
    try:
        with pytest.raises(HarnessError) as failure:
            await worker.execute(run_id)
        assert failure.value.status_code == 409
        assert len(model.calls) == 1
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "checkpoint"])
@pytest.mark.parametrize(
    "summary",
    ["答" * 131073, ["错误类型"], "答" * 131072],
    ids=["oversized", "nontext", "exact-limit"],
)
async def test_final_summary_is_bounded_on_new_response_and_recovery(
    tmp_path, monkeypatch, restored, summary
):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()]),
        "已知摘要" if restored else summary,
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def crash_after_final(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "final":
            async with service.store.transaction() as session:
                run = await session.get(Run, run_id)
                run.messages = [*run.messages[:-1], {**run.messages[-1], "content": summary}]
            raise RuntimeError("模拟最终摘要检查点后崩溃")

    try:
        if restored:
            monkeypatch.setattr(worker, "checkpoint", crash_after_final)
            with pytest.raises(RuntimeError):
                await worker.execute(run_id)
            worker = await reclaim(service, run_id)
        if isinstance(summary, str) and len(summary) == 131072:
            await worker.execute(run_id)
            saved = await persisted(service, run_id)
            assert saved.status == "completed"
            assert saved.answer == summary
        else:
            with pytest.raises(HarnessError) as failure:
                await worker.execute(run_id)
            assert failure.value.status_code == 422
            assert (await persisted(service, run_id)).status != "completed"
        assert len(model.calls) == 3
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["plan", "react"])
@pytest.mark.parametrize("approved", [False, True], ids=["no-approval", "trusted-approval"])
async def test_persisted_risk_deny_survives_checkpoint_crash_and_approval(
    tmp_path, monkeypatch, mode, approved
):
    model = (
        ScriptModel(
            {"steps": [plan_step(kind="tool_evidence", required=["write"])]},
            ("", [tool_message("write")]),
            {"risk": "deny", "reason": "明确拒绝"},
            "不得执行后的摘要",
        )
        if mode == "plan"
        else ScriptModel(
            ("", [tool_message("write")]),
            {"risk": "deny", "reason": "明确拒绝"},
            "不得执行后的答复",
        )
    )
    service, _, worker, run_id = await prepared(
        tmp_path, model, FakeTools("write"), risk_review_enabled=True
    )
    if mode == "react":
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            run.config = {**run.config, "mode": "react"}
    original = worker.checkpoint

    async def crash_after_deny(*args, **kwargs):
        await original(*args, **kwargs)
        if args[3] == "risk_review":
            raise RuntimeError("模拟deny检查点提交后崩溃")

    monkeypatch.setattr(worker, "checkpoint", crash_after_deny)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        if approved:
            async with service.store.transaction() as session:
                run = await session.get(Run, run_id)
                arg_hash = next(iter(run.config["risk_reviews"]))
                run.approval = {
                    "call_id": "c1",
                    "hash": arg_hash,
                    "approved": True,
                    "name": "write",
                    "arguments": {"value": "示例"},
                }
        resumed = await reclaim(service, run_id)
        with pytest.raises(HarnessError) as failure:
            await resumed.execute(run_id)
        assert failure.value.status_code == 403
        saved = await persisted(service, run_id)
        assert saved.status != "waiting_approval"
        assert service.tool_executor.executions == 0
        assert len(model.calls) == (3 if mode == "plan" else 2)
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("late_deny", [False, True], ids=["saved-deny", "late-deny"])
async def test_persisted_risk_deny_blocks_archived_side_effect_reuse(
    tmp_path, monkeypatch, late_deny
):
    from contextlib import asynccontextmanager

    from app.harness.plan_execution import normalize_plan
    from app.harness.security import canonical, digest

    model = ScriptModel()
    service, principal, worker, run_id = await prepared(tmp_path, model, FakeTools("write"))
    call = tool_message("write", "new-write")
    arguments = {"value": "示例"}
    arg_hash = digest(canonical({"call_id": "new-write", "name": "write", "arguments": arguments}))
    try:
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            plan = normalize_plan(
                {"steps": [plan_step(kind="tool_evidence", required=["write"])]},
                ["write"],
                revision=1,
            )
            plan["history"] = [{"revision": 0, "nodes": [{"tool_call_ids": ["source-ledger"]}]}]
            run.config = {
                **run.config,
                "budget_policy": "shared_actions_v1",
                "planned": True,
                "plan": plan,
                "risk_reviews": {} if late_deny else {arg_hash: {"risk": "deny"}},
            }
            session.add(
                ToolCall(
                    id="source-ledger",
                    tenant_id=principal.tenant_id,
                    owner_id=principal.user_id,
                    run_id=run_id,
                    call_id="original-write",
                    name="write",
                    arguments=arguments,
                    arguments_hash="original-hash",
                    status="done",
                    result={"status": "written"},
                )
            )
        run = await worker.load(run_id)
        if late_deny:
            original = worker.active_transaction
            injected = False

            @asynccontextmanager
            async def deny_after_source_lookup(identifier):
                nonlocal injected
                async with original(identifier) as active:
                    yield active
                if not injected:
                    injected = True
                    async with service.store.transaction() as session:
                        saved = await session.get(Run, run_id)
                        saved.config = {
                            **saved.config,
                            "risk_reviews": {arg_hash: {"risk": "deny"}},
                        }

            monkeypatch.setattr(worker, "active_transaction", deny_after_source_lookup)
        with pytest.raises(HarnessError) as failure:
            await worker.execute_tool(run, principal, call, list(run.messages), run.step, {"write"})
        assert failure.value.status_code == 403
        async with service.store.sessions() as session:
            reused = await session.scalar(
                select(Event).where(Event.run_id == run_id, Event.type == "tool_reused")
            )
        assert reused is None
        assert service.tool_executor.executions == 0
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False], ids=["approve-blocked", "cancel-allowed"])
async def test_trusted_approval_cannot_override_saved_risk_deny(tmp_path, approved):
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["write"])]},
        ("", [tool_message("write")]),
    )
    service, principal, worker, run_id = await prepared(tmp_path, model, FakeTools("write"))
    try:
        await worker.execute(run_id)
        waiting = await persisted(service, run_id)
        assert waiting.status == "waiting_approval"
        approval = waiting.approval
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            run.config = {**run.config, "risk_reviews": {approval["hash"]: {"risk": "deny"}}}
        if approved:
            with pytest.raises(HarnessError) as failure:
                await service.approve(
                    principal, run_id, True, approval["call_id"], approval["hash"]
                )
            assert failure.value.status_code == 403
            assert (await persisted(service, run_id)).status == "waiting_approval"
        else:
            await service.approve(principal, run_id, False, approval["call_id"], approval["hash"])
            assert (await persisted(service, run_id)).status == "cancelled"
        assert service.tool_executor.executions == 0
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["plan", "react"])
async def test_latest_risk_deny_blocks_tool_registration(tmp_path, monkeypatch, mode):
    responses = [
        ("", [tool_message("write")]),
        {"risk": "allow", "reason": "原审查允许"},
    ]
    if mode == "plan":
        responses.insert(0, {"steps": [plan_step(kind="tool_evidence", required=["write"])]})
    model = ScriptModel(*responses)
    service, _, worker, run_id = await prepared(
        tmp_path, model, FakeTools("write"), risk_review_enabled=True
    )
    if mode == "react":
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            run.config = {**run.config, "mode": "react"}
    original = worker.review_risk

    async def deny_before_registration(*args, **kwargs):
        step = await original(*args, **kwargs)
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            arg_hash = next(iter(run.config["risk_reviews"]))
            run.config = {**run.config, "risk_reviews": {arg_hash: {"risk": "deny"}}}
        return step

    monkeypatch.setattr(worker, "review_risk", deny_before_registration)
    try:
        with pytest.raises(HarnessError) as failure:
            await worker.execute(run_id)
        assert failure.value.status_code == 403
        saved = await persisted(service, run_id)
        assert saved.approval is None
        assert saved.status != "waiting_approval"
        async with service.store.sessions() as session:
            assert await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id)) is None
        assert service.tool_executor.executions == 0
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "checkpoint"])
@pytest.mark.parametrize(
    "answer",
    ["答" * 131073, ["非文本"], "答" * 131072],
    ids=["oversized", "nontext", "exact-limit"],
)
async def test_initial_plan_fallback_answer_is_bounded(tmp_path, monkeypatch, restored, answer):
    model = ScriptModel({"steps": []}, "有效答复" if restored else answer)
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=3)
    original = worker.checkpoint

    async def crash_after_fallback_answer(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "react":
            async with service.store.transaction() as session:
                run = await session.get(Run, run_id)
                run.messages = [*run.messages[:-1], {**run.messages[-1], "content": answer}]
            raise RuntimeError("模拟降级答复检查点后崩溃")

    try:
        if restored:
            monkeypatch.setattr(worker, "checkpoint", crash_after_fallback_answer)
            with pytest.raises(RuntimeError):
                await worker.execute(run_id)
            worker = await reclaim(service, run_id)
        if isinstance(answer, str) and len(answer) == 131072:
            await worker.execute(run_id)
            saved = await persisted(service, run_id)
            assert saved.status == "completed"
            assert saved.answer == answer
            assert saved.step == 3
        else:
            with pytest.raises(HarnessError) as failure:
                await worker.execute(run_id)
            assert failure.value.status_code == 422
            assert (await persisted(service, run_id)).status != "completed"
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker_patch",
    [{"purpose": "final"}, {"node_id": "foreign"}, {"plan_revision": 1}, {"step": 0}],
    ids=["wrong-purpose", "wrong-node", "wrong-revision", "wrong-step"],
)
async def test_fallback_answer_cannot_borrow_other_model_identity(
    tmp_path, monkeypatch, marker_patch
):
    model = ScriptModel({"steps": []}, "已知答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=3)
    original = worker.checkpoint

    async def corrupt_identity(*args, **kwargs):
        await original(*args, **kwargs)
        corrupted = False
        async with service.store.transaction() as session:
            run = await session.get(Run, run_id)
            pending = run.config.get("model_result_pending") or {}
            if pending.get("purpose") == "react":
                run.config = {**run.config, "model_result_pending": {**pending, **marker_patch}}
                corrupted = True
        if corrupted:
            raise RuntimeError("模拟检查点归属被替换")

    monkeypatch.setattr(worker, "checkpoint", corrupt_identity)
    try:
        with pytest.raises(RuntimeError):
            await worker.execute(run_id)
        resumed = await reclaim(service, run_id)
        with pytest.raises(HarnessError) as failure:
            await resumed.execute(run_id)
        assert failure.value.status_code == 409
        assert (await persisted(service, run_id)).status != "completed"
        assert len(model.calls) == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "checkpoint"])
@pytest.mark.parametrize(
    "batch", [17, 16, {"not": "a-list"}], ids=["seventeen", "sixteen", "nonlist"]
)
async def test_plan_tool_batch_bound_precedes_any_execution(tmp_path, monkeypatch, restored, batch):
    calls = (
        [tool_message("read", f"read-{index}") for index in range(batch)]
        if isinstance(batch, int)
        else batch
    )
    model = ScriptModel(
        {"steps": [plan_step(kind="tool_evidence", required=["read"])]},
        ("", [tool_message()] if restored else calls),
        "最终摘要",
    )
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=21)
    original = worker.checkpoint

    async def crash_after_tool_batch(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        pending = saved.config.get("model_result_pending") or {}
        if pending.get("purpose") == "node":
            async with service.store.transaction() as session:
                run = await session.get(Run, run_id)
                run.messages = [*run.messages[:-1], {**run.messages[-1], "tool_calls": calls}]
            raise RuntimeError("模拟工具批次检查点后崩溃")

    try:
        if restored:
            monkeypatch.setattr(worker, "checkpoint", crash_after_tool_batch)
            with pytest.raises(RuntimeError):
                await worker.execute(run_id)
            worker = await reclaim(service, run_id)
        if batch == 16:
            await worker.execute(run_id)
            saved = await persisted(service, run_id)
            assert saved.status == "completed", saved.error
            assert saved.step == 20
            assert service.tool_executor.executions == 16
        else:
            with pytest.raises(HarnessError) as failure:
                await worker.execute(run_id)
            assert failure.value.status_code == 422
            assert service.tool_executor.executions == 0
            async with service.store.sessions() as session:
                assert (
                    await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id)) is None
                )
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["答" * 131073, ["非文本"]], ids=["oversized", "nontext"])
async def test_fallback_final_transaction_rechecks_persisted_answer(tmp_path, monkeypatch, answer):
    from contextlib import asynccontextmanager

    model = ScriptModel({"steps": []}, "入口时合法的答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=3)
    original = worker.active_transaction

    @asynccontextmanager
    async def corrupt_before_finalization(identifier):
        async with original(identifier) as (session, run):
            if (run.config.get("model_result_pending") or {}).get("purpose") == "react":
                run.messages = [*run.messages[:-1], {**run.messages[-1], "content": answer}]
            yield session, run

    monkeypatch.setattr(worker, "active_transaction", corrupt_before_finalization)
    try:
        with pytest.raises(HarnessError) as failure:
            await worker.execute(run_id)
        assert failure.value.status_code == 422
        saved = await persisted(service, run_id)
        assert saved.status != "completed"
        assert saved.step == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "checkpoint"])
@pytest.mark.parametrize("invalid", [17, {"not": "a-list"}], ids=["seventeen", "nonlist"])
async def test_forbidden_planning_tool_batch_uses_original_fallback(
    tmp_path, monkeypatch, restored, invalid
):
    calls = (
        [tool_message("read", f"read-{index}") for index in range(invalid)]
        if isinstance(invalid, int)
        else invalid
    )
    model = ScriptModel(({"steps": ["节点"]}, [tool_message()] if restored else calls), "降级答复")
    service, _, worker, run_id = await prepared(tmp_path, model, max_steps=3)
    original = worker.checkpoint

    async def crash_after_planning(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        if (saved.config.get("model_result_pending") or {}).get("purpose") == "plan_initial":
            async with service.store.transaction() as session:
                run = await session.get(Run, run_id)
                run.messages = [*run.messages[:-1], {**run.messages[-1], "tool_calls": calls}]
            raise RuntimeError("模拟非法规划工具批次恢复")

    try:
        if restored:
            monkeypatch.setattr(worker, "checkpoint", crash_after_planning)
            with pytest.raises(RuntimeError):
                await worker.execute(run_id)
            worker = await reclaim(service, run_id)
        await worker.execute(run_id)
        saved = await persisted(service, run_id)
        assert saved.status == "completed"
        assert saved.answer == "降级答复"
        assert saved.config["plan_fallback"] is True
        assert saved.step == 3
        assert service.tool_executor.executions == 0
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "batch", [[tool_message("read", f"r-{i}") for i in range(17)], {"invalid": []}]
)
async def test_plan_preflight_rejects_batch_shape_before_execution(tmp_path, batch):
    from app.harness.plan_runtime import PlanRuntime

    service, principal, worker, run_id = await prepared(tmp_path, ScriptModel(), max_steps=21)
    try:
        planner = PlanRuntime(worker, run_id, principal, [], {"read"})
        with pytest.raises(HarnessError) as failure:
            await planner.preflight(await worker.load(run_id), batch)
        assert failure.value.status_code == 422
        assert service.tool_executor.executions == 0
    finally:
        await service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True], ids=["fresh", "raw-checkpoint"])
@pytest.mark.parametrize("invalid", [17, {"not": "a-list"}], ids=["seventeen", "nonlist"])
async def test_forbidden_replan_tool_batch_recovers_through_original_fallback(
    tmp_path, monkeypatch, restored, invalid
):
    calls = (
        [tool_message("read", f"replan-{index}") for index in range(invalid)]
        if isinstance(invalid, int)
        else invalid
    )
    model = ScriptModel(
        {"steps": [plan_step(required=["read"])]},
        "缺少必需工具",
        ({"steps": []}, [] if restored else calls),
        "仍未调用工具",
    )
    service, _, worker, run_id = await prepared(tmp_path, model)
    original = worker.checkpoint

    async def crash_after_raw_replan(*args, **kwargs):
        await original(*args, **kwargs)
        saved = await persisted(service, run_id)
        pending = saved.config.get("model_result_pending") or {}
        if pending.get("purpose") == "replan":
            async with service.store.transaction() as session:
                run = await session.get(Run, run_id)
                run.messages = [*run.messages[:-1], {**run.messages[-1], "tool_calls": calls}]
                run.config = {
                    **run.config,
                    "model_result_pending": {**pending, "kind": "tools"},
                }
            raise RuntimeError("模拟原始重规划调用检查点后崩溃")

    try:
        if restored:
            monkeypatch.setattr(worker, "checkpoint", crash_after_raw_replan)
            with pytest.raises(RuntimeError):
                await worker.execute(run_id)
            before = await persisted(service, run_id)
            assert before.step == 4
            assert len(model.calls) == 3
            worker = await reclaim(service, run_id)
        await drive(worker, run_id)
        saved = await persisted(service, run_id)
        plan = saved.config["plan"]
        assert plan["fallback"] is True
        assert plan["version"] == 1
        assert plan["replans_used"] == 1
        assert plan["nodes"][0]["acceptance"]["required_tools"] == ["read"]
        assert plan["nodes"][0]["acceptance_result"]["contract_satisfied"] is False
        assert saved.status == "failed"
        assert saved.step == 6
        assert len(model.calls) == 4
        assert service.tool_executor.executions == 0
        async with service.store.sessions() as session:
            assert await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id)) is None
    finally:
        await service.store.close()
