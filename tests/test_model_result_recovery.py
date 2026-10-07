"""完整模型检查点失租后消费已知结果；不重生成，不跳过反思与工具账本。"""

from types import SimpleNamespace

import pytest

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.models import Run
from app.harness.runtime import Worker
from tests.test_harness_runtime import FakeTools


class RecoveryModel:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            content="复核后答复" if not kwargs.get("tools") else "原始答复",
            model_id="recovery",
            usage={"total_tokens": 1},
            raw={},
        )


async def prepared(tmp_path, mode):
    model = RecoveryModel()
    tools = FakeTools()
    service = HarnessService(
        HarnessSettings(data_dir=tmp_path, evolution_enabled=False), model, tools
    )
    await service.store.initialize()
    user = await service.register("result-recovery", "test-password")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    created = await service.create_run(principal, "任务", mode, mode=mode)
    worker = Worker(service)
    assert await service.store.claim(worker.id) == created["id"]
    return service, principal, await worker.load(created["id"]), worker, model, tools


async def recover(service, run_id):
    async with service.store.sessions.begin() as session:
        run = await session.get(Run, run_id)
        run.lease_until = 0
    worker = Worker(service)
    assert await service.store.claim(worker.id) == run_id
    await worker.execute(run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["react", "reflection"])
async def test_known_answer_checkpoint_recovers_without_regeneration_and_keeps_quality_gate(
    tmp_path, mode
):
    service, principal, run, worker, model, tools = await prepared(tmp_path, mode)
    try:
        response = await worker.model_call(run, run.messages, tools.catalog(principal))
        messages = [*run.messages, {"role": "assistant", "content": response.content}]
        await worker.checkpoint(run.id, messages, 1, "model", {"tool_calls": []})
        await recover(service, run.id)
        result = await service.get_run(principal, run.id)
        assert result["status"] == "completed"
        assert model.calls == (1 if mode == "react" else 2)
        assert result["answer"] == ("原始答复" if mode == "react" else "复核后答复")
        async with service.store.sessions() as session:
            current = await session.get(Run, run.id)
            assert "model_result_pending" not in current.config
            assert "model_inflight" not in current.config
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_reflection_checkpoint_recovers_revised_answer_without_second_review(tmp_path):
    service, principal, run, worker, model, tools = await prepared(tmp_path, "reflection")
    try:
        first = await worker.model_call(run, run.messages, tools.catalog(principal))
        messages = [*run.messages, {"role": "assistant", "content": first.content}]
        await worker.checkpoint(run.id, messages, 1, "model", {"tool_calls": []})
        revised = await worker.model_call(run, messages)
        messages.append({"role": "assistant", "content": revised.content})
        await worker.checkpoint(
            run.id,
            messages,
            2,
            "reflection",
            {"verified": False},
            {**run.config, "reflected": True},
        )
        await recover(service, run.id)
        result = await service.get_run(principal, run.id)
        assert result["status"] == "completed"
        assert result["answer"] == "复核后答复"
        assert model.calls == 2
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_complete_tool_checkpoint_recovery_uses_ledger_once_and_keeps_evidence(tmp_path):
    service, principal, run, worker, model, tools = await prepared(tmp_path, "react")
    try:
        messages = [
            *run.messages,
            {
                "role": "assistant",
                "content": "准备读取",
                "tool_calls": [
                    {
                        "id": "read-evidence",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"value":"证据"}'},
                    }
                ],
            },
        ]
        await worker.checkpoint(
            run.id, messages, 1, "model", {"tool_calls": messages[-1]["tool_calls"]}
        )
        await worker.execute_tool(
            run, principal, messages[-1]["tool_calls"][0], messages, 1, {"read"}
        )
        await recover(service, run.id)
        result = await service.get_run(principal, run.id)
        assert result["status"] == "completed"
        assert tools.executions == 1
        assert model.calls == 1
        async with service.store.sessions() as session:
            current = await session.get(Run, run.id)
            assert any(
                message.get("role") == "tool"
                and message["tool_call_id"] == "read-evidence"
                and '"verified":true' in message["content"]
                for message in current.messages
            )
    finally:
        await service.store.close()
