"""真实 SQLite 与 Worker 验证委派闭环；受控模型不依赖外部服务。"""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness.models import Run, ToolCall
from app.harness_tools.catalog import HarnessTools


class DelegationModel:
    def __init__(self, child_tool=None, answer="子任务完成", gate=None, read_artifact=False):
        self.child_tool = child_tool
        self.answer = answer
        self.gate = gate
        self.child_started = asyncio.Event()
        self.child_catalogs = []
        self.read_artifact = read_artifact
        self.parent_pages = []

    async def chat(self, messages, model_preference=None, **kwargs):
        task = next(item["content"] for item in reversed(messages) if item["role"] == "user")
        has_result = any(item["role"] == "tool" for item in messages)
        calls = []
        answer = "主汇总完成"
        if task == "主任务" and not has_result:
            calls = [
                {
                    "id": "delegate-call",
                    "type": "function",
                    "function": {
                        "name": "delegate",
                        "arguments": json.dumps({"tasks": ["子任务甲", "子任务乙"]}),
                    },
                }
            ]
        elif task == "主任务" and self.read_artifact:
            page = next(
                (item for item in messages if item.get("tool_call_id") == "read-page"), None
            )
            if page:
                self.parent_pages.append(json.loads(page["content"]))
            else:
                result = next(
                    item for item in messages if item.get("tool_call_id") == "delegate-call"
                )
                artifact_id = json.loads(result["content"])[0]["artifact_id"]
                calls = [
                    {
                        "id": "read-page",
                        "type": "function",
                        "function": {
                            "name": "artifact_read",
                            "arguments": json.dumps(
                                {"asset_id": artifact_id, "start": 2000, "length": 8000}
                            ),
                        },
                    }
                ]
        elif task.startswith("子任务"):
            self.child_catalogs.append(
                [item["function"]["name"] for item in kwargs.get("tools", [])]
            )
            self.child_started.set()
            if self.gate:
                await self.gate.wait()
            answer = self.answer
            if self.child_tool and not has_result:
                arguments = {"expression": "1+1"} if self.child_tool == "calculator" else {}
                calls = [
                    {
                        "id": "child-call",
                        "type": "function",
                        "function": {"name": self.child_tool, "arguments": json.dumps(arguments)},
                    }
                ]
        return SimpleNamespace(
            content="" if calls else answer,
            model_id="受控模型",
            usage={},
            raw={"choices": [{"message": {"tool_calls": calls}}]},
        )


async def make_service(tmp_path, model=None, **options):
    settings = HarnessSettings(
        data_dir=tmp_path,
        database_url="",
        max_concurrent_runs=1,
        tool_output_chars=10000,
        **options,
    )
    tools = HarnessTools(settings)
    service = HarnessService(settings, model or DelegationModel(), tools)
    tools.service = service
    await service.initialize()
    user = await service.register("owner", "password123")
    return service, tools, Principal(user["user_id"], user["tenant_id"], user["role"])


async def finished(service, principal, run_id):
    async with asyncio.timeout(5):
        while True:
            run = await service.get_run(principal, run_id)
            if run["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                return run
            await asyncio.sleep(0.01)


async def child_rows(service, parent_id):
    async with service.store.sessions() as session:
        return list(await session.scalars(select(Run).where(Run.parent_run_id == parent_id)))


async def delegate_result(service, parent_id):
    async with service.store.sessions() as session:
        return (
            await session.scalar(
                select(ToolCall).where(ToolCall.run_id == parent_id, ToolCall.name == "delegate")
            )
        ).result


@pytest.mark.asyncio
async def test_seven_steps_two_children_leave_parent_summary(tmp_path):
    service, _, principal = await make_service(tmp_path, max_steps=7)
    try:
        parent = await service.create_run(principal, "主任务", "seven")
        done = await finished(service, principal, parent["id"])
        assert done["status"] == "completed", done
        assert done["answer"] == "主汇总完成"
        children = await child_rows(service, parent["id"])
        assert len(children) == 2 and all(child.status == "completed" for child in children)
        assert sum(child.config["max_steps"] for child in children) <= 5
        assert all(1 <= child.config["max_steps"] <= 3 for child in children)
        assert "config" not in done
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options,reason", [({"max_steps": 3}, "预算"), ({"max_user_runs": 2}, "限额")]
)
async def test_batch_preflight_failure_creates_no_children(tmp_path, options, reason):
    service, _, principal = await make_service(tmp_path, **options)
    try:
        parent = await service.create_run(principal, "主任务", "insufficient")
        done = await finished(service, principal, parent["id"])
        assert done["status"] == "completed", done
        result = await delegate_result(service, parent["id"])
        assert result["status"] == "failed" and reason in result["error"]
        assert result["children"] == [] and "未知" not in result["error"]
        assert await child_rows(service, parent["id"]) == []
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "child_tool,expected",
    [("calculator", "completed"), ("knowledge_search", "failed"), ("delegate", "failed")],
)
async def test_children_intersect_parent_permissions_and_reject_forgery(
    tmp_path, child_tool, expected
):
    model = DelegationModel(child_tool=child_tool)
    service, _, principal = await make_service(tmp_path, model, max_steps=7)
    try:
        parent = await service.create_run(
            principal, "主任务", "restricted", allowed_tools=["delegate", "calculator"]
        )
        done = await finished(service, principal, parent["id"])
        assert done["status"] == "completed", done
        children = await child_rows(service, parent["id"])
        assert len(children) == 2 and all(child.status == expected for child in children)
        assert all(child.config["allowed_tools"] == ["calculator"] for child in children)
        assert model.child_catalogs and all(
            names == ["calculator"] for names in model.child_catalogs
        )
        if expected == "failed":
            assert all("清单" in child.error for child in children)
            async with service.store.sessions() as session:
                assert not list(
                    await session.scalars(
                        select(ToolCall).where(
                            ToolCall.run_id.in_([child.id for child in children])
                        )
                    )
                )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_long_child_answers_are_complete_private_artifacts(tmp_path):
    answer = "完整证据" * 2000 + "最后关键证据"
    model = DelegationModel(answer=answer, read_artifact=True)
    service, tools, principal = await make_service(tmp_path, model, max_steps=10)
    try:
        parent = await service.create_run(
            principal, "主任务", "long", allowed_tools=["delegate", "artifact_read"]
        )
        assert (await finished(service, principal, parent["id"]))["status"] == "completed"
        assert model.parent_pages and model.parent_pages[0]["content"] == answer[2000:10000]
        results = await delegate_result(service, parent["id"])
        assert len(results) == 2
        for result in results:
            assert len(result["answer"]) <= 2000
            assert result["total_chars"] == len(answer)
            artifact = await service.get_asset(principal, result["artifact_id"])
            assert artifact["content"] == answer
            assert artifact["metadata"]["source_run_id"] == result["run_id"]
            page = await tools.execute(
                "artifact_read",
                {"asset_id": artifact["id"], "start": 2000, "length": 8000},
                principal,
                parent["id"],
            )
            assert page["content"] == answer[2000:10000]
            outsider = Principal("另一个用户", principal.tenant_id, "operator")
            with pytest.raises(HarnessError) as error:
                await tools.execute(
                    "artifact_read", {"asset_id": artifact["id"]}, outsider, parent["id"]
                )
            assert error.value.status_code in {403, 404}
    finally:
        await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("elapsed,expected", [(25, "completed"), (295, "cancelled")])
async def test_wait_uses_parent_deadline_and_returns_real_terminal_state(
    tmp_path, monkeypatch, elapsed, expected
):
    from app.harness_tools import catalog

    gate = asyncio.Event()
    model = DelegationModel(gate=gate)
    service, _, principal = await make_service(tmp_path, model, max_steps=10)
    clock = SimpleNamespace(value=time.time())
    original_sleep = asyncio.sleep
    advanced = asyncio.Event()

    async def controlled_sleep(seconds):
        if not advanced.is_set():
            await model.child_started.wait()
            clock.value += elapsed
            advanced.set()
            if expected == "completed":
                gate.set()
        await original_sleep(0.001)

    monkeypatch.setattr(catalog.delegation, "sleep", controlled_sleep)
    monkeypatch.setattr(catalog.delegation, "now", lambda: clock.value)
    try:
        parent = await service.create_run(principal, "主任务", "timed")
        assert (await finished(service, principal, parent["id"]))["status"] == "completed"
        result = await delegate_result(service, parent["id"])
        assert all(item["status"] == expected for item in result), result
        children = await child_rows(service, parent["id"])
        assert all(child.status == expected for child in children)
        if expected == "cancelled":
            assert all(item["error"] and item.get("message") for item in result)
    finally:
        gate.set()
        await service.close()


@pytest.mark.asyncio
async def test_backend_creation_reserves_summary_and_lifetime_child_limit(tmp_path):
    service, _, principal = await make_service(tmp_path, max_steps=4)
    try:
        # 停止调度，只保留真实数据库与服务创建边界。
        for worker in service.workers:
            worker.task.cancel()
        await asyncio.gather(*(worker.task for worker in service.workers), return_exceptions=True)
        parent = await service.create_run(principal, "父任务", "backend")
        async with service.store.sessions.begin() as session:
            row = await session.get(Run, parent["id"])
            row.status, row.step = "running", 1
        with pytest.raises(HarnessError, match="预算"):
            await service.create_run(
                principal,
                "子任务",
                "overspend",
                parent_run_id=parent["id"],
                allowed_tools=["calculator"],
                max_steps=3,
            )
        assert await child_rows(service, parent["id"]) == []
        for index in range(2):
            child = await service.create_run(
                principal,
                "子任务",
                f"child-{index}",
                parent_run_id=parent["id"],
                allowed_tools=["calculator"],
                max_steps=1,
            )
            await service.cancel(principal, child["id"])
        with pytest.raises(HarnessError, match="两个"):
            await service.create_run(
                principal,
                "第三任务",
                "third",
                parent_run_id=parent["id"],
                allowed_tools=["calculator"],
                max_steps=1,
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_parent_cancellation_stops_children_without_replaying_started_delegate(tmp_path):
    gate = asyncio.Event()
    model = DelegationModel(gate=gate)
    service, _, principal = await make_service(tmp_path, model)
    try:
        parent = await service.create_run(principal, "主任务", "parent-cancel")
        async with asyncio.timeout(5):
            await model.child_started.wait()
            while len(await child_rows(service, parent["id"])) < 2:
                await asyncio.sleep(0.01)
        await service.cancel(principal, parent["id"])
        assert (await finished(service, principal, parent["id"]))["status"] == "cancelled"
        children = await child_rows(service, parent["id"])
        assert len(children) == 2 and all(child.status == "cancelled" for child in children)
        async with service.store.sessions() as session:
            call = await session.scalar(
                select(ToolCall).where(ToolCall.run_id == parent["id"], ToolCall.name == "delegate")
            )
            assert call.status == "started"
        assert await service.store.claim("禁止重放的恢复工作者") is None
    finally:
        gate.set()
        await service.close()
