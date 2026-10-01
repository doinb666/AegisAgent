"""独立 PostgreSQL 验收库：审批竞争、重启恢复和未知副作用停止。"""

import asyncio
import json
import os
import time
from types import SimpleNamespace

from sqlalchemy import select, update

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.models import Run, ToolCall
from app.harness.store import uid


class WriteModel:
    async def chat(self, messages, **kwargs):
        completed = any(message.get("role") == "tool" for message in messages)
        calls = (
            []
            if completed
            else [
                {
                    "id": "pg-write",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{}"},
                }
            ]
        )
        return SimpleNamespace(
            content="已完成" if completed else "",
            model_id="test",
            usage={},
            raw={"choices": [{"message": {"tool_calls": calls}}]},
        )


class CountedTools:
    def __init__(self):
        self.executions = 0

    def catalog(self, principal):
        return [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "验收计数工具",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def requires_approval(self, name, arguments):
        return True

    async def execute(self, name, arguments, principal, run_id):
        self.executions += 1
        await asyncio.sleep(0.02)
        return {"count": self.executions}


async def main():
    settings = HarnessSettings(
        database_url=os.environ["AEGIS_TEST_DATABASE_URL"],
        max_concurrent_runs=1,
        max_child_runs=0,
        reflection_enabled=False,
        risk_review_enabled=False,
        evolution_enabled=False,
    )
    assert settings.resolved_database_url().startswith("postgresql+asyncpg://")
    tools = CountedTools()
    original = HarnessService(settings, WriteModel(), tools)
    second = HarnessService(settings, WriteModel(), tools)
    third = HarnessService(settings, WriteModel(), tools)
    await original.initialize()

    async def wait(service, principal, run_id, statuses):
        async with asyncio.timeout(15):
            while True:
                current = await service.get_run(principal, run_id)
                if current["status"] in statuses:
                    return current
                await asyncio.sleep(0.03)

    try:
        user = await original.register("pg-state-" + str(time.time_ns()), "acceptance-test-only")
        principal = Principal(user["user_id"], user["tenant_id"], user["role"])
        run = await original.create_run(principal, "审批后执行一次", "approval")
        waiting = await wait(original, principal, run["id"], {"waiting_approval", "failed"})
        assert waiting["status"] == "waiting_approval" and tools.executions == 0
        await original.close()
        await second.initialize()
        await third.initialize()
        approval = waiting["approval"]
        await asyncio.gather(
            *[
                service.approve(principal, run["id"], True, approval["call_id"], approval["hash"])
                for service in (second, third)
            ]
        )
        completed = await wait(second, principal, run["id"], {"completed", "failed"})
        assert completed["status"] == "completed" and tools.executions == 1
        async with second.store.sessions() as session:
            calls = (
                await session.scalars(select(ToolCall).where(ToolCall.run_id == run["id"]))
            ).all()
            assert len(calls) == 1 and calls[0].status == "done"
        cancelled = await second.create_run(principal, "取消审批任务", "cancel")
        pending = await wait(second, principal, cancelled["id"], {"waiting_approval"})
        await second.cancel(principal, cancelled["id"])
        cancelled_response = await third.approve(
            principal,
            cancelled["id"],
            True,
            pending["approval"]["call_id"],
            pending["approval"]["hash"],
        )
        assert cancelled_response["status"] == "cancelled", "取消后审批只能返回原终态"
        assert (await third.get_run(principal, cancelled["id"]))["status"] == "cancelled"
        await second.close()
        await third.close()
        manual = HarnessService(settings)
        try:
            unknown = await manual.create_run(principal, "未知副作用禁止重放", "unknown")
            async with manual.store.sessions.begin() as session:
                await session.execute(
                    update(Run)
                    .where(Run.id == unknown["id"])
                    .values(
                        status="running",
                        lease_owner="crashed",
                        lease_until=time.time() - 1,
                    )
                )
                session.add(
                    ToolCall(
                        id=uid(),
                        tenant_id=principal.tenant_id,
                        owner_id=principal.user_id,
                        run_id=unknown["id"],
                        call_id="unknown",
                        name="write",
                        arguments={},
                        arguments_hash="0" * 64,
                        status="started",
                    )
                )
            await manual.store.claim("recovery-worker")
            assert (await manual.get_run(principal, unknown["id"]))["status"] == "interrupted"
            assert tools.executions == 1
        finally:
            await manual.close()
        print(
            json.dumps(
                {
                    "数据库": "真实 PostgreSQL",
                    "模型工具": "确定性替身",
                    "审批竞争与重启": "通过",
                    "取消后审批拒绝": "通过",
                    "未知副作用禁止重放": "通过",
                },
                ensure_ascii=False,
            )
        )
    finally:
        for service in (original, second, third):
            if not service.closing:
                await service.close()


if __name__ == "__main__":
    asyncio.run(main())
