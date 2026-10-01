"""隔离验收库的 PG 多实例与有界负载；模型为确定性替身。"""

import asyncio
import json
import os
import time
from types import SimpleNamespace

from app.harness import HarnessService, HarnessSettings, Principal


class MeasuredModel:
    async def chat(self, messages, **kwargs):
        await asyncio.sleep(0.02)
        return SimpleNamespace(content="测试回答", model_id="deterministic-test", usage={}, raw={})


def percentile(values, fraction):
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))], 4)


async def main():
    settings = HarnessSettings(
        database_url=os.environ["AEGIS_TEST_DATABASE_URL"],
        max_concurrent_runs=2, max_child_runs=1, max_user_runs=32, evolution_enabled=False
    )
    assert settings.resolved_database_url().startswith("postgresql+asyncpg://"), "需要独立PG验收库"
    first = HarnessService(settings, MeasuredModel())
    await first.initialize()
    second = HarnessService(settings, MeasuredModel())
    await second.initialize()
    prefix = "load-" + str(time.time_ns())
    principals = []
    for index in range(5):
        user = await first.register(f"{prefix}-{index}", "acceptance-test-only")
        principals.append(Principal(user["user_id"], user["tenant_id"], user["role"]))

    async def wait_run(principal, run_id):
        async with asyncio.timeout(120):
            while True:
                run = await first.get_run(principal, run_id)
                if run["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                    return run
                await asyncio.sleep(0.05)

    results = []
    try:
        same = await asyncio.gather(
            *[
                (first if index % 2 else second).create_run(principals[0], "并发幂等", "same-key")
                for index in range(20)
            ]
        )
        assert len({run["id"] for run in same}) == 1
        await wait_run(principals[0], same[0]["id"])
        try:
            await second.get_run(principals[1], same[0]["id"])
            raise AssertionError("跨租户读取未被拒绝")
        except Exception as error:
            assert getattr(error, "status_code", None) == 404
        for count in (10, 50, 100):

            async def one(index):
                principal = principals[index % len(principals)]
                started = time.perf_counter()
                service = first if index % 2 else second
                run = await service.create_run(principal, "负载测试", f"{count}-{index}")
                created = time.time()
                result = await wait_run(principal, run["id"])
                assert result["status"] == "completed", result
                events = await service.events(principal, run["id"])
                running = [event for event in events if event["type"] == "running"]
                assert len(running) == 1, "任务被重复领取"
                queued = max(0, running[0]["data"]["claimed_at"] - created)
                return time.perf_counter() - started, queued

            started = time.perf_counter()
            measurements = await asyncio.gather(*(one(index) for index in range(count)))
            duration = time.perf_counter() - started
            latencies, queues = zip(*measurements)
            results.append(
                {
                    "tasks": count,
                    "p50_seconds": percentile(latencies, 0.50),
                    "p95_seconds": percentile(latencies, 0.95),
                    "p99_seconds": percentile(latencies, 0.99),
                    "queue_p95_seconds": percentile(queues, 0.95),
                    "errors": 0,
                    "throughput_per_second": round(count / duration, 2),
                }
            )
        print(
            json.dumps(
                {
                    "database": "真实PostgreSQL",
                    "instances": 2,
                    "model": "确定性替身20ms；不是实际LLM吞吐",
                    "idempotency_20": "通过",
                    "loads": results,
                },
                ensure_ascii=False,
            )
        )
    finally:
        await first.close()
        await second.close()


if __name__ == "__main__":
    asyncio.run(main())
