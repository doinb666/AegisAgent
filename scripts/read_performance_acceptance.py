"""固定上下文与轮询负载对比完整行读取和状态投影，仅使用临时 SQLite。"""

import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from sqlalchemy import select

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.models import Run
from app.harness.store import run_dict


async def measure():
    with tempfile.TemporaryDirectory(prefix="aegis-read-benchmark-") as directory:
        service = HarnessService(HarnessSettings(data_dir=Path(directory), database_url=""))
        await service.store.initialize()
        try:
            user = await service.register("read-benchmark", "benchmark-password")
            principal = Principal(user["user_id"], user["tenant_id"], user["role"])
            run = await service.create_run(principal, "固定读取负载", "read-benchmark")
            async with service.store.sessions.begin() as session:
                row = await session.scalar(select(Run).where(Run.id == run["id"]))
                row.messages = [{"role": "tool", "content": "x" * (2 * 1024 * 1024)}]

            async def baseline():
                async with service.store.sessions() as session:
                    return run_dict(await service.store.owned(session, Run, run["id"], principal))

            async def projection():
                return await service.get_run(principal, run["id"])

            results = {"full_row": [], "projection": []}
            for operation in (baseline, projection):
                assert await operation() == run
            # 交替测量以减少缓存和温度偏差，50 次相同请求，无模型推理。
            for _ in range(50):
                for name, operation in (("full_row", baseline), ("projection", projection)):
                    started = time.perf_counter()
                    assert await operation() == run
                    results[name].append((time.perf_counter() - started) * 1000)
            summary = {
                name: {
                    "requests": len(samples),
                    "median_ms": round(statistics.median(samples), 3),
                    "p95_ms": round(sorted(samples)[47], 3),
                }
                for name, samples in results.items()
            }
            return {
                "database": "temporary SQLite",
                "context_bytes": 2 * 1024 * 1024,
                "model_requests": 0,
                "results": summary,
                "median_reduction_percent": round(
                    100
                    * (1 - summary["projection"]["median_ms"] / summary["full_row"]["median_ms"]),
                    2,
                ),
            }
        finally:
            await service.close()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(measure()), ensure_ascii=False, indent=2))
