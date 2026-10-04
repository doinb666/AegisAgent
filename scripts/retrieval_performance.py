"""临时数据库与受控延迟下的串行/并行召回对比，不代表生产性能。"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness_tools.knowledge import KnowledgeService


class Embedding:
    def embed_query(self, text):
        return [1.0, 0.0]


class VectorBackend:
    async def search(self, collection, vector, top_k):
        await asyncio.sleep(0.08)
        return []


class MeasuredKnowledge(KnowledgeService):
    def _bm25(self, query, items):
        # 两路各注入 80ms，以观察重叠等待而不依赖网络或商业模型。
        time.sleep(0.08)
        return super()._bm25(query, items)


async def measure(rounds):
    with tempfile.TemporaryDirectory(prefix="aegis-retrieval-") as directory:
        settings = HarnessSettings(
            data_dir=Path(directory), database_url="", knowledge_backend="hybrid"
        )
        service = HarnessService(settings)
        await service.store.initialize()
        try:
            user = await service.register("performance-local", "temporary-password")
            principal = Principal(user["user_id"], user["tenant_id"], user["role"])
            await service.put_asset(
                principal, "document", "基准证据", "needle " * 500, status="active"
            )
            knowledge = MeasuredKnowledge(
                settings, service, vector_store=VectorBackend(), embedding_model=Embedding()
            )
            serial, parallel = [], []
            for _ in range(rounds):
                start = time.perf_counter()
                items, _, _ = await knowledge._load_chunks(principal)
                keyword = await asyncio.to_thread(knowledge._bm25, "needle", items)
                vector = await knowledge._bounded(
                    knowledge._search_vectors(principal, "needle", items)
                )
                expected = knowledge._fuse(keyword, vector)[: settings.knowledge_top_k]
                serial.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                actual = await knowledge.search(principal, "needle")
                parallel.append((time.perf_counter() - start) * 1000)
                assert actual["results"] == expected
            before, after = statistics.median(serial), statistics.median(parallel)
            print(json.dumps({
                "说明": "合成双路延迟，各80ms；不代表真实Milvus或生产吞吐",
                "轮数": rounds,
                "结果一致": True,
                "串行中位数毫秒": round(before, 2),
                "并行中位数毫秒": round(after, 2),
                "相对减少百分比": round((before - after) / before * 100, 2),
            }, ensure_ascii=False))
        finally:
            await service.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 100:
        parser.error("轮数须为1至100")
    asyncio.run(measure(args.rounds))
