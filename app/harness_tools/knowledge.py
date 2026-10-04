"""持久化分块 BM25；可选按用户独立集合的向量检索与重排。"""

import asyncio
import hashlib
import inspect
import math
import re
import threading
from collections import Counter

from sqlalchemy import func, select

from app.harness.errors import HarnessError
from app.harness.models import Asset
from app.harness.store import scope


def tokenize(text: str) -> list[str]:
    """英文词与中文双字切分，避免将整段中文作为单个词项。"""
    result = []
    for word in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", word):
            result.extend(word[index : index + 2] for index in range(len(word) - 1))
        else:
            result.append(word)
    return result


def chunk_id(document_id: str, version: int, index: int) -> str:
    return hashlib.sha256(f"{document_id}:{version}:{index}".encode()).hexdigest()


async def invoke(function, *args):
    """兼容已有同步 embedding 与异步基础设施，同步调用在线程执行。"""
    if inspect.iscoroutinefunction(function):
        return await function(*args)
    result = await asyncio.to_thread(function, *args)
    return await result if inspect.isawaitable(result) else result


class SentenceEmbedding:
    """仅在显式配置后懒加载 sentence-transformers 模型。"""

    def __init__(self, name: str, device: str):
        self.name, self.device = name, device
        self.model = None
        self.lock = threading.Lock()

    def embed_documents(self, texts):
        with self.lock:
            if self.model is None:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.name, device=self.device or None)
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class KnowledgeService:
    def __init__(
        self, settings, service, *, vector_store=None, embedding_model=None, reranker=None
    ):
        self.settings, self.service = settings, service
        self.vector_store, self.embedding_model = vector_store, embedding_model
        self.reranker = reranker
        self.configuration_warning = None
        if settings.knowledge_backend == "hybrid":
            if self.embedding_model is None and settings.knowledge_embedding_model:
                self.embedding_model = SentenceEmbedding(
                    settings.knowledge_embedding_model, settings.knowledge_embedding_device
                )
            if self.vector_store is None and settings.knowledge_milvus_host:
                from app.infrastructure.vectordb.milvus_client import MilvusManager

                password = settings.knowledge_milvus_password.get_secret_value()
                credentials = {}
                if settings.knowledge_milvus_user:
                    credentials = {"user": settings.knowledge_milvus_user, "password": password}
                self.vector_store = MilvusManager(
                    settings.knowledge_milvus_host,
                    settings.knowledge_milvus_port,
                    alias=f"harness_knowledge_{id(self):x}",
                    **credentials,
                )
            if self.vector_store is None or self.embedding_model is None:
                self.configuration_warning = "混合检索未配置完整的 embedding/Milvus，已降级为 BM25"
        if self.reranker is None and settings.knowledge_rerank_model:
            from app.core.rag.reranker import Reranker

            self.reranker = Reranker(
                settings.knowledge_rerank_model, device=settings.knowledge_embedding_device or None
            )

    def collection(self, principal) -> str:
        # 已有管理器没有 expr 支持；物理集合按租户与用户隔离，不访问全局集合。
        key = "\0".join(
            (self.settings.knowledge_milvus_collection, principal.tenant_id, principal.user_id)
        )
        return "aegis_knowledge_" + hashlib.sha256(key.encode()).hexdigest()[:40]

    @property
    def vector_enabled(self) -> bool:
        return (
            self.settings.knowledge_backend == "hybrid"
            and self.vector_store is not None
            and self.embedding_model is not None
        )

    async def index(self, principal, document, chunks) -> dict:
        if principal.role == "viewer":
            raise HarnessError(403, "只读角色不能建立知识索引")
        # 只信任由服务重新读取的资产，外部传入的归属和正文不能覆盖其他用户。
        current = await self.service.get_asset(principal, document["id"])
        if current["kind"] != "document":
            raise HarnessError(422, "只能对文档资产建立知识索引")
        if current["version"] != document["version"]:
            raise HarnessError(409, "文档已修改，不能为旧版本建立索引")
        if not isinstance(chunks, list) or any(not isinstance(text, str) for text in chunks):
            raise HarnessError(422, "知识分块必须是字符串列表")
        kept, remaining = [], self.settings.knowledge_max_chars
        truncated = False
        for text in chunks:
            if not text.strip():
                continue
            if not remaining or len(kept) >= self.settings.knowledge_max_chunks:
                truncated = True
                break
            kept.append(text[:remaining])
            truncated |= len(text) > remaining
            remaining -= len(kept[-1])
        if not kept:
            raise HarnessError(422, "没有可索引的非空分块")
        metadata = {
            **current["metadata"],
            "knowledge_chunks": kept,
            "knowledge_asset_version": current["version"] + 1,
            "knowledge_truncated": truncated,
        }
        saved = await self.service.put_asset(
            principal,
            "document",
            current["name"],
            current["content"],
            metadata,
            current["status"],
            current["id"],
            expected_version=current["version"],
        )
        warnings = self._warnings()
        if truncated:
            warnings.append("索引达到分块或字符预算，后续内容未入索引")
        if self.vector_enabled:
            try:
                await self._bounded(self._index_vectors(principal, saved, kept))
            except Exception as exc:
                warnings.append(self._failure("向量索引", exc))
        return {
            "document_id": saved["id"],
            "chunk_count": len(kept),
            "truncated": truncated,
            "warnings": warnings,
        }

    def _warnings(self) -> list[str]:
        return [self.configuration_warning] if self.configuration_warning else []

    async def _bounded(self, operation):
        return await asyncio.wait_for(operation, timeout=self.settings.knowledge_timeout_seconds)

    @staticmethod
    def _failure(stage, exc) -> str:
        # 外部错误可能包含连接凭据，禁止将原始异常字符串写入用户结果。
        reason = "超时" if isinstance(exc, TimeoutError) else type(exc).__name__
        return f"{stage}失败（{reason}），已保留 BM25 证据"

    @staticmethod
    def _vector(values) -> list[float]:
        vector = [float(value) for value in values]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding 向量为空或含非有限值")
        return vector

    async def _index_vectors(self, principal, document, chunks):
        vectors = [
            self._vector(values)
            for values in await invoke(self.embedding_model.embed_documents, chunks)
        ]
        if len(vectors) != len(chunks) or any(len(v) != len(vectors[0]) for v in vectors):
            raise ValueError("embedding 数量或维度不一致")
        collection = self.collection(principal)
        await self.vector_store.create_collection(collection, len(vectors[0]))
        metadata = [
            {"id": chunk_id(document["id"], document["version"], index)}
            for index in range(len(chunks))
        ]
        await self.vector_store.insert(collection, vectors, metadata)

    async def _load_chunks(self, principal):
        items, total_chars, truncated = [], 0, False
        rebuilt = False
        max_chars = self.settings.knowledge_max_chars
        # WHERE 同时限定租户和用户；流式读限定文档数，不扫描其他范围的资产。
        query = (
            select(
                Asset.id,
                Asset.name,
                Asset.version,
                Asset.attributes,
                func.substr(Asset.content, 1, max_chars + 1).label("text"),
            )
            .where(*scope(Asset, principal), Asset.kind == "document", Asset.status == "active")
            .order_by(Asset.name, Asset.id)
            .limit(self.settings.knowledge_max_documents + 1)
            .execution_options(yield_per=1)
        )
        async with self.service.store.sessions() as session:
            rows = await session.stream(query)
            document_count = 0
            async for row in rows:
                if document_count >= self.settings.knowledge_max_documents:
                    truncated = True
                    break
                document_count += 1
                metadata = row.attributes or {}
                chunks = metadata.get("knowledge_chunks")
                if (
                    metadata.get("knowledge_asset_version") != row.version
                    or not isinstance(chunks, list)
                    or any(not isinstance(text, str) for text in chunks)
                ):
                    rebuilt = True
                    chunks = [
                        row.text[index : index + 1500] for index in range(0, len(row.text), 1500)
                    ]
                truncated |= bool(metadata.get("knowledge_truncated"))
                for index, text in enumerate(chunks):
                    if not text.strip():
                        continue
                    remaining = max_chars - total_chars
                    if remaining <= 0 or len(items) >= self.settings.knowledge_max_chunks:
                        truncated = True
                        break
                    bounded = text[:remaining]
                    truncated |= len(bounded) < len(text)
                    total_chars += len(bounded)
                    items.append(
                        {
                            "id": chunk_id(row.id, row.version, index),
                            "document_id": row.id,
                            "version": row.version,
                            "source": row.name,
                            "chunk_index": index,
                            "content": bounded,
                        }
                    )
                if total_chars >= max_chars or len(items) >= self.settings.knowledge_max_chunks:
                    truncated = True
                    break
            await rows.close()
        return items, truncated, rebuilt

    def _bm25(self, query, items):
        frequencies = [Counter(tokenize(item["content"])) for item in items]
        lengths = [sum(counts.values()) for counts in frequencies]
        average = sum(lengths) / len(items) if items else 1
        document_frequency = Counter(term for counts in frequencies for term in counts)
        query_terms = set(tokenize(query))
        ranked = []
        for item, counts, length in zip(items, frequencies, lengths):
            score = 0.0
            for term in query_terms & counts.keys():
                df = document_frequency[term]
                idf = math.log(1 + (len(items) - df + 0.5) / (df + 0.5))
                frequency = counts[term]
                normalizer = frequency + 1.5 * (0.25 + 0.75 * length / (average or 1))
                score += idf * frequency * 2.5 / normalizer
            if score > 0:
                ranked.append({**item, "score": score})
        return sorted(ranked, key=lambda item: (-item["score"], item["id"]))

    async def _search_vectors(self, principal, query, items):
        vector = self._vector(await invoke(self.embedding_model.embed_query, query))
        raw = await self.vector_store.search(
            self.collection(principal), vector, top_k=self.settings.knowledge_top_k * 4
        )
        by_id = {item["id"]: item for item in items}
        # 只关联本次作用域、active 状态、当前版本且预算内的权威文档块。
        return [by_id[hit["id"]] for hit in raw if hit.get("id") in by_id]

    @staticmethod
    def _fuse(keyword, vector):
        scores, items = Counter(), {}
        for ranked in (keyword, vector):
            seen = set()
            for rank, item in enumerate(ranked):
                if item["id"] not in seen:
                    seen.add(item["id"])
                    scores[item["id"]] += 1 / (61 + rank)
                    items[item["id"]] = item
        return [{**items[key], "score": score} for key, score in scores.most_common()]

    async def _rerank(self, query, ranked):
        from app.models.schemas import RetrievalResult

        candidates = [
            RetrievalResult(
                id=item["id"], content=item["content"], score=item["score"], source="knowledge"
            )
            for item in ranked[: self.settings.knowledge_top_k * 4]
        ]
        reranked = await self.reranker.rerank(
            query, candidates, top_k=self.settings.knowledge_top_k
        )
        by_id = {item["id"]: item for item in ranked}
        seen, result = set(), []
        for hit in reranked:
            if hit.id in by_id and hit.id not in seen and math.isfinite(hit.score):
                seen.add(hit.id)
                result.append({**by_id[hit.id], "score": float(hit.score)})
        if candidates and not result:
            raise ValueError("重排没有返回有效的当前范围证据")
        return result

    async def search(self, principal, query) -> dict:
        if not isinstance(query, str) or len(query) > 12000:
            raise HarnessError(422, "知识检索问题必须是至多 12000 字符的文本")
        warnings = self._warnings()
        if not tokenize(query):
            return {"results": [], "backend": "bm25", "warnings": warnings, "truncated": False}
        items, truncated, rebuilt = await self._load_chunks(principal)
        if truncated:
            warnings.append("检索达到文档、分块或字符预算，仅覆盖当前范围内的有限证据")
        if rebuilt:
            warnings.append("部分文档未保存当前版本分块，已按当前正文重建词法分块")
        backend = "bm25"
        if self.vector_enabled:
            # 两路只读取同一份已鉴权的块；向量异常独立降级，取消仍向上传播。
            ranked, vector = await asyncio.gather(
                asyncio.to_thread(self._bm25, query, items),
                self._bounded(self._search_vectors(principal, query, items)),
                return_exceptions=True,
            )
            if isinstance(ranked, BaseException):
                raise ranked
            if isinstance(vector, Exception):
                warnings.append(self._failure("向量检索", vector))
            elif isinstance(vector, BaseException):
                raise vector
            else:
                ranked = self._fuse(ranked, vector)
                backend = "hybrid"
        else:
            ranked = await asyncio.to_thread(self._bm25, query, items)
        if self.reranker is not None and ranked:
            try:
                ranked = await self._bounded(self._rerank(query, ranked))
            except Exception as exc:
                warnings.append(self._failure("重排", exc))
        return {
            "results": ranked[: self.settings.knowledge_top_k],
            "backend": backend,
            "warnings": warnings,
            "truncated": truncated,
        }
