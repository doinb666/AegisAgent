# -*- coding: utf-8 -*-
"""长期记忆：向量库存储与按会话召回。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from app.models.schemas import MemoryItem


@runtime_checkable
class LTMEmbedProtocol(Protocol):
    """嵌入模型接口。"""

    def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class LTMCollectionProtocol(Protocol):
    """Milvus Collection 最小接口。"""

    def insert(self, data: Any, **kwargs: Any) -> Any:
        ...

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def delete(self, expr: str, **kwargs: Any) -> Any:
        ...

    def flush(self, **kwargs: Any) -> Any:
        ...


class LongTermMemory:
    """长期记忆：基于向量数据库的持久化记忆。"""

    vector_field: str = "embedding"
    content_field: str = "content"
    session_field: str = "session_id"
    pk_field: str = "pk"
    meta_field: str = "meta"

    metric_param: dict[str, Any] = {"metric_type": "L2", "params": {"nprobe": 16}}

    def __init__(self, milvus_collection: Any, embedding_model: Any) -> None:
        """
        :param milvus_collection: Milvus Collection，需包含向量、文本、会话 ID 等字段
        :param embedding_model: 含 ``embed_query`` 的嵌入模型
        """
        self._coll = milvus_collection
        self._embed = embedding_model

    def _ensure_embed(self) -> None:
        if not isinstance(self._embed, LTMEmbedProtocol):
            raise TypeError("embedding_model 需实现 embed_query")

    def _ensure_coll(self) -> None:
        if not isinstance(self._coll, LTMCollectionProtocol):
            raise TypeError("milvus_collection 需支持 insert/search/delete")

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        """写入一条长期记忆，返回 memory_id。"""
        self._ensure_embed()
        self._ensure_coll()

        memory_id = str(uuid.uuid4())
        meta = dict(metadata)
        meta["memory_id"] = memory_id

        def _sync() -> None:
            vec = self._embed.embed_query(content)
            row = {
                self.pk_field: memory_id,
                self.vector_field: vec,
                self.content_field: content,
                self.session_field: session_id,
                self.meta_field: json.dumps(meta, ensure_ascii=False),
            }
            # pymilvus 2.4+ 支持实体字典列表，字段名需与 Collection Schema 一致
            self._coll.insert([row])
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆写入失败: {}", e)
            raise RuntimeError(f"store 失败: {e}") from e

        return memory_id

    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        """按语义在指定会话内召回记忆。"""
        self._ensure_embed()
        self._ensure_coll()

        def _sync() -> list[MemoryItem]:
            vec = self._embed.embed_query(query)
            # 转义单引号，避免 expr 注入
            sid = session_id.replace("'", "\\'")
            expr = f'{self.session_field} == "{sid}"'
            out = self._coll.search(
                data=[vec],
                anns_field=self.vector_field,
                param=self.metric_param,
                limit=top_k,
                expr=expr,
                output_fields=[self.pk_field, self.content_field, self.meta_field],
            )
            items: list[MemoryItem] = []
            hits = out[0] if out else []
            for hit in hits:
                entity = getattr(hit, "entity", {}) or {}
                if hasattr(hit, "entity") and not isinstance(entity, dict):
                    try:
                        entity = hit.entity.to_dict()  # type: ignore[assignment]
                    except Exception:
                        entity = {}
                rid = str(entity.get(self.pk_field) or getattr(hit, "id", ""))
                text = str(entity.get(self.content_field) or "")
                meta_raw = entity.get(self.meta_field) or "{}"
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
                except json.JSONDecodeError:
                    meta = {}
                score = float(getattr(hit, "distance", 0.0) or 0.0)
                items.append(
                    MemoryItem(id=rid, content=text, score=score, metadata=meta),
                )
            return items

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆召回失败: {}", e)
            raise RuntimeError(f"recall 失败: {e}") from e

    async def forget(self, memory_id: str) -> None:
        """按主键删除一条记忆。"""
        self._ensure_coll()

        def _sync() -> None:
            mid = memory_id.replace("'", "\\'")
            expr = f'{self.pk_field} == "{mid}"'
            self._coll.delete(expr)
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆删除失败: {}", e)
            raise RuntimeError(f"forget 失败: {e}") from e
