# -*- coding: utf-8 -*-
"""Milvus 向量库管理器：集合生命周期与向量检索封装。"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
except ImportError:  # pragma: no cover
    Collection = Any  # type: ignore[misc, assignment]
    CollectionSchema = Any  # type: ignore[misc, assignment]
    DataType = Any  # type: ignore[misc, assignment]
    FieldSchema = Any  # type: ignore[misc, assignment]
    connections = Any  # type: ignore[misc, assignment]
    utility = Any  # type: ignore[misc, assignment]


def _run_sync(func, *args, **kwargs):
    """在线程池中执行同步 Milvus SDK 调用，避免阻塞事件循环。"""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))


class MilvusManager:
    """Milvus 向量数据库管理器（异步包装同步 SDK）。"""

    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        alias: str = "default",
        **conn_kwargs: Any,
    ) -> None:
        self._host = host
        self._port = port
        self._alias = alias
        self._conn_kwargs = conn_kwargs
        self._connected = False

    async def _ensure_connection(self) -> None:
        if self._connected:
            return
        try:

            def _connect() -> None:
                connections.connect(
                    alias=self._alias,
                    host=self._host,
                    port=self._port,
                    **self._conn_kwargs,
                )

            await _run_sync(_connect)
            self._connected = True
            logger.info("已连接 Milvus {}:{}", self._host, self._port)
        except Exception as exc:
            logger.exception("连接 Milvus 失败: {}", exc)
            raise

    async def create_collection(self, name: str, dim: int) -> None:
        """创建向量集合；若已存在则跳过。"""
        await self._ensure_connection()

        def _create() -> None:
            if utility.has_collection(name, using=self._alias):
                logger.info("集合 [{}] 已存在，跳过创建", name)
                return
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=64,
                ),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(
                fields=fields,
                description=f"向量集合 {name}",
            )
            col = Collection(name, schema, using=self._alias)
            idx = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
            col.create_index(field_name="embedding", index_params=idx)
            logger.info("已创建集合 [{}] dim={}", name, dim)

        try:
            await _run_sync(_create)
        except Exception as exc:
            logger.exception("创建集合失败 name={}: {}", name, exc)
            raise

    async def insert(
        self,
        collection: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]],
    ) -> list[str]:
        """插入向量与元数据；主键取自 metadata 中的 id 字段。"""
        await self._ensure_connection()
        if len(vectors) != len(metadata):
            raise ValueError("vectors 与 metadata 长度必须一致")

        ids: list[str] = []
        for i, m in enumerate(metadata):
            rid = m.get("id")
            if rid is None:
                rid = f"auto_{i}"
            ids.append(str(rid))

        def _insert() -> None:
            col = Collection(collection, using=self._alias)
            col.insert([ids, vectors])
            col.flush()

        try:
            await _run_sync(_insert)
            return ids
        except Exception as exc:
            logger.exception("Milvus 插入失败 collection={}: {}", collection, exc)
            raise

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
        **search_params: Any,
    ) -> list[dict[str, Any]]:
        """向量相似度检索，返回命中 id 与距离。"""
        await self._ensure_connection()
        params = search_params.get(
            "params",
            {"metric_type": "L2", "params": {"nprobe": 10}},
        )

        def _search():
            col = Collection(collection, using=self._alias)
            col.load()
            res = col.search(
                data=[query_vector],
                anns_field="embedding",
                param=params,
                limit=top_k,
                output_fields=["id"],
            )
            out: list[dict[str, Any]] = []
            for hits in res:
                for hit in hits:
                    out.append(
                        {
                            "id": hit.entity.get("id"),
                            "distance": float(hit.distance),
                        }
                    )
            return out

        try:
            return await _run_sync(_search)
        except Exception as exc:
            logger.exception("Milvus 检索失败 collection={}: {}", collection, exc)
            raise

    async def delete(self, collection: str, ids: list[str]) -> None:
        """按主键删除向量。"""
        await self._ensure_connection()
        if not ids:
            return

        expr = "id in [" + ", ".join(f'"{i}"' for i in ids) + "]"

        def _delete() -> None:
            col = Collection(collection, using=self._alias)
            col.delete(expr)

        try:
            await _run_sync(_delete)
        except Exception as exc:
            logger.exception("Milvus 删除失败 collection={}: {}", collection, exc)
            raise
