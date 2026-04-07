# -*- coding: utf-8 -*-
"""统一记忆管理：协调短期记忆与长期记忆。"""

from __future__ import annotations

from loguru import logger

from app.core.memory.long_term import LongTermMemory
from app.core.memory.short_term import ShortTermMemory
from app.models.schemas import MemoryContext, Message


class MemoryManager:
    """统一记忆管理器：协调短期记忆和长期记忆。"""

    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory) -> None:
        """
        :param short_term: 短期记忆实现（Redis + 窗口）
        :param long_term: 长期记忆实现（向量库）
        """
        self._stm = short_term
        self._ltm = long_term

    async def get_context(self, session_id: str, query: str) -> MemoryContext:
        """获取与当前查询相关的记忆上下文（短期历史 + 长期召回）。"""
        try:
            short_msgs = await self._stm.get_history(session_id)
        except Exception as e:
            logger.exception("读取短期记忆失败: {}", e)
            short_msgs = []

        try:
            long_items = await self._ltm.recall(query, session_id, top_k=5)
        except Exception as e:
            logger.exception("长期记忆召回失败: {}", e)
            long_items = []

        return MemoryContext(
            session_id=session_id,
            short_term_messages=short_msgs,
            long_term_items=long_items,
        )

    async def save(self, session_id: str, message: Message) -> None:
        """将新消息写入短期记忆（滑动窗口与压缩由 ShortTermMemory 负责）。"""
        try:
            await self._stm.add_message(session_id, message)
        except Exception as e:
            logger.exception("保存短期记忆失败: {}", e)
            raise RuntimeError(f"save 失败: {e}") from e
