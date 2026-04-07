# -*- coding: utf-8 -*-
"""RAG 答案生成：基于检索上下文与对话历史，支持引用标注。"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from app.models.schemas import Citation, Message, RAGResponse, RetrievalResult


@runtime_checkable
class RAGLLMProtocol(Protocol):
    """支持异步生成的 LLM 接口（如 LangChain Runnable）。"""

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        ...


class RAGGenerator:
    """RAG 答案生成器：基于检索到的上下文生成答案，支持引用标注。"""

    def __init__(
        self,
        llm: Any,
        system_prompt: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """
        :param llm: 实现 ``ainvoke`` 的模型对象，或兼容 OpenAI 风格的封装
        :param system_prompt: 可选系统提示，强调引用格式
        :param model_name: 写入 RAGResponse.model 的展示名
        """
        self._llm = llm
        self._system_prompt = system_prompt or (
            "你是严谨的知识助手。仅根据提供的上下文作答；若上下文不足请说明。"
            "回答中引用来源时使用 [1]、[2] 等形式，与上下文编号一致。"
        )
        self._model_name = model_name

    def _build_messages(
        self,
        query: str,
        contexts: list[RetrievalResult],
        chat_history: list[Any],
    ) -> list[dict[str, str]]:
        """构造 chat messages（OpenAI 风格）。"""
        ctx_lines = []
        for i, c in enumerate(contexts, start=1):
            ctx_lines.append(f"[{i}] (id={c.id})\n{c.content}")
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "（无检索上下文）"

        history_lines: list[str] = []
        for msg in chat_history:
            if isinstance(msg, Message):
                history_lines.append(f"{msg.role.value}: {msg.content}")
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_lines.append(f"{role}: {content}")
            else:
                history_lines.append(str(msg))

        user_content = (
            f"## 检索上下文\n{context_block}\n\n"
            f"## 历史对话（摘要参考）\n"
            f"{chr(10).join(history_lines) if history_lines else '（无）'}\n\n"
            f"## 用户问题\n{query}\n\n"
            "请作答并在必要时使用 [1]、[2] 引用上下文编号。"
        )

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _extract_citations(self, answer: str, contexts: list[RetrievalResult]) -> list[Citation]:
        """从回答中解析 [n] 引用并映射到 RetrievalResult。"""
        refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
        citations: list[Citation] = []
        seen: set[int] = set()
        for idx in refs:
            if idx in seen or idx < 1 or idx > len(contexts):
                continue
            seen.add(idx)
            r = contexts[idx - 1]
            snippet = r.content[:200] + ("..." if len(r.content) > 200 else "")
            citations.append(
                Citation(index=idx, result_id=r.id, snippet=snippet),
            )
        return citations

    def _parse_llm_output(self, raw: Any) -> str:
        """从 LLM 返回对象中提取文本。"""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "content"):
            return str(getattr(raw, "content", ""))
        if isinstance(raw, dict) and "content" in raw:
            return str(raw["content"])
        return str(raw)

    async def generate(
        self,
        query: str,
        contexts: list[RetrievalResult],
        chat_history: list[Any],
    ) -> RAGResponse:
        """基于上下文与历史生成回答，并解析引用。"""
        messages = self._build_messages(query, contexts, chat_history)

        try:
            if isinstance(self._llm, RAGLLMProtocol):
                raw = await self._llm.ainvoke(messages)
            elif callable(getattr(self._llm, "ainvoke", None)):
                raw = await self._llm.ainvoke(messages)  # type: ignore[misc]
            else:
                raise TypeError("llm 需实现异步 ainvoke(messages)")
        except Exception as e:
            logger.exception("RAG 生成调用失败: {}", e)
            raise RuntimeError(f"生成失败: {e}") from e

        answer = self._parse_llm_output(raw).strip()
        citations = self._extract_citations(answer, contexts)

        return RAGResponse(
            answer=answer,
            citations=citations,
            raw_contexts=list(contexts),
            model=self._model_name,
        )
