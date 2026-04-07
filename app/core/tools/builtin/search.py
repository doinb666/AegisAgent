# -*- coding: utf-8 -*-
"""内置网络搜索工具（基于 DuckDuckGo 即时答案或占位 HTTP）。"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter


class WebSearchTool(BaseTool):
    """通过公开 HTTP 接口执行轻量搜索（无 API Key 场景下的降级实现）。"""

    def __init__(self, timeout: float = 15.0) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = "在互联网上搜索关键词并返回摘要文本。"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词或完整问句",
                required=True,
            )
        ]
        self._timeout = timeout

    async def execute(self, **kwargs: Any) -> str:
        """执行搜索：优先 DuckDuckGo html 简化解析，失败则返回提示信息。"""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        url = "https://duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.post(url, data={"q": query})
                resp.raise_for_status()
                text = resp.text[:4000]
                logger.info("web_search 完成 query={} 响应长度={}", query, len(text))
                return f"搜索「{query}」的原始结果片段（已截断）：\n{text}"
        except Exception as exc:
            logger.exception("web_search 失败: {}", exc)
            return f"搜索暂时不可用，请稍后重试。错误: {exc!s}"
