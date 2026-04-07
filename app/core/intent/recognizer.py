# -*- coding: utf-8 -*-
"""意图识别：树形分类、置信度与澄清引导。"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """意图识别结果。"""

    intent: str = Field(description="主意图标签")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度 0~1")
    slots: dict[str, Any] = Field(default_factory=dict, description="槽位信息")
    sub_intent: str | None = Field(default=None, description="子意图（树形第二层）")
    rationale: str = Field(default="", description="简要理由（可给模型或日志用）")


# 简易树形意图：根 -> 子意图（关键词规则，可替换为分类模型）
_INTENT_TREE: dict[str, dict[str, list[str]]] = {
    "问答": {
        "知识问答": ["什么", "为什么", "如何", "解释", "含义"],
        "闲聊": ["你好", "谢谢", "再见", "聊天"],
    },
    "任务": {
        "搜索": ["搜索", "查一下", "帮我找", "search"],
        "计算": ["计算", "等于", "算", "加", "减"],
        "数据库": ["sql", "查询", "表", "统计"],
    },
    "文档": {
        "上传": ["上传", "文档", "pdf"],
        "总结": ["总结", "摘要", "概括"],
    },
}


class IntentRecognizer:
    """意图识别器：树形分类 + 置信度 + 引导澄清。"""

    def __init__(self, confidence_threshold: float = 0.55) -> None:
        self._threshold = confidence_threshold

    @property
    def confidence_threshold(self) -> float:
        """置信度阈值：低于该值时建议澄清。"""
        return self._threshold

    def _score_branch(self, query: str) -> tuple[str, str | None, float, str]:
        """返回 (根意图, 子意图, 置信度, 理由)。"""
        q = query.strip().lower()
        best_root = "未知"
        best_child: str | None = None
        best_hits = 0
        total_kw = 0

        for root, children in _INTENT_TREE.items():
            for child, kws in children.items():
                hits = sum(1 for kw in kws if kw.lower() in q)
                if hits > best_hits:
                    best_hits = hits
                    best_root = root
                    best_child = child
                total_kw += len(kws)

        if best_hits == 0:
            # 无关键词命中：根据长度与问号弱推断
            conf = 0.35 if "?" in q or "？" in q else 0.25
            return "问答", "知识问答", conf, "未命中关键词，弱规则推断"

        conf = min(0.95, 0.45 + 0.12 * best_hits)
        rationale = f"关键词命中 {best_hits} 次，归类为 {best_root}/{best_child}"
        return best_root, best_child, conf, rationale

    async def recognize(self, query: str, context: dict | None = None) -> IntentResult:
        """识别用户意图；context 可包含历史轮次等（当前规则引擎未使用）。"""
        _ = context
        root, child, conf, why = self._score_branch(query)
        slots: dict[str, Any] = {}
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", query)
        if nums:
            slots["numbers"] = nums[:5]

        result = IntentResult(
            intent=root,
            confidence=conf,
            slots=slots,
            sub_intent=child,
            rationale=why,
        )
        logger.info(
            "意图识别 query_preview={} -> {} / {} conf={}",
            query[:100],
            root,
            child,
            conf,
        )
        return result

    async def clarify(self, query: str, intent_result: IntentResult) -> str:
        """置信度不足时生成澄清提示语。"""
        if intent_result.confidence >= self._threshold:
            return ""

        return (
            "我不太确定您的具体需求。"
            f"您是想了解「{intent_result.intent}」相关（当前置信度 {intent_result.confidence:.2f}）吗？"
            "请补充场景、对象或期望的输出格式（例如：只要结论 / 需要步骤）。"
        )
