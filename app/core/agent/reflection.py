# -*- coding: utf-8 -*-
"""
反思 Agent：对输出做质量检查、幻觉与完整性评估，并给出改进建议。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


REFLECTION_SYSTEM_PROMPT = """你是严格的输出质量审查员。请对用户问题与助手回答进行审查。

## 你必须只输出一个 JSON 对象，格式如下：
{
  "quality_score": 0-100 的整数,
  "is_complete": true/false,
  "likely_hallucination": true/false,
  "hallucination_reasons": ["若怀疑幻觉，列出具体疑点；否则为空数组"],
  "completeness_notes": "是否遗漏关键要点",
  "suggestions": ["可执行的改进建议，面向助手"],
  "summary": "一句中文总结审查结论"
}

审查标准：
- 幻觉：回答是否包含无依据的具体事实、虚构来源或与用户问题无关的断言。
- 完整性：是否覆盖用户问题的核心子问题。
- 质量分：综合考虑正确性、清晰度与有用性。
"""


@dataclass
class ReflectionReport:
    """反思审查报告。"""

    quality_score: int
    is_complete: bool
    likely_hallucination: bool
    hallucination_reasons: List[str]
    completeness_notes: str
    suggestions: List[str]
    summary: str
    raw_model_output: Optional[str] = None
    parse_error: Optional[str] = None


class ReflectionLLM(Protocol):
    """反思阶段使用的 LLM。"""

    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        ...


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法解析 JSON")
    return json.loads(m.group(0))


class ReflectionAgent:
    """对 Agent 最终输出进行反思与质量把关。"""

    def __init__(self, llm: ReflectionLLM, min_quality_to_pass: int = 60) -> None:
        self._llm = llm
        self.min_quality_to_pass = max(0, min(100, min_quality_to_pass))

    async def reflect(
        self,
        user_query: str,
        agent_answer: str,
        evidence_snippets: Optional[List[str]] = None,
        trace_summary: Optional[str] = None,
    ) -> ReflectionReport:
        """
        对助手答案进行质量检查。

        evidence_snippets：可选的检索或工具观察片段，用于对照幻觉。
        trace_summary：可选的执行轨迹摘要，用于判断推理是否支撑结论。
        """
        ev_block = "\n".join(f"- {s}" for s in (evidence_snippets or [])) or "（无外部证据）"
        trace_block = trace_summary or "（无轨迹）"
        user_content = f"""## 用户问题
{user_query}

## 助手回答
{agent_answer}

## 外部证据/观察片段
{ev_block}

## 执行轨迹摘要
{trace_block}

请输出 JSON 审查结果。"""

        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            raw = await self._llm.acomplete(messages, temperature=0.1)
            data = _extract_json(raw)
        except Exception as e:  # noqa: BLE001
            logger.exception("反思模型调用或解析失败")
            return ReflectionReport(
                quality_score=50,
                is_complete=False,
                likely_hallucination=False,
                hallucination_reasons=[],
                completeness_notes="反思阶段解析失败，无法完成自动审查",
                suggestions=["请人工复核该回答"],
                summary="反思流程异常，已降级为保守评分",
                raw_model_output=None,
                parse_error=str(e),
            )

        try:
            score = int(data.get("quality_score", 0))
        except (TypeError, ValueError):
            score = 50

        report = ReflectionReport(
            quality_score=max(0, min(100, score)),
            is_complete=bool(data.get("is_complete", False)),
            likely_hallucination=bool(data.get("likely_hallucination", False)),
            hallucination_reasons=list(data.get("hallucination_reasons") or []),
            completeness_notes=str(data.get("completeness_notes", "")),
            suggestions=list(data.get("suggestions") or []),
            summary=str(data.get("summary", "")),
            raw_model_output=raw[:8000],
        )
        return report

    def should_retry_or_warn(self, report: ReflectionReport) -> Dict[str, Any]:
        """
        根据报告给出是否建议重试/告警的企业级决策结构。
        """
        warn = report.quality_score < self.min_quality_to_pass
        retry = report.likely_hallucination or not report.is_complete
        return {
            "warn_low_quality": warn,
            "suggest_retry": retry,
            "min_quality_threshold": self.min_quality_to_pass,
        }
