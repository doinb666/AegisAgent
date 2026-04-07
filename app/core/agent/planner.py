# -*- coding: utf-8 -*-
"""
规划 Agent：Plan-and-Execute，含任务分解、执行与重规划。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from app.core.agent.react_agent import AgentResult, LLMCallable, MemoryLike, ToolInvoker

logger = logging.getLogger(__name__)


PLAN_SYSTEM_PROMPT = """你是任务规划专家。请将用户目标拆分为有序、可执行的子任务列表。

## 输出格式（仅输出 JSON，不要其他文字）
{
  "subtasks": [
    {
      "id": "t1",
      "title": "子任务标题",
      "description": "具体要做什么",
      "action_type": "tool|reasoning",
      "tool_name": "若 action_type 为 tool 则填写工具名，否则 null",
      "tool_args_hint": "工具参数要点（自然语言提示）"
    }
  ]
}

规则：
- action_type 为 reasoning 表示主要依赖模型推理整合，无需调用工具。
- 子任务数量建议 2～8 个，避免过细或过粗。
- id 必须唯一。
"""


REPLAN_SYSTEM_PROMPT = """你是任务规划专家。根据已执行结果与错误信息，修订后续计划。

## 输出格式（仅输出 JSON）
{
  "subtasks": [ ... 同上结构 ... ],
  "notes": "简要说明为何如此调整"
}

若任务已完成，返回：
{ "subtasks": [], "notes": "已完成，原因说明" }
"""


@dataclass
class SubTask:
    """单个子任务。"""

    id: str
    title: str
    description: str
    action_type: str  # "tool" | "reasoning"
    tool_name: Optional[str] = None
    tool_args_hint: Optional[str] = None


@dataclass
class PlanExecuteState:
    """执行过程中的累积状态（便于重规划与追踪）。"""

    plan: List[SubTask] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON 对象。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("无法从模型输出中解析 JSON")
    return json.loads(m.group(0))


def _parse_subtasks(data: Dict[str, Any]) -> List[SubTask]:
    raw_list = data.get("subtasks") or []
    out: List[SubTask] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        out.append(
            SubTask(
                id=str(item.get("id", f"task_{len(out)}")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                action_type=str(item.get("action_type", "reasoning")).lower(),
                tool_name=item.get("tool_name"),
                tool_args_hint=item.get("tool_args_hint"),
            )
        )
    return out


class PlannerAgent:
    """规划与执行：生成计划、逐步执行、失败时重规划。"""

    def __init__(
        self,
        llm: LLMCallable,
        tools: ToolInvoker,
        memory: Optional[MemoryLike],
        max_replan_attempts: int = 2,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self.max_replan_attempts = max(0, max_replan_attempts)

    async def plan(self, query: str) -> List[SubTask]:
        """根据用户目标生成子任务列表。"""
        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": f"用户目标：\n{query}\n\n请输出 JSON 计划。"},
        ]
        try:
            raw = await self._llm.acomplete(messages, temperature=0.3)
            data = _extract_json_object(raw)
            return _parse_subtasks(data)
        except Exception as e:  # noqa: BLE001
            logger.exception("生成计划失败")
            # 降级：单步推理任务
            return [
                SubTask(
                    id="fallback_1",
                    title="直接回答",
                    description=query,
                    action_type="reasoning",
                )
            ]

    async def execute(
        self,
        plan: List[SubTask],
        query: str,
        session_id: str,
        tool_names: Optional[Sequence[str]] = None,
        trace_callback: Optional[Any] = None,
    ) -> AgentResult:
        """
        按顺序执行子任务；tool 类型调用工具，reasoning 类型用 LLM 综合上下文。
        """
        allowed = set(tool_names) if tool_names else None
        results: List[Dict[str, Any]] = []
        state = PlanExecuteState(plan=list(plan), results=[])

        for idx, task in enumerate(plan):
            rec: Dict[str, Any] = {
                "subtask_id": task.id,
                "title": task.title,
                "action_type": task.action_type,
            }
            try:
                if task.action_type == "tool" and task.tool_name:
                    if allowed is not None and task.tool_name not in allowed:
                        raise RuntimeError(f"工具 {task.tool_name} 不在允许列表中")
                    # 将 hint 作为简单参数传入（企业场景可换为结构化解析）
                    args = {"hint": task.tool_args_hint or "", "user_query": query}
                    obs = await self._tools.invoke(task.tool_name, args)
                    rec["observation"] = obs[:8000]
                    rec["status"] = "ok"
                else:
                    # 推理子任务：用 LLM 汇总已有结果
                    ctx = json.dumps(results, ensure_ascii=False, indent=2)[:12000]
                    msgs: Sequence[Dict[str, str]] = [
                        {
                            "role": "system",
                            "content": "你是执行专家。根据已有子任务结果，完成当前子任务描述，输出简洁结论。",
                        },
                        {
                            "role": "user",
                            "content": f"原始问题：{query}\n当前子任务：{task.title}\n详情：{task.description}\n已有结果：\n{ctx}",
                        },
                    ]
                    text = await self._llm.acomplete(msgs, temperature=0.3)
                    rec["llm_output"] = text[:8000]
                    rec["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                logger.exception("子任务执行失败: %s", task.id)
                rec["status"] = "error"
                rec["error"] = str(e)
                results.append(rec)
                state.results = results
                if trace_callback:
                    await trace_callback({"phase": "execute", "record": rec})
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=results,
                    error=str(e),
                )

            results.append(rec)
            state.results = results
            if trace_callback:
                await trace_callback({"phase": "execute", "record": rec})

        # 最终汇总答案
        try:
            summary_msgs: Sequence[Dict[str, str]] = [
                {
                    "role": "system",
                    "content": "你是总结助手。根据子任务执行记录，给出面向用户的完整最终答案。",
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n执行记录：\n{json.dumps(results, ensure_ascii=False, indent=2)[:14000]}",
                },
            ]
            final = await self._llm.acomplete(summary_msgs, temperature=0.2)
        except Exception as e:  # noqa: BLE001
            logger.exception("最终汇总失败")
            return AgentResult(
                success=False,
                final_answer="",
                steps=results,
                error=f"汇总阶段失败: {e}",
            )

        if self._memory is not None:
            try:
                await self._memory.append_turn(
                    session_id,
                    "assistant",
                    final,
                    metadata={"agent": "planner", "subtasks": len(plan)},
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("写入记忆失败: %s", e)

        return AgentResult(success=True, final_answer=final.strip(), steps=results)

    async def replan(
        self,
        plan: List[SubTask],
        results: List[Dict[str, Any]],
        error: Optional[str],
    ) -> List[SubTask]:
        """根据执行结果与错误重新生成子任务列表。"""
        payload = {
            "previous_plan": [task.__dict__ for task in plan],
            "results_so_far": results,
            "error": error,
        }
        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "上下文：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)[:16000]
                + "\n\n请输出修订后的 JSON 计划。",
            },
        ]
        try:
            raw = await self._llm.acomplete(messages, temperature=0.3)
            data = _extract_json_object(raw)
            return _parse_subtasks(data)
        except Exception as e:  # noqa: BLE001
            logger.exception("重规划失败")
            return [
                SubTask(
                    id="replan_fallback",
                    title="降级为单步推理",
                    description="基于已有结果直接整合",
                    action_type="reasoning",
                )
            ]

    async def run_with_replan(
        self,
        query: str,
        session_id: str,
        tool_names: Optional[Sequence[str]] = None,
        trace_callback: Optional[Any] = None,
    ) -> AgentResult:
        """
        高层封装：计划 → 执行；失败则重规划并最多重试 max_replan_attempts 次。
        """
        current_plan = await self.plan(query)
        last_error: Optional[str] = None
        aggregate_results: List[Dict[str, Any]] = []

        for attempt in range(self.max_replan_attempts + 1):
            res = await self.execute(
                current_plan,
                query,
                session_id,
                tool_names=tool_names,
                trace_callback=trace_callback,
            )
            if res.success:
                return res

            last_error = res.error
            aggregate_results.extend(res.steps)
            if attempt >= self.max_replan_attempts:
                break

            current_plan = await self.replan(current_plan, res.steps, last_error)
            if not current_plan:
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=aggregate_results,
                    error=last_error or "重规划后无子任务",
                )
            if trace_callback:
                await trace_callback(
                    {"phase": "replan", "attempt": attempt + 1, "new_plan": [t.id for t in current_plan]}
                )

        return AgentResult(
            success=False,
            final_answer="",
            steps=aggregate_results,
            error=last_error or "执行失败",
        )
