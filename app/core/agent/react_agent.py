# -*- coding: utf-8 -*-
"""
ReAct Agent：Thought → Action → Observation 循环。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ReAct 提示模板（中文说明 + 结构化输出约束）
# ---------------------------------------------------------------------------
REACT_SYSTEM_PROMPT = """你是一个严谨的智能助手，必须使用 ReAct（推理+行动）方式回答问题。

## 输出格式（严格遵守，每一步只输出一块内容）

### 若需要调用工具
先写思考，再写动作：
Thought: <用中文简要说明你为什么需要下一步、打算做什么>
Action: <工具名称，必须是可用工具列表中之一>
Action Input: <JSON 对象，工具的参数>

### 若已有足够信息可直接作答
Thought: <简要总结依据>
Final Answer: <面向用户的完整最终答案，使用用户使用的语言>

## 规则
- 不要编造工具名称或 Observation；Observation 由系统在你输出 Action 后自动追加。
- Action Input 必须是合法 JSON。
- 若某工具返回错误，在下一步 Thought 中分析并决定是否换工具或向用户说明限制。
"""


def build_react_user_prompt(
    query: str,
    tool_descriptions: str,
    history_block: str,
) -> str:
    """构造单轮用户侧提示（含工具说明与历史轨迹）。"""
    return f"""## 用户问题
{query}

## 可用工具
{tool_descriptions}

## 已执行的步骤与观察（如有）
{history_block}

请根据当前信息，输出下一步：要么 Action + Action Input，要么 Final Answer。"""


class LLMCallable(Protocol):
    """可被 ReAct 调用的最小 LLM 接口。"""

    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        """返回模型生成的文本。"""
        ...


class MemoryLike(Protocol):
    """记忆系统最小接口。"""

    async def get_relevant(self, session_id: str, query: str, limit: int = 8) -> List[str]:
        ...

    async def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...


class ToolInvoker(Protocol):
    """工具调用：按名称执行并返回字符串化观察结果。"""

    async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
        ...


@dataclass
class AgentResult:
    """ReAct 单次运行的结果。"""

    success: bool
    final_answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    trace_id: Optional[str] = None


def _parse_react_step(text: str) -> Dict[str, Any]:
    """
    从模型输出解析 Thought / Action / Action Input / Final Answer。
    解析失败时返回原始片段，由上层决定是否重试或报错。
    """
    out: Dict[str, Any] = {"raw": text.strip()}
    thought_m = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|\Z)", text, re.S | re.I)
    if thought_m:
        out["thought"] = thought_m.group(1).strip()

    if re.search(r"Final Answer:\s*", text, re.I):
        fa_m = re.search(r"Final Answer:\s*(.+)\Z", text, re.S | re.I)
        if fa_m:
            out["final_answer"] = fa_m.group(1).strip()
            out["done"] = True
        return out

    action_m = re.search(r"Action:\s*(\S+)", text, re.I)
    input_m = re.search(r"Action Input:\s*(\{[\s\S]*\})", text)
    if action_m:
        out["action"] = action_m.group(1).strip()
    if input_m:
        try:
            out["action_input"] = json.loads(input_m.group(1))
        except json.JSONDecodeError:
            out["action_input"] = {}
            out["parse_error"] = "Action Input 不是合法 JSON"
    else:
        # 尝试宽松匹配非 JSON 块后的 JSON
        loose = re.search(r"Action Input:\s*([\s\S]+?)(?=\n\n|\Z)", text, re.I)
        if loose:
            raw_json = loose.group(1).strip()
            try:
                out["action_input"] = json.loads(raw_json)
            except json.JSONDecodeError:
                out["action_input"] = {}
                out["parse_error"] = "无法解析 Action Input"

    return out


class ReActAgent:
    """ReAct 循环：思考、行动、观察，直到最终答案或达到最大步数。"""

    def __init__(
        self,
        llm: LLMCallable,
        tools: ToolInvoker,
        memory: Optional[MemoryLike],
        max_steps: int = 10,
        session_id: Optional[str] = None,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._memory = memory
        self.max_steps = max(1, max_steps)
        self.session_id = session_id or "default"

    def _tool_catalog_text(self, tool_names: Sequence[str]) -> str:
        """工具列表描述（若注册表无描述则仅列名）。"""
        lines = []
        for name in tool_names:
            lines.append(f"- {name}")
        return "\n".join(lines) if lines else "（无外部工具，请直接 Final Answer）"

    async def run(self, query: str, context: Dict[str, Any]) -> AgentResult:
        """
        执行 ReAct 循环。

        context 可包含：
        - tool_names: 允许使用的工具名列表
        - session_id: 会话 ID（覆盖实例默认值）
        - trace_callback: 可选 async 回调(step_dict) 用于外部追踪
        """
        session_id = str(context.get("session_id", self.session_id))
        trace_cb = context.get("trace_callback")
        tool_names: List[str] = list(context.get("tool_names") or [])
        extra_system = str(context.get("extra_system", ""))

        steps: List[Dict[str, Any]] = []
        history_lines: List[str] = []

        mem_snippets: List[str] = []
        if self._memory is not None:
            try:
                mem_snippets = await self._memory.get_relevant(session_id, query, limit=8)
            except Exception as e:  # noqa: BLE001
                logger.warning("读取记忆失败，将继续无记忆上下文: %s", e)

        mem_block = "\n".join(f"- {s}" for s in mem_snippets) if mem_snippets else "（无）"

        for step_idx in range(self.max_steps):
            tool_desc = self._tool_catalog_text(tool_names)
            history_block = "\n".join(history_lines) if history_lines else "（尚无）"
            user_prompt = build_react_user_prompt(query, tool_desc, history_block)
            messages: List[Dict[str, str]] = [
                {
                    "role": "system",
                    "content": REACT_SYSTEM_PROMPT
                    + ("\n\n" + extra_system if extra_system else "")
                    + f"\n\n## 检索记忆摘要\n{mem_block}",
                },
                {"role": "user", "content": user_prompt},
            ]

            try:
                raw = await self._llm.acomplete(messages, temperature=0.2)
            except Exception as e:  # noqa: BLE001
                err = f"LLM 调用失败: {e}"
                logger.exception(err)
                rec = {"step": step_idx, "phase": "llm", "error": err}
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=steps,
                    error=err,
                )

            parsed = _parse_react_step(raw)
            rec: Dict[str, Any] = {
                "step": step_idx,
                "phase": "react",
                "raw_llm": raw[:4000],
                "parsed": {k: v for k, v in parsed.items() if k != "raw"},
            }

            if parsed.get("done") and parsed.get("final_answer"):
                answer = str(parsed["final_answer"])
                rec["final"] = True
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                if self._memory is not None:
                    try:
                        await self._memory.append_turn(
                            session_id,
                            "assistant",
                            answer,
                            metadata={"agent": "react", "steps": len(steps)},
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("写入记忆失败: %s", e)
                return AgentResult(success=True, final_answer=answer, steps=steps)

            action = parsed.get("action")
            action_input = parsed.get("action_input")
            if not action:
                # 无法继续：模型未给出有效动作或答案
                msg = parsed.get("parse_error") or "未解析到 Action 或 Final Answer"
                rec["error"] = msg
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                return AgentResult(
                    success=False,
                    final_answer="",
                    steps=steps,
                    error=msg,
                )

            if tool_names and action not in tool_names:
                obs = f"错误：工具 [{action}] 不在允许列表中。"
            else:
                try:
                    obs = await self._tools.invoke(action, action_input or {})
                except Exception as e:  # noqa: BLE001
                    obs = f"工具执行异常: {e}"
                    logger.exception("工具调用失败")

            rec["action"] = action
            rec["action_input"] = action_input
            rec["observation"] = obs[:8000]
            steps.append(rec)
            if trace_cb:
                await trace_cb(rec)

            history_lines.append(
                f"Step {step_idx + 1}\nThought: {parsed.get('thought', '')}\n"
                f"Action: {action}\nObservation: {obs}\n"
            )

        return AgentResult(
            success=False,
            final_answer="",
            steps=steps,
            error=f"已达到最大步数限制 ({self.max_steps})，未得到 Final Answer。",
        )
