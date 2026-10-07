"""只监听本机的浏览器验收服务；使用明确的确定性测试模型。"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from app.harness.context import PLAN_PROMPT
from app.harness.settings import HarnessSettings
from app.infrastructure.llm.model_parameters import validate_model_selection
from app.main import create_app
from tests.test_harness_api import ToolModel


def plan_response(content="", calls=None):
    """统一计划夹具的公开响应，保留既有测试模型协议。"""
    raw = {"choices": [{"message": {"tool_calls": calls}}]} if calls is not None else {}
    return SimpleNamespace(content=content, model_id="test", usage={}, raw=raw)


class AcceptanceModel(ToolModel):
    supports_incremental = True

    async def plan_fixture(self, messages, kwargs):
        """仅对明确前缀提供计划夹具；不改变既有工具和风险审查。"""
        content = next(
            (item.get("content", "") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        original = next(
            (
                item.get("content", "")
                for item in messages
                if item.get("role") == "user" and item.get("content", "").startswith("验收：计划")
            ),
            "",
        )
        try:
            node = json.loads(content)
        except (ValueError, TypeError):
            node = None
        if isinstance(node, dict) and "当前节点" in node:
            original = node.get("原始目标", "")
        if not original.startswith("验收：计划"):
            return None
        if content.startswith("所有计划节点已完成结构验收。仅基于已知输出给出非空交付摘要"):
            return plan_response("计划最终交付，结构通过仍需核对结果正确性。")
        if PLAN_PROMPT in content:
            repair = "仅规划剩余后缀" in content
            if repair and "重规划" in original:
                await asyncio.sleep(1.5)
            if "回退" in original:
                answer = "无效计划"
            else:
                tools = (
                    ["calculator"]
                    if any(word in original for word in ("证据", "重规划", "失败"))
                    else []
                )
                if "拒绝" in original:
                    tools = ["file_write"]
                objectives = (
                    ["核对公开工具证据"]
                    if repair
                    else ["整理任务边界", "核对工具证据" if tools else "根据前序回答形成交付"]
                )
                if "恶意" in original:
                    objectives[0] = '<img src=x onerror="alert(1)"> 安全目标'
                if "预算" in original:
                    tools = ["calculator"]
                    objectives[-1] = "工具批次触及共享预算"
                steps = [
                    {
                        "objective": objective,
                        "inputs": {"instruction": objective, "from_steps": [i - 1] if i else []},
                        "acceptance": {
                            "output_kind": "tool_evidence"
                            if tools and (repair or i > 0)
                            else "answer",
                            "required_tools": tools if repair or i > 0 else [],
                        },
                    }
                    for i, objective in enumerate(objectives)
                ]
                answer = json.dumps({"steps": steps}, ensure_ascii=False)
            return plan_response(answer)
        if isinstance(node, dict) and "当前节点" in node:
            identifier = node["当前节点"]
            required = node["验收契约"]["required_tools"]
            if "慢任务" in original and identifier == "n0-2":
                await asyncio.sleep(8)
            current_tools = [item for item in messages if item.get("role") == "tool"]
            if required and (not current_tools or "预算" in original):
                if "重规划" in original and identifier == "n0-2":
                    return plan_response("暂缺必需工具证据")
                name = required[0]
                arguments = {"expression": "1/0" if "失败" in original else "2+3"}
                if name == "file_write":
                    arguments = {"path": "plan-result.txt", "content": "计划验收内容"}
                calls = [
                    {
                        "id": "plan-" + identifier + "-" + str(len(current_tools)),
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
                if "预算" in original:
                    calls = [
                        {
                            "id": "budget-" + str(i),
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"expression": f"2+{i}"}),
                            },
                        }
                        for i in range(16)
                    ]
                return plan_response(calls=calls)
            return plan_response(f"节点 {identifier} 的公开回答，正确性需核对。")
        # 风险审查是独立输入，不含原任务，仍走既有默认审查。
        if kwargs.get("tools"):
            return plan_response("计划最终交付，结构通过仍需核对结果正确性。")
        return None

    def public_routes(self):
        return [
            {
                "id": "acceptance-primary",
                "model": "acceptance-primary",
                "label": "验收主模型",
                "provider": "openai",
                "priority": 0,
                "parameters": {
                    "temperature": True,
                    "output_token_parameter": "max_completion_tokens",
                    "max_output_tokens": 2048,
                    "reasoning_efforts": ["low", "medium", "high"],
                },
            },
            {
                "id": "acceptance-backup",
                "model": "acceptance-backup",
                "label": "验收备用模型",
                "provider": "openai",
                "priority": 1,
                "parameters": {
                    "temperature": False,
                    "output_token_parameter": "max_tokens",
                    "max_output_tokens": 512,
                    "reasoning_efforts": ["medium"],
                },
            },
        ]

    async def chat(self, messages, **kwargs):
        validate_model_selection(
            self, kwargs.get("model_preference"), kwargs.get("model_parameters")
        )
        fixture = await self.plan_fixture(messages, kwargs)
        if fixture is not None:
            return fixture
        if any(
            item.get("role") == "user" and item.get("content") == "验收：模拟模型失败"
            for item in messages
        ):
            raise RuntimeError("确定性模型故障注入")
        task = next(
            (item.get("content", "") for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        on_delta = kwargs.get("on_delta")
        if (
            kwargs.get("tools")
            and task == "验收：差异长文写文件"
            and not any(item.get("role") == "tool" for item in messages)
        ):
            return SimpleNamespace(
                content="",
                model_id="test",
                usage={},
                raw={
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "call-file-large",
                                        "type": "function",
                                        "function": {
                                            "name": "file_write",
                                            "arguments": json.dumps(
                                                {
                                                    "path": "review.txt",
                                                    "content": '<img src=x onerror="alert(1)">\n'
                                                    + "边界说明\n" * 2000,
                                                }
                                            ),
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
            )
        if on_delta and task.startswith("验收：流式"):
            if task == "验收：流式工具" and any(item.get("role") == "tool" for item in messages):
                return await super().chat(messages, **kwargs)
            first = "第一段：正在核对公开证据（中文与🧪）。\n"
            await on_delta({"type": "content", "text": first})
            await asyncio.sleep(1.2)
            if task == "验收：流式中断":
                raise RuntimeError("确定性流中断")
            if task == "验收：流式慢任务":
                await asyncio.sleep(20)
            if task == "验收：流式工具":
                await on_delta({"type": "tools"})
                return await super().chat(messages, **kwargs)
            second = '第二段：保留来源与约束。<img src=x onerror="alert(1)">\n'
            await on_delta({"type": "content", "text": second})
            await asyncio.sleep(1.2)
            final = "第三段：完整答复由任务结束后保存。"
            await on_delta({"type": "content", "text": final})
            return SimpleNamespace(
                content=first + second + final, model_id="test", usage={}, raw={}
            )
        return await super().chat(messages, **kwargs)


app = create_app(
    HarnessSettings(
        data_dir=Path(os.getenv("AEGIS_UI_TEST_DATA", "data/ui-test")),
        **({"max_steps": 20, "max_model_calls": 20} if os.getenv("AEGIS_UI_PLAN") == "1" else {}),
    ),
    AcceptanceModel(),
)
