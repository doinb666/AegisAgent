"""只监听本机的浏览器验收服务；使用明确的确定性测试模型。"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

from app.harness.settings import HarnessSettings
from app.infrastructure.llm.model_parameters import validate_model_selection
from app.main import create_app
from tests.test_harness_api import ToolModel


class AcceptanceModel(ToolModel):
    supports_incremental = True

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
    HarnessSettings(data_dir=Path(os.getenv("AEGIS_UI_TEST_DATA", "data/ui-test"))),
    AcceptanceModel(),
)
