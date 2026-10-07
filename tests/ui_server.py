"""只监听本机的浏览器验收服务；使用明确的确定性测试模型。"""

import os
from pathlib import Path

from app.harness.settings import HarnessSettings
from app.infrastructure.llm.model_parameters import validate_model_selection
from app.main import create_app
from tests.test_harness_api import ToolModel


class AcceptanceModel(ToolModel):
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
        return await super().chat(messages, **kwargs)


app = create_app(
    HarnessSettings(data_dir=Path(os.getenv("AEGIS_UI_TEST_DATA", "data/ui-test"))),
    AcceptanceModel(),
)
