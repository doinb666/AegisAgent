"""只监听本机的浏览器验收服务；使用明确的确定性测试模型。"""

import os
from pathlib import Path

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel


class AcceptanceModel(ToolModel):
    async def chat(self, messages, **kwargs):
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
