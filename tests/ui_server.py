"""只监听本机的浏览器验收服务；使用明确的确定性测试模型。"""

import os
from pathlib import Path

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel

app = create_app(
    HarnessSettings(data_dir=Path(os.getenv("AEGIS_UI_TEST_DATA", "data/ui-test"))), ToolModel()
)
