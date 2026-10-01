"""工作台资产编辑需保留元数据并拒绝覆盖已更新版本。"""

import httpx
import pytest

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel, account


@pytest.mark.asyncio
async def test_http_asset_edit_rejects_stale_version(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path, evolution_enabled=False), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "cas-editor")
        payload = {
            "kind": "skill",
            "name": "review-python",
            "content": "先审查",
            "status": "draft",
            "metadata": {"directory": "coding/review", "resources": {"notes.md": "检查清单"}},
        }
        created = await client.post("/api/v1/assets", headers=headers, json=payload)
        asset = created.json()
        updated = {**payload, "content": "先读取再审查", "expected_version": asset["version"]}
        endpoint = f"/api/v1/assets/{asset['id']}"
        first = await client.put(endpoint, headers=headers, json=updated)
        assert first.status_code == 200
        stale = await client.put(endpoint, headers=headers, json=updated)
        assert stale.status_code == 409
        current = (await client.get(endpoint, headers=headers)).json()
        assert current["metadata"] == payload["metadata"]
        assert current["version"] == asset["version"] + 1
