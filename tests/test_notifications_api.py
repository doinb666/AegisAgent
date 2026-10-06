"""通知HTTP边界：身份、分页、读状态和任务生命周期。"""

import httpx
import pytest

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel, account, wait_run


@pytest.mark.asyncio
async def test_notification_http_completion_and_read(tmp_path):
    app = create_app(
        HarnessSettings(data_dir=tmp_path, database_url="", evolution_enabled=False), ToolModel()
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "notice-http-owner")
        assert (await client.get("/api/v1/notifications")).status_code == 401
        assert (await client.get("/api/v1/capabilities", headers=headers)).json()["notifications"][
            "enabled"
        ]
        response = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "notification"},
            json={"message": "计算"},
        )
        await wait_run(client, headers, response.json()["id"], {"completed"})
        notices = (await client.get("/api/v1/notifications", headers=headers)).json()
        assert len(notices) == 1 and notices[0]["type"] == "completed"
        assert (await client.get("/api/v1/notifications/unread", headers=headers)).json()[
            "count"
        ] == 1
        outsider = await account(client, "notice-http-other")
        endpoint = f"/api/v1/notifications/{notices[0]['id']}/read"
        assert (await client.post(endpoint, headers=outsider)).status_code == 404
        assert (await client.post(endpoint, headers=headers)).status_code == 200
        assert (await client.get("/api/v1/notifications/unread", headers=headers)).json()[
            "count"
        ] == 0
        for query in ("limit=0", "limit=51", "before=-1", "before=wrong"):
            assert (
                await client.get("/api/v1/notifications?" + query, headers=headers)
            ).status_code == 422
