"""项目会话HTTP契约及严格参数校验。"""

import httpx
import pytest

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel, account, wait_run


@pytest.mark.asyncio
async def test_threads_http_lifecycle_and_validation(tmp_path):
    app = create_app(
        HarnessSettings(data_dir=tmp_path, database_url="", evolution_enabled=False), ToolModel()
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "thread-http-owner")
        assert (await client.get("/api/v1/capabilities", headers=headers)).json()["threads"][
            "enabled"
        ]
        assert (await client.get("/api/v1/threads")).status_code == 401
        body = {"title": "HTTP会话"}
        result = await client.post(
            "/api/v1/threads", json=body, headers={**headers, "Idempotency-Key": "thread-http"}
        )
        assert result.status_code == 201
        thread = result.json()
        replay = await client.post(
            "/api/v1/threads", json=body, headers={**headers, "Idempotency-Key": "thread-http"}
        )
        assert replay.json()["id"] == thread["id"]
        assert (
            await client.post(
                "/api/v1/threads",
                json={"title": "不同"},
                headers={**headers, "Idempotency-Key": "thread-http"},
            )
        ).status_code == 409
        response = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "thread-task"},
            json={"message": "计算", "thread_id": thread["id"]},
        )
        assert response.status_code == 202, response.text
        run = await wait_run(client, headers, response.json()["id"], {"completed"})
        assert run["thread_id"] == thread["id"]
        detail = (await client.get(f"/api/v1/threads/{thread['id']}", headers=headers)).json()
        assert detail["latest_run_id"] == run["id"]
        assert (
            await client.patch(
                f"/api/v1/threads/{thread['id']}",
                headers=headers,
                json={"archived": True, "title": "已整理"},
            )
        ).status_code == 200
        assert (await client.get("/api/v1/threads", headers=headers)).json() == []
        assert len((await client.get("/api/v1/threads?archived=true", headers=headers)).json()) == 1
        for payload in (
            {"title": " "},
            {"title": "x" * 201},
            {"title": "会话", "owner_id": "forged"},
        ):
            assert (
                await client.post("/api/v1/threads", headers=headers, json=payload)
            ).status_code == 422
        for payload in ({"archived": "true"}, {"title": ""}, {"tenant_id": "forged"}):
            assert (
                await client.patch(f"/api/v1/threads/{thread['id']}", headers=headers, json=payload)
            ).status_code == 422
        for endpoint in (
            "/threads?limit=0",
            "/threads?limit=51",
            "/threads?archived=wrong",
            "/workspace/projects?limit=51",
            "/threads?before=",
        ):
            assert (await client.get("/api/v1" + endpoint, headers=headers)).status_code == 422
        outsider = await account(client, "thread-http-outsider")
        assert (
            await client.get(f"/api/v1/threads/{thread['id']}", headers=outsider)
        ).status_code == 404
        assert (await client.get("/api/v1/threads?archived=true", headers=outsider)).json() == []
