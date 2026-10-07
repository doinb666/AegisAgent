"""计划 HTTP 严格请求边界、只读工具内核执行与私有隔离。"""

from datetime import UTC, datetime

import httpx
import pytest

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel, account, wait_run


@pytest.mark.asyncio
async def test_schedule_http_validation_private_scope_and_readonly_execution(tmp_path):
    app = create_app(
        HarnessSettings(data_dir=tmp_path, database_url="", evolution_enabled=False), ToolModel()
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        # HTTP 任务由受控时钟单轮触发；后台轮询生命周期另有真实 loop 专项验证。
        await app.state.harness.schedules.close()
        owner = await account(client, "schedule-http-owner")
        other = await account(client, "schedule-http-other")
        now = datetime.now(UTC).timestamp()
        body = {
            "title": "计算",
            "message": "计算2+3",
            "scheduled_at": datetime.fromtimestamp(now + 300, UTC).isoformat(),
        }
        assert (await client.get("/api/v1/schedules")).status_code == 401
        assert (await client.get("/api/v1/capabilities", headers=owner)).json()["schedules"][
            "enabled"
        ]
        for invalid in (
            {"scheduled_at": "2026-10-01T12:00:00"},
            {"scheduled_at": 12345},
            {"scheduled_at": "12345"},
            {"scheduled_at": "9999-12-31T23:59:59.999999+00:00"},
            {"scheduled_at": "0001-01-01T00:00:00+14:00"},
            {"title": " "},
            {"message": "\n"},
            {"interval_seconds": 59},
            {"interval_seconds": 31 * 86400 + 1},
            {"interval_seconds": True},
            {"allowed_tools": ["file_write"]},
            {"repository_root": "C:/secret"},
        ):
            result = await client.post(
                "/api/v1/schedules",
                headers={**owner, "Idempotency-Key": "invalid"},
                json={**body, **invalid},
            )
            assert result.status_code == 422, result.text
        response = await client.post(
            "/api/v1/schedules", headers={**owner, "Idempotency-Key": "http-schedule"}, json=body
        )
        assert response.status_code == 201
        schedule = response.json()
        assert (
            await client.get(f"/api/v1/schedules/{schedule['id']}", headers=other)
        ).status_code == 404
        assert (await client.get("/api/v1/schedules", headers=other)).json() == []
        for query in ("limit=0", "limit=51", "before=", "status=wrong"):
            assert (
                await client.get("/api/v1/schedules?" + query, headers=owner)
            ).status_code == 422
        member = await client.post(
            "/api/v1/auth/members",
            headers=owner,
            json={"username": "schedule-viewer", "password": "password123", "role": "viewer"},
        )
        assert member.status_code == 201
        login = await client.post(
            "/api/v1/auth/login", json={"username": "schedule-viewer", "password": "password123"}
        )
        viewer = {"Authorization": "Bearer " + login.json()["token"]}
        assert (await client.get("/api/v1/schedules", headers=viewer)).status_code == 200
        assert (
            await client.post(
                "/api/v1/schedules", headers={**viewer, "Idempotency-Key": "viewer"}, json=body
            )
        ).status_code == 403
        app.state.harness.schedules.clock = lambda: now + 300
        await app.state.harness.schedules.tick()
        detail = (await client.get(f"/api/v1/schedules/{schedule['id']}", headers=owner)).json()
        assert detail["occurrences"][0]["run_id"], detail
        run = await wait_run(client, owner, detail["occurrences"][0]["run_id"], {"completed"})
        assert run["answer"] == "计算结果为5"
        # 确定性模型恶意请求非白名单工具，内核拒绝而不是自动审批/执行。
        rejected = await client.post(
            "/api/v1/schedules",
            headers={**owner, "Idempotency-Key": "forbidden"},
            json={**body, "message": "写文件"},
        )
        await app.state.harness.schedules.tick()
        rejected_detail = (
            await client.get(f"/api/v1/schedules/{rejected.json()['id']}", headers=owner)
        ).json()
        failed = await wait_run(
            client, owner, rejected_detail["occurrences"][0]["run_id"], {"failed"}
        )
        assert "允许清单" in failed["error"]
