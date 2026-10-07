"""真实 HTTP/SQLite 的任务参数、旧哈希及配置变更恢复。"""

import json

import httpx
import pytest

from app.harness.models import Run
from app.harness.security import canonical, digest
from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel, account, wait_run


class ParameterModel(ToolModel):
    def __init__(self):
        self.routes = [
            {
                "id": "route-test",
                "model": "test",
                "parameters": {
                    "temperature": True,
                    "output_token_parameter": "max_completion_tokens",
                    "max_output_tokens": 512,
                    "reasoning_efforts": ["medium"],
                },
            }
        ]
        self.calls = []

    def public_routes(self):
        return self.routes

    async def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        return await super().chat(messages, **kwargs)


@pytest.mark.asyncio
async def test_all_persisted_model_events_filter_untrusted_usage(tmp_path):
    class UntrustedUsageModel(ParameterModel):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            response.usage = {
                "prompt_tokens": True,
                "completion_tokens": -1,
                "total_tokens": 10**15,
                "cached_tokens": 7,
                "unexpected_usage_key": "不应持久化的供应商数据",
            }
            return response

    app = create_app(
        HarnessSettings(data_dir=tmp_path, evolution_enabled=False), UntrustedUsageModel()
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "usage-filter")
        created = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "usage-filter"},
            json={"message": "计算", "model_parameters": {"temperature": 0}},
        )
        assert created.status_code == 202
        run = await wait_run(client, headers, created.json()["id"], {"completed", "failed"})
        assert run["status"] == "completed"
        stream = await client.get(f"/api/v1/runs/{run['id']}/events", headers=headers)
        events = [
            json.loads(block.split("data: ", 1)[1])
            for block in stream.text.split("\n\n")
            if "event: model\n" in block or "event: model_usage\n" in block
        ]
        assert len(events) == 4
        assert all(event["usage"] == {"cached_tokens": 7} for event in events)
        assert "unexpected_usage_key" not in stream.text
        assert "不应持久化的供应商数据" not in stream.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["react", "plan", "reflection"])
async def test_run_parameters_persist_hash_and_recover_before_current_capabilities(tmp_path, mode):
    model = ParameterModel()
    # 本用例只统计任务调用；后台演进由独立专项覆盖。
    app = create_app(HarnessSettings(data_dir=tmp_path, evolution_enabled=False), model)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "parameters-owner")
        request_headers = {**headers, "Idempotency-Key": "configured"}
        body = {
            "message": "计算",
            "model": "route-test",
            "mode": mode,
            "model_parameters": {
                "temperature": 0,
                "max_output_tokens": 128,
                "reasoning_effort": "medium",
            },
        }
        result = await client.post("/api/v1/runs", headers=request_headers, json=body)
        assert result.status_code == 202, result.text
        run = await wait_run(client, headers, result.json()["id"], {"completed", "failed"})
        assert run["status"] == "completed"
        assert run["model_parameters"] == body["model_parameters"]
        assert all(call["model_parameters"] == body["model_parameters"] for call in model.calls)
        stream = await client.get(f"/api/v1/runs/{run['id']}/events", headers=headers)
        assert '"model_parameters": {"temperature": 0' in stream.text
        usage_events = [
            json.loads(block.split("data: ", 1)[1])
            for block in stream.text.split("\n\n")
            if "event: model_usage\n" in block
        ]
        assert len(usage_events) == len(model.calls)
        assert all(event["model_parameters"] == body["model_parameters"] for event in usage_events)
        assert len(model.calls) == (2 if mode == "react" else 3)
        assert (
            await client.post(
                "/api/v1/runs",
                headers=request_headers,
                json={**body, "model_parameters": {"temperature": 1}},
            )
        ).status_code == 409
        model.routes = [{"id": "route-other", "model": "other"}]
        assert (await client.post("/api/v1/runs", headers=request_headers, json=body)).json()[
            "id"
        ] == run["id"]
        assert (
            await client.post(
                "/api/v1/runs", headers={**headers, "Idempotency-Key": "new"}, json=body
            )
        ).status_code == 422
        old_headers = {**headers, "Idempotency-Key": "old"}
        old = await client.post("/api/v1/runs", headers=old_headers, json={"message": "计算"})
        async with app.state.harness.store.sessions() as session:
            persisted = await session.get(Run, old.json()["id"])
            assert persisted.payload_hash == digest(
                canonical(
                    dict(
                        message="计算",
                        session_id=None,
                        mode="react",
                        model=None,
                        parent_run_id=None,
                        allowed_tools=None,
                        max_steps=None,
                    )
                )
            )
        for empty in (None, {}):
            restored = await client.post(
                "/api/v1/runs",
                headers=old_headers,
                json={"message": "计算", "model_parameters": empty},
            )
            assert restored.json()["id"] == old.json()["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parameters",
    [
        {"temperature": True},
        {"temperature": "0"},
        {"temperature": 3},
        {"temperature": 10**400},
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"max_output_tokens": 513},
        {"reasoning_effort": "high"},
        {"unknown": 1},
        {"temperature": None},
    ],
)
async def test_invalid_task_parameters_rejected_before_creation(tmp_path, parameters):
    app = create_app(HarnessSettings(data_dir=tmp_path), ParameterModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "invalid-parameters")
        result = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "invalid"},
            json={"message": "计算", "model_parameters": parameters},
        )
        assert result.status_code == 422, result.text
        assert (await client.get("/api/v1/runs", headers=headers)).json() == []


@pytest.mark.asyncio
async def test_nan_and_undeclared_parameters_are_rejected(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = {
            **await account(client, "undeclared-parameters"),
            "Idempotency-Key": "invalid",
            "Content-Type": "application/json",
        }
        for content in (
            '{"message":"计算","model_parameters":{"temperature":NaN}}',
            '{"message":"计算","model_parameters":{"temperature":0}}',
        ):
            result = await client.post("/api/v1/runs", headers=headers, content=content.encode())
            assert result.status_code == 422, result.text


@pytest.mark.asyncio
async def test_acceptance_common_capabilities(tmp_path):
    from tests.ui_server import AcceptanceModel

    app = create_app(HarnessSettings(data_dir=tmp_path), AcceptanceModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "common-capabilities")
        cap = (await client.get("/api/v1/capabilities", headers=headers)).json()
        assert cap["model_parameters"] == {
            "temperature": False,
            "output_tokens": True,
            "max_output_tokens": 512,
            "reasoning_efforts": ["medium"],
        }


@pytest.mark.asyncio
async def test_unconfigured_model_rejected_and_temperature_numeric_form_is_idempotent(tmp_path):
    model = ParameterModel()
    app = create_app(HarnessSettings(data_dir=tmp_path), model)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = {**await account(client, "numeric-form"), "Idempotency-Key": "zero"}
        integer = await client.post(
            "/api/v1/runs",
            headers=headers,
            json={"message": "计算", "model_parameters": {"temperature": 0}},
        )
        floating = await client.post(
            "/api/v1/runs",
            headers=headers,
            json={"message": "计算", "model_parameters": {"temperature": 0.0}},
        )
        assert integer.status_code == floating.status_code == 202
        assert integer.json()["id"] == floating.json()["id"]
        model.routes = []
        unknown = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "unknown"},
            json={"message": "计算", "model": "missing"},
        )
        assert unknown.status_code == 422
        app.state.harness.model_router = None
        unconfigured = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "unconfigured"},
            json={"message": "计算", "model": "missing"},
        )
        assert unconfigured.status_code == 422
