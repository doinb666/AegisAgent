"""任务参数声明、实际协议映射与不兼容路由隔离。"""

import json

import httpx
import pytest

from app.infrastructure.llm.circuit_breaker import CircuitState
from app.infrastructure.llm.model_config import parse_entries
from tests.test_model_protocols import configured, mock_transport


@pytest.mark.parametrize(
    "parameters",
    [
        {"temperature": 1},
        {"max_output_tokens": True},
        {"max_output_tokens": 0},
        {"max_output_tokens": 32769},
        {"output_token_parameter": "tokens"},
        {"reasoning_efforts": [""]},
        {"reasoning_efforts": ["low"] * 9},
    ],
)
def test_parameter_declarations_are_strict(parameters):
    with pytest.raises(ValueError, match="AEGIS_MODELS_JSON"):
        parse_entries(json.dumps([{"model": "test", "parameters": parameters}]))


def test_parameter_contract_public_metadata_and_anthropic_restrictions(monkeypatch):
    router = configured(monkeypatch, parameters={"temperature": True})
    assert router.public_routes()[0]["parameters"] == {
        "temperature": True,
        "output_token_parameter": None,
        "max_output_tokens": 4096,
        "reasoning_efforts": [],
    }
    assert "protocol-test-key" not in json.dumps(router.public_routes())
    for parameters in (
        {"reasoning_efforts": ["low"]},
        {"output_token_parameter": "max_completion_tokens"},
    ):
        with pytest.raises(ValueError):
            parse_entries(
                json.dumps([{"model": "test", "provider": "anthropic", "parameters": parameters}])
            )


@pytest.mark.parametrize(
    "extra",
    [
        {"max_tokens": 1, "max_completion_tokens": 1},
        {"extra_body": {"model_parameters": {"temperature": 1}}},
        {"extra_body": {"max_output_tokens": 1}},
    ],
)
def test_admin_extra_cannot_conflict_with_dynamic_fields(extra):
    from app.infrastructure.llm.model_router import ModelConfig

    with pytest.raises(ValueError):
        parse_entries(json.dumps([{"model": "test", "extra": extra}]))
    with pytest.raises(ValueError):
        ModelConfig("test", "secret-not-logged", extra=extra)


def test_direct_model_config_keeps_anthropic_protocol_restrictions():
    from app.infrastructure.llm.model_router import ModelConfig
    from app.infrastructure.llm.types import ModelProvider

    with pytest.raises(ValueError):
        ModelConfig(
            "test",
            "secret-not-logged",
            provider=ModelProvider.ANTHROPIC,
            extra={"extra_body": {"reasoning_effort": "high"}},
        )


@pytest.mark.asyncio
async def test_fallback_maps_token_fields_and_keeps_legacy_default(monkeypatch):
    from app.harness.settings import HarnessSettings
    from app.infrastructure.llm.model_router import build_harness_router

    requests = []

    def respond(request):
        requests.append((request.url.host, json.loads(request.content)))
        if request.url.host == "primary.invalid":
            return httpx.Response(503, json={"error": {"message": "测试故障"}})
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "备用"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_transport(monkeypatch, respond)
    monkeypatch.setenv("TEST_MODEL_KEY", "test-secret")
    entries = [
        {
            "model": "test",
            "api_key_env": "TEST_MODEL_KEY",
            "base_url": f"https://{host}/v1",
            "priority": priority,
            "parameters": {"output_token_parameter": field},
        }
        for priority, host, field in [
            (0, "primary.invalid", "max_completion_tokens"),
            (1, "backup.invalid", "max_tokens"),
        ]
    ]
    router = build_harness_router(HarnessSettings(models_json=json.dumps(entries)))
    try:
        await router.chat(
            [{"role": "user", "content": "测试"}], model_parameters={"max_output_tokens": 128}
        )
        assert requests[0][1]["max_completion_tokens"] == 128
        assert "max_tokens" not in requests[0][1]
        assert requests[1][1]["max_tokens"] == 128
        assert "max_completion_tokens" not in requests[1][1]
        requests.clear()
        await router.chat([{"role": "user", "content": "旧调用"}])
        assert requests[-1][1]["temperature"] == 0.7
        assert "model_parameters" not in requests[-1][1]
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_dynamic_parameters_map_at_http_boundary(monkeypatch):
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "回答"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_transport(monkeypatch, respond)
    router = configured(
        monkeypatch,
        parameters={
            "output_token_parameter": "max_completion_tokens",
            "max_output_tokens": 512,
            "reasoning_efforts": ["medium"],
        },
        extra={"max_tokens": 1000, "temperature": 0.4, "extra_body": {"reasoning_effort": "high"}},
    )
    try:
        await router.chat(
            [{"role": "user", "content": "测试"}],
            model_parameters={
                "max_output_tokens": 128,
                "reasoning_effort": "medium",
            },
        )
        assert bodies[0]["max_completion_tokens"] == 128
        assert bodies[0]["reasoning_effort"] == "medium"
        assert not {"max_tokens", "model_parameters", "temperature"}.intersection(bodies[0])
        await router.chat([{"role": "user", "content": "旧请求"}])
        assert bodies[1]["reasoning_effort"] == "high"
        assert bodies[1]["temperature"] == 0.4
        assert router._configs[0].extra["extra_body"] == {"reasoning_effort": "high"}
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_incompatible_candidates_are_skipped_without_breaker_failure(monkeypatch):
    from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
    from tests.test_model_router import Client

    clients = []

    def factory(**kwargs):
        client = Client(fail="primary" in kwargs["base_url"])
        clients.append(client)
        return client

    monkeypatch.setattr("app.infrastructure.llm.model_router.AsyncOpenAI", factory)
    router = ModelRouter(
        [
            ModelConfig(
                "primary",
                "test",
                "https://primary.invalid",
                priority=0,
                parameters={"temperature": True},
            ),
            ModelConfig("skip", "test", "https://skip.invalid", priority=1),
            ModelConfig(
                "backup",
                "test",
                "https://backup.invalid",
                priority=2,
                parameters={"temperature": True},
            ),
        ],
        failure_threshold=1,
    )
    try:
        await router.chat(
            [{"role": "user", "content": "测试"}], model_parameters={"temperature": 0}
        )
        assert [client.calls for client in clients] == [1, 0, 1]
        assert [breaker.state for breaker in router._breakers.values()] == [
            CircuitState.OPEN,
            CircuitState.CLOSED,
            CircuitState.CLOSED,
        ]
    finally:
        await router.aclose()
