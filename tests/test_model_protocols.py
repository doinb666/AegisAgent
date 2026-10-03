"""使用真实 HTTP 传输替身验收协议适配，不调用付费模型。"""

import asyncio
import json
import traceback

import httpx
import httpx2
import pytest

from app.harness.settings import HarnessSettings
from app.infrastructure.llm.model_router import build_harness_router


def configured(monkeypatch, **overrides):
    monkeypatch.setenv("TEST_MODEL_KEY", "protocol-test-key")
    entry = {"model": "test-model", "api_key_env": "TEST_MODEL_KEY", **overrides}
    return build_harness_router(HarnessSettings(models_json=json.dumps([entry])))


def mock_transport(monkeypatch, respond):
    # 在 HTTP 层替换传输，保留 SDK 客户端类型与序列化行为。
    for module in (httpx, httpx2):
        original = module.AsyncClient.__init__

        def initialize(self, *args, _original=original, _module=module, **kwargs):
            def handle(request):
                response = respond(request)
                return _module.Response(
                    response.status_code, content=response.content, headers=dict(response.headers)
                )

            kwargs["transport"] = _module.MockTransport(handle)
            _original(self, *args, **kwargs)

        monkeypatch.setattr(module.AsyncClient, "__init__", initialize)


@pytest.mark.asyncio
async def test_anthropic_tool_request_result_and_normalized_response(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [
                    {"type": "text", "text": "检查文件"},
                    {
                        "type": "tool_use",
                        "id": "call_2",
                        "name": "file_read",
                        "input": {"path": "说明.md"},
                    },
                ],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 10, "cache_read_input_tokens": 5},
            },
        )

    mock_transport(monkeypatch, respond)
    router = configured(monkeypatch, provider="anthropic", base_url="https://anthropic.invalid/v1")
    try:
        result = await router.chat(
            [
                {"role": "system", "content": "只读分析"},
                {"role": "user", "content": "检查"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression":"2+3"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"value":5}'},
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "description": "读文件",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="auto",
            max_tokens=100,
        )
        request = requests[0]
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "protocol-test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["system"] == "只读分析" and body["max_tokens"] == 100
        assert body["tools"][0]["input_schema"] == {"type": "object"}
        assert body["messages"][-1] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"value":5}'}],
        }
        assert body["messages"][-2]["content"][0]["type"] == "tool_use"
        assert result.content == "检查文件"
        call = result.raw["choices"][0]["message"]["tool_calls"][0]
        assert call["id"] == "call_2" and json.loads(call["function"]["arguments"]) == {
            "path": "说明.md"
        }
        assert result.usage["cached_tokens"] == 5 and result.usage["total_tokens"] == 35
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_ollama_without_key_and_custom_parameters(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "local",
                "object": "chat.completion",
                "created": 0,
                "model": "local-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "本地回答"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_transport(monkeypatch, respond)
    router = build_harness_router(
        HarnessSettings(
            models_json=json.dumps(
                [
                    {
                        "model": "local-model",
                        "provider": "ollama",
                        "extra": {"temperature": 0.2, "extra_body": {"think": False}},
                    }
                ]
            )
        )
    )
    try:
        result = await router.chat([{"role": "user", "content": "你好"}])
        assert result.content == "本地回答"
        assert str(requests[0].url) == "http://127.0.0.1:11434/v1/chat/completions"
        body = json.loads(requests[0].content)
        assert body["temperature"] == 0.2 and body["think"] is False
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_azure_deployment_endpoint_and_version(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "azure",
                "object": "chat.completion",
                "created": 0,
                "model": "deployment",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "云端回答"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_transport(monkeypatch, respond)
    router = configured(
        monkeypatch,
        provider="azure",
        base_url="https://resource.openai.azure.com",
        api_version="2024-10-21",
    )
    try:
        assert (await router.chat([{"role": "user", "content": "测试"}])).content == "云端回答"
        assert requests[0].url.path == "/openai/deployments/test-model/chat/completions"
        assert requests[0].url.params["api-version"] == "2024-10-21"
        assert requests[0].headers["api-key"] == "protocol-test-key"
    finally:
        await router.aclose()


@pytest.mark.parametrize(
    "patch",
    [
        {"provider": "unknown"},
        {"base_url": "file:///etc/passwd"},
        {"base_url": "https://name:password@model.invalid/v1"},
        {"weight": 0},
        {"extra": {"messages": []}},
        {"extra": {"stream": True}},
        {"extra": {"extra_body": {"tools": []}}},
        {"api_key": "embedded-secret"},
        {"provider": "azure"},
        {"priority": True},
    ],
)
def test_invalid_configuration_rejected_before_network(monkeypatch, patch):
    with pytest.raises(ValueError):
        configured(monkeypatch, **patch)


@pytest.mark.asyncio
async def test_routes_select_same_model_from_distinct_sources(monkeypatch):
    router = build_harness_router(
        HarnessSettings(
            models_json=json.dumps(
                [
                    {
                        "model": "shared",
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "label": "本机",
                    },
                    {
                        "model": "shared",
                        "provider": "ollama",
                        "base_url": "http://127.0.0.1:11435/v1",
                        "label": "备用",
                        "priority": 1,
                    },
                ]
            )
        )
    )
    try:
        routes = router.public_routes()
        assert len(routes) == 2 and routes[0]["id"] != routes[1]["id"]
        assert "api_key" not in json.dumps(routes) and "11434" not in json.dumps(routes)
        assert router._select_candidates(routes[1]["id"])[0].base_url.endswith("11435/v1")
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_native_failure_falls_back_without_replaying_tool_results(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "primary.invalid":
            return httpx.Response(
                503, json={"error": {"type": "overloaded_error", "message": "服务繁忙"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "backup",
                "object": "chat.completion",
                "created": 0,
                "model": "shared",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "备用完成"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    mock_transport(monkeypatch, respond)
    monkeypatch.setenv("TEST_MODEL_KEY", "protocol-test-key")
    router = build_harness_router(
        HarnessSettings(
            models_json=json.dumps(
                [
                    {
                        "model": "shared",
                        "provider": "anthropic",
                        "base_url": "https://primary.invalid/v1",
                        "api_key_env": "TEST_MODEL_KEY",
                    },
                    {
                        "model": "shared",
                        "provider": "custom",
                        "base_url": "https://backup.invalid/v1",
                        "api_key_env": "TEST_MODEL_KEY",
                        "priority": 1,
                    },
                ]
            )
        )
    )
    messages = [
        {"role": "user", "content": "复核"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expression":"1+1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "one", "content": "2"},
    ]
    try:
        result = await router.chat(messages)
        assert result.content == "备用完成" and len(requests) == 2
        assert json.loads(requests[-1].content)["messages"] == messages
        assert messages[-1]["role"] == "tool"
    finally:
        await router.aclose()


@pytest.mark.parametrize(
    "value", ["{}", "null", "[null]", '[{"model":" "}]', '[{"model":"x","weight":"NaN"}]']
)
def test_malformed_entries_have_redacted_error(value):
    with pytest.raises(ValueError, match="AEGIS_MODELS_JSON 无效"):
        build_harness_router(HarnessSettings(models_json=value))


@pytest.mark.parametrize(
    "patch",
    [
        {"base_url": "https://model.invalid:bad/v1"},
        {"base_url": "https://model.invalid:99999/v1"},
        {"weight": True},
    ],
)
def test_invalid_port_and_boolean_weight_are_redacted(monkeypatch, patch):
    with pytest.raises(ValueError, match="AEGIS_MODELS_JSON 无效"):
        configured(monkeypatch, **patch)


@pytest.mark.asyncio
async def test_all_sources_failed_do_not_expose_private_endpoint(monkeypatch):
    mock_transport(
        monkeypatch, lambda request: httpx.Response(503, json={"error": "protocol-test-key"})
    )
    router = configured(monkeypatch, provider="anthropic", base_url="https://private.invalid/v1")
    try:
        with pytest.raises(RuntimeError) as error:
            await router.chat([{"role": "user", "content": "测试"}])
        assert error.value.__suppress_context__ and error.value.__cause__ is None
        rendered = "".join(traceback.format_exception(error.value))
        assert "private.invalid" not in rendered and "protocol-test-key" not in rendered
    finally:
        await router.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocks", [[], [{"type": "unknown", "text": "伪成功"}], [{"type": "text", "text": " "}]]
)
async def test_native_empty_or_unsupported_response_rejected(monkeypatch, blocks):
    mock_transport(monkeypatch, lambda request: httpx.Response(200, json={"content": blocks}))
    router = configured(monkeypatch, provider="anthropic")
    try:
        with pytest.raises(RuntimeError):
            await router.chat([{"role": "user", "content": "测试"}])
    finally:
        await router.aclose()


@pytest.mark.parametrize(
    "extra",
    [
        {"max_tokens": None},
        {"max_tokens": True},
        {"temperature": "hot"},
        {"stop": {}},
        {"top_p": float("nan")},
        {"extra_body": {"temperature": "hot"}},
    ],
)
def test_bad_sampling_parameters_rejected(monkeypatch, extra):
    with pytest.raises(ValueError):
        configured(monkeypatch, extra=extra)


@pytest.mark.asyncio
async def test_cancelled_half_open_probe_can_recover(monkeypatch):
    from app.infrastructure.llm.circuit_breaker import CircuitBreaker, CircuitState

    breaker = CircuitBreaker(failure_threshold=1, half_open_max=1, recovery_timeout=0.1)
    monotonic = [0.0]
    monkeypatch.setattr(
        "app.infrastructure.llm.circuit_breaker.time.monotonic", lambda: monotonic[0]
    )

    async def fail():
        raise TimeoutError()

    async def cancel():
        raise asyncio.CancelledError()

    async def succeed():
        return "恢复"

    with pytest.raises(TimeoutError):
        await breaker.call(fail)
    monotonic[0] = 1
    with pytest.raises(asyncio.CancelledError):
        await breaker.call(cancel)
    assert breaker.state == CircuitState.OPEN
    monotonic[0] = 2
    assert await breaker.call(succeed) == "恢复"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls,should_fail",
    [
        (None, True),
        (
            [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{}"},
                }
            ],
            False,
        ),
    ],
)
async def test_compatible_empty_answer_requires_real_tool_call(monkeypatch, calls, should_fail):
    mock_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "", "tool_calls": calls},
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
            },
        ),
    )
    router = configured(monkeypatch)
    try:
        if should_fail:
            with pytest.raises(RuntimeError):
                await router.chat([{"role": "user", "content": "测试"}])
        else:
            result = await router.chat([{"role": "user", "content": "测试"}])
            assert result.raw["choices"][0]["message"]["tool_calls"][0]["id"] == "call"
    finally:
        await router.aclose()
