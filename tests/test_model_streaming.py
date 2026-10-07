"""真实路由流消费、组装与资源释放；不调用付费模型。"""

import asyncio
import json

import httpx
import httpx2
import pytest

from app.infrastructure.llm.model_router import ModelConfig, ModelRouter
from tests.test_model_protocols import configured, mock_transport
from tests.test_model_router import Client


def chunk(content=None, tools=None, finish=None, usage=None, **extra):
    delta = {**extra}
    if content is not None:
        delta["content"] = content
    if tools is not None:
        delta["tool_calls"] = tools
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage}


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for value in self.chunks:
            if isinstance(value, BaseException):
                raise value
            if isinstance(value, asyncio.Event):
                await value.wait()
            else:
                yield value

    async def close(self):
        self.closed = True


class StreamClient(Client):
    def __init__(self, chunks):
        super().__init__()
        self.stream = Stream(chunks)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs.get("stream"):
            return self.stream
        return await super().create(**kwargs)


def routed(monkeypatch, chunks, **config):
    client = StreamClient(chunks)
    monkeypatch.setattr("app.infrastructure.llm.model_router.AsyncOpenAI", lambda **kwargs: client)
    return ModelRouter([ModelConfig("test", "test-key", **config)]), client


@pytest.mark.asyncio
async def test_stream_usage_keeps_only_bounded_standard_counters(monkeypatch):
    router, client = routed(
        monkeypatch,
        [
            chunk(
                "回答",
                finish="stop",
                usage={
                    "total_tokens": 3,
                    "prompt_tokens": True,
                    "completion_tokens": 10**15,
                    "unknown": "不保存",
                },
            )
        ],
    )

    async def receive(delta):
        pass

    try:
        result = await router.chat([], on_delta=receive)
        assert result.usage == {"total_tokens": 3}
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_anthropic_fallback_is_complete_response_without_stream_request(monkeypatch):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "原生完整答复"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    mock_transport(monkeypatch, respond)
    router = configured(monkeypatch, provider="anthropic", base_url="https://anthropic.invalid/v1")
    received = []

    async def receive(delta):
        received.append(delta)

    try:
        result = await router.chat([{"role": "user", "content": "任务"}], on_delta=receive)
        assert result.content == "原生完整答复"
        assert "stream" not in requests[0]
        assert not received
        assert router.public_routes()[0]["streaming"] is False
    finally:
        await router.aclose()


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_streaming_declaration_is_strict_boolean(value):
    with pytest.raises(ValueError):
        ModelConfig("test", "test-key", streaming=value)


@pytest.mark.asyncio
async def test_real_sdk_delayed_http_bytes_publish_before_stream_finishes(monkeypatch):
    release = asyncio.Event()
    first = asyncio.Event()
    closed = []
    for module in (httpx, httpx2):
        original = module.AsyncClient.__init__

        class HTTPBytes(module.AsyncByteStream):
            async def __aiter__(self):
                for value in (chunk("首片"), release, chunk("末片", finish="stop")):
                    if isinstance(value, asyncio.Event):
                        await value.wait()
                    else:
                        body = {
                            "id": "test",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": "test",
                            **value,
                        }
                        yield ("data: " + json.dumps(body) + "\n\n").encode()
                yield b"data: [DONE]\n\n"

            async def aclose(self):
                closed.append(True)

        def initialize(
            self, *args, _original=original, _module=module, _stream=HTTPBytes, **kwargs
        ):
            kwargs["transport"] = _module.MockTransport(
                lambda request: _module.Response(
                    200, stream=_stream(), headers={"Content-Type": "text/event-stream"}
                )
            )
            _original(self, *args, **kwargs)

        monkeypatch.setattr(module.AsyncClient, "__init__", initialize)
    router = configured(monkeypatch)
    received = []

    async def receive(delta):
        received.append(delta)
        first.set()

    task = asyncio.create_task(router.chat([], on_delta=receive))
    try:
        await asyncio.wait_for(first.wait(), 1)
        assert not task.done()
        assert received == [{"type": "content", "text": "首片"}]
        release.set()
        assert (await task).content == "首片末片"
        assert closed
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await router.aclose()


@pytest.mark.asyncio
async def test_stream_text_arrives_before_full_response_and_reasoning_is_hidden(monkeypatch):
    release = asyncio.Event()
    router, client = routed(
        monkeypatch,
        [
            chunk(reasoning_content="内部推理"),
            chunk("第一片"),
            release,
            chunk("第二片", finish="stop"),
            {"choices": [], "usage": {"total_tokens": 3}},
        ],
    )
    deltas = []
    first = asyncio.Event()

    async def receive(delta):
        deltas.append(delta)
        first.set()

    task = asyncio.create_task(router.chat([{"role": "user", "content": "任务"}], on_delta=receive))
    try:
        await asyncio.wait_for(first.wait(), 1)
        assert not task.done()
        assert deltas == [{"type": "content", "text": "第一片"}]
        release.set()
        result = await task
        assert result.content == "第一片第二片"
        assert result.usage == {"total_tokens": 3}
        assert client.requests[0]["stream"] is True
        assert client.requests[0]["stream_options"] == {"include_usage": True}
        assert client.stream.closed
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await router.aclose()


@pytest.mark.asyncio
async def test_interleaved_tools_are_assembled_without_publishing_fragments(monkeypatch):
    router, client = routed(
        monkeypatch,
        [
            chunk("临时说明"),
            chunk(
                tools=[
                    {
                        "index": 1,
                        "id": "call-b",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"value":'},
                    }
                ]
            ),
            chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call-a",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"value":"甲"}'},
                    },
                    {"index": 1, "function": {"arguments": '"乙"}'}},
                ]
            ),
            chunk(finish="tool_calls"),
        ],
    )
    deltas = []

    async def receive(delta):
        deltas.append(delta)

    try:
        result = await router.chat([{"role": "user", "content": "任务"}], on_delta=receive)
        calls = result.raw["choices"][0]["message"]["tool_calls"]
        assert [call["id"] for call in calls] == ["call-a", "call-b"]
        assert calls[1]["function"]["arguments"] == '{"value":"乙"}'
        assert deltas == [{"type": "content", "text": "临时说明"}, {"type": "tools"}]
        assert client.stream.closed
    finally:
        await router.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values",
    [
        [chunk("未结束")],
        [chunk("截断", finish="length")],
        [
            chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call",
                        "function": {"name": "read", "arguments": '{"value":'},
                    }
                ]
            ),
            chunk(finish="tool_calls"),
        ],
        [
            chunk(
                tools=[{"index": 16, "id": "call", "function": {"name": "read", "arguments": "{}"}}]
            ),
            chunk(finish="tool_calls"),
        ],
        [chunk("x" * 131073, finish="stop")],
        [
            chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call",
                        "function": {"name": "read", "arguments": '{"value":1e400}'},
                    }
                ]
            ),
            chunk(finish="tool_calls"),
        ],
        [
            chunk(
                tools=[
                    {
                        "index": 0,
                        "id": "call",
                        "function": {"name": "read", "arguments": "x" * 16385},
                    }
                ]
            ),
            chunk(finish="tool_calls"),
        ],
        [
            chunk(
                tools=[
                    {
                        "index": index,
                        "id": str(index),
                        "function": {
                            "name": "read",
                            "arguments": '{"value":"' + "x" * 15000 + '"}',
                        },
                    }
                    for index in range(9)
                ]
            ),
            chunk(finish="tool_calls"),
        ],
        [chunk(role="assistant")] * 65537,
    ],
)
async def test_incomplete_and_unbounded_streams_fail_closed(monkeypatch, values):
    router, client = routed(monkeypatch, values)

    async def receive(delta):
        pass

    try:
        with pytest.raises(RuntimeError):
            await router.chat([{"role": "user", "content": "任务"}], on_delta=receive)
        assert client.stream.closed
    finally:
        await router.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic", [False, True])
async def test_fallback_only_before_first_semantic_delta(monkeypatch, semantic):
    primary = StreamClient(
        ([chunk("已开始")] if semantic else [chunk(role="assistant")]) + [TimeoutError("流中断")]
    )
    backup = StreamClient([chunk("备用", finish="stop")])
    monkeypatch.setattr(
        "app.infrastructure.llm.model_router.AsyncOpenAI",
        lambda **kwargs: primary if "primary" in kwargs["base_url"] else backup,
    )
    router = ModelRouter(
        [
            ModelConfig("primary", "test", "https://primary.invalid"),
            ModelConfig("backup", "test", "https://backup.invalid", priority=1),
        ]
    )

    async def receive(delta):
        pass

    try:
        if semantic:
            with pytest.raises(RuntimeError):
                await router.chat([], on_delta=receive)
            assert not backup.requests
        else:
            assert (await router.chat([], on_delta=receive)).content == "备用"
        assert primary.stream.closed
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_sdk_http_stream_maps_task_parameters_and_locks_stream_options(monkeypatch):
    requests = []
    values = [
        chunk("HTTP首片"),
        chunk("完整", finish="stop"),
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "total_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 1},
            },
        },
    ]
    payload = (
        "".join(
            "data: "
            + json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "test",
                    **value,
                },
                ensure_ascii=False,
            )
            + "\n\n"
            for value in values
        )
        + "data: [DONE]\n\n"
    )

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, content=payload.encode(), headers={"Content-Type": "text/event-stream"}
        )

    mock_transport(monkeypatch, respond)
    router = configured(
        monkeypatch,
        extra={"extra_body": {"stream_options": {"include_usage": False}}},
        parameters={
            "output_token_parameter": "max_completion_tokens",
            "reasoning_efforts": ["medium"],
        },
    )
    deltas = []

    async def receive(delta):
        deltas.append(delta)

    try:
        response = await router.chat(
            [],
            model_parameters={"max_output_tokens": 128, "reasoning_effort": "medium"},
            on_delta=receive,
        )
        assert response.content == "HTTP首片完整"
        assert response.usage["total_tokens"] == 4
        assert response.usage["cached_tokens"] == 1
        assert requests[0]["stream_options"] == {"include_usage": True}
        assert requests[0]["max_completion_tokens"] == 128
        assert requests[0]["reasoning_effort"] == "medium"
        assert "temperature" not in requests[0]
        assert len(deltas) == 2
    finally:
        await router.aclose()


@pytest.mark.asyncio
async def test_explicit_nonstream_route_keeps_full_response_without_fake_deltas(monkeypatch):
    router, client = routed(monkeypatch, [], streaming=False)
    deltas = []

    async def receive(delta):
        deltas.append(delta)

    try:
        result = await router.chat([], on_delta=receive)
        assert result.content == "备用结果"
        assert "stream" not in client.requests[0]
        assert not deltas
        assert router.public_routes()[0]["streaming"] is False
    finally:
        await router.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancel_closes_stream_and_releases_slot(monkeypatch, cancel):
    release = asyncio.Event()
    router, client = routed(monkeypatch, [chunk("已开始"), release, chunk("结束", finish="stop")])
    router._semaphore = asyncio.Semaphore(1)
    first = asyncio.Event()

    async def receive(delta):
        first.set()

    task = asyncio.create_task(router.chat([], on_delta=receive))
    try:
        await asyncio.wait_for(first.wait(), 1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, 0.01)
        assert client.stream.closed
        client.stream = Stream([chunk("再次调用", finish="stop")])
        assert (await asyncio.wait_for(router.chat([], on_delta=receive), 1)).content == "再次调用"
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await router.aclose()
