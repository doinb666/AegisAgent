"""路由连接替身验证，真实模型费用与网络留给外部联测。"""

from types import SimpleNamespace

import pytest

from app.infrastructure.llm.circuit_breaker import CircuitState
from app.infrastructure.llm.model_router import ModelConfig, ModelRouter


class Client:
    def __init__(self, fail=False):
        self.chat = SimpleNamespace(completions=self)
        self.fail, self.calls, self.closed = fail, 0, False

    async def create(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise TimeoutError("模拟超时")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="备用结果"))], usage=None
        )

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_preferred_endpoint_fails_and_distinct_breakers(monkeypatch):
    clients = []

    def factory(**kwargs):
        client = Client(fail="primary" in kwargs["base_url"])
        clients.append(client)
        return client

    monkeypatch.setattr("app.infrastructure.llm.model_router.AsyncOpenAI", factory)
    router = ModelRouter(
        [
            ModelConfig("same-model", "test", "https://primary.invalid/v1", priority=0),
            ModelConfig("same-model", "test", "https://backup.invalid/v1", priority=1),
        ],
        failure_threshold=1,
    )
    result = await router.chat([{"role": "user", "content": "测试"}], model_preference="same-model")
    assert result.content == "备用结果"
    assert [item.state for item in router._breakers.values()] == [
        CircuitState.OPEN,
        CircuitState.CLOSED,
    ]
    await router.chat([{"role": "user", "content": "第二次"}])
    assert [item.calls for item in clients] == [1, 2]
    await router.aclose()
    assert all(item.closed for item in clients)
