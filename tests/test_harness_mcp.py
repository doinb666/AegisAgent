"""真实本地 Streamable HTTP/SSE 协议、允许列表与接收预算验收。"""

import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from jsonschema import ValidationError

from app.harness_tools.mcp import LimitedStream, LimitedTransport, MCPGateway

SERVER_LOGS = {}


def server_log(url):
    logs = SERVER_LOGS[url]
    logs.seek(0)
    return logs.read().decode(errors="replace")[-6000:]


def exception_messages(exc):
    if isinstance(exc, BaseExceptionGroup):
        return " ".join(exception_messages(item) for item in exc.exceptions)
    return str(exc)


@pytest.fixture(scope="module")
def local_mcp_url():
    # Windows 动态端口范围可能被系统预留，选取并检测较低的回环端口。
    for port in random.sample(range(8000, 18000), 50):
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", port))
                break
            except OSError:
                continue
    else:
        raise RuntimeError("没有可用的本地测试端口")
    environment = {**os.environ, "AEGIS_TEST_MCP_TOKEN": "test-local-token"}
    logs = tempfile.TemporaryFile()
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("mcp_test_server.py")), str(port)],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=logs,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 15
        with httpx.Client(trust_env=False, timeout=0.5) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    logs.seek(0)
                    raise RuntimeError(
                        "本地 MCP 服务启动失败：" + logs.read().decode(errors="replace")
                    )
                try:
                    if client.get(url + "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
            else:
                logs.seek(0)
                raise RuntimeError(
                    "本地 MCP 服务未在预算内就绪：" + logs.read().decode(errors="replace")
                )
        SERVER_LOGS[url] = logs
        yield url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        logs.close()
        SERVER_LOGS.pop(url, None)


def gateway(url, monkeypatch, *, path="/mcp", tools=None):
    monkeypatch.setenv("AEGIS_TEST_MCP_TOKEN", "test-local-token")
    return MCPGateway(
        json.dumps(
            [
                {
                    "name": "local",
                    "url": url + path,
                    "token_env": "AEGIS_TEST_MCP_TOKEN",
                    "tools": tools
                    if tools is not None
                    else ["sum_values", "large_output", "failing_tool"],
                }
            ]
        )
    )


@pytest.mark.asyncio
async def test_official_sdk_real_initialize_list_and_call(local_mcp_url, monkeypatch):
    client = gateway(local_mcp_url, monkeypatch)
    tools = await client.list_tools("local")
    assert {tool["name"] for tool in tools} == {"sum_values", "large_output", "failing_tool"}
    result = await client.call("local", "sum_values", {"left": 7, "right": 8})
    assert result["isError"] is False
    assert result["structuredContent"] == {"sum": 15}


@pytest.mark.asyncio
async def test_allowlist_and_schema_before_call(local_mcp_url, monkeypatch):
    client = gateway(local_mcp_url, monkeypatch)
    with pytest.raises(ValueError, match="允许列表"):
        await client.call("local", "forbidden_tool", {})
    with pytest.raises(ValueError, match="允许列表"):
        await client.list_tools("unconfigured")
    with pytest.raises(Exception) as error:
        await client.call("local", "sum_values", {"left": "wrong", "right": 1})
    assert "integer" in exception_messages(error.value)
    assert isinstance(error.value, (ValidationError, BaseExceptionGroup))


@pytest.mark.asyncio
async def test_missing_token_rejected_without_connection(local_mcp_url, monkeypatch):
    client = gateway(local_mcp_url, monkeypatch)
    monkeypatch.delenv("AEGIS_TEST_MCP_TOKEN")
    with pytest.raises(RuntimeError, match="凭据环境变量"):
        await client.list_tools("local")


@pytest.mark.asyncio
async def test_real_protocol_tool_error(local_mcp_url, monkeypatch):
    with pytest.raises(Exception) as error:
        await gateway(local_mcp_url, monkeypatch).call("local", "failing_tool", {})
    assert "执行失败" in exception_messages(error.value)


@pytest.mark.asyncio
async def test_compressed_http_response_is_rejected(local_mcp_url, monkeypatch):
    with pytest.raises(Exception) as error:
        await gateway(local_mcp_url, monkeypatch, path="/compressed").list_tools("local")
    assert "压缩响应" in exception_messages(error.value)


@pytest.mark.asyncio
async def test_real_mcp_response_receiving_budget(local_mcp_url, monkeypatch):
    failures = []
    original_iterator = LimitedStream.__aiter__

    async def observe_real_stream(self):
        try:
            async for chunk in original_iterator(self):
                yield chunk
        except ValueError as exc:
            failures.append(str(exc))
            raise

    # SDK 会将底层异常改为 SSE 结束错误；观察真实流，不替换网络或响应数据。
    monkeypatch.setattr(LimitedStream, "__aiter__", observe_real_stream)
    with pytest.raises(Exception) as error:
        await gateway(local_mcp_url, monkeypatch, path="/budget", tools=["large_output"]).call(
            "local", "large_output", {}
        )
    assert any("接收预算" in failure for failure in failures), server_log(local_mcp_url)
    assert "SSE" in exception_messages(error.value) or "接收预算" in exception_messages(error.value)


@pytest.mark.asyncio
async def test_empty_allowlist_denies_all_real_server_tools(local_mcp_url, monkeypatch):
    client = gateway(local_mcp_url, monkeypatch, tools=[])
    assert await client.list_tools("local") == []
    with pytest.raises(ValueError, match="允许列表"):
        await client.call("local", "sum_values", {"left": 1, "right": 2})


@pytest.mark.asyncio
async def test_real_server_rejects_wrong_bearer_token(local_mcp_url, monkeypatch):
    statuses = []
    original_transport = LimitedTransport.handle_async_request

    async def observe_real_response(self, request):
        response = await original_transport(self, request)
        statuses.append(response.status_code)
        return response

    monkeypatch.setattr(LimitedTransport, "handle_async_request", observe_real_response)
    client = gateway(local_mcp_url, monkeypatch)
    monkeypatch.setenv("AEGIS_TEST_MCP_TOKEN", "incorrect-test-token")
    with pytest.raises(Exception) as error:
        await client.list_tools("local")
    assert 401 in statuses
    assert "error response" in exception_messages(error.value)


@pytest.mark.parametrize(
    "url",
    [
        "file:///private",
        "http://user:pass@127.0.0.1/mcp",
        "http://127.0.0.1/mcp#fragment",
        "http:///missing-host",
    ],
)
def test_endpoint_configuration_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        MCPGateway(json.dumps([{"name": "unsafe", "url": url, "tools": []}]))
