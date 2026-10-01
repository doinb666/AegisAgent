"""测试专用官方 MCP SDK 服务，仅由测试启动在本机回环地址。"""

import gzip
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from mcp.server import MCPServer
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.responses import Response

server = MCPServer("AegisAgent 本地协议验收")


class SumResult(BaseModel):
    sum: int


@server.tool()
def sum_values(left: int, right: int) -> SumResult:
    """返回两个整数之和。"""
    return SumResult(sum=left + right)


@server.tool()
def forbidden_tool() -> str:
    """存在于服务但不在客户端允许列表。"""
    return "不应通过允许列表调用"


@server.tool(structured_output=False)
def large_output() -> str:
    """用真实协议发送超过接收预算的响应。"""
    return "x" * 2_100_000


@server.tool()
def failing_tool() -> str:
    """返回官方协议的工具错误。"""
    raise ValueError("本地测试工具故意失败")


mcp_app = server.streamable_http_app()
budget_server = MCPServer("AegisAgent 大响应协议验收")
budget_server.tool(structured_output=False)(large_output)
budget_app = budget_server.streamable_http_app(
    streamable_http_path="/budget", json_response=True, stateless_http=True
)


@asynccontextmanager
async def lifespan(application):
    async with mcp_app.router.lifespan_context(mcp_app):
        async with budget_app.router.lifespan_context(budget_app):
            yield


lifecycle_app = Starlette(lifespan=lifespan)


async def app(scope, receive, send):
    if scope["type"] != "http":
        await lifecycle_app(scope, receive, send)
        return
    if scope["path"] == "/health":
        await Response("ready")(scope, receive, send)
        return
    headers = dict(scope["headers"])
    expected = ("Bearer " + os.environ["AEGIS_TEST_MCP_TOKEN"]).encode()
    if headers.get(b"authorization") != expected:
        await Response("unauthorized", status_code=401)(scope, receive, send)
        return
    if scope["path"] == "/compressed":
        await Response(
            gzip.compress(b"compressed local test"), headers={"Content-Encoding": "gzip"}
        )(scope, receive, send)
        return
    selected_app = budget_app if scope["path"] == "/budget" else mcp_app
    await selected_app(scope, receive, send)


if __name__ == "__main__":
    # 本地测试使用 Selector，避免 Windows Proactor AcceptEx 的 WinError 64。
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(sys.argv[1]),
        log_level="warning",
        access_log=False,
        loop="asyncio:SelectorEventLoop",
    )
