"""使用官方 MCP Client SDK，端点及凭据仅由运维配置。"""

import json
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx2 as httpx


class LimitedStream(httpx.AsyncByteStream):
    """对单次 HTTP 响应含 SSE 的累计接收量设限。"""

    def __init__(self, stream, limit):
        self.stream = stream
        self.limit = limit

    async def __aiter__(self):
        received = 0
        async for chunk in self.stream:
            received += len(chunk)
            if received > self.limit:
                raise ValueError("MCP 响应超过接收预算")
            yield chunk

    async def aclose(self):
        await self.stream.aclose()


class LimitedTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.transport = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request):
        response = await self.transport.handle_async_request(request)
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            await response.aclose()
            raise ValueError("MCP 服务不能返回压缩响应，防止解压预算绕过")
        response.stream = LimitedStream(response.stream, 2_000_000)
        return response

    async def aclose(self):
        await self.transport.aclose()


class MCPGateway:
    def __init__(self, configuration: str):
        servers = json.loads(configuration)
        if not isinstance(servers, list):
            raise ValueError("MCP_SERVERS_JSON 必须为数组")
        self.servers = {}
        for item in servers:
            name, url = item["name"], item["url"]
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError("MCP 地址必须为 HTTP/HTTPS")
            if parsed.username or parsed.password or parsed.fragment:
                raise ValueError("MCP 地址不得包含内嵌凭据或片段")
            if name in self.servers:
                raise ValueError("MCP 服务名称重复")
            for field in ("tenant_ids", "user_ids"):
                identifiers = item.get(field, [])
                if not isinstance(identifiers, list) or any(
                    not isinstance(value, str) for value in identifiers
                ):
                    raise ValueError("MCP 租户与用户授权必须是字符串数组")
            self.servers[name] = item

    def authorized_servers(self, principal):
        if principal.role == "viewer":
            return []
        allowed = []
        for name, config in self.servers.items():
            tenants, users = config.get("tenant_ids", []), config.get("user_ids", [])
            if not tenants and not users:
                continue
            if tenants and principal.tenant_id not in tenants:
                continue
            if users and principal.user_id not in users:
                continue
            allowed.append(name)
        return sorted(allowed)

    def authorize(self, server, principal):
        if server not in self.authorized_servers(principal):
            raise PermissionError("没有访问此 MCP 服务的租户或用户授权")

    @asynccontextmanager
    async def session(self, server: str):
        if server not in self.servers:
            raise ValueError("MCP 服务不在运维允许列表")
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        config = self.servers[server]
        headers = {"Accept-Encoding": "identity"}
        if config.get("token_env"):
            token = os.environ.get(config["token_env"], "")
            if not token:
                raise RuntimeError("MCP 凭据环境变量未配置")
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(
            headers=headers,
            timeout=20,
            follow_redirects=False,
            trust_env=False,
            transport=LimitedTransport(),
        ) as client:
            async with streamable_http_client(config["url"], http_client=client) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield session

    async def list_tools(self, server: str) -> list[dict]:
        async with self.session(server) as session:
            result = await session.list_tools()
            allowed = set(self.servers[server].get("tools", []))
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema,
                }
                for tool in result.tools
                if tool.name in allowed
            ]

    async def call(self, server: str, tool: str, arguments: dict):
        config = self.servers.get(server)
        if not config or tool not in config.get("tools", []):
            raise ValueError("MCP 工具不在运维允许列表")
        async with self.session(server) as session:
            catalog = await session.list_tools()
            match = next((t for t in catalog.tools if t.name == tool), None)
            if match is None:
                raise ValueError("MCP 服务未提供该工具")
            from jsonschema import validate

            validate(arguments, match.input_schema)
            result = await session.call_tool(tool, arguments)
            if result.is_error:
                raise RuntimeError("MCP 工具返回执行失败")
            return result.model_dump(mode="json", by_alias=True)
