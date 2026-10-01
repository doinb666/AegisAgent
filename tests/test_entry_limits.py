"""入口长度与认证限流，避免先分配无限上传再拒绝。"""

import httpx
import pytest
from starlette.responses import JSONResponse

from app.api.limits import EntryLimits


async def read_body(scope, receive, send):
    while True:
        message = await receive()
        if not message.get("more_body"):
            break
    await JSONResponse({"ok": True})(scope, receive, send)


@pytest.mark.asyncio
async def test_length_and_chunked_requests():
    app = EntryLimits(read_body, max_bytes=10)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.post("/upload", content=b"12345678901")).status_code == 413

        async def chunks():
            yield b"123456"
            yield b"78901"

        assert (await client.post("/upload", content=chunks())).status_code == 413
        assert (await client.post("/upload", content=b"123")).status_code == 200


@pytest.mark.asyncio
async def test_authentication_rate_limit():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=EntryLimits(read_body)), base_url="http://test"
    ) as client:
        for _ in range(20):
            assert (await client.post("/auth/login")).status_code == 200
        result = await client.post("/auth/login")
        assert result.status_code == 429 and result.headers["retry-after"] == "60"
