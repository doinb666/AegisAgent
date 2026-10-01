"""入口字节预算与个人版认证限流；多副本入口需另配共享限流。"""

import time
from collections import OrderedDict, deque

from starlette.responses import JSONResponse


class RequestTooLarge(Exception):
    pass


class EntryLimits:
    def __init__(self, app, max_bytes=11_000_000):
        self.app = app
        self.max_bytes = max_bytes
        self.attempts = OrderedDict()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            return await JSONResponse({"detail": "无效请求长度"}, 400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse({"detail": "请求超过大小限制"}, 413)(scope, receive, send)
        if scope["path"].endswith(("/auth/login", "/auth/register")):
            address = (scope.get("client") or ("unknown",))[0]
            now = time.monotonic()
            entries = self.attempts.setdefault(address, deque())
            self.attempts.move_to_end(address)
            while entries and entries[0] < now - 60:
                entries.popleft()
            if len(entries) >= 20:
                return await JSONResponse(
                    {"detail": "认证请求过于频繁，请稍后重试"},
                    429,
                    headers={"Retry-After": "60"},
                )(scope, receive, send)
            entries.append(now)
            if len(self.attempts) > 2048:
                self.attempts.popitem(last=False)
        received = 0
        started = False

        async def bounded_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise RequestTooLarge()
            return message

        async def record_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, bounded_receive, record_send)
        except RequestTooLarge:
            if started:
                raise
            await JSONResponse({"detail": "请求超过大小限制"}, 413)(scope, receive, send)
