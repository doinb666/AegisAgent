"""只调用隔离执行服务，没有宿主执行降级路径。"""

import asyncio

import httpx

from app.harness_tools.workspace import workspace_key


class SandboxClient:
    def __init__(self, settings):
        self.settings = settings

    async def execute(self, principal, run_id: str, code: str) -> dict:
        if not self.settings.sandbox_url or not self.settings.sandbox_token:
            raise RuntimeError("尚未配置隔离执行服务，已拒绝执行；请启用 Docker 沙箱")
        if len(code.encode()) > 12000:
            raise ValueError("代码超过沙箱输入上限")
        async with httpx.AsyncClient(
            timeout=self.settings.sandbox_timeout_seconds + 10,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            response = await client.post(
                self.settings.sandbox_url.rstrip("/") + "/execute",
                headers={"Authorization": f"Bearer {self.settings.sandbox_token}"},
                json={"workspace_id": workspace_key(principal, run_id), "code": code},
            )
            response.raise_for_status()
            return response.json()


async def bounded(coro, timeout: float = 30):
    """所有外部工具都有独立超时预算。"""
    return await asyncio.wait_for(coro, timeout)
