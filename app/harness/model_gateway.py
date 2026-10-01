"""统一限制前台、反思和后台经验作业的模型并发。"""

import asyncio


class SharedModelGateway:
    def __init__(self, delegate, limit):
        self.delegate = delegate
        self.semaphore = asyncio.Semaphore(limit)

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    async def chat(self, *args, **kwargs):
        async with self.semaphore:
            return await self.delegate.chat(*args, **kwargs)
