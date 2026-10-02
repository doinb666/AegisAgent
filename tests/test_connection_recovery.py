"""连接取消失效后的真实 SQLite 连接池恢复回归。"""

import asyncio

import anyio
import pytest
from sqlalchemy import select

from app.harness.models import User
from app.harness.settings import HarnessSettings
from app.harness.store import Store


@pytest.mark.asyncio
async def test_cancelled_invalidation_preserves_next_read_and_persisted_data(tmp_path):
    store = Store(HarnessSettings(data_dir=tmp_path, database_url=""))
    await store.initialize()
    try:
        async with store.sessions.begin() as session:
            session.add(
                User(
                    id="saved-user",
                    tenant_id="saved-tenant",
                    username="saved-user",
                    password="测试哈希",
                    role="admin",
                )
            )

        async with store.engine.connect() as connection:
            with anyio.CancelScope() as cancellation:
                cancellation.cancel()
                await connection.invalidate()
            assert cancellation.cancelled_caught
            # 失效流程被取消后，等待驱动完成后台关闭，再归还池内记录。
            await asyncio.sleep(0.03)

        async with store.sessions() as session:
            user = await session.scalar(select(User).where(User.id == "saved-user"))
            assert user is not None
            assert (user.tenant_id, user.username, user.role) == (
                "saved-tenant",
                "saved-user",
                "admin",
            )
    finally:
        await store.close()
