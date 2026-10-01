"""资产修订保留来源且撤销验证，反馈不能验证新提炼步骤。"""

import pytest
from sqlalchemy import select

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.models import Asset
from app.harness.runtime import Worker


@pytest.mark.asyncio
async def test_revision_preserves_server_provenance(tmp_path):
    service = HarnessService(HarnessSettings(data_dir=tmp_path))
    await service.store.initialize()
    user = await service.register("revision-test", "test-password")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    run = await service.create_run(principal, "失败案例", "revision")
    try:
        worker = Worker(service)
        await service.store.claim(worker.id)
        await worker.finish(run["id"], "failed", error="工具输入无效")
        assets = await service.list_assets(principal, kind="skill")
        original = assets[0]
        changed = await service.put_asset(
            principal,
            "skill",
            "新修复建议",
            "仍需独立验证",
            asset_id=original["id"],
            status="active",
        )
        assert changed["metadata"]["source_run_id"] == run["id"]
        assert changed["metadata"]["edited"] and changed["status"] == "draft"
        assert not changed["metadata"]["repair_verified"]
        again = await service.put_asset(
            principal,
            "skill",
            changed["name"],
            changed["content"],
            metadata=changed["metadata"],
            asset_id=changed["id"],
        )
        assert again["metadata"]["source_run_id"] == run["id"]
        await service.feedback(principal, run["id"], True)
        async with service.store.sessions() as session:
            saved = await session.scalar(select(Asset).where(Asset.id == original["id"]))
            assert saved.status == "draft"
    finally:
        await service.store.close()
