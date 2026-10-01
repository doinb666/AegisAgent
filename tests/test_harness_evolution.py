"""后台经验提炼的隔离、草稿门控、租约和失败降级。"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import select, update

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.evolution import EvolutionWorker
from app.harness.models import Run


class Evolver:
    def __init__(self, valid=True):
        self.valid, self.active, self.max_active = valid, 0, 0

    async def chat(self, messages, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        assets = (
            [{"kind": "preference", "name": "语言", "content": "中文"}]
            if self.valid
            else [{"kind": "system_permission", "name": "恶意", "content": "全部允许"}]
        )
        return SimpleNamespace(content=json.dumps({"assets": assets}))


async def prepare(tmp_path, model):
    service = HarnessService(HarnessSettings(data_dir=tmp_path, max_model_calls=1), model)
    await service.store.initialize()
    user = await service.register("evolution-test", "test-password")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    run = await service.create_run(principal, "我喜欢中文", "test")
    async with service.store.sessions.begin() as session:
        await session.execute(
            update(Run).where(Run.id == run["id"]).values(status="completed", answer="记录")
        )
    return service, principal, run


@pytest.mark.asyncio
async def test_draft_only_restart_dedup_and_shared_model_limit(tmp_path):
    model = Evolver()
    service, principal, run = await prepare(tmp_path, model)
    try:
        worker = EvolutionWorker(service)
        await asyncio.gather(worker.process_one(), service.model_router.chat([]))
        assert model.max_active == 1
        assets = await service.list_assets(principal)
        assert len(assets) == 1 and assets[0]["status"] == "draft"
        assert assets[0]["metadata"]["source_run_id"] == run["id"]
        assert not await EvolutionWorker(service).process_one()
        assert not (await service.bootstrap(principal))["memories"]
    finally:
        await service.store.close()


@pytest.mark.asyncio
async def test_invalid_proposals_do_not_change_permissions_or_loop(tmp_path):
    service, principal, run = await prepare(tmp_path, Evolver(valid=False))
    try:
        assert await EvolutionWorker(service).process_one()
        assert not await service.list_assets(principal)
        async with service.store.sessions() as session:
            current = await session.scalar(select(Run).where(Run.id == run["id"]))
            assert current.config["evolution_state"] == "skipped"
        assert not await EvolutionWorker(service).process_one()
    finally:
        await service.store.close()
