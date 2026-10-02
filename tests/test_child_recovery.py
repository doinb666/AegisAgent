"""真实 SQLite 父子恢复、私有范围与执行边界回归。"""

import asyncio
import time
from types import MethodType, SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.harness import HarnessService, HarnessSettings, Principal
from app.harness.models import Event, Run, ToolCall
from app.harness.runtime import Worker
from app.harness.store import Store, uid


@pytest_asyncio.fixture
async def stores(tmp_path):
    settings = HarnessSettings(data_dir=tmp_path, database_url="").model_copy(
        update={"lease_seconds": 0.15}
    )
    first, second = Store(settings), Store(settings)
    await first.initialize()
    try:
        yield first, second
    finally:
        await first.close()
        await second.close()


def task(status="running", parent=None, owner="owner", tenant="tenant", worker="worker"):
    identifier = uid()
    return Run(
        id=identifier,
        tenant_id=tenant,
        owner_id=owner,
        idempotency_key=identifier,
        payload_hash="hash",
        session_id="session",
        trace_id=uid(),
        message="恢复测试",
        status=status,
        parent_run_id=parent.id if parent else None,
        created=time.time(),
        lease_owner=worker if status == "running" else None,
        lease_until=time.time() + 60 if status == "running" else None,
        config={"mode": "react", "max_steps": 4},
        messages=[],
    )


def tool(run, status="started", name="read"):
    return ToolCall(
        id=uid(), tenant_id=run.tenant_id, owner_id=run.owner_id, run_id=run.id,
        call_id="call", name=name, arguments={}, arguments_hash="hash", status=status,
        result={"证据": "已完成"} if status == "done" else None,
    )


async def save(store, *rows):
    async with store.sessions.begin() as session:
        session.add_all(rows)


async def read(store, identifier):
    async with store.sessions() as session:
        return await session.get(Run, identifier)


def worker(store):
    service = SimpleNamespace(store=store, settings=store.settings)
    service.snapshot = MethodType(HarnessService.snapshot, service)
    result = Worker(service, child_only=True)
    result.id = "worker"
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_status", ["completed", "failed", "cancelled", "interrupted"])
@pytest.mark.parametrize("child_status", ["queued", "running", "waiting_approval"])
@pytest.mark.parametrize("unknown", [False, True])
async def test_terminal_parent_stops_only_owned_active_children(
    stores, parent_status, child_status, unknown
):
    store, _ = stores
    parent = task(parent_status)
    child = task(child_status, parent)
    completed = task("completed", parent)
    foreign_owner = task("queued", parent, owner="other")
    foreign_tenant = task("queued", parent, tenant="other")
    rows = [parent, child, completed, foreign_owner, foreign_tenant]
    if unknown:
        rows.append(tool(child))
    await save(store, *rows)

    assert await store.claim("next", child_only=True) is None
    stopped = await read(store, child.id)
    assert stopped.status == ("interrupted" if unknown else "cancelled")
    assert "父运行" in stopped.error
    if unknown:
        assert "未知" in stopped.error
    assert stopped.lease_owner is None and stopped.lease_until is None
    assert (await read(store, completed.id)).status == "completed"
    assert (await read(store, foreign_owner.id)).status == "queued"
    assert (await read(store, foreign_tenant.id)).status == "queued"
    async with store.sessions() as session:
        events = (await session.scalars(select(Event).where(Event.run_id == child.id))).all()
        assert len(events) == 1 and events[0].type == stopped.status
        assert events[0].data["parent_run_id"] == parent.id
    # 重复恢复不重复记录终态事件。
    assert await store.claim("again", child_only=True) is None


@pytest.mark.asyncio
async def test_expired_delegate_parent_never_claims_queued_child(stores):
    store, _ = stores
    parent = task()
    parent.lease_until = time.time() - 1
    queued = task("queued", parent)
    running = task("running", parent)
    unknown = task("running", parent)
    completed = task("completed", parent)
    await save(store, parent, queued, running, unknown, completed,
               tool(parent, name="delegate"), tool(unknown), tool(completed, "done"))

    assert await store.claim("child-worker", child_only=True) is None
    assert (await read(store, parent.id)).status == "interrupted"
    assert (await read(store, queued.id)).status == "cancelled"
    assert (await read(store, running.id)).status == "cancelled"
    assert (await read(store, unknown.id)).status == "interrupted"
    assert (await read(store, completed.id)).status == "completed"
    async with store.sessions() as session:
        saved = await session.scalar(select(ToolCall).where(ToolCall.run_id == unknown.id))
        assert saved.status == "started" and saved.result is None


@pytest.mark.asyncio
async def test_ineligible_first_child_does_not_block_valid_child(stores):
    store, _ = stores
    invalid_parent, valid_parent = task("queued"), task()
    invalid = task("queued", invalid_parent)
    valid = task("queued", valid_parent)
    await save(store, invalid_parent, valid_parent, invalid, valid)
    assert await store.claim("next", child_only=True) == valid.id
    assert (await read(store, invalid.id)).status == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_two_stores_have_one_child_claim_or_no_orphan_claim(stores, expired):
    first, second = stores
    parent = task()
    child = task("queued", parent)
    rows = [parent, child]
    if expired:
        parent.lease_until = time.time() - 1
        rows.append(tool(parent, name="delegate"))
    await save(first, *rows)
    claims = await asyncio.gather(
        first.claim("first", child_only=True), second.claim("second", child_only=True)
    )
    assert sum(item is not None for item in claims) == (0 if expired else 1)
    assert (await read(first, child.id)).status == ("cancelled" if expired else "running")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_status", ["completed", "failed", "interrupted", "queued", "expired"]
)
async def test_load_checks_parent_and_preserves_unknown_child(stores, parent_status):
    store, _ = stores
    parent = task("running" if parent_status == "expired" else parent_status)
    if parent_status == "expired":
        parent.lease_until = time.time() - 1
    child = task("running", parent)
    await save(store, parent, child, tool(child))
    with pytest.raises(asyncio.CancelledError):
        await worker(store).load(child.id)
    stopped = await read(store, child.id)
    assert stopped.status == "interrupted" and "未知" in stopped.error


@pytest.mark.asyncio
async def test_heartbeat_stops_live_execution_when_parent_lease_expires(stores):
    store, _ = stores
    parent, executing = task(), worker(store)
    child = task("running", parent)
    await save(store, parent, child, tool(child))
    entered = asyncio.Event()

    async def blocked():
        entered.set()
        await asyncio.Event().wait()

    executing.execution = asyncio.create_task(blocked())
    await entered.wait()
    async with store.sessions.begin() as session:
        persisted = await session.get(Run, parent.id)
        persisted.lease_until = time.time() - 1
    heartbeat = asyncio.create_task(executing.heartbeat(child.id))
    try:
        async with asyncio.timeout(0.5):
            await heartbeat
        assert executing.execution.cancelled()
        assert (await read(store, child.id)).status == "interrupted"
    finally:
        heartbeat.cancel()
        executing.execution.cancel()
        await asyncio.gather(heartbeat, executing.execution, return_exceptions=True)


@pytest.mark.asyncio
async def test_tool_boundary_stops_child_without_invoking_tool(stores):
    store, _ = stores
    parent = task("failed")
    child = task("running", parent)
    await save(store, parent, child)
    executing = worker(store)
    calls = []

    async def execute(*args):
        calls.append(args)
        return {}

    executing.service.tool_executor = SimpleNamespace(
        requires_approval=lambda *args: False, execute=execute
    )
    call = {"id": "call", "function": {"name": "read", "arguments": "{}"}}
    principal = Principal(child.owner_id, child.tenant_id, "operator")
    with pytest.raises(asyncio.CancelledError):
        await executing.execute_tool(child, principal, call, [], 0, {"read"})
    assert calls == []
    assert (await read(store, child.id)).status == "cancelled"


@pytest.mark.asyncio
async def test_finishing_parent_stops_child_in_same_transaction(stores):
    store, _ = stores
    parent = task()
    child = task("running", parent)
    await save(store, parent, child, tool(child))
    await worker(store).finish(parent.id, "failed", error="父任务失败")
    assert (await read(store, parent.id)).status == "failed"
    assert (await read(store, child.id)).status == "interrupted"
