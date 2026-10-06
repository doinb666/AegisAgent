"""历史问答分页：共享会话 ID 不能突破账号范围，子任务独立展示。"""

import asyncio

import pytest
from sqlalchemy import event

from app.harness.errors import HarnessError
from app.harness.models import Run
from tests.test_workspace_inspection import get, seed_run
from tests.test_workspace_inspection import inspection as inspection_fixture

inspection = inspection_fixture


async def set_session(env, ids, session_id="shared", parent=None):
    async with env.service.store.sessions.begin() as transaction:
        for identifier in ids:
            run = await transaction.get(Run, identifier)
            run.session_id = session_id
            run.parent_run_id = parent
            run.answer = "回答 " + identifier
            run.messages = [{"role": "system", "content": "内部上下文不可返回"}]


@pytest.mark.asyncio
async def test_stable_bounded_pages_and_no_future_turns(inspection):
    env = inspection
    ids = [f"turn-{index:02}" for index in range(30)]
    for identifier in ids:
        await seed_run(env, identifier, created=10)
    await set_session(env, ids)
    received = []
    params = {"limit": 7}
    while True:
        response = await get(env, "/api/v1/runs/turn-28/thread", **params)
        assert response.status_code == 200
        page = response.json()
        assert len(page["items"]) <= 7
        received = [row["id"] for row in page["items"]] + received
        if not page["has_more"]:
            assert page["next_before"] is None
            break
        params["before"] = page["next_before"]
    assert received == ids[:29]
    assert (await get(env, "/api/v1/runs/turn-28/thread", before="turn-29")).status_code == 404


@pytest.mark.asyncio
async def test_owner_scope_child_isolation_and_cursor_validation(inspection):
    env = inspection
    for identifier, role in (
        ("anchor", "owner"),
        ("child", "owner"),
        ("foreign", "viewer"),
        ("outsider", "outsider"),
        ("unrelated", "owner"),
    ):
        await seed_run(env, identifier, role=role)
    await set_session(env, ["anchor", "foreign", "outsider"])
    await set_session(env, ["child"], parent="anchor")
    page = (await get(env, "/api/v1/runs/anchor/thread")).json()
    assert [row["id"] for row in page["items"]] == ["anchor"]
    child_page = (await get(env, "/api/v1/runs/child/thread")).json()
    assert [row["id"] for row in child_page["items"]] == ["child"]
    assert not child_page["has_more"]
    for cursor in ("foreign", "outsider", "missing", "unrelated", "child"):
        assert (await get(env, "/api/v1/runs/anchor/thread", before=cursor)).status_code == 404
    for role in ("viewer", "outsider"):
        assert (await get(env, "/api/v1/runs/anchor/thread", role=role)).status_code == 404
    assert (await get(env, "/api/v1/runs/child/thread", before="anchor")).status_code == 404
    assert (await env.client.get("/api/v1/runs/anchor/thread")).status_code == 401


@pytest.mark.asyncio
async def test_thread_projection_excludes_internal_context(inspection):
    env = inspection
    # SQL捕获只检查接口投影，后台队列领取有自己的上下文读取职责。
    for worker in env.service.workers:
        worker.task.cancel()
    await asyncio.gather(*(worker.task for worker in env.service.workers), return_exceptions=True)
    await seed_run(env, "projection")
    await set_session(env, ["projection"])
    statements = []

    def capture(_, __, statement, *args):
        statements.append(statement)

    event.listen(env.service.store.engine.sync_engine, "before_cursor_execute", capture)
    try:
        result = await get(env, "/api/v1/runs/projection/thread")
    finally:
        event.remove(env.service.store.engine.sync_engine, "before_cursor_execute", capture)
    assert result.status_code == 200
    item = result.json()["items"][0]
    assert set(item) == {"id", "message", "answer", "error", "status", "created"}
    assert "内部上下文不可返回" not in result.text
    assert all("harness_runs.messages" not in statement for statement in statements)


@pytest.mark.asyncio
async def test_http_and_service_page_limits(inspection):
    env = inspection
    await seed_run(env, "limits")
    for limit in (0, -1, 51, "invalid"):
        assert (await get(env, "/api/v1/runs/limits/thread", limit=limit)).status_code == 422
    assert (await get(env, "/api/v1/runs/limits/thread", before="x" * 129)).status_code == 422
    for limit in (True, None, 0, 51):
        with pytest.raises(HarnessError) as error:
            await env.service.inspection.thread(env.principals["owner"], "limits", limit)
        assert error.value.status_code == 422
