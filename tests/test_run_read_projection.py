"""状态轮询不应读取内部上下文；私有范围和游标仍须完整验证。"""

import pytest
from sqlalchemy import event

from app.harness import HarnessError, Principal
from tests.test_harness_store import identity, make_service


@pytest.mark.asyncio
async def test_status_list_events_and_files_exclude_execution_context(tmp_path):
    service = make_service(tmp_path)
    await service.store.initialize()
    statements = []

    def record_query(_, __, statement, ___, ____, _____):
        if statement.lstrip().upper().startswith("SELECT") and "harness_runs" in statement:
            statements.append(statement.split("FROM", 1)[0])

    try:
        owner = await identity(service, "read-owner")
        colleague_data = await service.create_user(owner, "read-colleague", "password123", "viewer")
        colleague = Principal(colleague_data["user_id"], owner.tenant_id, "viewer")
        first = await service.create_run(owner, "第一次读取", "read-first")
        second = await service.create_run(owner, "第二次读取", "read-second")
        event.listen(service.store.engine.sync_engine, "before_cursor_execute", record_query)
        assert await service.get_run(owner, first["id"]) == first
        assert [run["id"] for run in await service.list_runs(owner, limit=1)] == [second["id"]]
        assert [run["id"] for run in await service.list_runs(owner, before=second["id"])] == [
            first["id"]
        ]
        events = await service.events(owner, first["id"])
        assert len(events) == 2
        assert await service.events(owner, first["id"], events[-1]["id"]) == []
        assert (await service.list_run_files(owner, first["id"]))["files"] == []
        for operation in (
            service.get_run(colleague, first["id"]),
            service.events(colleague, first["id"]),
            service.list_runs(colleague, before=first["id"]),
            service.list_run_files(colleague, first["id"]),
        ):
            with pytest.raises(HarnessError) as caught:
                await operation
            assert caught.value.status_code == 404
        assert statements
        assert all("harness_runs.messages" not in query for query in statements), statements
        assert all("harness_runs.payload_hash" not in query for query in statements), statements
    finally:
        event.remove(service.store.engine.sync_engine, "before_cursor_execute", record_query)
        await service.close()
