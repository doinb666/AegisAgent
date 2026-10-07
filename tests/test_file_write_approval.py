"""文件写入审批使用真实 HTTP、SQLite 和工作区文件。"""

import asyncio
import json
import os
import subprocess
import threading

import httpx
import pytest
from sqlalchemy import select

from app.harness.models import Run, ToolCall
from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import ToolModel, account, wait_run


@pytest.mark.asyncio
async def test_file_write_approval_requires_frozen_baseline_and_rejects_stale(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "diff-owner")
        other = await account(client, "diff-other")
        run = (
            await client.post(
                "/api/v1/runs",
                headers={**headers, "Idempotency-Key": "diff"},
                json={"message": "写文件"},
            )
        ).json()
        pending = await wait_run(client, headers, run["id"], {"waiting_approval", "failed"})
        approval = pending["approval"]
        assert "file_write" in approval, "审批未冻结文件基线"
        assert not (tmp_path / "workspaces").exists()
        body = {"approved": True, "call_id": approval["call_id"], "args_hash": approval["hash"]}
        url = f"/api/v1/runs/{run['id']}/approval"
        assert (await client.post(url, headers=headers, json=body)).status_code == 422
        body["baseline_hash"] = "0" * 64
        assert (await client.post(url, headers=headers, json=body)).status_code == 409
        body["baseline_hash"] = approval["file_write"]["baseline_hash"]
        assert (await client.post(url, headers=other, json=body)).status_code == 404
        principal = await app.state.harness.authenticate(headers["Authorization"][7:])
        workspace = app.state.harness.tool_executor.workspace
        target = workspace.directory(principal, run["id"]) / "result.txt"
        target.write_bytes("外部先写".encode())
        assert (await client.post(url, headers=headers, json=body)).status_code == 200
        done = await wait_run(client, headers, run["id"], {"completed", "failed", "interrupted"})
        assert done["status"] == "completed", done
        assert target.read_bytes() == "外部先写".encode()
        async with app.state.harness.store.sessions() as session:
            tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run["id"]))
            assert tool.status == "done"
            assert tool.result["code"] == "baseline_conflict"


async def pending_write(app, client, name):
    headers = await account(client, name)
    run = (
        await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": name},
            json={"message": "写文件"},
        )
    ).json()
    pending = await wait_run(client, headers, run["id"], {"waiting_approval", "failed"})
    assert pending["status"] == "waiting_approval", pending
    return headers, run["id"], pending["approval"]


def approval_body(approval):
    return {
        "approved": True,
        "call_id": approval["call_id"],
        "args_hash": approval["hash"],
        "baseline_hash": approval["file_write"]["baseline_hash"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True], ids=["unchanged", "changed"])
async def test_restored_file_approval_never_refreshes_baseline(tmp_path, monkeypatch, changed):
    from app.harness_tools.file_write import FileWriter

    settings = HarnessSettings(data_dir=tmp_path)
    first = create_app(settings, ToolModel())
    async with (
        first.router.lifespan_context(first),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first), base_url="http://test"
        ) as client,
    ):
        headers, run_id, approval = await pending_write(first, client, "restore-owner")
        principal = await first.state.harness.authenticate(headers["Authorization"][7:])
        workspace = first.state.harness.tool_executor.workspace
        if changed:
            target = workspace.directory(principal, run_id) / "result.txt"
            target.write_bytes("恢复期间外部写入".encode())

    async def forbidden(*args, **kwargs):
        raise AssertionError("恢复必须复用原冻结审批")

    monkeypatch.setattr(FileWriter, "freeze", forbidden)
    restored = create_app(settings, ToolModel())
    async with (
        restored.router.lifespan_context(restored),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restored), base_url="http://test"
        ) as client,
    ):
        waiting = (await client.get(f"/api/v1/runs/{run_id}", headers=headers)).json()
        assert waiting["approval"] == approval
        response = await client.post(
            f"/api/v1/runs/{run_id}/approval", headers=headers, json=approval_body(approval)
        )
        assert response.status_code == 200, response.text
        done = await wait_run(client, headers, run_id, {"completed", "failed", "interrupted"})
        assert done["status"] == "completed", done
        async with restored.state.harness.store.sessions() as session:
            tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id))
            assert tool.status == "done"
            if changed:
                assert tool.result["code"] == "baseline_conflict"
                assert target.read_bytes() == "恢复期间外部写入".encode()
            else:
                assert tool.result["status"] == "written"


@pytest.mark.asyncio
async def test_legacy_already_approved_write_fails_closed_without_freeze(tmp_path, monkeypatch):
    from app.harness_tools.file_write import FileWriter

    settings = HarnessSettings(data_dir=tmp_path)
    first = create_app(settings, ToolModel())
    async with (
        first.router.lifespan_context(first),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=first), base_url="http://test"
        ) as client,
    ):
        headers, run_id, approval = await pending_write(first, client, "legacy-owner")
    async with first.state.harness.store.sessions.begin() as session:
        run = await session.get(Run, run_id)
        run.approval = {key: value for key, value in approval.items() if key != "file_write"}
        run.approval = {**run.approval, "approved": True}
        run.status = "queued"
    await first.state.harness.store.close()

    async def forbidden(*args, **kwargs):
        raise AssertionError("旧批准缺少基线，不能补拍")

    monkeypatch.setattr(FileWriter, "freeze", forbidden)
    restored = create_app(settings, ToolModel())
    async with (
        restored.router.lifespan_context(restored),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restored), base_url="http://test"
        ) as client,
    ):
        done = await wait_run(client, headers, run_id, {"completed", "failed", "interrupted"})
        assert done["status"] == "failed", done
        assert "冻结基线" in done["error"]
        assert not list(tmp_path.glob("workspaces/*/result.txt"))


@pytest.mark.asyncio
async def test_publish_uncertainty_keeps_started_and_never_replays(tmp_path, monkeypatch):
    from app.harness_tools import file_write

    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers, run_id, approval = await pending_write(app, client, "unknown-owner")
        method = "rename" if os.name == "nt" else "link"
        original = getattr(file_write.os, method)

        def unknown(source, target, **kwargs):
            original(source, target, **kwargs)
            raise TimeoutError("发布后连接丢失")

        monkeypatch.setattr(file_write.os, method, unknown)
        assert (
            await client.post(
                f"/api/v1/runs/{run_id}/approval", headers=headers, json=approval_body(approval)
            )
        ).status_code == 200
        done = await wait_run(client, headers, run_id, {"completed", "failed", "interrupted"})
        assert done["status"] == "interrupted", done
        async with app.state.harness.store.sessions() as session:
            tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id))
            assert tool.status == "started"
        assert await app.state.harness.store.claim("recovery-worker") is None
        assert next(tmp_path.glob("workspaces/*/result.txt")).read_bytes() == "已批准".encode()


@pytest.mark.asyncio
async def test_cancelled_thread_write_keeps_started_even_when_file_later_changes(
    tmp_path, monkeypatch
):
    from app.harness_tools.file_write import FileWriter

    ready, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = FileWriter._locked_write

    def paused(self, *args):
        ready.set()
        if not release.wait(10):
            raise RuntimeError("测试未释放写入线程")
        try:
            return original(self, *args)
        finally:
            finished.set()

    monkeypatch.setattr(FileWriter, "_locked_write", paused)
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers, run_id, approval = await pending_write(app, client, "cancel-write-owner")
        try:
            assert (
                await client.post(
                    f"/api/v1/runs/{run_id}/approval", headers=headers, json=approval_body(approval)
                )
            ).status_code == 200
            assert await asyncio.to_thread(ready.wait, 5)
            assert (
                await client.post(f"/api/v1/runs/{run_id}/cancel", headers=headers)
            ).status_code == 200
            release.set()
            assert await asyncio.to_thread(finished.wait, 5)
            await asyncio.sleep(0.05)
            current = (await client.get(f"/api/v1/runs/{run_id}", headers=headers)).json()
            assert current["status"] == "cancelled"
            async with app.state.harness.store.sessions() as session:
                tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id))
                assert tool.status == "started"
            assert next(tmp_path.glob("workspaces/*/result.txt")).read_bytes() == "已批准".encode()
            assert await app.state.harness.store.claim("retry-cancelled-write") is None
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("restriction", ["child", "schedule", "viewer"])
async def test_file_write_rechecks_readonly_restrictions_at_both_layers(tmp_path, restriction):
    from app.harness import HarnessError, Principal
    from app.harness.runtime import Worker

    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers, run_id, approval = await pending_write(app, client, "readonly-write-owner")
        service = app.state.harness
        principal = await service.authenticate(headers["Authorization"][7:])
        worker = Worker(service)
        async with service.store.sessions.begin() as session:
            run = await session.get(Run, run_id)
            if restriction == "child":
                run.parent_run_id = run_id
            elif restriction == "schedule":
                run.config = {**run.config, "schedule_id": "test-schedule"}
            run.status = "running"
            run.lease_owner = worker.id
        if restriction == "viewer":
            principal = Principal(principal.user_id, principal.tenant_id, "viewer")
        try:
            tools = service.tool_executor
            with pytest.raises(PermissionError):
                await tools.prepare_file_write(
                    principal, run_id, approval["arguments"], approval["call_id"], approval["hash"]
                )
            with pytest.raises(PermissionError):
                await tools.execute(
                    "file_write",
                    approval["arguments"],
                    principal,
                    run_id,
                    approval_context=approval["file_write"],
                )
            async with service.store.sessions() as session:
                run = await session.get(Run, run_id)
            call = {
                "id": approval["call_id"],
                "function": {"name": "file_write", "arguments": json.dumps(approval["arguments"])},
            }
            with pytest.raises(HarnessError) as error:
                await worker.execute_tool(
                    run, principal, call, run.messages, run.step, {"file_write"}
                )
            assert error.value.status_code == 403
            assert not list(tmp_path.glob("workspaces/*/result.txt"))
        finally:
            # 测试不启动这个 Worker，也不留下可被其他 Worker 认领的运行。
            async with service.store.sessions.begin() as session:
                run = await session.get(Run, run_id)
                run.status = "cancelled"
                run.lease_owner = None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["changed", "unreadable"])
async def test_permissions_after_real_replace_are_unknown_and_preserve_ledger(
    tmp_path, monkeypatch, failure
):
    from app.harness_tools import file_permissions, file_write

    gate = asyncio.Event()

    class PausedModel(ToolModel):
        async def chat(self, messages, **kwargs):
            if kwargs.get("tools"):
                await gate.wait()
            return await super().chat(messages, **kwargs)

    app = create_app(HarnessSettings(data_dir=tmp_path), PausedModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "post-publish-permission")
        created = (
            await client.post(
                "/api/v1/runs",
                headers={**headers, "Idempotency-Key": "post-permission"},
                json={"message": "写文件"},
            )
        ).json()
        run_id = created["id"]
        principal = await app.state.harness.authenticate(headers["Authorization"][7:])
        target = (
            app.state.harness.tool_executor.workspace.directory(principal, run_id) / "result.txt"
        )
        target.write_bytes("原正文".encode())
        gate.set()
        pending = await wait_run(client, headers, run_id, {"waiting_approval", "failed"})
        assert pending["status"] == "waiting_approval", pending
        approval = pending["approval"]
        assert approval["file_write"]["baseline"]["exists"] is True
        replace = file_write.os.replace
        inspect = file_permissions.permission_identity
        published = False

        def altered(source, destination):
            nonlocal published
            replace(source, destination)
            published = True
            if failure == "changed":
                if os.name == "nt":
                    _, protected = file_permissions.windows_security(target)
                    changed = subprocess.run(
                        [
                            "icacls",
                            str(target),
                            "/inheritance:e" if protected else "/inheritance:d",
                        ],
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    assert changed.returncode == 0, changed.stderr
                else:
                    target.chmod(0o400)
                assert inspect(target) != approval["file_write"]["baseline"]["permissions"]

        def unreadable(path):
            if published and path == target and failure == "unreadable":
                raise PermissionError("发布后的目标权限无法读取")
            return inspect(path)

        monkeypatch.setattr(file_write.os, "replace", altered)
        monkeypatch.setattr(file_permissions, "permission_identity", unreadable)
        assert (
            await client.post(
                f"/api/v1/runs/{run_id}/approval", headers=headers, json=approval_body(approval)
            )
        ).status_code == 200
        done = await wait_run(client, headers, run_id, {"completed", "failed", "interrupted"})
        assert done["status"] == "interrupted", done
        assert target.read_bytes() == "已批准".encode()
        async with app.state.harness.store.sessions() as session:
            tool = await session.scalar(select(ToolCall).where(ToolCall.run_id == run_id))
            assert tool.status == "started"
        assert await app.state.harness.store.claim("permission-replay") is None
