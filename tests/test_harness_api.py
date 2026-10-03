"""真实 HTTP 边界与 SQLite 验证，模型为确定性测试替身。"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.harness.settings import HarnessSettings
from app.main import create_app


class ToolModel:
    async def chat(self, messages, **kwargs):
        if not kwargs.get("tools"):
            return SimpleNamespace(
                content='{"risk":"review","reason":"测试审查"}', model_id="test", usage={}, raw={}
            )
        calls = []
        if not any(item.get("role") == "tool" for item in messages):
            task = next(item["content"] for item in reversed(messages) if item["role"] == "user")
            name = "file_write" if "写文件" in task else "calculator"
            arguments = (
                {"path": "result.txt", "content": "已批准"}
                if name == "file_write"
                else {"expression": "2+3"}
            )
            calls = [
                {
                    "id": "call-test",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        return SimpleNamespace(
            content="计算结果为5" if not calls else "",
            model_id="test",
            usage={"total_tokens": 1},
            raw={"choices": [{"message": {"tool_calls": calls}}]},
        )


async def account(client, name):
    credentials = {"username": name, "password": "test-password-123"}
    assert (await client.post("/api/v1/auth/register", json=credentials)).status_code == 201
    result = await client.post("/api/v1/auth/login", json=credentials)
    assert result.status_code == 200
    return {"Authorization": "Bearer " + result.json()["token"]}


async def wait_run(client, headers, run_id, statuses, timeout=10):
    async with asyncio.timeout(timeout):
        while True:
            result = (await client.get(f"/api/v1/runs/{run_id}", headers=headers)).json()
            if result["status"] in statuses:
                return result
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_auth_scope_and_sse_reconnect(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        assert (await client.get("/api/v1/runs")).status_code == 401
        alice = await account(client, "api-alice")
        assert (await client.get("/api/v1/capabilities", headers=alice)).status_code == 200
        bob = await account(client, "api-bob")
        headers = {**alice, "Idempotency-Key": "same-request"}
        created = await client.post("/api/v1/runs", headers=headers, json={"message": "计算"})
        assert created.status_code == 202
        run_id = created.json()["id"]
        assert (
            await client.post("/api/v1/runs", headers=headers, json={"message": "计算"})
        ).json()["id"] == run_id
        assert (
            await client.post("/api/v1/runs", headers=headers, json={"message": "其他"})
        ).status_code == 409
        assert (await client.get(f"/api/v1/runs/{run_id}", headers=bob)).status_code == 404
        completed = await wait_run(client, alice, run_id, {"completed", "failed"})
        assert completed["status"] == "completed", completed
        stream = await client.get(f"/api/v1/runs/{run_id}/events", headers=alice)
        ids = [int(line[4:]) for line in stream.text.splitlines() if line.startswith("id: ")]
        assert ids == sorted(set(ids)) and len(ids) >= 3
        resumed = await client.get(
            f"/api/v1/runs/{run_id}/events", headers={**alice, "Last-Event-ID": str(ids[-2])}
        )
        assert f"id: {ids[-1]}\n" in resumed.text
        assert f"id: {ids[0]}\n" not in resumed.text
        assert (
            (await client.get("/"))
            .headers["content-security-policy"]
            .startswith("default-src 'self'")
        )
        assert (await client.post("/api/v1/auth/logout", headers=alice)).status_code == 200
        assert (await client.get("/api/v1/runs", headers=alice)).status_code == 401


@pytest.mark.asyncio
async def test_approval_upload_memory_and_restore(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "api-owner")
        created = await client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "write"},
            json={"message": "写文件"},
        )
        run_id = created.json()["id"]
        pending = await wait_run(client, headers, run_id, {"waiting_approval", "failed"})
        assert pending["status"] == "waiting_approval", pending
        assert not list(tmp_path.glob("workspaces/*/result.txt"))
        assert (
            await client.post(
                f"/api/v1/runs/{run_id}/approval", headers=headers, json={"approved": True}
            )
        ).status_code == 422
        body = {"approved": True, "call_id": pending["approval"]["call_id"], "args_hash": "0" * 64}
        assert (
            await client.post(f"/api/v1/runs/{run_id}/approval", headers=headers, json=body)
        ).status_code == 409
        body["args_hash"] = pending["approval"]["hash"]
        assert (
            await client.post(f"/api/v1/runs/{run_id}/approval", headers=headers, json=body)
        ).status_code == 200
        assert (await wait_run(client, headers, run_id, {"completed", "failed"}))[
            "status"
        ] == "completed"
        assert len(list(tmp_path.glob("workspaces/*/result.txt"))) == 1
        upload = await client.post(
            "/api/v1/documents/upload",
            headers=headers,
            files={"file": ("../notes.md", "隔离沙箱只能使用临时文件".encode(), "text/markdown")},
        )
        assert upload.status_code == 201, upload.text
        assert upload.json()["filename"] == "notes.md"
        principal = await app.state.harness.authenticate(headers["Authorization"][7:])
        knowledge = await app.state.harness.tool_executor.execute(
            "knowledge_search", {"query": "隔离沙箱"}, principal, run_id
        )
        assert knowledge["results"] and knowledge["backend"] == "bm25"
        asset = (
            await client.post(
                "/api/v1/assets",
                headers=headers,
                json={"kind": "profile", "name": "语言", "content": "中文", "status": "active"},
            )
        ).json()
        changed = await client.put(
            f"/api/v1/assets/{asset['id']}",
            headers=headers,
            json={"kind": "profile", "name": "语言", "content": "简洁中文"},
        )
        assert changed.status_code == 200
        restored = await client.post(
            f"/api/v1/assets/{asset['id']}/restore", headers=headers, json={"version": 1}
        )
        assert restored.status_code == 200 and restored.json()["content"] == "中文"
        me = (await client.get("/api/v1/auth/me", headers=headers)).json()
        assert any(memory["id"] == asset["id"] for memory in me["bootstrap"]["memories"])


@pytest.mark.asyncio
async def test_upload_initialization_does_not_block_api(tmp_path, monkeypatch):
    from app.api.routes import harness

    started = asyncio.Event()
    release = asyncio.Event()
    original_parse = harness.parse_isolated
    observed_mime = []

    async def slow_parse(raw, filename, mime_type):
        started.set()
        await release.wait()
        observed_mime.append(mime_type)
        return await original_parse(raw, filename, mime_type)

    monkeypatch.setattr(harness, "parse_isolated", slow_parse)
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "upload-cold-start")
        upload = asyncio.create_task(
            client.post(
                "/api/v1/documents/upload",
                headers=headers,
                files={"file": ("NOTES.MD", "中文文档".encode(), "application/pdf")},
            )
        )
        try:
            await asyncio.wait_for(started.wait(), 1)
            response = await asyncio.wait_for(client.get("/api/v1/auth/me", headers=headers), 0.5)
            assert response.status_code == 200
        finally:
            release.set()
            result = await upload
        assert result.status_code == 201, result.text
        assert observed_mime == ["text/plain"], "使用受控后缀，不信客户端 MIME"
        assert app.state.etl_limit._value == 2


@pytest.mark.asyncio
async def test_upload_retry_reuses_document_after_index_failure(tmp_path, monkeypatch):
    app = create_app(HarnessSettings(data_dir=tmp_path), ToolModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = {**await account(client, "upload-retry"), "Idempotency-Key": "document-request"}
        knowledge = app.state.harness.tool_executor.knowledge
        original_index = knowledge.index

        async def failed_index(*args):
            raise RuntimeError("验收索引异常")

        monkeypatch.setattr(knowledge, "index", failed_index)
        files = {"file": ("notes.md", "幂等文档".encode(), "text/markdown")}
        first = await client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert first.status_code == 201 and first.json()["indexing"]["status"] == "pending"
        monkeypatch.setattr(knowledge, "index", original_index)
        second = await client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert second.status_code == 201 and second.json()["reused"]
        assert first.json()["document_id"] == second.json()["document_id"]
        assert len((await client.get("/api/v1/documents", headers=headers)).json()) == 1
        conflict = await client.post(
            "/api/v1/documents/upload",
            headers=headers,
            files={
                "file": ("notes.md", b"different", "text/plain"),
            },
        )
        assert conflict.status_code == 409
        document_id = first.json()["document_id"]
        principal = await app.state.harness.authenticate(headers["Authorization"][7:])
        await app.state.harness.put_asset(
            principal,
            "document",
            "notes.md",
            "更新的正文",
            status="active",
            asset_id=document_id,
        )
        retried = await client.post("/api/v1/documents/upload", headers=headers, files=files)
        assert retried.status_code == 201 and retried.json()["indexing"]["warnings"]
        assert (await client.get(f"/api/v1/assets/{document_id}", headers=headers)).json()[
            "content"
        ] == "更新的正文"
