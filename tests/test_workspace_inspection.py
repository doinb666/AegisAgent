"""工作台查询与文件预览的真实 SQLite、HTTP 和文件系统边界。"""

import asyncio
import os
import subprocess
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from app.harness.models import Asset, Run
from app.harness.settings import HarnessSettings
from app.harness_tools.workspace import workspace_key
from app.main import create_app


class QuietModel:
    async def chat(self, *args, **kwargs):
        return SimpleNamespace(content="测试", model_id="test", usage={}, raw={})


@pytest_asyncio.fixture
async def inspection(tmp_path):
    app = create_app(HarnessSettings(data_dir=tmp_path, evolution_enabled=False), QuietModel())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        service = app.state.harness
        admin = await service.register("inspection-owner", "test-password")
        from app.harness.errors import Principal

        owner = Principal(admin["user_id"], admin["tenant_id"], "admin")
        member = await service.create_user(owner, "inspection-viewer", "test-password", "viewer")
        outsider = await service.register("inspection-outsider", "test-password")
        principals = {
            "owner": owner,
            "viewer": Principal(member["user_id"], member["tenant_id"], "viewer"),
            "outsider": Principal(outsider["user_id"], outsider["tenant_id"], "admin"),
        }
        headers = {}
        for role, username in (
            ("owner", "inspection-owner"),
            ("viewer", "inspection-viewer"),
            ("outsider", "inspection-outsider"),
        ):
            token = (await service.login(username, "test-password"))["token"]
            headers[role] = {"Authorization": "Bearer " + token}
        yield SimpleNamespace(
            app=app,
            service=service,
            client=client,
            principals=principals,
            headers=headers,
            tmp_path=tmp_path,
        )


async def seed_run(env, identifier, role="owner", message="测试", status="completed", created=10):
    principal = env.principals[role]
    async with env.service.store.sessions.begin() as session:
        session.add(
            Run(
                id=identifier,
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                idempotency_key=identifier,
                payload_hash="0" * 64,
                session_id=identifier,
                message=message,
                status=status,
                trace_id=identifier,
                created=created,
            )
        )
    return env.tmp_path / "workspaces" / workspace_key(principal, identifier)


async def get(env, endpoint, role="owner", **params):
    return await env.client.get(endpoint, headers=env.headers[role], params=params)


@pytest.mark.asyncio
async def test_private_list_literal_search_and_stable_cursor(inspection):
    env = inspection
    await seed_run(env, "a", message="百分比%_字面", created=20)
    await seed_run(env, "b", message="百分比XY字面", created=20)
    await seed_run(env, "c", message="百分比%_字面", status="failed", created=20)
    await seed_run(env, "old", created=10)
    await seed_run(env, "foreign", role="viewer", created=30)
    await seed_run(env, "other-tenant", role="outsider", created=30)
    async with env.service.store.sessions.begin() as session:
        child = await session.get(Run, "a")
        child.parent_run_id = "b"
    response = await get(env, "/api/v1/runs", limit=2)
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["c", "b"]
    assert all(row["created"] == 20 for row in response.json())
    assert [row["id"] for row in (await get(env, "/api/v1/runs", before="b")).json()] == [
        "a",
        "old",
    ]
    assert [row["id"] for row in (await get(env, "/api/v1/runs", query="%_")).json()] == ["c", "a"]
    assert [
        row["id"] for row in (await get(env, "/api/v1/runs", query="%_", status="completed")).json()
    ] == ["a"]
    for cursor in ("foreign", "other-tenant", "missing"):
        assert (await get(env, "/api/v1/runs", before=cursor)).status_code == 404
    assert len((await get(env, "/api/v1/runs", role="viewer")).json()) == 1


@pytest.mark.asyncio
async def test_query_validation_auth_and_default_limit(inspection):
    env = inspection
    async with env.service.store.sessions.begin() as session:
        principal = env.principals["owner"]
        session.add_all(
            [
                Run(
                    id=f"r{i:03}",
                    tenant_id=principal.tenant_id,
                    owner_id=principal.user_id,
                    idempotency_key=f"r{i:03}",
                    payload_hash="0" * 64,
                    session_id="s",
                    message="列表",
                    status="completed",
                    trace_id="trace",
                    created=i,
                )
                for i in range(105)
            ]
        )
    assert len((await get(env, "/api/v1/runs")).json()) == 100
    for params in (
        {"limit": 0},
        {"limit": 101},
        {"limit": "abc"},
        {"status": "unknown"},
        {"query": "x" * 201},
        {"before": "x" * 129},
    ):
        assert (await get(env, "/api/v1/runs", **params)).status_code == 422
    for status in (
        "queued",
        "running",
        "waiting_approval",
        "completed",
        "failed",
        "cancelled",
        "interrupted",
    ):
        assert (await get(env, "/api/v1/runs", status=status)).status_code == 200
    for endpoint in (
        "/api/v1/runs",
        "/api/v1/workspace/overview",
        "/api/v1/runs/missing/files",
        "/api/v1/runs/missing/file?path=notes.txt",
    ):
        assert (await env.client.get(endpoint)).status_code == 401


@pytest.mark.asyncio
async def test_real_overview_scope_without_list_limit(inspection):
    env = inspection
    await seed_run(env, "done")
    await seed_run(env, "failed", status="failed")
    await seed_run(env, "viewer-run", role="viewer")
    await seed_run(env, "outside-run", role="outsider")
    async with env.service.store.sessions.begin() as session:
        for i in range(105):
            principal = env.principals["owner"]
            session.add(
                Asset(
                    id=f"asset{i}",
                    tenant_id=principal.tenant_id,
                    owner_id=principal.user_id,
                    kind="memory" if i < 103 else "document",
                    name="资产",
                    content="正文",
                    status="active" if i < 104 else "draft",
                    version=1,
                )
            )
        for role in ("viewer", "outsider"):
            principal = env.principals[role]
            session.add(
                Asset(
                    id=role,
                    tenant_id=principal.tenant_id,
                    owner_id=principal.user_id,
                    kind="skill",
                    name="私有",
                    content="正文",
                    status="retired",
                )
            )
    response = await get(env, "/api/v1/workspace/overview")
    assert response.status_code == 200
    assert response.json() == {
        "runs": {"total": 2, "by_status": {"completed": 1, "failed": 1}},
        "assets": {
            "total": 105,
            "by_kind": {"memory": 103, "document": 2},
            "by_status": {"active": 104, "draft": 1},
        },
    }
    overview = (await get(env, "/api/v1/workspace/overview", role="viewer")).json()
    assert overview["runs"]["total"] == 1 and overview["assets"]["total"] == 1


@pytest.mark.asyncio
async def test_files_empty_scope_and_viewer_read_only(inspection):
    env = inspection
    root = await seed_run(env, "empty")
    assert not root.exists()
    assert (await get(env, "/api/v1/runs/empty/files")).json() == {"files": [], "truncated": False}
    assert (await get(env, "/api/v1/runs/empty/file", path="missing.txt")).status_code == 404
    assert not root.exists() and not root.parent.exists()
    for role in ("viewer", "outsider"):
        for endpoint in ("files", "file"):
            assert (
                await get(env, f"/api/v1/runs/empty/{endpoint}", role=role, path="../private")
            ).status_code == 404
    viewer_root = await seed_run(env, "viewer-own", role="viewer")
    viewer_root.mkdir(parents=True)
    (viewer_root / "notes.txt").write_text("只读中文", encoding="utf8")
    result = await get(env, "/api/v1/runs/viewer-own/file", role="viewer", path="notes.txt")
    assert result.status_code == 200 and result.json()["content"] == "只读中文"
    assert (await get(env, "/api/v1/runs/viewer-own/files", role="viewer")).json() == {
        "files": [{"path": "notes.txt", "bytes": 12}],
        "truncated": False,
    }


@pytest.mark.asyncio
async def test_preview_path_and_content_boundaries(inspection):
    env = inspection
    root = await seed_run(env, "preview")
    root.mkdir(parents=True)
    (root / "notes.txt").write_text("正文", encoding="utf8")
    (root / "nul.bin").write_bytes(b"hello\x00world")
    (root / "invalid.bin").write_bytes(b"\xff\xfe")
    (root / "late.bin").write_bytes(b"a" * 65536 + b"\x00")
    (root / "huge.txt").write_bytes(b"x" * 1_000_001)
    raw = b"a" * 65535 + "中".encode() + b"tail"
    (root / "unicode.txt").write_bytes(raw)
    result = await get(env, "/api/v1/runs/preview/file", path="unicode.txt")
    assert result.status_code == 200
    assert result.json() == {
        "path": "unicode.txt",
        "content": "a" * 65535,
        "bytes": len(raw),
        "truncated": True,
    }
    for path in (
        "",
        ".",
        "..",
        "./notes.txt",
        "dir/../notes.txt",
        "/etc/passwd",
        "C:/private",
        "C:notes.txt",
        "notes.txt:stream",
        "dir\\notes.txt",
        "\\\\server\\share\\file",
        ".git/config",
        ".GIT/config",
        "notes.txt/",
        "x" * 513,
        "CON",
        "NUL.txt",
        "AUX/notes.txt",
        "notes.txt.",
        "notes.txt ",
        "nul.bin",
        "invalid.bin",
        "late.bin",
        "huge.txt",
    ):
        response = await get(env, "/api/v1/runs/preview/file", path=path)
        assert response.status_code == 422, (path, response.text)
        assert str(env.tmp_path) not in response.text
    assert (await get(env, "/api/v1/runs/preview/file", path="missing.txt")).status_code == 404


@pytest.mark.asyncio
async def test_file_scan_budgets_depth_and_hidden_git(inspection):
    env = inspection
    root = await seed_run(env, "budget")
    root.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".git" / "secret").write_text("不展示", encoding="utf8")
    for i in range(205):
        (root / f"file{i:03}.txt").write_text("x", encoding="utf8")
    listing = (await get(env, "/api/v1/runs/budget/files")).json()
    assert len(listing["files"]) == 200 and listing["truncated"]
    assert all(not row["path"].startswith(".git") for row in listing["files"])
    exact = await seed_run(env, "exact")
    exact.mkdir(parents=True)
    for i in range(200):
        (exact / f"file{i:03}.txt").write_text("x", encoding="utf8")
    assert not (await get(env, "/api/v1/runs/exact/files")).json()["truncated"]
    depth_root = await seed_run(env, "depth")
    target = depth_root
    for _ in range(9):
        target /= "dir"
        target.mkdir(parents=True)
        (target / "notes.txt").write_text("x", encoding="utf8")
    listing = (await get(env, "/api/v1/runs/depth/files")).json()
    assert listing["truncated"] and listing["files"]
    assert all(len(row["path"].split("/")) <= 8 for row in listing["files"])
    scans = await seed_run(env, "scans")
    scans.mkdir(parents=True)
    for i in range(1005):
        (scans / f"dir{i:04}").mkdir()
    assert (await get(env, "/api/v1/runs/scans/files")).json() == {"files": [], "truncated": True}


@pytest.mark.asyncio
async def test_links_are_not_followed(inspection):
    env = inspection
    root = await seed_run(env, "links")
    root.mkdir(parents=True)
    outside = env.tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("秘密", encoding="utf8")
    linked = False
    try:
        (root / "linked.txt").symlink_to(outside / "secret.txt")
        (root / "linked-dir").symlink_to(outside, target_is_directory=True)
        linked = True
    except OSError:
        pass
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(root / "junction"), str(outside)],
            capture_output=True,
            check=False,
        )
        linked = linked or result.returncode == 0
        if result.returncode == 0:
            response = await get(env, "/api/v1/runs/links/file", path="junction/secret.txt")
            assert response.status_code == 422
    if not linked:
        pytest.skip("当前环境不允许创建符号链接或 junction")
    listing = (await get(env, "/api/v1/runs/links/files")).json()
    assert listing["files"] == []
    if (root / "linked.txt").is_symlink():
        assert (await get(env, "/api/v1/runs/links/file", path="linked.txt")).status_code == 422


@pytest.mark.asyncio
async def test_file_inspection_does_not_block_event_loop(inspection, monkeypatch):
    env = inspection
    root = await seed_run(env, "thread")
    root.mkdir(parents=True)
    (root / "notes.txt").write_text("正文", encoding="utf8")
    workspace = env.service.tool_executor.workspace
    original = workspace._preview_file
    import threading

    started = threading.Event()
    release = threading.Event()

    def slow_preview(*args):
        started.set()
        release.wait(timeout=2)
        return original(*args)

    monkeypatch.setattr(workspace, "_preview_file", slow_preview)
    task = asyncio.create_task(get(env, "/api/v1/runs/thread/file", path="notes.txt"))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        response = await asyncio.wait_for(get(env, "/api/v1/runs"), timeout=0.5)
        assert response.status_code == 200
    finally:
        release.set()
        result = await task
    assert result.status_code == 200
