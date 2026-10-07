"""差异审批的实际文件与独立进程边界；仅操作临时目录。"""

import asyncio
import hashlib
import importlib
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import pytest

from app.harness import Principal
from app.harness_tools.workspace import Workspace, workspace_key

OWNER = Principal("write-owner", "write-tenant", "operator")


def writer(workspace):
    assert importlib.util.find_spec("app.harness_tools.file_write"), "尚未实现冻结写入契约"
    return importlib.import_module("app.harness_tools.file_write").FileWriter(workspace)


async def freeze(workspace, path="说明.txt", content="新正文", call_id="call-write"):
    return await writer(workspace).freeze(
        OWNER, "run", {"path": path, "content": content}, call_id, "a" * 64
    )


async def write(workspace, contract, content="新正文"):
    return await writer(workspace).execute(
        OWNER, "run", {"path": contract["path"], "content": content}, contract
    )


@pytest.mark.asyncio
async def test_freeze_missing_workspace_does_not_create_target_directories(tmp_path):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace, "中文目录/说明.txt")
    assert not workspace.root.exists()
    assert contract["baseline"]["exists"] is False
    assert contract["baseline"]["missing_directories"]
    assert contract["workspace_key"] == workspace_key(OWNER, "run")
    assert contract["proposed_sha256"] == hashlib.sha256("新正文".encode()).hexdigest()
    assert len(contract["baseline_hash"]) == 64
    assert contract["preview"]["before"] == ""
    result = await write(workspace, contract)
    assert result["status"] == "written"
    assert (workspace.root / contract["workspace_key"] / contract["path"]).read_bytes() == (
        "新正文".encode()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "replace", "new", "parent"])
async def test_changed_baseline_never_overwrites_file(tmp_path, change):
    workspace = Workspace(tmp_path)
    root = workspace.directory(OWNER, "run")
    parent = root / "nested"
    parent.mkdir()
    target = parent / "说明.txt"
    if change != "new":
        target.write_text("原正文", encoding="utf-8")
    contract = await freeze(workspace, "nested/说明.txt")
    if change == "replace":
        target.unlink()
        target.write_text("原正文", encoding="utf-8")
    elif change == "parent":
        parent.rename(root / "old")
        parent.mkdir()
        target.write_text("原正文", encoding="utf-8")
    else:
        target.write_text("外部修改", encoding="utf-8")
    before = target.read_bytes()
    assert (await write(workspace, contract))["code"] == "baseline_conflict"
    assert target.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw", [b"\xff", b"binary\x00", b"x" * 1_000_001], ids=["encoding", "binary", "oversize"]
)
async def test_freeze_rejects_nontext_and_oversize(tmp_path, raw):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes(raw)
    with pytest.raises(ValueError):
        await freeze(workspace)
    assert target.read_bytes() == raw


@pytest.mark.asyncio
async def test_freeze_rejects_hardlink(tmp_path):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_text("保留", encoding="utf-8")
    os.link(target, tmp_path / "outside.txt")
    with pytest.raises(ValueError):
        await freeze(workspace)
    assert target.read_text(encoding="utf-8") == "保留"


@pytest.mark.asyncio
async def test_hardlink_added_during_snapshot_is_rejected(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes("保留".encode())
    module = importlib.import_module("app.harness_tools.file_write")
    original = module.os.fstat
    watched_inode = target.stat().st_ino
    checked = 0

    def race(descriptor):
        nonlocal checked
        info = original(descriptor)
        if info.st_ino == watched_inode:
            checked += 1
            if checked == 2:
                os.link(target, tmp_path / "outside.txt")
                info = original(descriptor)
        return info

    monkeypatch.setattr(module.os, "fstat", race)
    with pytest.raises(ValueError):
        await freeze(workspace)
    assert target.read_bytes() == "保留".encode()


@pytest.mark.asyncio
async def test_preview_and_diff_input_are_bounded(tmp_path):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_text("原" * 200_000 + "\n" * 20_000, encoding="utf-8")
    contract = await freeze(workspace, content="新" * 12_000)
    preview = contract["preview"]
    assert len(preview["before"]) <= 8192
    assert len(preview["after"]) <= 8192
    assert len(preview["diff"]) <= 16384
    assert preview["truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("旧正文", "新正文", ["-旧正文", "+新正文"]),
        ("相同正文", "相同正文\n", ["-相同正文", "+相同正文"]),
        ("相同正文\n", "相同正文", ["-相同正文", "+相同正文"]),
    ],
    ids=["single-line", "add-ending-newline", "remove-ending-newline"],
)
async def test_diff_preserves_line_boundaries_and_ending_newline(tmp_path, before, after, expected):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes(before.encode())
    contract = await freeze(workspace, content=after)
    diff = contract["preview"]["diff"]
    assert all(line in diff.splitlines() for line in expected), diff
    assert "\\ No newline at end of file" in diff
    assert contract["preview"]["before"] == before
    assert contract["preview"]["after"] == after
    assert contract["baseline"]["sha256"] == hashlib.sha256(before.encode()).hexdigest()
    assert contract["proposed_sha256"] == hashlib.sha256(after.encode()).hexdigest()


@pytest.mark.asyncio
async def test_context_tampering_and_wrong_owner_cannot_write(tmp_path):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    with pytest.raises(ValueError):
        await write(workspace, contract, content="篡改")
    other = Principal("other", OWNER.tenant_id, "operator")
    with pytest.raises(ValueError):
        await writer(workspace).execute(
            other, "run", {"path": "说明.txt", "content": "新正文"}, contract
        )
    assert not workspace.root.exists()


def hold_lock(data_dir, key, ready, release):
    from app.harness_tools.workspace_lock import WorkspaceLock

    with WorkspaceLock(Path(data_dir), key):
        ready.set()
        release.wait(10)


@pytest.mark.asyncio
async def test_lock_timeout_is_cross_process_and_stable(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    module = importlib.import_module("app.harness_tools.workspace_lock")
    monkeypatch.setattr(module, "LOCK_TIMEOUT", 0.15)
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    process = ctx.Process(
        target=hold_lock, args=(tmp_path, contract["workspace_key"], ready, release)
    )
    process.start()
    try:
        assert await asyncio.to_thread(ready.wait, 10)
        before = time.monotonic()
        result = await write(workspace, contract)
        assert result["code"] == "lock_timeout"
        assert time.monotonic() - before < 2
        assert not workspace.root.exists()
    finally:
        release.set()
        await asyncio.to_thread(process.join, 10)
        if process.is_alive():
            process.kill()
    assert process.exitcode == 0
    assert (await write(workspace, contract))["status"] == "written"


@pytest.mark.asyncio
async def test_permission_copy_failure_never_publishes(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes("保留".encode())
    contract = await freeze(workspace)
    module = importlib.import_module("app.harness_tools.file_permissions")

    def denied(*args):
        raise PermissionError("测试权限复制失败")

    monkeypatch.setattr(module, "preserve_permissions", denied)
    result = await write(workspace, contract)
    assert result["code"] == "write_rejected"
    assert target.read_bytes() == "保留".encode()
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_definite_prepublication_rejection(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes("保留".encode())
    unrelated = target.parent / "unrelated.txt"
    unrelated.write_bytes(b"untouched")
    contract = await freeze(workspace)
    permissions = importlib.import_module("app.harness_tools.file_permissions")
    unlink = Path.unlink

    def denied(*args):
        raise PermissionError("测试权限复制失败")

    def cleanup_denied(path, *args, **kwargs):
        if path.name.startswith(".aegis-write-"):
            raise PermissionError("测试临时文件清理失败")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(permissions, "preserve_permissions", denied)
    monkeypatch.setattr(Path, "unlink", cleanup_denied)
    result = await write(workspace, contract)
    assert result["code"] == "write_rejected"
    assert "清理失败" in result["error"]
    assert target.read_bytes() == "保留".encode()
    assert unrelated.read_bytes() == b"untouched"


@pytest.mark.asyncio
async def test_unlock_failure_does_not_hide_definite_rejection(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes("保留".encode())
    module = importlib.import_module("app.harness_tools.workspace_lock")
    unlock = module.WorkspaceLock.__exit__

    def failed(self, *args):
        unlock(self, *args)
        raise OSError("测试解锁失败")

    monkeypatch.setattr(module.WorkspaceLock, "__exit__", failed)
    result = await write(workspace, contract)
    assert result["status"] == "rejected"
    assert "解锁失败" in result["error"]
    assert target.read_bytes() == "保留".encode()


@pytest.mark.asyncio
async def test_unlock_failure_after_publish_remains_unknown(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    module = importlib.import_module("app.harness_tools.workspace_lock")
    unlock = module.WorkspaceLock.__exit__

    def failed(self, *args):
        unlock(self, *args)
        raise OSError("测试解锁失败")

    monkeypatch.setattr(module.WorkspaceLock, "__exit__", failed)
    with pytest.raises(OSError, match="解锁失败"):
        await write(workspace, contract)
    assert next(workspace.root.rglob("*.txt")).read_bytes() == "新正文".encode()


@pytest.mark.asyncio
async def test_timeout_after_publish_is_unknown_not_lock_timeout(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_text("原正文", encoding="utf-8")
    contract = await freeze(workspace)
    module = importlib.import_module("app.harness_tools.file_write")
    replace = module.os.replace

    def uncertain(source, destination):
        replace(source, destination)
        raise TimeoutError("发布后结果丢失")

    monkeypatch.setattr(module.os, "replace", uncertain)
    with pytest.raises(TimeoutError, match="发布后"):
        await write(workspace, contract)
    assert target.read_bytes() == "新正文".encode()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="验证真实 Windows DACL")
@pytest.mark.parametrize("inheritance", ["d", "e"], ids=["protected", "inherited"])
async def test_windows_acl_is_preserved_on_atomic_replace(tmp_path, inheritance):
    import getpass

    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes("原正文".encode())
    changed = subprocess.run(
        [
            "icacls",
            str(target),
            f"/inheritance:{inheritance}",
            "/grant:r",
            getpass.getuser() + ":(R,W)",
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert changed.returncode == 0, changed.stderr
    module = importlib.import_module("app.harness_tools.file_permissions")
    original = module.permission_identity(target)
    assert original["protected"] is (inheritance == "d")
    sample = target.parent / "permission-sample.txt"
    sample.write_bytes(b"sample")
    descriptor, protected = module.windows_security(target)
    module.set_windows_security(sample, descriptor, protected)
    assert module.permission_identity(sample) == original, (
        descriptor.hex(),
        module.windows_security(sample)[0].hex(),
    )
    sample.unlink()
    contract = await freeze(workspace)
    result = await write(workspace, contract)
    assert result["status"] == "written", result
    assert module.permission_identity(target) == original


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="验证 POSIX mode")
async def test_posix_mode_is_preserved_on_atomic_replace(tmp_path):
    workspace = Workspace(tmp_path)
    target = workspace.directory(OWNER, "run") / "说明.txt"
    target.write_bytes("原正文".encode())
    target.chmod(0o640)
    contract = await freeze(workspace)
    assert (await write(workspace, contract))["status"] == "written"
    assert target.stat().st_mode & 0o777 == 0o640


@pytest.mark.asyncio
async def test_hardlinked_lock_file_is_rejected_before_target_write(tmp_path):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    lock_path = tmp_path / "workspace-locks" / (contract["workspace_key"] + ".lock")
    os.link(lock_path, tmp_path / "outside.lock")
    result = await write(workspace, contract)
    assert result["status"] == "rejected"
    assert not workspace.root.exists()


@pytest.mark.asyncio
async def test_freeze_records_existing_data_directory_identity(tmp_path):
    contract = await freeze(Workspace(tmp_path))
    assert contract["baseline"]["ancestors"], "冻结必须记录已存在的数据目录祖先"


@pytest.mark.asyncio
async def test_new_file_race_does_not_clobber_external_content(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    module = importlib.import_module("app.harness_tools.file_write")
    method = "rename" if os.name == "nt" else "link"
    original = getattr(module.os, method)

    def race(source, target, **kwargs):
        Path(target).write_bytes("外部竞态".encode())
        return original(source, target, **kwargs)

    monkeypatch.setattr(module.os, method, race)
    result = await write(workspace, contract)
    assert result["code"] == "baseline_conflict"
    target = workspace.root / contract["workspace_key"] / contract["path"]
    assert target.read_bytes() == "外部竞态".encode()
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.asyncio
async def test_unsupported_new_file_publish_is_definite_rejection(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    module = importlib.import_module("app.harness_tools.file_write")

    def unavailable(*args, **kwargs):
        raise NotImplementedError("当前平台没有原子无覆盖发布")

    monkeypatch.setattr(module.os, "rename" if os.name == "nt" else "link", unavailable)
    result = await write(workspace, contract)
    assert result["code"] == "write_rejected"
    assert not list(workspace.root.rglob("*.txt"))


def process_write(data_dir, contract, release, result_queue):
    workspace = Workspace(Path(data_dir))
    release.wait(10)
    result = asyncio.run(write(workspace, contract))
    result_queue.put(result)


@pytest.mark.asyncio
async def test_two_processes_using_same_baseline_have_only_one_winner(tmp_path):
    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    lock_path = tmp_path / "workspace-locks" / (contract["workspace_key"] + ".lock")
    before = lock_path.stat().st_ino
    ctx = multiprocessing.get_context("spawn")
    release, results = ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(target=process_write, args=(tmp_path, contract, release, results))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    release.set()
    try:
        outcomes = [await asyncio.to_thread(results.get, True, 15) for _ in range(2)]
        assert sorted(item["status"] for item in outcomes) == ["rejected", "written"]
        assert next(item for item in outcomes if item["status"] == "rejected")["code"] == (
            "baseline_conflict"
        )
    finally:
        for process in processes:
            await asyncio.to_thread(process.join, 10)
            if process.is_alive():
                process.kill()
        results.close()
    assert all(process.exitcode == 0 for process in processes)
    assert lock_path.stat().st_ino == before
    assert next(workspace.root.rglob("*.txt")).read_bytes() == "新正文".encode()


@pytest.mark.asyncio
async def test_linked_lock_directory_is_rejected(tmp_path):
    from tests.test_workspace_tool_boundary import directory_link

    workspace = Workspace(tmp_path)
    contract = await freeze(workspace)
    lock_directory = tmp_path / "workspace-locks"
    lock_directory.rename(tmp_path / "old-locks")
    outside = tmp_path / "outside"
    outside.mkdir()
    directory_link(lock_directory, outside)
    result = await write(workspace, contract)
    assert result["code"] == "write_rejected"
    assert not list(outside.iterdir())
    assert not workspace.root.exists()
