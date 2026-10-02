"""工具读写与归档导入的真实路径边界；仅操作临时目录。"""

import hashlib
import io
import os
import stat
import subprocess
import zipfile

import pytest

from app.harness import Principal
from app.harness_tools.project import ProjectManager
from app.harness_tools.workspace import Workspace, workspace_key

OWNER = Principal("boundary-owner", "boundary-tenant", "operator")
OTHER = Principal("boundary-other", "boundary-tenant", "operator")
INVALID_PATHS = (
    "",
    ".",
    "..",
    "./notes.txt",
    "dir/../notes.txt",
    "/absolute.txt",
    "C:/private",
    "C:notes.txt",
    "notes.txt:stream",
    "dir\\notes.txt",
    "\\\\server\\share\\file",
    ".git",
    ".GiT/config",
    ".git.",
    ".git ",
    "nested/.GiT./config",
    "notes.txt.",
    "notes.txt ",
    "dir./notes.txt",
    "dir /notes.txt",
    "CON",
    "con.txt",
    "NUL.txt",
    "AUX/notes.txt",
    "COM1.txt",
    "LPT9/notes.txt",
    "notes.txt/",
    "dir//notes.txt",
    "nul\x00.txt",
    "notes\x00.txt",
    "x" * 513,
)


def archive_bytes(*entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        for name, content in entries:
            entry = zipfile.ZipInfo()
            entry.filename = entry.orig_filename = name
            archive.writestr(entry, content)
    return data.getvalue()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_link(link, target):
    """Windows 使用无需管理员权限的 junction，其余平台使用符号链接。"""
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode:
            pytest.skip("当前环境不能创建 junction")
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("path", INVALID_PATHS)
def test_tool_resolve_rejects_invalid_paths_before_creating_workspace(tmp_path, path):
    workspace = Workspace(tmp_path)
    with pytest.raises(ValueError):
        workspace.resolve(OWNER, "run", path)
    assert not workspace.root.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="验证 Windows 尾点/空格的真实文件别名")
@pytest.mark.parametrize("path", [".git.", ".git ", ".GiT.", ".GiT "])
async def test_windows_git_alias_read_is_rejected(tmp_path, path):
    workspace = Workspace(tmp_path)
    root = workspace.directory(OWNER, "run")
    metadata = root / ".git"
    metadata.write_text("gitdir: 临时 Worktree 元数据", encoding="utf-8")
    before = file_hash(metadata)
    with pytest.raises(ValueError):
        await workspace.read(OWNER, "run", path)
    assert file_hash(metadata) == before


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="验证 Windows 尾点/空格的真实文件别名")
@pytest.mark.parametrize("path", [".git.", ".git ", ".GiT.", ".GiT "])
async def test_windows_git_alias_write_preserves_metadata(tmp_path, path):
    workspace = Workspace(tmp_path)
    root = workspace.directory(OWNER, "run")
    metadata = root / ".git"
    metadata.write_text("gitdir: 临时 Worktree 元数据", encoding="utf-8")
    before = file_hash(metadata)
    with pytest.raises(ValueError):
        await workspace.write(OWNER, "run", path, "恶意覆盖")
    assert file_hash(metadata) == before


@pytest.mark.parametrize("path", [path for path in INVALID_PATHS if path and path != "notes.txt/"])
def test_archive_rejects_invalid_paths(tmp_path, path):
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError):
        ProjectManager._extract(archive_bytes((path, "恶意内容")), destination)


@pytest.mark.skipif(os.name != "nt", reason="验证 Windows 归档路径别名覆盖")
@pytest.mark.parametrize("path", [".git.", ".git ", ".GiT.", ".GiT "])
def test_windows_archive_git_alias_preserves_metadata(tmp_path, path):
    metadata = tmp_path / ".git"
    metadata.write_text("gitdir: 临时 Worktree 元数据", encoding="utf-8")
    before = file_hash(metadata)
    with pytest.raises(ValueError):
        ProjectManager._extract(archive_bytes((path, "恶意覆盖")), tmp_path)
    assert file_hash(metadata) == before


@pytest.mark.asyncio
async def test_linked_workspace_cannot_access_other_workspace(tmp_path):
    workspace = Workspace(tmp_path)
    other_root = workspace.directory(OTHER, "run")
    secret = other_root / "说明.txt"
    secret.write_text("他人的私有正文", encoding="utf-8")
    before = file_hash(secret)
    linked_root = workspace.root / workspace_key(OWNER, "run")
    directory_link(linked_root, other_root)
    with pytest.raises(ValueError):
        workspace.directory(OWNER, "run")
    with pytest.raises(ValueError):
        await workspace.read(OWNER, "run", "说明.txt")
    with pytest.raises(ValueError):
        await workspace.write(OWNER, "run", "说明.txt", "恶意覆盖")
    assert file_hash(secret) == before


def test_linked_global_root_is_rejected_before_creating_workspace(tmp_path):
    workspace = Workspace(tmp_path)
    target = tmp_path / "target"
    target.mkdir()
    directory_link(workspace.root, target)
    with pytest.raises(ValueError):
        workspace.directory(OWNER, "run")
    assert not list(target.iterdir())


@pytest.mark.asyncio
async def test_middle_link_is_rejected_by_tools_and_archive(tmp_path):
    workspace = Workspace(tmp_path)
    root = workspace.directory(OWNER, "run")
    target = root / "正常目录"
    target.mkdir()
    secret = target / "说明.txt"
    secret.write_text("保留正文", encoding="utf-8")
    before = file_hash(secret)
    directory_link(root / "别名", target)
    with pytest.raises(ValueError):
        await workspace.read(OWNER, "run", "别名/说明.txt")
    with pytest.raises(ValueError):
        await workspace.write(OWNER, "run", "别名/新建.txt", "恶意内容")
    with pytest.raises(ValueError):
        ProjectManager._extract(archive_bytes(("别名/说明.txt", "恶意覆盖")), root)
    assert file_hash(secret) == before
    assert not (target / "新建.txt").exists()


def test_archive_rejects_linked_destination(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "destination"
    directory_link(destination, target)
    with pytest.raises(ValueError):
        ProjectManager._extract(archive_bytes(("说明.txt", "恶意内容")), destination)
    assert not (target / "说明.txt").exists()


@pytest.mark.asyncio
async def test_normal_nested_unicode_files_preserve_worktree_metadata(tmp_path):
    workspace = Workspace(tmp_path)
    root = workspace.directory(OWNER, "run")
    metadata = root / ".git"
    metadata.write_text("gitdir: 临时 Worktree 元数据", encoding="utf-8")
    before = file_hash(metadata)
    ProjectManager._extract(
        archive_bytes(("中文目录/", ""), ("中文目录/嵌套/说明.txt", "原始正文")), root
    )
    assert await workspace.read(OWNER, "run", "中文目录/嵌套/说明.txt") == "原始正文"
    await workspace.write(OWNER, "run", "中文目录/嵌套/说明.txt", "更新正文")
    assert await workspace.read(OWNER, "run", "中文目录/嵌套/说明.txt") == "更新正文"
    assert file_hash(metadata) == before


@pytest.mark.parametrize("name", ["链接", "链接目录/"])
def test_archive_does_not_extract_symbolic_links(tmp_path, name):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        entry = zipfile.ZipInfo(name)
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(entry, "../outside")
        archive.writestr("说明.txt", "正常正文")
    ProjectManager._extract(data.getvalue(), tmp_path)
    assert not (tmp_path / name).exists()
    assert (tmp_path / "说明.txt").read_text(encoding="utf-8") == "正常正文"
