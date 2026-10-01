"""真实本地工具边界；临时文件与仓库均不执行用户代码。"""

import asyncio
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from jsonschema import ValidationError

from app.core.tools.builtin.calculator import CalculatorTool
from app.harness import HarnessSettings, Principal
from app.harness_tools.catalog import HarnessTools
from app.harness_tools.project import ProjectManager
from app.harness_tools.sandbox import SandboxClient
from app.harness_tools.workspace import Workspace

OWNER = Principal("test-owner", "test-tenant", "operator")
VIEWER = Principal("test-viewer", "test-tenant", "viewer")


@pytest.mark.asyncio
async def test_mcp_credentials_require_explicit_tenant_or_user_scope(tmp_path, monkeypatch):
    servers = [
        {
            "name": "private",
            "url": "https://example.invalid/mcp",
            "tools": ["search"],
            "tenant_ids": [OWNER.tenant_id],
            "user_ids": [OWNER.user_id],
        },
        {"name": "unbound", "url": "https://example.invalid/mcp", "tools": ["search"]},
    ]
    tools = HarnessTools(HarnessSettings(data_dir=tmp_path, mcp_servers_json=json.dumps(servers)))
    assert tools.mcp.authorized_servers(OWNER) == ["private"]
    outsiders = [
        VIEWER,
        Principal("another", OWNER.tenant_id, "operator"),
        Principal(OWNER.user_id, "another-tenant", "admin"),
    ]
    for principal in outsiders:
        assert not tools.mcp.authorized_servers(principal)
        assert all(
            not item["function"]["name"].startswith("mcp_") for item in tools.catalog(principal)
        )
        with pytest.raises(PermissionError):
            await tools.execute("mcp_list", {"server": "private"}, principal, "run")
    calls = []

    async def listed(server):
        calls.append(server)
        return [{"name": "search"}]

    monkeypatch.setattr(tools.mcp, "list_tools", listed)
    assert await tools.execute("mcp_list", {"server": "private"}, OWNER, "run") == [
        {"name": "search"}
    ]
    with pytest.raises(ValidationError):
        await tools.execute("mcp_list", {"server": "unbound"}, OWNER, "run")
    assert calls == ["private"]


@pytest.mark.asyncio
async def test_workspace_file_roundtrip_and_scope(tmp_path):
    workspace = Workspace(tmp_path)
    result = await workspace.write(OWNER, "run-one", "nested/说明.txt", "仅本人可读")
    assert result["bytes"] == len("仅本人可读".encode())
    assert await workspace.read(OWNER, "run-one", "nested/说明.txt") == "仅本人可读"
    others = [
        Principal("other-owner", OWNER.tenant_id, "operator"),
        Principal(OWNER.user_id, "other-tenant", "operator"),
    ]
    for principal, run_id in [(other, "run-one") for other in others] + [(OWNER, "run-two")]:
        with pytest.raises(ValueError):
            await workspace.read(principal, run_id, "nested/说明.txt")
    assert len({workspace.directory(principal, "run-one") for principal in [OWNER, *others]}) == 3


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "nested/../../escape.txt",
        ".git/config",
        ".GIT/config",
        "nested/.GiT/config",
        "",
        "/escape.txt",
    ],
)
def test_workspace_rejects_escape_and_git_metadata(tmp_path, path):
    with pytest.raises(ValueError):
        Workspace(tmp_path).resolve(OWNER, "run", path)


@pytest.mark.asyncio
async def test_workspace_rejects_symlink_escape_when_platform_supports_it(tmp_path):
    workspace = Workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("不应读取", encoding="utf-8")
    link = workspace.directory(OWNER, "run") / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"当前 Windows 权限不支持创建符号链接：{type(exc).__name__}")
    with pytest.raises(ValueError):
        await workspace.read(OWNER, "run", "link/private.txt")
    with pytest.raises(ValueError):
        await workspace.write(OWNER, "run", "link/changed.txt", "不能写出工作区")
    assert not (outside / "changed.txt").exists()


@pytest.mark.asyncio
async def test_file_size_and_viewer_tool_permission(tmp_path):
    tools = HarnessTools(HarnessSettings(data_dir=tmp_path, repository_root=tmp_path))
    with pytest.raises(PermissionError):
        await tools.execute("file_write", {"path": "note.txt", "content": "越权"}, VIEWER, "run")
    with pytest.raises(ValueError):
        await tools.workspace.write(OWNER, "run", "large.txt", "字" * 400_000)
    large = tools.workspace.directory(OWNER, "run") / "large.txt"
    large.write_bytes(b"x" * 1_000_001)
    with pytest.raises(ValueError):
        await tools.workspace.read(OWNER, "run", "large.txt")
    with pytest.raises(ValidationError):
        await tools.execute("project_prepare", {"mode": "invalid"}, OWNER, "run")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression",
    [
        "2**101",
        "2**-101",
        "9**9**9",
        "1e309",
        "1/0",
        "__import__('os').getcwd()",
        "1+" * 60 + "1",
        "1" * 513,
    ],
)
async def test_calculator_rejects_expensive_or_unsafe_expressions(expression):
    with pytest.raises(ValueError):
        await CalculatorTool().execute(expression=expression)


@pytest.mark.asyncio
async def test_calculator_bounded_arithmetic():
    calculator = CalculatorTool()
    assert await calculator.execute(expression="(3+5)*2 % 7") == "2.0"
    assert float(await calculator.execute(expression="2**100")) == 2.0**100


@pytest.mark.asyncio
async def test_missing_sandbox_rejects_without_host_fallback(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("不得调用宿主执行器")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    sandbox = SandboxClient(HarnessSettings(data_dir=tmp_path, sandbox_url="", sandbox_token=""))
    with pytest.raises(RuntimeError, match="拒绝执行"):
        await sandbox.execute(OWNER, "run", "raise RuntimeError('不应在宿主执行')")


def git_executable():
    executable = shutil.which("git")
    if not executable:
        executable = next(
            (
                str(path)
                for path in [
                    Path("C:/Program Files/Git/cmd/git.exe"),
                    Path("C:/Program Files/Git/bin/git.exe"),
                ]
                if path.is_file()
            ),
            None,
        )
    if not executable:
        pytest.skip("本机未安装 Git，真实 Fork/Worktree 验收未执行")
    return executable


def run_git(executable, root, *arguments):
    return subprocess.run(
        [executable, "-C", str(root), "-c", "core.fsmonitor=false", *arguments],
        check=True,
        capture_output=True,
        timeout=15,
    ).stdout


@pytest.fixture
def temporary_repository(tmp_path):
    executable = git_executable()
    root = tmp_path / "source-repository"
    root.mkdir()
    run_git(executable, root, "init")
    (root / "README.md").write_text("临时仓库，不执行脚本", encoding="utf-8")
    run_git(executable, root, "add", "README.md")
    run_git(
        executable,
        root,
        "-c",
        "user.name=测试用户",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "core.hooksPath=" + str(tmp_path / "empty-test-hooks"),
        "commit",
        "-m",
        "test: 临时仓库初始化",
    )
    return executable, root


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fork", "worktree"])
async def test_real_git_project_prepare(tmp_path, temporary_repository, mode):
    executable, source = temporary_repository
    settings = HarnessSettings(
        data_dir=tmp_path / "data", repository_root=source, git_executable=executable
    )
    workspace = Workspace(settings.data_dir)
    manager = ProjectManager(settings, workspace)
    before = run_git(executable, source, "rev-parse", "HEAD")
    result = await manager.prepare(OWNER, "run", mode)
    destination = workspace.directory(OWNER, "run")
    assert result["mode"] == mode
    assert (destination / "README.md").read_text(encoding="utf-8") == "临时仓库，不执行脚本"
    assert (destination / ".git").exists() == (mode == "worktree")
    assert run_git(executable, source, "rev-parse", "HEAD") == before
    await workspace.write(OWNER, "run", "README.md", "只修改副本")
    assert (source / "README.md").read_text(encoding="utf-8") == "临时仓库，不执行脚本"
    with pytest.raises(ValueError):
        await manager.prepare(OWNER, "run", mode)
    with pytest.raises(PermissionError):
        await manager.prepare(VIEWER, "viewer-run", mode)
    if mode == "worktree":
        with pytest.raises(ValueError):
            await workspace.read(OWNER, "run", ".GIT")


@pytest.mark.parametrize("filename", ["../escape.txt", ".GIT/config"])
def test_project_archive_rejects_unsafe_paths(tmp_path, filename):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(filename, "不能覆盖元数据")
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(ValueError):
        ProjectManager._extract(data.getvalue(), destination)


class DelegationService:
    def __init__(self, *, fail_second=False, completed=True):
        self.fail_second, self.completed = fail_second, completed
        self.children, self.cancelled, self.created_arguments = {}, [], []

    async def get_run(self, principal, run_id):
        if run_id == "parent":
            return {
                "id": run_id,
                "parent_run_id": None,
                "session_id": "parent-session",
                "status": "running",
            }
        return self.children[run_id]

    async def create_run(self, principal, message, idempotency_key, **kwargs):
        self.created_arguments.append(kwargs)
        if self.fail_second and self.children:
            raise RuntimeError("第二子任务创建失败")
        run_id = f"child-{len(self.children)}"
        child = {
            "id": run_id,
            "status": "completed" if self.completed else "running",
            "answer": None,
            "error": None,
        }
        self.children[run_id] = child
        return child

    async def cancel(self, principal, run_id):
        self.cancelled.append(run_id)
        self.children[run_id]["status"] = "cancelled"


@pytest.mark.asyncio
async def test_delegate_none_answer_and_read_only_child_contract(tmp_path):
    tools = HarnessTools(HarnessSettings(data_dir=tmp_path))
    tools.service = DelegationService()
    result = await tools.execute(
        "delegate", {"tasks": ["只读任务"], "mode": "fork"}, OWNER, "parent"
    )
    assert result[0]["answer"] == ""
    arguments = tools.service.created_arguments[0]
    assert arguments["allowed_tools"] == ["calculator", "knowledge_search", "artifact_read"]
    assert arguments["max_steps"] == 3 and arguments["session_id"] == "parent-session"


@pytest.mark.asyncio
async def test_delegate_second_creation_failure_cancels_first(tmp_path):
    tools = HarnessTools(HarnessSettings(data_dir=tmp_path))
    tools.service = DelegationService(fail_second=True, completed=False)
    with pytest.raises(RuntimeError, match="第二子任务"):
        await tools.execute(
            "delegate", {"tasks": ["第一任务", "第二任务"], "mode": "team"}, OWNER, "parent"
        )
    assert tools.service.cancelled == ["child-0"]
    assert tools.service.created_arguments[0]["session_id"] is None


@pytest.mark.asyncio
async def test_delegate_cancellation_cleans_up_children(tmp_path):
    tools = HarnessTools(HarnessSettings(data_dir=tmp_path))
    tools.service = DelegationService(completed=False)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            tools.execute("delegate", {"tasks": ["未完成任务"]}, OWNER, "parent"), timeout=0.02
        )
    assert tools.service.cancelled == ["child-0"]
