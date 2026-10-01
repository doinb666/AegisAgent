"""受控本地仓库的 Fork/Worktree 准备，不执行仓库脚本或 Git hooks。"""

import asyncio
import io
import os
import zipfile
from pathlib import Path


class ProjectManager:
    def __init__(self, settings, workspace):
        self.settings = settings
        self.workspace = workspace

    async def prepare(self, principal, run_id: str, mode: str):
        if principal.role == "viewer":
            raise PermissionError("viewer不能创建代码工作区")
        if not self.settings.repository_root:
            raise RuntimeError("管理员尚未配置 AEGIS_REPOSITORY_ROOT")
        source = self.settings.repository_root.resolve()
        destination = self.workspace.directory(principal, run_id)
        if any(destination.iterdir()):
            raise ValueError("工作区已含文件，拒绝覆盖")
        await self._git(source, "rev-parse", "--show-toplevel")
        if mode == "worktree":
            await self._git(
                source, "worktree", "add", "--detach", "--no-checkout", str(destination), "HEAD"
            )
        archive = await self._git(source, "archive", "--format=zip", "HEAD")
        await asyncio.to_thread(self._extract, archive, destination)
        return {
            "mode": mode,
            "files": len(list(destination.rglob("*"))),
            "message": "已准备隔离代码副本；代码执行仍需沙箱和审批",
        }

    @staticmethod
    def _extract(data: bytes, destination: Path):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 30_000_000:
                raise ValueError("仓库归档超过工作区配额")
            for item in archive.infolist():
                parts = Path(item.filename).parts
                target = (destination / item.filename).resolve()
                if any(
                    part.casefold() in {".git", ".."} for part in parts
                ) or not target.is_relative_to(destination):
                    raise ValueError("仓库归档路径不安全")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                elif (item.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(item))

    async def _git(self, root: Path, *arguments) -> bytes:
        environment = {
            k: v
            for k, v in os.environ.items()
            if k.upper() in {"PATH", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
        }
        environment.update(
            GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0"
        )
        hooks = (self.settings.data_dir / "empty-hooks").resolve()
        hooks.mkdir(parents=True, exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            self.settings.git_executable,
            "-C",
            str(root),
            "-c",
            f"core.hooksPath={hooks}",
            "-c",
            "core.fsmonitor=false",
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError("Git 工作区操作失败，请检查仓库配置")
        return stdout
