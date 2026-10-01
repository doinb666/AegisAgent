"""按身份和任务隔离文件，拒绝路径穿越与符号链接逃逸。"""

import asyncio
import hashlib
from pathlib import Path


def workspace_key(principal, run_id: str) -> str:
    return hashlib.sha256(
        f"{principal.tenant_id}:{principal.user_id}:{run_id}".encode()
    ).hexdigest()


class Workspace:
    def __init__(self, data_dir: Path):
        self.root = (data_dir / "workspaces").resolve()

    def directory(self, principal, run_id: str) -> Path:
        path = self.root / workspace_key(principal, run_id)
        path.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("工作区路径不安全")
        return path

    def resolve(self, principal, run_id: str, relative_path: str) -> Path:
        if not relative_path or len(relative_path) > 512:
            raise ValueError("文件路径为空或过长")
        root = self.directory(principal, run_id)
        relative = Path(relative_path)
        if relative.is_absolute() or any(p.casefold() in {"..", ".git"} for p in relative.parts):
            raise ValueError("文件路径越界")
        result = (root / relative).resolve()
        if not result.is_relative_to(root) or result == root:
            raise ValueError("文件路径越界")
        return result

    async def read(self, principal, run_id: str, path: str) -> str:
        target = self.resolve(principal, run_id, path)
        if not target.is_file() or target.stat().st_size > 1_000_000:
            raise ValueError("文件不存在或超过读取上限")
        return await asyncio.to_thread(target.read_text, encoding="utf-8")

    async def write(self, principal, run_id: str, path: str, content: str) -> dict:
        if len(content.encode()) > 1_000_000:
            raise ValueError("文件超过写入上限")
        target = self.resolve(principal, run_id, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_text, content, encoding="utf-8")
        return {"path": path, "bytes": len(content.encode())}
