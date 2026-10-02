"""按身份和任务隔离文件，拒绝路径穿越与符号链接逃逸。"""

import asyncio
import codecs
import hashlib
import os
import stat
from pathlib import Path, PureWindowsPath


def workspace_key(principal, run_id: str) -> str:
    return hashlib.sha256(
        f"{principal.tenant_id}:{principal.user_id}:{run_id}".encode()
    ).hexdigest()


class Workspace:
    def __init__(self, data_dir: Path):
        self.root = data_dir.resolve() / "workspaces"

    @staticmethod
    def _linked(info):
        return stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
        )

    @classmethod
    def _check_directory(cls, directory: Path) -> bool:
        try:
            info = directory.lstat()
        except FileNotFoundError:
            return False
        if cls._linked(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("工作区路径不安全")
        return True

    def _read_only_directory(self, principal, run_id: str) -> Path | None:
        root = self.root / workspace_key(principal, run_id)
        for directory in (self.root, root):
            if not self._check_directory(directory):
                return None
        return root

    @staticmethod
    def _path_parts(path: str) -> list[str]:
        if not path or len(path) > 512 or any(char in path for char in ("\\", ":", "\x00")):
            raise ValueError("文件路径无效")
        parts = path.split("/")
        if any(
            part.casefold() in {"", ".", "..", ".git"}
            or part.endswith((".", " "))
            or PureWindowsPath(part).is_reserved()
            for part in parts
        ):
            raise ValueError("文件路径越界或无效")
        return parts

    @classmethod
    def _checked_path(cls, root: Path, parts: list[str]) -> Path:
        if not cls._check_directory(root):
            raise FileNotFoundError("工作区不存在")
        target = root
        for index, part in enumerate(parts):
            target /= part
            try:
                info = target.lstat()
            except FileNotFoundError:
                continue
            if cls._linked(info):
                raise ValueError("禁止访问链接文件")
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise ValueError("文件路径包含非目录项")
        return target

    async def list_files(self, principal, run_id: str) -> dict:
        return await asyncio.to_thread(self._list_files, principal, run_id)

    def _list_files(self, principal, run_id: str) -> dict:
        root = self._read_only_directory(principal, run_id)
        if root is None:
            return {"files": [], "truncated": False}
        files = []
        pending = [(root, ())]
        scanned = 0
        truncated = False
        while pending:
            directory, parent_parts = pending.pop()
            # 显式队列与逐项扫描限定成本，禁止先 rglob 全量遍历再排序。
            with os.scandir(directory) as entries:
                for entry in entries:
                    scanned += 1
                    parts = (*parent_parts, entry.name)
                    relative = "/".join(parts)
                    try:
                        self._path_parts(relative)
                    except ValueError:
                        if entry.name.casefold() != ".git":
                            truncated = True
                    else:
                        info = entry.stat(follow_symlinks=False)
                        if not self._linked(info):
                            if stat.S_ISREG(info.st_mode):
                                if len(files) >= 200:
                                    return self._file_listing(files, True)
                                files.append({"path": relative, "bytes": info.st_size})
                            elif stat.S_ISDIR(info.st_mode):
                                if len(parts) < 8:
                                    pending.append((Path(entry.path), parts))
                                else:
                                    truncated = True
                    if scanned >= 1000:
                        return self._file_listing(files, True)
        return self._file_listing(files, truncated)

    @staticmethod
    def _file_listing(files: list[dict], truncated: bool) -> dict:
        return {"files": sorted(files, key=lambda item: item["path"]), "truncated": truncated}

    async def preview_file(self, principal, run_id: str, path: str) -> dict:
        return await asyncio.to_thread(self._preview_file, principal, run_id, path)

    def _preview_file(self, principal, run_id: str, path: str) -> dict:
        parts = self._path_parts(path)
        target = self._read_only_directory(principal, run_id)
        if target is None:
            raise FileNotFoundError("工作区不存在")
        for index, part in enumerate(parts):
            target /= part
            info = target.lstat()
            if self._linked(info):
                raise ValueError("禁止读取链接文件")
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise FileNotFoundError("文件不存在")
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("仅支持预览普通文本文件")
        if info.st_size > 1_000_000:
            raise ValueError("文件超过预览大小上限")
        with target.open("rb") as stream:
            raw = stream.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("文件超过预览大小上限")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("仅支持 UTF-8 纯文本预览") from exc
        if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
            raise ValueError("禁止预览二进制文件")
        truncated = len(raw) > 65536
        decoder = codecs.getincrementaldecoder("utf-8")()
        content = decoder.decode(raw[:65536], final=not truncated)
        return {"path": path, "content": content, "bytes": len(raw), "truncated": truncated}

    def directory(self, principal, run_id: str) -> Path:
        path = self.root / workspace_key(principal, run_id)
        for directory in (self.root, path):
            if not self._check_directory(directory):
                directory.mkdir(parents=True, exist_ok=True)
                self._check_directory(directory)
        return path

    def resolve(self, principal, run_id: str, relative_path: str) -> Path:
        parts = self._path_parts(relative_path)
        root = self.directory(principal, run_id)
        return self._checked_path(root, parts)

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
