"""服务端冻结差异审批与有基线的原子写入；不提供任意宿主路径接口。"""

import asyncio
import difflib
import errno
import hashlib
import os
import stat
import tempfile
from itertools import islice

from app.harness.security import canonical, digest

from . import file_permissions
from .workspace import workspace_key
from .workspace_lock import WorkspaceLock

MAX_BYTES = 1_000_000
PREVIEW_CHARS = 8192
DIFF_CHARS = 16384
DIFF_LINES = 128
DIFF_LINE_CHARS = 256


def identity(info):
    value = {"device": info.st_dev, "inode": info.st_ino}
    if os.name == "nt":
        value["birthtime_ns"] = info.st_birthtime_ns
    return value


def rejected(code, message):
    return {"status": "rejected", "code": code, "error": message}


def publish_new(temporary, target):
    """仅归类确定尚未发布的异常，其余异常由内核按未知副作用处理。"""
    try:
        if os.name == "nt":
            # Windows rename 在目标存在时失败，不会静默覆盖。
            os.rename(temporary, target)
        else:
            # 同卷 hardlink 是原子 no-clobber 发布，随后清理自己的临时名字。
            os.link(temporary, target, follow_symlinks=False)
    except FileExistsError:
        return rejected("baseline_conflict", "新文件已存在，拒绝覆盖")
    except NotImplementedError:
        return rejected("write_rejected", "当前平台不支持原子无覆盖发布")
    except OSError as exc:
        if exc.errno not in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV}:
            raise
        return rejected("write_rejected", "文件系统不支持原子无覆盖发布")
    return None


def contract_hash(contract):
    return digest(
        canonical(
            {
                key: value
                for key, value in contract.items()
                if key not in {"baseline_hash", "preview"}
            }
        )
    )


def preview(before, after):
    def lines(text):
        # 先截字符，再限制行数、行长，避免完整大文件进入 SequenceMatcher。
        clipped = text[:PREVIEW_CHARS]
        raw = clipped.splitlines(keepends=True)
        result = [line[:DIFF_LINE_CHARS] for line in raw[:DIFF_LINES]]
        return result, len(text) > PREVIEW_CHARS or result != raw

    old, old_cut = lines(before)
    new, new_cut = lines(after)
    delta = "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in islice(
            difflib.unified_diff(old, new, fromfile="写入前", tofile="写入后"), DIFF_LINES * 2 + 8
        )
    )
    return {
        "before": before[:PREVIEW_CHARS],
        "after": after[:PREVIEW_CHARS],
        "diff": delta[:DIFF_CHARS],
        "truncated": old_cut or new_cut or len(delta) > DIFF_CHARS,
    }


class FileWriter:
    def __init__(self, workspace):
        self.workspace = workspace

    def _arguments(self, arguments):
        if set(arguments) != {"path", "content"} or not all(
            isinstance(value, str) for value in arguments.values()
        ):
            raise ValueError("文件写入参数必须且仅包含 path/content")
        parts = self.workspace._path_parts(arguments["path"])
        raw = arguments["content"].encode("utf-8")
        if len(raw) > MAX_BYTES:
            raise ValueError("文件超过写入上限")
        return parts, raw

    def _snapshot(self, principal, run_id, parts):
        root = self.workspace.root / workspace_key(principal, run_id)
        directories = [self.workspace.root.parent, self.workspace.root, root]
        cursor = root
        for part in parts[:-1]:
            cursor /= part
            directories.append(cursor)
        ancestors, missing = [], []
        for index, directory in enumerate(directories):
            try:
                info = directory.lstat()
            except FileNotFoundError:
                missing.append(index)
                continue
            if missing or self.workspace._linked(info) or not stat.S_ISDIR(info.st_mode):
                raise ValueError("文件祖先路径不安全")
            ancestors.append({"index": index, **identity(info)})
        target = cursor / parts[-1]
        baseline = {
            "exists": False,
            "sha256": None,
            "bytes": 0,
            "file_identity": None,
            "ancestors": ancestors,
            "missing_directories": missing,
        }
        try:
            info = target.lstat()
        except FileNotFoundError:
            return baseline, "", target, directories
        if missing or self.workspace._linked(info) or not stat.S_ISREG(info.st_mode):
            raise ValueError("仅允许普通文本文件，禁止链接和特殊文件")
        if info.st_nlink != 1:
            raise ValueError("禁止覆盖硬链接文件")
        if info.st_size > MAX_BYTES:
            raise ValueError("原文件超过大小上限")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        descriptor = os.open(target, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if identity(opened) != identity(info) or opened.st_nlink != 1:
                raise ValueError("读取时原文件身份改变")
            raw = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        current = target.lstat()
        if len(raw) > MAX_BYTES:
            raise ValueError("原文件超过大小上限")
        if self.workspace._linked(current) or any(
            item.st_nlink != 1
            or (identity(item), item.st_size, item.st_mtime_ns)
            != (identity(info), info.st_size, info.st_mtime_ns)
            for item in (opened, after, current)
        ):
            raise ValueError("读取时原文件发生变化")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("仅支持 UTF-8 原文件") from exc
        if any(ord(char) < 32 and char not in "\t\r\n\f" for char in text):
            raise ValueError("禁止覆盖二进制文件")
        baseline.update(
            exists=True,
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
            file_identity=identity(info),
            permissions=file_permissions.permission_identity(target),
        )
        return baseline, text, target, directories

    async def freeze(self, principal, run_id, arguments, call_id, args_hash):
        parts, raw = self._arguments(arguments)
        if principal.role == "viewer":
            raise PermissionError("viewer 无权冻结文件写入审批")
        return await asyncio.to_thread(
            self._freeze, principal, run_id, arguments, parts, raw, call_id, args_hash
        )

    def _freeze(self, principal, run_id, arguments, parts, raw, call_id, args_hash):
        key = workspace_key(principal, run_id)
        with WorkspaceLock(self.workspace.root.parent, key):
            baseline, before, _, _ = self._snapshot(principal, run_id, parts)
            contract = {
                "version": 1,
                "workspace_key": key,
                "path": arguments["path"],
                "call_id": call_id,
                "args_hash": args_hash,
                "baseline": baseline,
                "proposed_sha256": hashlib.sha256(raw).hexdigest(),
                "preview": preview(before, arguments["content"]),
            }
            contract["baseline_hash"] = contract_hash(contract)
            return contract

    async def execute(self, principal, run_id, arguments, contract):
        parts, raw = self._arguments(arguments)
        if principal.role == "viewer":
            raise PermissionError("viewer 无权执行文件写入")
        if not isinstance(contract, dict) or (
            contract.get("version") != 1
            or contract.get("workspace_key") != workspace_key(principal, run_id)
            or contract.get("path") != arguments["path"]
            or contract.get("proposed_sha256") != hashlib.sha256(raw).hexdigest()
            or contract.get("baseline_hash") != contract_hash(contract)
        ):
            raise ValueError("文件写入冻结上下文缺失或不匹配")
        # 取消 await 不会终止线程；内核须保留 started，绝不能据此重放。
        return await asyncio.to_thread(self._execute, principal, run_id, parts, raw, contract)

    def _execute(self, principal, run_id, parts, raw, contract):
        lock = WorkspaceLock(self.workspace.root.parent, contract["workspace_key"])
        try:
            lock.__enter__()
        except TimeoutError:
            return rejected("lock_timeout", "工作区锁等待超时，未写入目标文件")
        except (OSError, ValueError):
            return rejected("write_rejected", "工作区锁不可用或路径不安全，未写入目标文件")
        result = None
        try:
            result = self._locked_write(principal, run_id, parts, raw, contract)
        finally:
            try:
                lock.__exit__(None, None, None)
            except (OSError, ValueError):
                if result is None or result.get("status") != "rejected":
                    raise
                result["error"] += "；工作区锁解锁失败，目标文件未发布"
        return result

    def _locked_write(self, principal, run_id, parts, raw, contract):
        temporary = None
        publishing = False
        refusal = None
        try:
            try:
                baseline, _, target, directories = self._snapshot(principal, run_id, parts)
            except (OSError, ValueError):
                return rejected("baseline_conflict", "文件路径或原文件已改变，拒绝覆盖")
            if baseline != contract["baseline"]:
                return rejected("baseline_conflict", "审批基线已改变，请使用新调用重新审批")
            for directory in directories:
                try:
                    directory.mkdir()
                except FileExistsError:
                    pass
                if not self.workspace._check_directory(directory):
                    raise ValueError("创建后的目录不安全")
            # 创建缺失目录后再次冻结；已存在祖先必须仍是原身份。
            created, _, _, _ = self._snapshot(principal, run_id, parts)
            original = dict(baseline, ancestors=created["ancestors"], missing_directories=[])
            if created != original or any(
                ancestor not in created["ancestors"] for ancestor in baseline["ancestors"]
            ):
                return rejected("baseline_conflict", "创建目录期间文件路径改变，拒绝覆盖")
            descriptor, name = tempfile.mkstemp(prefix=".aegis-write-", dir=target.parent)
            temporary = target.parent / os.path.basename(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if baseline["exists"]:
                file_permissions.preserve_permissions(target, temporary)
            final, _, _, _ = self._snapshot(principal, run_id, parts)
            if final != created:
                refusal = rejected("baseline_conflict", "发布前文件已改变，拒绝覆盖")
                return refusal
            publishing = True
            if baseline["exists"]:
                os.replace(temporary, target)
                temporary = None
                if file_permissions.permission_identity(target) != baseline["permissions"]:
                    raise PermissionError("发布后目标访问权限与审批基线不一致，结果未知")
            else:
                refusal = publish_new(temporary, target)
                if refusal is not None:
                    publishing = False
                    return refusal
                if os.name == "nt":
                    temporary = None
            if os.name != "nt":
                directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            return {
                "status": "written",
                "path": contract["path"],
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        except (OSError, ValueError) as exc:
            if publishing:
                raise
            refusal = rejected("write_rejected", "发布前写入被拒绝：" + str(exc)[:500])
            return refusal
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    if publishing or refusal is None:
                        raise
                    refusal["code"] = "write_rejected"
                    refusal["error"] += "；本次临时文件清理失败，目标文件未发布"
