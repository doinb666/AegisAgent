"""只协调本机遵守协议的工作区写入；锁文件始终保留，防止锁 inode 分叉。"""

import errno
import os
import re
import stat
import time
from pathlib import Path

LOCK_TIMEOUT = 5.0


def linked(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
    )


class WorkspaceLock:
    def __init__(self, data_dir: Path, key: str):
        if not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("工作区锁标识无效")
        self.directory = data_dir / "workspace-locks"
        self.path = self.directory / (key + ".lock")
        self.stream = None

    @staticmethod
    def _check_file(info):
        if linked(info) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("工作区锁文件不安全")

    def __enter__(self):
        try:
            self.directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = self.directory.lstat()
        if linked(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("工作区锁目录不安全")
        try:
            self._check_file(self.path.lstat())
        except FileNotFoundError:
            pass
        descriptor = os.open(
            self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
        self.stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            opened = os.fstat(descriptor)
            self._check_file(opened)
            current = self.path.lstat()
            self._check_file(current)
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                raise ValueError("工作区锁文件已被替换")
            deadline = time.monotonic() + LOCK_TIMEOUT
            while True:
                try:
                    self._lock()
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError("工作区锁等待超时") from exc
                    time.sleep(min(0.025, max(0, deadline - time.monotonic())))
            if opened.st_size == 0:
                self.stream.write(b"\0")
            return self
        except BaseException:
            self.stream.close()
            self.stream = None
            raise

    def _lock(self):
        if os.name == "nt":
            import msvcrt

            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def __exit__(self, *exc):
        try:
            if os.name == "nt":
                import msvcrt

                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.stream.close()
            self.stream = None
