"""固定解析入口；资源受限，不接受脚本、命令或任意文件路径。"""

import asyncio
import base64
import ctypes
import json
import os
import sys
from dataclasses import asdict

_job_handle = None
MEMORY_BYTES = 512 * 1024 * 1024


def apply_limits():
    """失败则拒绝解析，不静默回退到无预算进程。"""
    if os.name != "nt":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        return
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("process_time", ctypes.c_int64),
            ("job_time", ctypes.c_int64),
            ("flags", wintypes.DWORD),
            ("minimum_working_set", ctypes.c_size_t),
            ("maximum_working_set", ctypes.c_size_t),
            ("active_processes", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority", wintypes.DWORD),
            ("scheduling", wintypes.DWORD),
        ]

    class IOCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in (
                "read_ops",
                "write_ops",
                "other_ops",
                "read_bytes",
                "write_bytes",
                "other_bytes",
            )
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io", IOCounters),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    global _job_handle
    _job_handle = kernel.CreateJobObjectW(None, None)
    if not _job_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    # PROCESS_TIME、ACTIVE_PROCESS、PROCESS_MEMORY、KILL_ON_JOB_CLOSE。
    limits.basic.flags = 0x2 | 0x8 | 0x100 | 0x2000
    limits.basic.process_time = 30 * 10_000_000
    limits.basic.active_processes = 1
    limits.process_memory = MEMORY_BYTES
    if not kernel.SetInformationJobObject(
        _job_handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(_job_handle, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())


def main():
    apply_limits()
    payload = json.loads(sys.stdin.buffer.read(16_000_001))
    from app.etl.pipeline import ETLPipeline

    raw = base64.b64decode(payload["data"], validate=True)
    if len(raw) > 10_000_000:
        raise ValueError("解析工作进程输入超限")
    result = asyncio.run(ETLPipeline().run_bytes(raw, payload["filename"], payload["mime_type"]))
    output = json.dumps(asdict(result), ensure_ascii=False).encode("utf-8")
    if len(output) > 8_000_000:
        raise ValueError("解析工作进程输出超限")
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
