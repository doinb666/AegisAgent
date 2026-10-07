"""原子替换前保留访问权限；Windows DACL 不能用 chmod 代替。"""

import ctypes
import hashlib
import os
import stat
from ctypes import wintypes
from pathlib import Path

DACL_SECURITY_INFORMATION = 0x00000004
PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
UNPROTECTED_DACL_SECURITY_INFORMATION = 0x20000000
SE_DACL_PROTECTED = 0x1000


def windows_security(path: Path) -> tuple[bytes, bool]:
    api = ctypes.WinDLL("advapi32", use_last_error=True)
    get = api.GetFileSecurityW
    get.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    get(str(path), DACL_SECURITY_INFORMATION, None, 0, ctypes.byref(needed))
    if ctypes.get_last_error() != 122 or not 0 < needed.value <= 1_000_000:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(needed.value)
    if not get(
        str(path), DACL_SECURITY_INFORMATION, descriptor, needed.value, ctypes.byref(needed)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    control, revision = wintypes.WORD(), wintypes.DWORD()
    inspect = api.GetSecurityDescriptorControl
    inspect.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    inspect.restype = wintypes.BOOL
    if not inspect(descriptor, ctypes.byref(control), ctypes.byref(revision)):
        raise ctypes.WinError(ctypes.get_last_error())
    return descriptor.raw, bool(control.value & SE_DACL_PROTECTED)


def set_windows_security(path: Path, descriptor: bytes, protected: bool) -> None:
    api = ctypes.WinDLL("advapi32", use_last_error=True)
    flags = DACL_SECURITY_INFORMATION | (
        PROTECTED_DACL_SECURITY_INFORMATION if protected else UNPROTECTED_DACL_SECURITY_INFORMATION
    )
    buffer = ctypes.create_string_buffer(descriptor)
    # 两种接口对 AUTO_INHERITED 的处理不同；按原控制标志选择，复制后再逐字验证。
    if not int.from_bytes(descriptor[2:4], "little") & 0x0400:
        legacy = api.SetFileSecurityW
        legacy.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        legacy.restype = wintypes.BOOL
        if not legacy(str(path), flags, buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        return
    get_dacl = api.GetSecurityDescriptorDacl
    get_dacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    get_dacl.restype = wintypes.BOOL
    set_security = api.SetNamedSecurityInfoW
    set_security.argtypes = [
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    set_security.restype = wintypes.DWORD
    present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), ctypes.c_void_p()
    if not get_dacl(buffer, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not present.value:
        raise PermissionError("原文件缺少可复制的 DACL")
    error = set_security(str(path), 1, flags, None, None, dacl, None)
    if error:
        raise ctypes.WinError(error)


def permission_identity(path: Path) -> dict:
    if os.name == "nt":
        descriptor, protected = windows_security(path)
        return {"dacl_sha256": hashlib.sha256(descriptor).hexdigest(), "protected": protected}
    info = path.stat(follow_symlinks=False)
    return {"mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid, "gid": info.st_gid}


def preserve_permissions(source: Path, temporary: Path) -> None:
    before = permission_identity(source)
    if os.name == "nt":
        descriptor, protected = windows_security(source)
        set_windows_security(temporary, descriptor, protected)
    else:
        info = source.stat(follow_symlinks=False)
        temp_info = temporary.stat(follow_symlinks=False)
        # 本轮不扩展提权接口；不同 owner/group 时宁可拒绝替换。
        if (info.st_uid, info.st_gid) != (temp_info.st_uid, temp_info.st_gid):
            raise PermissionError("原文件 owner/group 与服务不一致，拒绝改变归属")
        os.chmod(temporary, stat.S_IMODE(info.st_mode), follow_symlinks=False)
    if permission_identity(temporary) != before:
        raise PermissionError("无法完整保留原文件访问权限")
