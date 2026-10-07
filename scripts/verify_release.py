"""独立下载公开发布资产，逐项核对本地清单、大小及SHA256；不使用发布凭据。"""

import argparse
import hashlib
import re
import time
from pathlib import Path

import httpx


def verify_download(client, url, expected_size, expected_digest, name, label="公开文件"):
    """只重试传输中断；每次从零验证，不接受部分正文或完整性失败。"""
    for attempt in range(3):
        digest, size = hashlib.sha256(), 0
        try:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                for block in response.iter_bytes():
                    size += len(block)
                    if size > expected_size:
                        raise ValueError(f"{label}超出预期大小：{name}")
                    digest.update(block)
            if size != expected_size or digest.hexdigest() != expected_digest:
                raise ValueError(f"{label}大小或摘要不一致：{name}")
            return
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"公开下载返回HTTP {error.response.status_code}：{name}") from None
        except httpx.TransportError:
            if attempt == 2:
                raise RuntimeError(f"公开下载连接失败，已尝试3次：{name}") from None
            print(f"下载连接中断，准备第{attempt + 2}/3次尝试：{name}", flush=True)
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.tag):
        parser.error("版本标签须为v主版本.次版本.补丁版本")
    version = args.tag[1:]
    names = {
        f"AegisAgent-source-v{version}.zip",
        f"aegiscode-{version}-py3-none-any.whl",
        "AegisCode-portable.zip",
        "AegisCode-Setup.exe",
    }
    manifest = (args.directory / "SHA256SUMS.txt").read_bytes()
    checksums = {}
    for row in manifest.decode("utf-8").splitlines():
        digest, name = row.split("  ", 1)
        if name not in names or name in checksums or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("校验清单含不合法或重复资产")
        checksums[name] = digest
    if set(checksums) != names:
        raise ValueError("校验清单须包含四项发布资产")
    base = f"https://github.com/doinb666/AegisAgent/releases/download/{args.tag}/"
    with httpx.Client(follow_redirects=True, timeout=180) as client:
        verify_download(
            client,
            base + "SHA256SUMS.txt",
            len(manifest),
            hashlib.sha256(manifest).hexdigest(),
            "SHA256SUMS.txt",
            label="公开清单",
        )
        print("公开校验清单一致", flush=True)
        for name in sorted(names):
            path = args.directory / name
            expected_size = path.stat().st_size
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != checksums[name]:
                    raise ValueError(f"本地文件摘要不一致：{name}")
            verify_download(client, base + name, expected_size, checksums[name], name)
            print(f"独立下载通过：{name}（{expected_size}字节）", flush=True)


if __name__ == "__main__":
    main()
