"""独立下载公开发布资产，逐项核对本地清单、大小及SHA256；不使用发布凭据。"""

import argparse
import hashlib
import re
from pathlib import Path

import httpx


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
        response = client.get(base + "SHA256SUMS.txt")
        response.raise_for_status()
        if response.content != manifest:
            raise ValueError("公开清单与本地清单不一致")
        print("公开校验清单一致", flush=True)
        for name in sorted(names):
            path = args.directory / name
            expected_size = path.stat().st_size
            with path.open("rb") as source:
                if hashlib.file_digest(source, "sha256").hexdigest() != checksums[name]:
                    raise ValueError(f"本地文件摘要不一致：{name}")
            digest, size = hashlib.sha256(), 0
            with client.stream("GET", base + name) as response:
                response.raise_for_status()
                for block in response.iter_bytes():
                    size += len(block)
                    if size > expected_size:
                        raise ValueError(f"公开文件超出预期大小：{name}")
                    digest.update(block)
            if size != expected_size or digest.hexdigest() != checksums[name]:
                raise ValueError(f"公开文件大小或摘要不一致：{name}")
            print(f"独立下载通过：{name}（{size}字节）", flush=True)


if __name__ == "__main__":
    main()
