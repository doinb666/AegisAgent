"""用 Git 凭据发布 GitHub Release；资产必须与本地大小及 SHA256 一致。"""

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import httpx

REPOSITORY = "doinb666/AegisAgent"
API = f"https://api.github.com/repos/{REPOSITORY}"


def checked(response, allowed=(200, 201)):
    if response.status_code not in allowed:
        raise RuntimeError(f"GitHub API 返回 HTTP {response.status_code}")
    return response.json()


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def chunks(path):
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            yield block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--notes", required=True, type=Path)
    args = parser.parse_args()
    git = shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
    credential = subprocess.run(
        [git, "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    fields = dict(line.split("=", 1) for line in credential.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if not token:
        raise RuntimeError("未找到 GitHub 凭据")
    files = sorted(path for path in args.directory.iterdir() if path.is_file())
    if len(files) != 5 or not (args.directory / "SHA256SUMS.txt").is_file():
        raise ValueError("发布目录应仅含五个已验收资产")
    with httpx.Client(
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"},
        timeout=httpx.Timeout(300, connect=20),
    ) as client:
        response = client.get(f"{API}/releases/tags/{args.tag}")
        if response.status_code == 404:
            release = checked(
                client.post(
                    f"{API}/releases",
                    json={
                        "tag_name": args.tag,
                        "target_commitish": args.commit,
                        "name": f"AegisCode {args.tag}",
                        "draft": True,
                        "body": args.notes.read_text(encoding="utf-8"),
                    },
                )
            )
        else:
            release = checked(response)
        assets = {item["name"]: item for item in release["assets"]}
        for path in files:
            expected = "sha256:" + sha256(path)
            asset = assets.get(path.name)
            if not asset:
                asset = checked(
                    client.post(
                        f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release['id']}/assets",
                        params={"name": path.name},
                        content=chunks(path),
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(path.stat().st_size),
                        },
                    )
                )
            if asset["size"] != path.stat().st_size or asset.get("digest") != expected:
                raise RuntimeError(f"资产大小或摘要不一致，停止发布：{path.name}")
            print(f"资产验证通过：{path.name} ({asset['size']} 字节)", flush=True)
        if release["draft"]:
            release = checked(
                client.patch(f"{API}/releases/{release['id']}", json={"draft": False})
            )
        print("已公开发布：" + release["html_url"])


if __name__ == "__main__":
    main()
