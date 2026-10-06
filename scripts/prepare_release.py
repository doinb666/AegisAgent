"""从已提交Git快照构建五个发布资产，避免打包未提交的用户文件。"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = metadata["project"]["version"]
    git = shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
    revision = subprocess.check_output([git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    destination = ROOT / "dist/releases" / version
    if destination.exists():
        raise SystemExit("发布目录已存在，请保留并先核对，程序不覆盖旧资产")
    destination.mkdir(parents=True)
    source = destination / f"AegisAgent-source-v{version}.zip"
    subprocess.run(
        [git, "archive", "--format=zip", f"--output={source}", revision], cwd=ROOT, check=True
    )
    with tempfile.TemporaryDirectory(prefix="aegis-release-") as temporary:
        snapshot = Path(temporary)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(snapshot)
        snapshot_version = tomllib.loads((snapshot / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["version"]
        if snapshot_version != version:
            raise SystemExit("版本信息尚未提交，请先提交再构建")
        npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
        if not npm:
            raise SystemExit("发布构建需要npm")
        subprocess.run([npm, "ci"], cwd=snapshot, check=True)
        for builder in ("build_wheel.py", "build_windows.py"):
            subprocess.run(
                [sys.executable, str(snapshot / "scripts" / builder)], cwd=snapshot, check=True
            )
        names = [
            f"aegiscode-{version}-py3-none-any.whl",
            "AegisCode-Setup.exe",
            "AegisCode-portable.zip",
        ]
        for name in names:
            shutil.copy2(snapshot / "dist" / name, destination / name)
    files = sorted(destination.iterdir())
    sums = []
    for artifact in files:
        with artifact.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        sums.append(f"{digest}  {artifact.name}")
    (destination / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"构建完成：{destination}，来源提交：{revision}")


if __name__ == "__main__":
    main()
