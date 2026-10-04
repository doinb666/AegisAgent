"""构建便携目录、ZIP 与包含便携目录的 Windows 安装器。"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    if sys.platform != "win32":
        raise SystemExit("Windows EXE 必须在 Windows 构建")
    npm = shutil.which("npm.cmd")
    if not npm:
        raise SystemExit("构建 Windows 发布包需要 Node.js 与 npm")
    subprocess.run([npm, "run", "frontend:verify"], cwd=ROOT, check=True)
    import tiktoken

    tokenizer_cache = ROOT / "build/tokenizer-cache"
    tokenizer_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tokenizer_cache)
    tiktoken.get_encoding("cl100k_base")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onedir",
            "--name",
            "AegisCode",
            "--paths",
            str(ROOT),
            "--add-data",
            f"{ROOT / 'app/web'};app/web",
            "--add-data",
            f"{tokenizer_cache};tokenizer-cache",
            "--collect-submodules",
            "tiktoken_ext",
            "--collect-submodules",
            "app.harness",
            "--collect-submodules",
            "app.harness_tools",
            "--collect-submodules",
            "app.etl",
            "--collect-submodules",
            "sqlalchemy.dialects.sqlite",
            "--collect-submodules",
            "sqlalchemy.dialects.postgresql",
            "--collect-submodules",
            "uvicorn",
            "--collect-submodules",
            "mcp.client",
            "--collect-submodules",
            "mcp.shared",
            "--copy-metadata",
            "mcp",
            "--hidden-import",
            "aiosqlite",
            "--hidden-import",
            "asyncpg",
            str(ROOT / "scripts/run_aegis.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = Path(
        shutil.make_archive(
            str(ROOT / "dist/AegisCode-portable"), "zip", ROOT / "dist", "AegisCode"
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--onefile",
            "--windowed",
            "--name",
            "AegisCode-Setup",
            "--add-data",
            f"{payload};payload",
            str(ROOT / "scripts/windows_installer.py"),
        ],
        cwd=ROOT,
        check=True,
    )
    print("产物：dist/AegisCode/AegisCode.exe、便携ZIP、AegisCode-Setup.exe")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
