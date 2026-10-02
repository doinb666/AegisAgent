"""在临时源码副本构建 wheel，避免历史 build/lib 重新带入已删除模块。"""

import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    filename = f"{project['name'].replace('-', '_')}-{project['version']}-py3-none-any.whl"
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aegis-wheel-") as directory:
        snapshot = Path(directory)
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(ROOT / name, snapshot / name)
        shutil.copytree(
            ROOT / "app", snapshot / "app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(destination),
            ],
            cwd=snapshot,
            check=True,
        )
        with zipfile.ZipFile(destination / filename) as archive:
            for name in archive.namelist():
                if name.startswith("app/") and not name.endswith("/"):
                    source = snapshot / name
                    if not source.is_file() or archive.read(name) != source.read_bytes():
                        raise RuntimeError(f"安装包包含残留或不一致文件：{name}")
        print(f"构建与源码一致性校验通过：{destination / filename}")


if __name__ == "__main__":
    build()
