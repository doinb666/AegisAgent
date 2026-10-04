"""在独立中文目录验收 wheel 或 Setup：静态产物、登录和重启持久化。"""

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx


def wait_ready(client, process):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("安装包服务提前退出，请检查验收日志")
        try:
            response = client.get("/api/v1/health/ready")
            if response.status_code == 200 and response.json()["database"] == "up":
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError("安装包启动超过 60 秒")


def verify_session(client, prepare):
    credentials = {"username": "package-acceptance", "password": "acceptance-only-2026"}
    if prepare:
        assert client.post("/api/v1/auth/register", json=credentials).status_code == 201
    login = client.post("/api/v1/auth/login", json=credentials)
    assert login.status_code == 200
    headers = {"Authorization": "Bearer " + login.json()["token"]}
    if prepare:
        response = client.post(
            "/api/v1/assets",
            headers=headers,
            json={
                "kind": "profile",
                "name": "安装验收偏好",
                "content": "使用中文",
                "status": "active",
            },
        )
        assert response.status_code == 201
    current = client.get("/api/v1/auth/me", headers=headers)
    assert current.status_code == 200
    assert any(item["name"] == "安装验收偏好" for item in current.json()["bootstrap"]["memories"])
    capabilities = client.get("/api/v1/capabilities", headers=headers)
    assert capabilities.status_code == 200
    assert capabilities.json()["collaboration"]["project_modes"] == ["fork", "worktree"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--static", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    if bool(args.wheel) == bool(args.setup):
        parser.error("只能选择 wheel 或 Setup 一种产物")
    expected = hashlib.sha256(args.static.read_bytes()).digest()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AEGIS_", "OPENAI_", "ANTHROPIC_", "AZURE_", "OLLAMA_"))
    }
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="aegis-安装验收-") as temporary:
        root = Path(temporary)
        if args.wheel:
            # 应用及依赖安装到独立 venv，避免源码遮蔽或本机依赖掩盖缺失。
            subprocess.run(
                [sys.executable, "-m", "venv", str(root / "venv")],
                check=True,
            )
            python = root / "venv/Scripts/python.exe"
            subprocess.run(
                [str(python), "-m", "pip", "install", str(args.wheel.resolve())],
                cwd=root,
                env=environment,
                check=True,
            )
            probe = (
                subprocess.check_output(
                    [str(python), "-c", "import app; print(app.__file__)"],
                    cwd=root,
                    env=environment,
                )
                .decode()
                .strip()
            )
            assert Path(probe).is_relative_to(root / "venv"), "检测到源码遮蔽 wheel"
            command = [str(python), "-m", "app.launcher"]
        else:
            target = root / "中文安装目录"
            subprocess.run(
                [str(args.setup.resolve()), "--target", str(target)],
                cwd=root,
                env=environment,
                check=True,
                timeout=120,
            )
            command = [str(target / "AegisCode.exe")]
        command += ["--no-browser", "--port", str(args.port), "--data-dir", str(root / "data")]
        with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=5) as client:
            for prepare in (True, False):
                with (root / "service.log").open("ab") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=root,
                        env=environment,
                        stdout=log,
                        stderr=log,
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                    )
                    try:
                        wait_ready(client, process)
                        assert client.get("/").status_code == 200
                        assert (
                            hashlib.sha256(client.get("/static/app.js").content).digest()
                            == expected
                        )
                        verify_session(client, prepare)
                    except Exception:
                        log.flush()
                        output = (root / "service.log").read_text(
                            encoding="utf-8", errors="replace"
                        )
                        print(output[-5000:])
                        raise
                    finally:
                        # 只回收本脚本创建的独立验收进程，不操作正式服务。
                        if process.poll() is None:
                            if os.name == "nt":
                                process.send_signal(signal.CTRL_BREAK_EVENT)
                            else:
                                process.terminate()
                        process.wait(timeout=30)
            print("安装包验收通过：独立中文目录、静态校验、登录、能力接口、重启后偏好保留")


if __name__ == "__main__":
    main()
