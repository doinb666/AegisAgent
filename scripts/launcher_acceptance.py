"""在独立临时目录验证真实启动器，绝不回收已有工作台进程。"""

import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from app.launcher import bind_listener


def main():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    with tempfile.TemporaryDirectory(prefix="aegis-launcher-") as temporary:
        task_dir = Path(temporary)
        with bind_listener("127.0.0.1", 0) as occupied:
            occupied.listen()
            port = occupied.getsockname()[1]
            command = [
                sys.executable,
                "-m",
                "app.launcher",
                "--no-browser",
                "--port",
                str(port),
                "--data-dir",
                str(task_dir / "data"),
            ]
            conflict = subprocess.run(
                command, cwd=root, env=environment, capture_output=True, timeout=30
            )
            assert conflict.returncode == 2
            assert "--auto-port" in conflict.stderr.decode("utf-8")
            with (task_dir / "service.log").open("wb") as log:
                process = subprocess.Popen(
                    [*command, "--auto-port"],
                    cwd=root,
                    env=environment,
                    stdout=log,
                    stderr=log,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                )
                try:
                    deadline = time.monotonic() + 60
                    address = None
                    with httpx.Client(timeout=2, trust_env=False) as client:
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                raise RuntimeError("独立启动器提前退出")
                            output = (task_dir / "service.log").read_text(encoding="utf-8")
                            match = re.search(r"工作台地址：(http://127\.0\.0\.1:\d+)", output)
                            if match:
                                address = match.group(1)
                                try:
                                    ready = client.get(address + "/api/v1/health/ready")
                                    if ready.status_code == 200:
                                        break
                                except httpx.HTTPError:
                                    pass
                            time.sleep(0.1)
                        else:
                            raise TimeoutError("独立启动器未能在60秒内就绪")
                        assert int(address.rsplit(":", 1)[1]) != port
                        assert ready.json()["database"] == "up"
                        assert client.get(address + "/").status_code == 200
                        credentials = {"username": "launcher-test", "password": "temporary-test"}
                        assert (
                            client.post(
                                address + "/api/v1/auth/register", json=credentials
                            ).status_code
                            == 201
                        )
                        login = client.post(address + "/api/v1/auth/login", json=credentials)
                        assert login.status_code == 200
                        headers = {"Authorization": "Bearer " + login.json()["token"]}
                        assert (
                            client.get(
                                address + "/api/v1/runs/missing/thread", headers=headers
                            ).status_code
                            == 404
                        )
                        assert (
                            client.get(
                                address + "/api/v1/runs/missing/thread?limit=51", headers=headers
                            ).status_code
                            == 422
                        )
                finally:
                    if process.poll() is None:
                        if os.name == "nt":
                            process.send_signal(signal.CTRL_BREAK_EVENT)
                        else:
                            process.terminate()
                    process.wait(timeout=30)
    print("真实启动验收通过：冲突提示、预绑定空闲端口、就绪、首页、登录和会话边界")


if __name__ == "__main__":
    main()
