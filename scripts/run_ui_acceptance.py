"""独立数据库启动验收服务，仅回收本脚本创建的进程。"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", action="append", required=True)
    args = parser.parse_args()
    allowed = {
        "scripts.project_threads_browser_acceptance",
        "scripts.workbench_browser_acceptance",
        "scripts.notifications_browser_acceptance",
        "scripts.layout_browser_acceptance",
        "scripts.assets_browser_acceptance",
        "scripts.skill_browser_acceptance",
        "scripts.skill_revision_browser_acceptance",
        "scripts.schedules_browser_acceptance",
        "scripts.task_inputs_browser_acceptance",
        "scripts.model_parameters_browser_acceptance",
        "scripts.model_output_browser_acceptance",
        "scripts.file_changes_browser_acceptance",
        "scripts.plan_browser_acceptance",
    }
    if not set(args.module) <= allowed:
        parser.error("请选择已登记的验收模块")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="aegis-browser-") as temporary:
        environment = {
            **os.environ,
            "AEGIS_UI_TEST_DATA": str(Path(temporary) / "data"),
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "AEGIS_UI_PLAN": "1" if "scripts.plan_browser_acceptance" in args.module else "0",
            "AEGIS_UI_SKILL_REVISION": "1"
            if "scripts.skill_revision_browser_acceptance" in args.module
            else "0",
        }
        with (Path(temporary) / "server.log").open("w", encoding="utf-8") as log:
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "tests.ui_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ]
            if os.name == "nt" and environment["AEGIS_UI_PLAN"] == "1":
                # Windows Proactor偶发中止本机accept；仅计划验收选择Selector，不改变生产启动。
                command = [
                    sys.executable,
                    "-c",
                    "import asyncio, sys, uvicorn; "
                    "asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()); "
                    "uvicorn.run('tests.ui_server:app', host='127.0.0.1', "
                    "port=int(sys.argv[1]), loop='asyncio')",
                    str(port),
                ]
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=log,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            url = f"http://127.0.0.1:{port}"
            last_probe = "尚未探测"
            try:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("测试服务退出，请检查测试环境")
                    try:
                        with urllib.request.urlopen(url, timeout=5) as response:
                            if response.status == 200:
                                break
                            last_probe = f"HTTP {response.status}"
                    except OSError as error:
                        last_probe = f"{type(error).__name__}: {error}"
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"测试服务未就绪，最后探测：{last_probe}")
                    time.sleep(0.1)
                for module in args.module:
                    extra = []
                    if module == "scripts.workbench_browser_acceptance":
                        extra = ["--output", str(Path(temporary) / "workbench-screenshots")]
                    elif module == "scripts.skill_browser_acceptance":
                        extra = ["--output", str(Path(temporary) / "skill-screenshots")]
                    subprocess.run(
                        [sys.executable, "-m", module, "--url", url, *extra],
                        env=environment,
                        check=True,
                        timeout=240,
                    )
            except Exception:
                # 仅输出本脚本的确定性验收服务日志，避免临时目录回收后丢失故障证据。
                log.flush()
                print(
                    (Path(temporary) / "server.log").read_text(encoding="utf-8")[-4000:],
                    file=sys.stderr,
                )
                raise
            finally:
                if process.poll() is None:
                    process.send_signal(
                        signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM
                    )
                    process.wait(timeout=30)


if __name__ == "__main__":
    main()
