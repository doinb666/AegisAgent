"""源码、wheel 与 Windows 可执行文件共用的个人启动入口。"""

from __future__ import annotations

import argparse
import errno
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uvicorn import Server

logger = logging.getLogger("uvicorn.error")


def bind_listener(host: str, port: int, auto_port: bool = False) -> socket.socket:
    """占住实际监听端口，避免先检测再绑定期间的竞争。"""
    listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET)
    try:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((host, port))
        except OSError as error:
            if not auto_port or error.errno not in {errno.EADDRINUSE, 10048}:
                raise
            listener.bind((host, 0))
        return listener
    except OSError:
        listener.close()
        raise


def browser_address(host: str, port: int) -> str:
    """通配监听地址转换为本机访问地址，正确包裹 IPv6。"""
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    authority = f"[{host}]" if ":" in host else host
    return f"http://{authority}:{port}"


def open_when_started(
    server: Server, address: str, stopped: threading.Event, timeout: float = 60
) -> None:
    """只根据本进程初始化状态打开页面，不探测可能属于其他应用的端口。"""
    deadline = time.monotonic() + timeout
    while not stopped.is_set() and not server.should_exit:
        if server.started:
            try:
                if not webbrowser.open(address):
                    logger.warning("浏览器未能自动打开，请手动访问 %s", address)
            except (OSError, webbrowser.Error):
                logger.warning("浏览器启动失败，请手动访问 %s", address)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning("服务仍在初始化，就绪后请手动访问 %s", address)
            return
        stopped.wait(min(0.1, remaining))


def main() -> None:
    if "--migrate-schedules" in sys.argv:
        from app.harness.schedule_migration import main as schedule_main

        schedule_main([argument for argument in sys.argv[1:] if argument != "--migrate-schedules"])
        return
    if "--migrate-notifications" in sys.argv:
        from app.harness.notification_migration import main as notification_main

        notification_main(
            [argument for argument in sys.argv[1:] if argument != "--migrate-notifications"]
        )
        return
    if "--migrate-threads" in sys.argv:
        from app.harness.thread_migration import main as migration_main

        migration_main([argument for argument in sys.argv[1:] if argument != "--migrate-threads"])
        return
    if "--etl-worker" in sys.argv:
        if getattr(sys, "frozen", False):
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(sys._MEIPASS) / "tokenizer-cache"))
        from app.etl.worker import main as parse_main

        parse_main()
        return
    parser = argparse.ArgumentParser(description="AegisCode 私有任务工作台")
    parser.add_argument("--host", default="127.0.0.1", help="默认仅监听本机")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--auto-port", action="store_true", help="端口被占用时选择空闲端口")
    parser.add_argument("--data-dir", type=Path, help="独立持久化目录")
    parser.add_argument("--config", type=Path, help="含 .env 的配置目录")
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.port <= 65535:
        parser.error("端口必须为1至65535")
    frozen = getattr(sys, "frozen", False)
    if frozen:
        tokenizer_cache = Path(sys._MEIPASS) / "tokenizer-cache"
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(tokenizer_cache))
    default_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AegisCode"
    data_dir = (arguments.data_dir or (default_data if frozen else Path("data"))).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    if arguments.config:
        os.chdir(arguments.config.resolve())
    elif frozen:
        os.chdir(data_dir)
    os.environ["AEGIS_DATA_DIR"] = str(data_dir)
    try:
        listener = bind_listener(arguments.host, arguments.port, arguments.auto_port)
    except OSError as error:
        if error.errno in {errno.EADDRINUSE, 10048}:
            parser.error("端口已被占用，请使用 --port 指定其他端口或添加 --auto-port")
        parser.error("无法绑定监听地址，请检查 --host、--port 和本机网络配置")
    with listener:
        import uvicorn

        from app.main import app

        port = listener.getsockname()[1]
        address = browser_address(arguments.host, port)
        server = uvicorn.Server(
            uvicorn.Config(app, host=arguments.host, port=port, log_level="info")
        )
        logger.info("工作台地址：%s", address)
        stopped = threading.Event()
        browser_thread = None
        if not arguments.no_browser:
            browser_thread = threading.Thread(
                target=open_when_started, args=(server, address, stopped), daemon=True
            )
            browser_thread.start()
        try:
            server.run(sockets=[listener])
        finally:
            stopped.set()
            if browser_thread:
                browser_thread.join(timeout=1)


if __name__ == "__main__":
    main()
