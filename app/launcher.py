"""源码、wheel 与 Windows 可执行文件共用的个人启动入口。"""

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path


def main():
    if "--etl-worker" in sys.argv:
        if getattr(sys, "frozen", False):
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(sys._MEIPASS) / "tokenizer-cache"))
        from app.etl.worker import main as parse_main

        parse_main()
        return
    parser = argparse.ArgumentParser(description="AegisCode 私有任务工作台")
    parser.add_argument("--host", default="127.0.0.1", help="默认仅监听本机")
    parser.add_argument("--port", type=int, default=8000)
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
    import uvicorn

    from app.main import app

    if not arguments.no_browser:
        # 不依赖外部桌面服务；默认浏览器只连接本机地址。
        address = "127.0.0.1" if arguments.host in {"0.0.0.0", "::"} else arguments.host
        timer = threading.Timer(2, webbrowser.open, args=(f"http://{address}:{arguments.port}",))
        timer.daemon = True
        timer.start()
    uvicorn.run(app, host=arguments.host, port=arguments.port, log_level="info")


if __name__ == "__main__":
    main()
