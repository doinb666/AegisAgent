"""独立进程解析与有界管道；超时或取消必须终止工作进程。"""

import asyncio
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from app.etl.parser import ParsedDocument
from app.etl.pipeline import ETLResult


async def parse_isolated(
    raw: bytes, filename: str, mime_type: str, timeout: float = 30
) -> ETLResult:
    command = (
        [sys.executable, "--etl-worker"]
        if getattr(sys, "frozen", False)
        else [
            sys.executable,
            "-m",
            "app.etl.worker",
        ]
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper()
        in {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "TIKTOKEN_CACHE_DIR",
        }
    }
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
        cwd=None if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    async def send():
        payload = json.dumps(
            {
                "data": base64.b64encode(raw).decode("ascii"),
                "filename": filename,
                "mime_type": mime_type,
            }
        ).encode("utf-8")
        process.stdin.write(payload)
        await process.stdin.drain()
        process.stdin.close()

    async def receive(stream, limit):
        output = bytearray()
        while chunk := await stream.read(65536):
            output.extend(chunk)
            if len(output) > limit:
                raise ValueError("解析进程输出超过预算")
        return bytes(output)

    tasks = [
        asyncio.create_task(send()),
        asyncio.create_task(receive(process.stdout, 8_000_000)),
        asyncio.create_task(receive(process.stderr, 65536)),
    ]
    try:
        async with asyncio.timeout(timeout):
            _, output, _ = await asyncio.gather(*tasks)
            if await process.wait() != 0:
                raise ValueError("解析进程失败或超过资源预算")
        result = json.loads(output)
        return ETLResult(result["chunks"], ParsedDocument(**result["parsed"]), result["meta"])
    finally:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # 先排空已终止进程的管道，避免满管道让 wait 等待传输关闭。
        async def drain(stream):
            while await stream.read(65536):
                pass

        await asyncio.gather(drain(process.stdout), drain(process.stderr))
        await process.wait()
