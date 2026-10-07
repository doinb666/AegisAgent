"""仅绑定本机临时端口的安装包模型夹具，不加载仓库 app 或外部模型。"""

import argparse
import asyncio
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import closing, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_REQUEST_BYTES = 256_000
MAX_REQUESTS = 8
MAX_FRAME_BYTES = 4096
MODEL = "package-fixture"
CALL_ID = "call-package-write"
TASK_MESSAGE = "安装包功能验收：写入中文文件并返回确定性答复。"
FILE_PATH = "中文验收/结果.txt"
FILE_CONTENT = "安装包真实写入🛡️\n中文与表情保持 UTF-8 字节。\n"
INTRO = "准备写入安装验收文件🛡️。\n"
ANSWER = "安装包验收完成🛡️。\n中文文件已按审批保存。"
KEY_ENV = "AEGIS_PACKAGE_MOCK_KEY"


def arguments():
    return {"path": FILE_PATH, "content": FILE_CONTENT}


def completion(phase):
    message = {"role": "assistant", "content": ANSWER if phase == "answer" else INTRO}
    if phase == "write":
        message["tool_calls"] = [
            {
                "id": CALL_ID,
                "type": "function",
                "function": {
                    "name": "file_write",
                    "arguments": json.dumps(arguments(), ensure_ascii=False, separators=(",", ":")),
                },
            }
        ]
    return {
        "id": "fixture-" + phase,
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop" if phase == "answer" else "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def sse_frames(phase):
    def chunk(delta, finish=None):
        payload = {
            "id": "fixture-" + phase,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        frame = ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()
        assert len(frame) <= MAX_FRAME_BYTES, "夹具 SSE 帧超过固定上限"
        return frame

    frames = [chunk({"role": "assistant", "content": ""})]
    if phase == "write":
        frames.append(chunk({"content": INTRO}))
        raw = json.dumps(arguments(), ensure_ascii=False, separators=(",", ":"))
        pieces = [raw[start : start + 17] for start in range(0, len(raw), 17)]
        assert len(pieces) <= 32 and len(raw.encode()) < 1024
        for index, piece in enumerate(pieces):
            tool = {"index": 0, "function": {"arguments": piece}}
            if index == 0:
                tool.update(id="call-package-", type="function")
                tool["function"]["name"] = "file_"
            elif index == 1:
                tool["id"] = "write"
                tool["function"]["name"] = "write"
            frames.append(chunk({"tool_calls": [tool]}))
        frames.append(chunk({}, "tool_calls"))
    else:
        for start in range(0, len(ANSWER), 7):
            frames.append(chunk({"content": ANSWER[start : start + 7]}))
        frames.append(chunk({}, "stop"))
    frames.append(b"data: [DONE]\n\n")
    assert len(frames) <= 32
    return frames


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 4

    def __init__(self, handler):
        self.slots = threading.BoundedSemaphore(4)
        self.connection_lock = threading.Lock()
        self.connections = 0
        super().__init__(("127.0.0.1", 0), handler)

    def get_request(self):
        request = super().get_request()
        with self.connection_lock:
            self.connections += 1
        return request

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class LocalModelFixture:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []
        self.errors = []
        self.stopped = threading.Event()
        self.stalled_sent = threading.Event()
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.settimeout(10)

            def log_message(self, *args):
                # 不记录认证头和请求正文，即使夹具只含临时测试数据。
                pass

            def send_body(self, status, body, content_type="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_body(404, b'{"error":"unknown fixture route"}')
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_REQUEST_BYTES:
                    self.send_body(413, b'{"error":"bounded request required"}')
                    return
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("夹具请求正文不完整")
                    request = json.loads(raw)
                    phase = fixture.record(request)
                    if request.get("stream") is True:
                        frames = sse_frames(phase)
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Content-Length", str(sum(map(len, frames))))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        for frame in frames:
                            self.wfile.write(frame)
                            self.wfile.flush()
                        self.close_connection = True
                    else:
                        self.send_body(
                            200, json.dumps(completion(phase), ensure_ascii=False).encode()
                        )
                except (ValueError, AssertionError, OSError, TypeError, RecursionError) as exc:
                    fixture.error(type(exc).__name__ + ": " + str(exc)[:200])
                    self.close_connection = True
                    try:
                        self.send_body(400, b'{"error":"invalid fixture request"}')
                    except OSError:
                        pass

            def do_GET(self):
                if self.path not in {"/api/v1/runs/slow/events", "/api/v1/runs/stalled/events"}:
                    self.send_body(404, b'{"error":"unknown fixture route"}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                deadline = time.monotonic() + 2
                try:
                    if self.path == "/api/v1/runs/stalled/events":
                        fixture.stopped.wait(0.25)
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        fixture.stalled_sent.set()
                        fixture.stopped.wait(2)
                        return
                    while time.monotonic() < deadline and not fixture.stopped.is_set():
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                        fixture.stopped.wait(0.02)
                except OSError:
                    pass

        self.server = FixtureServer(Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def environment(self):
        return {
            "AEGIS_MODELS_JSON": json.dumps(
                [
                    {
                        "model": MODEL,
                        "provider": "openai",
                        "base_url": self.base_url,
                        "api_key_env": KEY_ENV,
                    }
                ]
            ),
            KEY_ENV: "fixture-not-a-real-credential",
            # 仅禁用测试中不相关的额外 LLM 工作；人工审批和工具边界仍由安装包执行。
            "AEGIS_RISK_REVIEW_ENABLED": "false",
            "AEGIS_EVOLUTION_ENABLED": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }

    def record(self, request):
        if not isinstance(request, dict) or request.get("model") != MODEL:
            raise ValueError("夹具模型名或请求类型不正确")
        messages = request.get("messages")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 64:
            raise ValueError("夹具 messages 数量无效")
        if any(not isinstance(item, dict) for item in messages):
            raise ValueError("夹具消息必须为对象")
        if not any(
            item.get("role") == "user" and TASK_MESSAGE in str(item.get("content", ""))
            for item in messages
        ):
            raise ValueError("请求不是本次确定性安装任务")
        tools = request.get("tools", [])
        if (
            not isinstance(tools, list)
            or len(tools) > 32
            or any(
                not isinstance(tool, dict) or not isinstance(tool.get("function"), dict)
                for tool in tools
            )
        ):
            raise ValueError("夹具工具目录必须是有界对象列表")
        schema = next(
            (
                tool.get("function", {}).get("parameters", {})
                for tool in tools
                if tool.get("function", {}).get("name") == "file_write"
            ),
            None,
        )
        if (
            not isinstance(schema, dict)
            or not isinstance(schema.get("properties"), dict)
            or (
                not isinstance(schema.get("required"), list)
                or set(schema.get("properties", {})) != {"path", "content"}
                or set(schema.get("required", [])) != {"path", "content"}
                or schema.get("additionalProperties") is not False
            )
        ):
            raise ValueError("安装包未提供严格的 file_write 参数协议")
        results = [item for item in messages if item.get("role") == "tool"]
        phase = "answer" if results else "write"
        if results:
            result = json.loads(results[-1].get("content", "null"))
            if results[-1].get("tool_call_id") != CALL_ID or (
                not isinstance(result, dict)
                or result.get("status") != "written"
                or result.get("path") != FILE_PATH
                or result.get("sha256") != hashlib.sha256(FILE_CONTENT.encode()).hexdigest()
                or result.get("bytes") != len(FILE_CONTENT.encode())
            ):
                raise ValueError("模型收到的实际文件工具结果不正确")
        with self.lock:
            if len(self.requests) >= MAX_REQUESTS:
                raise ValueError("夹具请求次数超过固定上限")
            self.requests.append({"phase": phase, "stream": request.get("stream") is True})
        return phase

    def error(self, message):
        with self.lock:
            if len(self.errors) < MAX_REQUESTS:
                self.errors.append(message)

    def snapshot(self):
        with self.lock:
            return {"requests": list(self.requests), "errors": list(self.errors)}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "本机夹具线程未停止"


def acceptance_self_test():
    """以真实慢流和 Windows 文件操作验证验收辅助函数的失败边界。"""
    import httpx

    try:
        from . import package_acceptance as acceptance
    except ImportError:
        import package_acceptance as acceptance

    address = "http://127.0.0.1:8948"
    for encoding in ("utf-8", "gb18030"):
        output = (
            f"工作台地址：{address}\nStarted server process [1234]\nApplication startup complete.\n"
        ).encode(encoding)
        assert acceptance.startup_logged(output, address)
        assert not acceptance.startup_logged(output, "http://127.0.0.1:894")
        assert not acceptance.startup_logged(output.replace(b"complete.", b"waiting."), address)
        assert acceptance.port_conflict_logged(
            "error: 端口已被占用，请使用 --port 指定其他端口或添加 --auto-port\n".encode(encoding)
        )
        assert not acceptance.port_conflict_logged(
            "error: 无法绑定监听地址，请检查 --host、--port 和本机网络配置\n".encode(encoding)
        )
        assert not acceptance.port_conflict_logged(output)

    with tempfile.TemporaryDirectory(prefix="aegis-退出码自检-") as temporary:
        root = Path(temporary)
        for encoding in ("utf-8", "gb18030"):
            for reason, returncode, accepted in (
                ("端口已被占用", 0, False),
                ("端口已被占用", 2, True),
                ("无法绑定监听地址", 2, False),
            ):
                command = [
                    sys.executable,
                    "-c",
                    "import sys; "
                    f"sys.stdout.buffer.write('{reason}\\n'.encode('{encoding}')); "
                    f"sys.exit({returncode})",
                ]
                try:
                    with redirect_stdout(io.StringIO()):
                        acceptance.verify_port_conflict(command, root, dict(os.environ), b"")
                except AssertionError:
                    assert not accepted, "非零端口占用退出未通过"
                else:
                    assert accepted, "占用消息掩盖退出码0，或旧日志掩盖无关错误"
        print("占用端口退出码自检通过：两编码退出0拒绝、退出2通过、无关错误拒绝")
        if os.name == "nt":
            windows_process_tree_self_test(acceptance, root)

    delta = {
        "type": "model_output_delta",
        "data": {"call_id": "public-write", "text": INTRO, "offset": 0, "seq": 1},
    }
    finished = {
        "type": "model_output_finished",
        "data": {"call_id": "public-write", "reason": "tools", "streamed": True, "seq": 3},
    }
    for retracted_call in (None, "unrelated-call"):
        events = [delta, finished]
        if retracted_call:
            events.insert(
                1,
                {
                    "type": "model_output_retracted",
                    "data": {"call_id": retracted_call, "reason": "tools", "seq": 2},
                },
            )
        try:
            acceptance.verify_feature_frames(events)
        except AssertionError:
            pass
        else:
            raise AssertionError("验收未拒绝缺失或不同 call_id 的撤销事件")
    valid_events = [
        delta,
        {
            "type": "model_output_retracted",
            "data": {"call_id": "public-write", "reason": "tools", "seq": 2},
        },
        finished,
    ]
    acceptance.verify_feature_frames(valid_events)
    with LocalModelFixture() as fixture:
        with httpx.Client(base_url=fixture.base_url.removesuffix("/v1"), trust_env=False) as client:
            # 停顿场景留足首次心跳调度余量；旧闲置超时需约1.25秒，仍超过1.15秒门。
            for run_id, timeout, limit in (("slow", 0.12, 0.19), ("stalled", 1.0, 1.15)):
                started = time.monotonic()
                try:
                    acceptance.read_events(client, {}, run_id, timeout=timeout)
                except TimeoutError:
                    elapsed = time.monotonic() - started
                    assert elapsed < limit, f"{run_id} 未遵守总截止时间：{elapsed:.3f}秒"
                    if run_id == "stalled":
                        assert fixture.stalled_sent.is_set(), "停顿负例未实际发送首个心跳"
                    print(f"事件硬截止负例通过：{run_id}，预算{timeout:.3f}秒，实际{elapsed:.3f}秒")
                except httpx.TimeoutException as error:
                    elapsed = time.monotonic() - started
                    raise AssertionError(f"HTTP闲置超时替代了总截止：{elapsed:.3f}秒") from error
                else:
                    raise AssertionError("持续 keepalive 或心跳后停顿未触发事件总超时")
    with tempfile.TemporaryDirectory(prefix="aegis-账本关闭自检-") as temporary:
        data_dir = Path(temporary)
        database = data_dir / "aegis.db"
        with closing(sqlite3.connect(database)) as connection:
            connection.execute(
                "CREATE TABLE harness_tool_calls "
                "(run_id TEXT, call_id TEXT, name TEXT, status TEXT, arguments TEXT, result TEXT)"
            )
            connection.execute(
                "INSERT INTO harness_tool_calls VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "self-test",
                    CALL_ID,
                    "file_write",
                    "done",
                    json.dumps(arguments()),
                    '{"status":"written"}',
                ),
            )
            connection.commit()
        acceptance.verify_feature_ledger(data_dir, "self-test")
        renamed = database.with_name("closed.db")
        database.rename(renamed)
        renamed.unlink()


def windows_process_tree_self_test(acceptance, root):
    """只创建自有父子进程，以保留句柄验证超时后整棵进程树均退出。"""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    child_pid_file = root / "owned-child.pid"
    child_code = (
        "import os,signal,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent_code = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN); "
        "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2],sys.argv[3]]); time.sleep(30)"
    )
    # 直接使用基础解释器，让父子都明确忽略 CtrlBreak，保证走强制超时回收分支。
    process = subprocess.Popen(
        [sys._base_executable, "-c", parent_code, "-c", child_code, str(child_pid_file)],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    child_handle = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if child_pid_file.exists() and child_pid_file.stat().st_size:
                break
            assert process.poll() is None, "自检父进程提前退出"
            time.sleep(0.01)
        assert child_pid_file.exists(), "自检子进程未启动"
        child_pid = int(child_pid_file.read_text())
        child_handle = kernel.OpenProcess(0x00100000, False, child_pid)
        assert child_handle, "无法保留自建子进程的同步句柄"
        assert kernel.WaitForSingleObject(child_handle, 0) == 258, "子进程已提前退出"
        acceptance.stop_process(process, graceful_timeout=0.1)
        assert process.poll() is not None
        assert kernel.WaitForSingleObject(child_handle, 5000) == 0, "强制回收后自建子进程仍存活"
    finally:
        acceptance.stop_process(process, graceful_timeout=0.1)
        if child_handle:
            kernel.CloseHandle(child_handle)
    print("Windows 自建进程树超时回收自检通过：父子进程均退出")


async def self_test():
    """夹具自身以真实 OpenAI SDK 读流，确认碎片和有界拒绝；不加载 app。"""
    import httpx
    from openai import AsyncOpenAI

    tools = [
        {
            "type": "function",
            "function": {
                "name": "file_write",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    messages = [{"role": "user", "content": TASK_MESSAGE}]
    with LocalModelFixture() as fixture:
        async with AsyncOpenAI(
            api_key="fixture-only",
            base_url=fixture.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(trust_env=False),
        ) as sdk:
            stream = await sdk.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, stream=True
            )
            text, name, call_id, raw = "", "", "", ""
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text += delta.content or ""
                for tool in delta.tool_calls or []:
                    call_id += tool.id or ""
                    name += tool.function.name or ""
                    raw += tool.function.arguments or ""
            await stream.close()
            assert text == INTRO and name == "file_write" and call_id == CALL_ID
            assert json.loads(raw) == arguments()
            result = {
                "status": "written",
                "path": FILE_PATH,
                "bytes": len(FILE_CONTENT.encode()),
                "sha256": hashlib.sha256(FILE_CONTENT.encode()).hexdigest(),
            }
            messages += [{"role": "tool", "tool_call_id": CALL_ID, "content": json.dumps(result)}]
            stream = await sdk.chat.completions.create(
                model=MODEL, messages=messages, tools=tools, stream=True
            )
            answer = "".join(
                [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
            )
            await stream.close()
            assert answer == ANSWER
            # 旧包兼容响应仅用于让旧包红测到真正的功能缺口。
            response = await sdk.chat.completions.create(
                model=MODEL, messages=messages[:1], tools=tools
            )
            assert response.choices[0].message.tool_calls[0].function.name == "file_write"
        async with httpx.AsyncClient(trust_env=False) as client:
            # 未知路由提前拒绝，不发送未读取的正文，避免 Windows 关闭连接时复位。
            assert (
                await client.post(fixture.base_url + "/unknown", content=b"")
            ).status_code == 404
        # 只声明超限正文，验证服务器先拒绝；发送未读取的巨量正文会导致 Windows 复位。
        reader, writer = await asyncio.open_connection("127.0.0.1", fixture.server.server_port)
        try:
            writer.write(
                (
                    "POST /v1/chat/completions HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    f"Content-Length: {MAX_REQUEST_BYTES + 1}\r\n"
                    "Connection: close\r\n\r\n"
                ).encode("ascii")
            )
            await writer.drain()
            assert await asyncio.wait_for(reader.readline(), timeout=2) == (
                b"HTTP/1.1 413 Request Entity Too Large\r\n"
            )
        finally:
            writer.close()
            await writer.wait_closed()
        snapshot = fixture.snapshot()
        assert len(snapshot["requests"]) == 3 and not snapshot["errors"], snapshot
    await asyncio.to_thread(acceptance_self_test)
    print("本机模型夹具自检通过：真实 SDK、SSE 工具碎片、请求边界、撤销配对、慢流截止与账本关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", required=True)
    parser.parse_args()
    asyncio.run(self_test())
