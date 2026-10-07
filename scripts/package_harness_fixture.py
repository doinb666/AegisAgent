"""本机安装专项协议替身：真实 SDK 流/非流，不导入 app 或调用外部工具。"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler

try:
    from .package_model_fixture import FixtureServer
except ImportError:
    from package_model_fixture import FixtureServer

MODEL = "package-harness-fixture"
MAX_REQUEST_BYTES = 256_000
MAX_REQUESTS = 48
PLAN_TASK = "安装专项计算器：两节点计划，先计算2+3，再解释结果。"
TOOL_TASK = "安装专项计算器：计算2+3并整理可复用方法。"
RECALL_TASK = "安装专项计算器：召回计算2+3的可复用方法。"
AFTER_TASK = "安装专项计算器：再次召回计算2+3的可复用方法。"
SKILL_NAME = "安装专项计算器可复用方法"
SKILL_CONTENT = (
    "安装专项计算器：计算2+3时使用calculator，检查返回5；"
    "仅相同输入与权限适用，业务正确性需人工验证。"
)
REVISION_CONTENT = (
    "安装专项计算器未验证修订建议：检查输入边界并人工复核calculator结果，禁止声称已修复。"
)
CALL_IDS = {"plan_tool": "call-harness-plan", "tool": "call-harness-source"}
CALCULATOR_RESULT = "5.0"


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def plan():
    return {
        "steps": [
            {
                "objective": "计算2+3",
                "inputs": {"instruction": "调用calculator计算2+3", "from_steps": []},
                "acceptance": {"output_kind": "tool_evidence", "required_tools": ["calculator"]},
            },
            {
                "objective": "解释结果",
                "inputs": {"instruction": "根据前序证据解释结果", "from_steps": [0]},
                "acceptance": {"output_kind": "answer", "required_tools": []},
            },
        ]
    }


def completion(phase, text):
    message = {"role": "assistant", "content": text}
    if phase in CALL_IDS:
        message["tool_calls"] = [
            {
                "id": CALL_IDS[phase],
                "type": "function",
                "function": {"name": "calculator", "arguments": encoded({"expression": "2+3"})},
            }
        ]
    return {
        "id": "harness-" + phase,
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if phase in CALL_IDS else "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


def frames(response):
    choice = response["choices"][0]
    message = choice["message"]
    deltas = [{"role": "assistant"}]
    text = message["content"]
    deltas.extend({"content": text[start : start + 11]} for start in range(0, len(text), 11))
    for call in message.get("tool_calls", []):
        deltas.append({"tool_calls": [{"index": 0, **call}]})
    result = []
    for delta, finish in [*((item, None) for item in deltas), ({}, choice["finish_reason"])]:
        chunk = {key: response[key] for key in ("id", "created", "model")}
        chunk.update(
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": delta, "finish_reason": finish}],
        )
        frame = ("data: " + encoded(chunk) + "\n\n").encode()
        assert len(frame) <= 4096
        result.append(frame)
    result.append(b"data: [DONE]\n\n")
    assert len(result) <= 128
    return result


class LocalHarnessFixture:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = []
        self.errors = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                self.connection.settimeout(5)

            def log_message(self, *args):
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
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if self.headers.get("Transfer-Encoding") or not 0 < length <= MAX_REQUEST_BYTES:
                    if (
                        self.path == "/v1/chat/completions"
                        and not self.headers.get("Transfer-Encoding")
                        and length == MAX_REQUEST_BYTES + 1
                    ):
                        # 仅兼容模型路由的超限字节回归，未知路由非法长度立即拒绝。
                        self.rfile.read(length)
                    self.send_body(413, b'{"error":"bounded request required"}')
                    return
                try:
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("正文不完整")
                    if self.path != "/v1/chat/completions":
                        # 正常有界正文先读完整，避免 Windows 关闭时未读正文触发连接重置。
                        self.send_body(404, b'{"error":"unknown route"}')
                        return
                    request = json.loads(raw)
                    response = fixture.respond(request)
                    if request.get("stream") is True:
                        self.send_body(
                            200, b"".join(frames(response)), "text/event-stream; charset=utf-8"
                        )
                    else:
                        self.send_body(200, encoded(response).encode())
                except (ValueError, TypeError, OSError, AssertionError, RecursionError) as exc:
                    with fixture.lock:
                        if len(fixture.errors) < MAX_REQUESTS:
                            fixture.errors.append(type(exc).__name__ + ": " + str(exc)[:200])
                    try:
                        self.send_body(400, b'{"error":"invalid fixture request"}')
                    except OSError:
                        pass

        self.server = FixtureServer(Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def environment(self):
        return {
            "AEGIS_MODELS_JSON": encoded(
                [
                    {
                        "model": MODEL,
                        "provider": "openai",
                        "base_url": self.base_url,
                        "api_key_env": "AEGIS_HARNESS_FIXTURE_KEY",
                    }
                ]
            ),
            "AEGIS_HARNESS_FIXTURE_KEY": "fixture-local-only",
            "AEGIS_RISK_REVIEW_ENABLED": "false",
            "AEGIS_EVOLUTION_ENABLED": "true",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }

    def respond(self, request):
        if not isinstance(request, dict) or request.get("model") != MODEL:
            raise ValueError("模型名错误")
        messages = request.get("messages")
        if (
            not isinstance(messages, list)
            or not 1 <= len(messages) <= 64
            or any(not isinstance(item, dict) for item in messages)
        ):
            raise ValueError("messages必须为有界对象列表")
        tools = request.get("tools", [])
        if (
            not isinstance(tools, list)
            or len(tools) > 32
            or any(
                not isinstance(tool, dict) or not isinstance(tool.get("function"), dict)
                for tool in tools
            )
        ):
            raise ValueError("工具目录无效")
        system = "\n".join(
            str(item.get("content", "")) for item in messages if item.get("role") == "system"
        )
        users = [str(item.get("content", "")) for item in messages if item.get("role") == "user"]
        latest = users[-1] if users else ""
        if "revisions" in system:
            context = json.loads(latest)
            if not isinstance(context, dict):
                raise ValueError("修订上下文必须为对象")
            skills = context.get("skills", [])
            if (
                not isinstance(skills, list)
                or len(skills) != 1
                or not isinstance(skills[0], dict)
                or not isinstance(skills[0].get("asset_id"), str)
                or not skills[0]["asset_id"].strip()
            ):
                raise ValueError("修订缺少唯一可信来源")
            phase, text = (
                "revision",
                encoded(
                    {
                        "revisions": [
                            {"asset_id": skills[0]["asset_id"], "content": REVISION_CONTENT}
                        ]
                    }
                ),
            )
        elif "经验提炼器" in system:
            trajectory = json.loads(latest)
            if not isinstance(trajectory, dict):
                raise ValueError("提炼轨迹必须为对象")
            source = trajectory.get("用户原始要求") == TOOL_TASK
            phase = "extract"
            text = encoded(
                {
                    "assets": [{"kind": "skill", "name": SKILL_NAME, "content": SKILL_CONTENT}]
                    if source
                    else []
                }
            )
        elif PLAN_TASK in "\n".join(users):
            if "当前节点" in latest:
                node_context = json.loads(latest)
                if not isinstance(node_context, dict) or node_context.get("当前节点") not in (
                    "n0-1",
                    "n0-2",
                ):
                    raise ValueError("节点上下文必须为对象且包含本次有效节点")
                node = node_context["当前节点"]
                phase, text = (
                    ("plan_tool", "计算节点调用本机计算器")
                    if node == "n0-1"
                    else ("plan_answer", "前序工具证据为5；结构通过不代表业务验证。")
                )
            elif "允许工具" in latest:
                phase, text = "plan", encoded(plan())
            else:
                phase, text = "final", "安装专项计算器两节点完成，结果为5，业务正确性未独立验证。"
        elif TOOL_TASK in users:
            results = [item for item in messages if item.get("role") == "tool"]
            phase, text = (
                ("answer", "安装专项计算器结果为5。") if results else ("tool", "执行本机计算器")
            )
            if results and (
                results[-1].get("tool_call_id") != CALL_IDS["tool"]
                or json.loads(results[-1].get("content", "null")) != CALCULATOR_RESULT
            ):
                raise ValueError("实际calculator证据不正确")
        elif any(task in users for task in (RECALL_TASK, AFTER_TASK)):
            phase, text = "recall", "安装专项计算器召回完成；输入边界仍需人工确认。"
        else:
            raise ValueError("未知安装专项任务")
        if phase in CALL_IDS:
            calculator = next(
                (
                    tool["function"]
                    for tool in tools
                    if tool["function"].get("name") == "calculator"
                ),
                None,
            )
            parameters = calculator.get("parameters") if calculator is not None else None
            if not isinstance(parameters, dict) or parameters.get("required") != ["expression"]:
                raise ValueError("缺少calculator协议")
        with self.lock:
            if len(self.requests) >= MAX_REQUESTS:
                raise ValueError("请求次数超限")
            self.requests.append({"phase": phase, "stream": request.get("stream") is True})
        return completion(phase, text)

    def snapshot(self):
        with self.lock:
            return {"requests": list(self.requests), "errors": list(self.errors)}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def self_check():
    from openai import OpenAI

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ]
    cases = [
        ("plan", [{"role": "user", "content": PLAN_TASK}, {"role": "user", "content": "允许工具"}]),
        (
            "plan_tool",
            [
                {"role": "user", "content": PLAN_TASK},
                {"role": "user", "content": encoded({"当前节点": "n0-1"})},
            ],
        ),
        (
            "plan_answer",
            [
                {"role": "user", "content": PLAN_TASK},
                {"role": "user", "content": encoded({"当前节点": "n0-2"})},
            ],
        ),
        ("final", [{"role": "user", "content": PLAN_TASK}]),
        ("tool", [{"role": "user", "content": TOOL_TASK}]),
        (
            "answer",
            [
                {"role": "user", "content": TOOL_TASK},
                {
                    "role": "tool",
                    "tool_call_id": CALL_IDS["tool"],
                    "content": encoded(CALCULATOR_RESULT),
                },
            ],
        ),
        (
            "extract",
            [
                {"role": "system", "content": "经验提炼器"},
                {"role": "user", "content": encoded({"用户原始要求": TOOL_TASK})},
            ],
        ),
        ("recall", [{"role": "user", "content": RECALL_TASK}]),
        (
            "revision",
            [
                {"role": "system", "content": "revisions"},
                {"role": "user", "content": encoded({"skills": [{"asset_id": "sdk-only-id"}]})},
            ],
        ),
    ]
    with (
        LocalHarnessFixture() as fixture,
        OpenAI(base_url=fixture.base_url, api_key="local", max_retries=0, timeout=5) as sdk,
    ):
        for stream in (False, True):
            for phase, messages in cases:
                response = sdk.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools, stream=stream
                )
                if stream:
                    text, calls = "", []
                    with response:
                        for chunk in response:
                            text += chunk.choices[0].delta.content or ""
                            calls.extend(chunk.choices[0].delta.tool_calls or [])
                    assert bool(calls) == (phase in CALL_IDS)
                    if phase in CALL_IDS:
                        assert calls[0].function.name == "calculator"
                        assert json.loads(calls[0].function.arguments) == {"expression": "2+3"}
                else:
                    message = response.choices[0].message
                    text = message.content
                    assert bool(message.tool_calls) == (phase in CALL_IDS)
                if phase == "revision":
                    assert json.loads(text) == {
                        "revisions": [{"asset_id": "sdk-only-id", "content": REVISION_CONTENT}]
                    }
                assert text
        assert not fixture.snapshot()["errors"]
    return {"sdk_modes": [False, True], "phases": [phase for phase, _ in cases], "requests": 18}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", required=True)
    parser.parse_args()
    print(encoded(self_check()))


if __name__ == "__main__":
    main()
