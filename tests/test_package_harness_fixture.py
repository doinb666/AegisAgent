"""安装专项夹具通过真实 HTTP 与 SDK 验证，不加载应用。"""

import importlib.util
import json
import socket
import sqlite3
import threading

import httpx
import pytest
from openai import BadRequestError, OpenAI


def fixture_module():
    spec = importlib.util.find_spec("scripts.package_harness_fixture")
    assert spec is not None, "缺少独立 Harness 安装专项协议夹具"
    from scripts import package_harness_fixture

    return package_harness_fixture


def test_sdk_stream_and_nonstream_self_check():
    module = fixture_module()
    report = module.self_check()
    assert report["sdk_modes"] == [False, True]
    assert report["phases"] == [
        "plan",
        "plan_tool",
        "plan_answer",
        "final",
        "tool",
        "answer",
        "extract",
        "recall",
        "revision",
    ]


def test_fixture_bounds_and_text_only_revision():
    module = fixture_module()
    with module.LocalHarnessFixture() as fixture:
        with httpx.Client(base_url=fixture.base_url, timeout=2, trust_env=False) as client:
            assert (
                client.post(
                    "/chat/completions", content=b"x" * (module.MAX_REQUEST_BYTES + 1)
                ).status_code
                == 413
            )
            assert (
                client.post(
                    "/chat/completions", json={"model": "wrong", "messages": []}
                ).status_code
                == 400
            )
            assert client.post("/missing", json={}).status_code == 404
        with OpenAI(base_url=fixture.base_url, api_key="local", max_retries=0, timeout=2) as sdk:
            with pytest.raises(BadRequestError):
                sdk.chat.completions.create(
                    model=module.MODEL, messages=[{"role": "user", "content": "未知任务"}]
                )
        assert not fixture.snapshot()["requests"]


def test_fixture_rejects_unbounded_messages_and_request_count():
    module = fixture_module()
    fixture = module.LocalHarnessFixture()
    try:
        with pytest.raises(ValueError, match="messages"):
            fixture.respond({"model": module.MODEL, "messages": [{}] * 65})
        request = {
            "model": module.MODEL,
            "messages": [{"role": "user", "content": module.RECALL_TASK}],
        }
        for _ in range(module.MAX_REQUESTS):
            fixture.respond(request)
        with pytest.raises(ValueError, match="次数"):
            fixture.respond(request)
    finally:
        fixture.server.server_close()


def test_tool_result_and_revision_source_boundaries():
    module = fixture_module()
    with module.LocalHarnessFixture() as fixture:
        with pytest.raises(ValueError, match="证据"):
            fixture.respond(
                {
                    "model": module.MODEL,
                    "messages": [
                        {"role": "user", "content": module.TOOL_TASK},
                        {
                            "role": "tool",
                            "tool_call_id": module.CALL_IDS["tool"],
                            "content": "null",
                        },
                    ],
                }
            )
        with pytest.raises(ValueError, match="唯一可信来源"):
            fixture.respond(
                {
                    "model": module.MODEL,
                    "messages": [
                        {"role": "system", "content": "revisions"},
                        {"role": "user", "content": '{"skills":[]}'},
                    ],
                }
            )
        assert not fixture.snapshot()["requests"]


def test_missing_plan_and_portable_path_traversal(tmp_path):
    import zipfile

    from scripts.package_harness_acceptance import extract_portable, verify_plan

    with pytest.raises(AssertionError, match="功能缺失"):
        verify_plan([])
    archive = tmp_path / "非法.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../非法.exe", b"x")
    with pytest.raises(AssertionError, match="路径遍历"):
        extract_portable(archive, tmp_path / "隔离目录")
    assert not (tmp_path / "非法.exe").exists()


@pytest.mark.parametrize("context", ["[]", '{"skills":[null]}', '{"skills":[{"asset_id":""}]}'])
def test_revision_malformed_context_returns_http_400(context):
    module = fixture_module()
    with module.LocalHarnessFixture() as fixture:
        with httpx.Client(base_url=fixture.base_url, timeout=2, trust_env=False) as client:
            response = client.post(
                "/chat/completions",
                json={
                    "model": module.MODEL,
                    "messages": [
                        {"role": "system", "content": "revisions"},
                        {"role": "user", "content": context},
                    ],
                },
            )
            assert response.status_code == 400
        assert not fixture.snapshot()["requests"]


@pytest.mark.parametrize("field", ["id", "tenant_id", "owner_id", "arguments_hash", "future_field"])
def test_complete_ledger_snapshot_detects_source_row_changes(tmp_path, field):
    from scripts.package_harness_acceptance import ledger_snapshot
    from scripts.package_harness_fixture import CALCULATOR_RESULT, CALL_IDS

    database = tmp_path / "aegis.db"
    plan = {
        "status": "completed",
        "nodes": [
            {
                "status": "completed",
                "tool_call_ids": ["plan-ledger"],
                "acceptance_result": {
                    "ledger_sources": [
                        {"id": "plan-ledger", "run_id": "plan-run", "name": "calculator"}
                    ]
                },
            },
            {"status": "completed"},
        ],
    }
    # 仅独立测试数据库注入改变，生产验收继续只读真实账本。
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE harness_tool_calls ("
            "id TEXT PRIMARY KEY, tenant_id TEXT, owner_id TEXT, run_id TEXT, call_id TEXT, "
            "name TEXT, arguments TEXT, arguments_hash TEXT, status TEXT, result TEXT, "
            "future_field TEXT);"
            "CREATE TABLE harness_runs (id TEXT PRIMARY KEY, config TEXT);"
        )
        for run_id, ledger_id, call_id in (
            ("plan-run", "plan-ledger", CALL_IDS["plan_tool"]),
            ("source-run", "source-ledger", CALL_IDS["tool"]),
        ):
            connection.execute(
                "INSERT INTO harness_tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ledger_id,
                    "tenant",
                    "owner",
                    run_id,
                    call_id,
                    "calculator",
                    json.dumps({"expression": "2+3"}),
                    "hash",
                    "done",
                    json.dumps(CALCULATOR_RESULT),
                    "初始附加列",
                ),
            )
            config = {"evolution_state": "done"}
            if run_id == "plan-run":
                config["plan"] = plan
            connection.execute(
                "INSERT INTO harness_runs VALUES (?,?)", (run_id, json.dumps(config))
            )
    before = ledger_snapshot(tmp_path, "plan-run", "source-run")
    with sqlite3.connect(database) as connection:
        # 字段名只来自上方固定测试参数，数值继续参数化。
        connection.execute(
            f"UPDATE harness_tool_calls SET {field}=? WHERE run_id=?",
            ("改变后的测试值", "source-run"),
        )
    after = ledger_snapshot(tmp_path, "plan-run", "source-run")
    assert after != before, f"完整账本快照漏检普通来源行的 {field} 改变"


def instrument_body_reads(fixture):
    records = []
    responded = threading.Event()
    handler = fixture.server.RequestHandlerClass

    class CountingReader:
        def __init__(self, reader):
            self.reader = reader
            self.bytes_read = 0

        def read(self, size):
            result = self.reader.read(size)
            self.bytes_read += len(result)
            return result

        def __getattr__(self, name):
            return getattr(self.reader, name)

    class ObservedHandler(handler):
        def do_POST(self):
            self.rfile = CountingReader(self.rfile)
            super().do_POST()

        def send_body(self, status, body, content_type="application/json"):
            super().send_body(status, body, content_type)
            records.append(
                {
                    "status": status,
                    "bytes_read": self.rfile.bytes_read,
                    "close_connection": self.close_connection,
                }
            )
            responded.set()

    fixture.server.RequestHandlerClass = ObservedHandler
    return records, responded


@pytest.mark.parametrize("attempt", range(5))
def test_unknown_route_consumes_segmented_body_before_complete_404(attempt):
    del attempt
    module = fixture_module()
    fixture = module.LocalHarnessFixture()
    records, responded = instrument_body_reads(fixture)
    with fixture, socket.create_connection(fixture.server.server_address, timeout=2) as client:
        client.sendall(
            b"POST /v1/missing HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Length: 2\r\nContent-Type: application/json\r\n\r\n{"
        )
        assert not responded.wait(0.1), f"正文未完整发送已关闭响应：{records}"
        client.sendall(b"}")
        chunks = []
        while chunk := client.recv(4096):
            chunks.append(chunk)
        headers, body = b"".join(chunks).split(b"\r\n\r\n", 1)
        assert headers.startswith(b"HTTP/1.1 404 ")
        assert body == b'{"error":"unknown route"}'
        assert f"Content-Length: {len(body)}".encode() in headers
        assert responded.wait(1)
        assert records == [{"status": 404, "bytes_read": 2, "close_connection": True}]


@pytest.mark.parametrize(
    "framing",
    [
        b"",
        b"Content-Length: 999999999\r\n",
        b"Content-Length: 256001\r\n",
        b"Content-Length: invalid\r\n",
        b"Transfer-Encoding: chunked\r\n",
    ],
)
def test_unknown_route_rejects_invalid_framing_without_unbounded_read(framing):
    module = fixture_module()
    fixture = module.LocalHarnessFixture()
    records, responded = instrument_body_reads(fixture)
    with fixture, socket.create_connection(fixture.server.server_address, timeout=2) as client:
        client.sendall(b"POST /v1/missing HTTP/1.1\r\nHost: 127.0.0.1\r\n" + framing + b"\r\n")
        chunks = []
        while chunk := client.recv(4096):
            chunks.append(chunk)
        assert b"".join(chunks).startswith(b"HTTP/1.1 413 ")
        assert responded.wait(1)
        assert records == [{"status": 413, "bytes_read": 0, "close_connection": True}]


def test_unknown_route_repeated_json_preserves_complete_404():
    module = fixture_module()
    fixture = module.LocalHarnessFixture()
    records, _ = instrument_body_reads(fixture)
    with fixture, httpx.Client(base_url=fixture.base_url, timeout=2, trust_env=False) as client:
        for _ in range(20):
            response = client.post("/missing", json={})
            assert response.status_code == 404
            assert response.content == b'{"error":"unknown route"}'
    assert len(records) == 20
    assert all(
        record == {"status": 404, "bytes_read": 2, "close_connection": True} for record in records
    )


@pytest.mark.parametrize(
    ("stage", "value"),
    [
        ("extract", []),
        ("extract", None),
        ("calculator", []),
        ("calculator", None),
        ("calculator", {"required": {"expression": True}}),
        ("calculator", {"required": [{}]}),
        ("node", {"说明": "当前节点"}),
        ("node", {"当前节点": []}),
    ],
)
def test_structured_inputs_reject_invalid_types_with_recorded_http_400(stage, value):
    module = fixture_module()
    request = {"model": module.MODEL}
    if stage == "extract":
        request["messages"] = [
            {"role": "system", "content": "经验提炼器"},
            {"role": "user", "content": json.dumps(value)},
        ]
    elif stage == "calculator":
        request["messages"] = [{"role": "user", "content": module.TOOL_TASK}]
        request["tools"] = [
            {"type": "function", "function": {"name": "calculator", "parameters": value}}
        ]
    else:
        request["messages"] = [
            {"role": "user", "content": module.PLAN_TASK},
            {"role": "user", "content": json.dumps(value, ensure_ascii=False)},
        ]
    with module.LocalHarnessFixture() as fixture:
        with httpx.Client(base_url=fixture.base_url, timeout=2, trust_env=False) as client:
            response = client.post("/chat/completions", json=request)
            assert response.status_code == 400
            assert response.json() == {"error": "invalid fixture request"}
        snapshot = fixture.snapshot()
        assert snapshot["requests"] == []
        assert len(snapshot["errors"]) == 1
