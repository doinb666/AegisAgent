"""独立中文目录验收 wheel 或 Setup：安装与重启，可选 SDK 流和文件审批。"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

try:
    from . import package_model_fixture as model_fixture
except ImportError:
    import package_model_fixture as model_fixture


class ProcessExited(RuntimeError):
    """记录本次 Popen 的实际退出码与独立日志片段。"""

    def __init__(self, process, output):
        self.pid = process.pid
        self.returncode = process.returncode
        self.output = output
        super().__init__(f"安装包服务提前退出，PID={self.pid}，退出码={self.returncode}")


def wait_ready(client, process, log_path, log_offset):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ProcessExited(process, log_path.read_bytes()[log_offset:])
        # Windows venv 启动器的子 Python PID 可不同；仅信任本次启动且预绑定后的完整日志。
        output = log_path.read_bytes()[log_offset:]
        if not startup_logged(output, client.base_url):
            time.sleep(0.2)
            continue
        try:
            response = client.get("/api/v1/health/ready")
            if response.status_code == 200 and response.json()["database"] == "up":
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise TimeoutError("安装包启动超过 60 秒")


def startup_logged(output, base_url):
    """仅匹配完整 ASCII 地址和启动标记，兼容冻结程序的 GBK 中文日志。"""
    address = str(base_url).rstrip("/").encode("ascii")
    return (
        any(line.rstrip().endswith(address) for line in output.splitlines())
        and b"Started server process [" in output
        and b"Application startup complete." in output
    )


def verify_task_inputs(client, headers, prepare):
    templates = client.get("/api/v1/task-templates", headers=headers)
    assert templates.status_code == 200 and len(templates.json()) == 4
    assert client.get("/api/v1/mcp/servers", headers=headers).json() == []
    if prepare:
        document = client.post(
            "/api/v1/documents/upload",
            headers={**headers, "Idempotency-Key": "package-document"},
            files={
                "file": (
                    "安装验证资料.md",
                    "# 安装证据\n重启后保留本人资料。".encode(),
                    "text/markdown",
                )
            },
        )
        assert document.status_code == 201
        schedule = client.post(
            "/api/v1/schedules",
            headers={**headers, "Idempotency-Key": "package-schedule"},
            json={
                "title": "安装包暂停计划",
                "message": "只读核对安装证据",
                "scheduled_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
            },
        )
        assert schedule.status_code == 201
        paused = client.patch(
            f"/api/v1/schedules/{schedule.json()['id']}",
            headers=headers,
            json={"action": "pause", "expected_version": schedule.json()["version"]},
        )
        assert paused.status_code == 200 and paused.json()["status"] == "paused"
    references = client.get("/api/v1/documents/references", headers=headers)
    assert references.status_code == 200
    assert len(references.json()) == 1 and references.json()[0]["name"] == "安装验证资料.md"
    schedules = client.get("/api/v1/schedules", headers=headers).json()
    assert len(schedules) == 1 and schedules[0]["status"] == "paused"
    assert schedules[0]["title"] == "安装包暂停计划"
    return references.json()[0]["id"]


def login_session(client, prepare, username="package-acceptance"):
    credentials = {"username": username, "password": "acceptance-only-2026"}
    if prepare:
        assert client.post("/api/v1/auth/register", json=credentials).status_code == 201
    login = client.post("/api/v1/auth/login", json=credentials)
    assert login.status_code == 200
    body = login.json()
    return {"Authorization": "Bearer " + body["token"]}, body["user"]


def verify_session(client, prepare):
    headers, _ = login_session(client, prepare)
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
    details = capabilities.json()
    assert details["collaboration"]["modes"] == ["fork", "team"]
    assert details["collaboration"]["max_children"] == 2
    assert details["collaboration"]["project_modes"] == [], "未绑定仓库时应隐藏 Git 模式"
    assert details["model_protocols"] == ["openai", "custom", "anthropic", "azure", "ollama"]
    assert details["threads"]["enabled"] and details["notifications"]["enabled"]
    assert details["task_inputs"]["enabled"] and details["schedules"]["enabled"]
    assert not details["model_parameters"]["temperature"]
    assert not details["model_parameters"]["output_tokens"]
    document_id = verify_task_inputs(client, headers, prepare)
    if prepare:
        project = client.post(
            "/api/v1/assets",
            headers=headers,
            json={
                "kind": "project",
                "name": "安装包项目",
                "content": "只做确定性安装验收",
                "status": "active",
            },
        )
        assert project.status_code == 201
        for title in ("参数边界", "并发恢复"):
            response = client.post(
                "/api/v1/threads",
                headers=headers,
                json={
                    "title": title,
                    "project_id": project.json()["id"],
                },
            )
            assert response.status_code == 201
    threads = client.get("/api/v1/threads", headers=headers).json()
    assert {thread["title"] for thread in threads} == {"参数边界", "并发恢复"}
    assert len({thread["session_id"] for thread in threads}) == 2
    if prepare:
        response = client.post(
            "/api/v1/runs",
            headers={**headers, "Idempotency-Key": "package-failure"},
            json={
                "message": "未配置模型时应明确失败",
                "thread_id": threads[0]["id"],
                "document_ids": [document_id],
            },
        )
        assert response.status_code == 202
        deadline = time.monotonic() + 20
        while True:
            run = client.get("/api/v1/runs/" + response.json()["id"], headers=headers).json()
            if run["status"] == "failed":
                assert "模型未配置" in run["error"], "失败原因不是缺少模型配置"
                break
            assert time.monotonic() < deadline, "缺模型任务未明确失败"
            time.sleep(0.1)
    runs = client.get("/api/v1/runs", headers=headers).json()
    assert len(runs) == 1 and runs[0]["document_references"][0]["id"] == document_id
    assert runs[0]["model_parameters"] == {}
    assert runs[0]["status"] == "failed" and "模型未配置" in runs[0]["error"]
    notifications = client.get("/api/v1/notifications", headers=headers).json()
    assert len(notifications) == 1 and notifications[0]["type"] == "failed"
    if prepare:
        assert (
            client.post(
                f"/api/v1/notifications/{notifications[0]['id']}/read", headers=headers
            ).status_code
            == 200
        )
    assert client.get("/api/v1/notifications/unread", headers=headers).json()["count"] == 0


def stop_process(process, graceful_timeout=30):
    """只停止本脚本启动的进程，不按端口或程序名回收其他服务。"""
    if process.poll() is None:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.terminate()
    try:
        process.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            taskkill = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"
            result = subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode and process.poll() is None:
                result.check_returncode()
        else:
            process.kill()
        process.wait(timeout=10)


@contextmanager
def running_service(command, root, data_dir, environment, port, expected, log_name):
    service_command = [
        *command,
        "--no-browser",
        "--port",
        str(port),
        "--data-dir",
        str(data_dir),
    ]
    log_path = root / log_name
    with (
        log_path.open("ab") as log,
        httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5, trust_env=False) as client,
    ):
        log_offset = log.tell()
        process = subprocess.Popen(
            service_command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=log,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        try:
            wait_ready(client, process, log_path, log_offset)
            assert client.get("/").status_code == 200
            assert hashlib.sha256(client.get("/static/app.js").content).digest() == expected
            yield client
        except Exception:
            log.flush()
            print(log_path.read_text(encoding="utf-8", errors="replace")[-5000:])
            raise
        finally:
            stop_process(process)


def wait_run(client, headers, run_id, statuses):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}", headers=headers)
        assert response.status_code == 200
        run = response.json()
        if run["status"] in statuses:
            return run
        time.sleep(0.1)
    raise AssertionError("安装包功能任务未在限定时间内到达验收状态")


async def read_event_chunks(base_url, headers, run_id, timeout):
    address = httpx.URL(base_url)
    assert address.scheme == "http" and address.host == "127.0.0.1", "验收事件仅允许本机 HTTP"
    chunks, size = [], 0
    try:
        async with asyncio.timeout(timeout):
            async with httpx.AsyncClient(
                # 固定本机 HTTP 验收，不加载未使用的 TLS 证书库，避免同步初始化阻塞截止。
                base_url=base_url,
                timeout=timeout,
                trust_env=False,
                verify=False,
            ) as client:
                async with client.stream(
                    "GET", f"/api/v1/runs/{run_id}/events", headers=headers
                ) as response:
                    assert response.status_code == 200
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        assert size <= 2_000_000, "确定性事件流超过验收上限"
                        chunks.append(chunk)
    except TimeoutError as error:
        raise TimeoutError("安装包事件流超过总截止时间") from error
    return chunks


def read_events(client, headers, run_id, timeout=20):
    chunks = asyncio.run(read_event_chunks(client.base_url, headers, run_id, timeout))
    text = b"".join(chunks).decode("utf-8").replace("\r\n", "\n")
    events = []
    for block in text.split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        if "event" in fields and "data" in fields:
            events.append(
                {
                    "id": int(fields["id"]),
                    "type": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    assert len(events) <= 256 and events, "夹具事件数量无效"
    ids = [event["id"] for event in events]
    assert ids == sorted(set(ids))
    return events


def verify_feature_frames(events, final=False):
    deltas = [event["data"] for event in events if event["type"] == "model_output_delta"]
    assert deltas and all(isinstance(delta.get("text"), str) for delta in deltas)
    calls = {}
    for delta in deltas:
        previous = calls.setdefault(delta["call_id"], {"text": "", "seq": 0})
        assert delta["offset"] == len(previous["text"]) and delta["seq"] > previous["seq"]
        previous["text"] += delta["text"]
        previous["seq"] = delta["seq"]
    assert any(value["text"] == model_fixture.INTRO for value in calls.values())
    finished = [event["data"] for event in events if event["type"] == "model_output_finished"]
    tools_finished = [item for item in finished if item["reason"] == "tools"]
    retracted = [event["data"] for event in events if event["type"] == "model_output_retracted"]
    assert tools_finished and all(item["streamed"] is True for item in tools_finished)
    for item in tools_finished:
        assert item["call_id"] in calls
        assert any(
            retract["call_id"] == item["call_id"]
            and retract["reason"] == "tools"
            and calls[item["call_id"]]["seq"] < retract["seq"] < item["seq"]
            for retract in retracted
        ), "工具输出缺少同一 call_id 的先行撤销事件"
    if final:
        assert any(value["text"] == model_fixture.ANSWER for value in calls.values())
        assert any(item["reason"] == "answer" and item["streamed"] is True for item in finished)


def verify_completed_feature(client, headers, run_id, target):
    response = client.get(f"/api/v1/runs/{run_id}", headers=headers)
    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "completed" and run["answer"] == model_fixture.ANSWER, run
    assert run["message"] == model_fixture.TASK_MESSAGE
    assert target.read_bytes() == model_fixture.FILE_CONTENT.encode()
    preview = client.get(
        f"/api/v1/runs/{run_id}/file", headers=headers, params={"path": model_fixture.FILE_PATH}
    )
    assert preview.status_code == 200 and preview.json()["content"] == model_fixture.FILE_CONTENT
    events = read_events(client, headers, run_id)
    verify_feature_frames(events, final=True)
    results = [event["data"] for event in events if event["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["call_id"] == model_fixture.CALL_ID
    assert results[0]["result"] == {
        "status": "written",
        "path": model_fixture.FILE_PATH,
        "bytes": len(model_fixture.FILE_CONTENT.encode()),
        "sha256": hashlib.sha256(model_fixture.FILE_CONTENT.encode()).hexdigest(),
    }
    return events


def verify_feature_ledger(data_dir, run_id):
    # 服务停机后只读安装包真实 SQLite，核验 done，而不调用仓库内核替代安装程序。
    database = data_dir / "aegis.db"
    assert database.is_file()
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT call_id, name, status, arguments, result "
            "FROM harness_tool_calls WHERE run_id=?",
            (run_id,),
        ).fetchall()
    assert len(rows) == 1
    call_id, name, status, arguments, result = rows[0]
    assert (call_id, name, status) == (model_fixture.CALL_ID, "file_write", "done")
    assert json.loads(arguments) == model_fixture.arguments()
    assert json.loads(result)["status"] == "written"


def port_conflict_logged(output):
    """准确匹配端口占用原因，兼容 Python 与冻结程序的中文错误编码。"""
    return any("端口已被占用".encode(encoding) in output for encoding in ("utf-8", "gb18030"))


def verify_port_conflict(command, root, environment, expected):
    """真实占用端口必须使自建服务失败，且不得探测占用者。"""
    with model_fixture.LocalModelFixture() as occupied:
        log_name = "port-conflict.log"
        try:
            with running_service(
                command,
                root,
                root / "port-conflict-data",
                environment,
                occupied.server.server_port,
                expected,
                log_name,
            ):
                raise AssertionError("占用端口时错误接受了其他服务")
        except ProcessExited as error:
            assert error.returncode not in {None, 0}, "端口占用必须使本次进程非零退出"
            assert port_conflict_logged(error.output), "启动失败原因不是端口已被占用"
        with occupied.server.connection_lock:
            assert occupied.server.connections == 0, "向端口占用者发送了连接或业务 HTTP"
    print("安装包占用端口负例通过：启动明确失败且占用者零连接")


def verify_features(command, root, environment, port, expected):
    verify_port_conflict(command, root, environment, expected)
    data_dir = root / "feature-data"
    with model_fixture.LocalModelFixture() as fixture:
        feature_environment = {**environment, **fixture.environment()}
        with running_service(
            command, root, data_dir, feature_environment, port, expected, "feature-service.log"
        ) as client:
            headers, user = login_session(client, True, "package-feature-acceptance")
            response = client.post(
                "/api/v1/runs",
                headers={**headers, "Idempotency-Key": "package-features"},
                json={"message": model_fixture.TASK_MESSAGE, "mode": "react"},
            )
            assert response.status_code == 202, response.text
            run_id = response.json()["id"]
            pending = wait_run(
                client, headers, run_id, {"waiting_approval", "failed", "interrupted"}
            )
            assert pending["status"] == "waiting_approval", pending
            approval = pending["approval"]
            events = read_events(client, headers, run_id)
            frozen = approval.get("file_write")
            missing = []
            if not any(event["type"] == "model_output_delta" for event in events):
                missing.append("公开模型增量帧")
            if not isinstance(frozen, dict) or not frozen.get("baseline_hash"):
                missing.append("file_write 冻结基线与差异字段")
            assert not missing, "安装包功能缺失：" + "、".join(missing)
            verify_feature_frames(events)
            assert approval["name"] == "file_write" and approval["call_id"] == model_fixture.CALL_ID
            assert approval["arguments"] == model_fixture.arguments()
            assert frozen["version"] == 1 and frozen["path"] == model_fixture.FILE_PATH
            assert frozen["baseline"]["exists"] is False and frozen["baseline"]["sha256"] is None
            assert frozen["baseline"]["bytes"] == 0
            assert frozen["preview"]["before"] == ""
            assert frozen["preview"]["after"] == model_fixture.FILE_CONTENT
            assert "+" in frozen["preview"]["diff"] and frozen["preview"]["truncated"] is False
            assert len(frozen["baseline_hash"]) == 64
            assert (
                frozen["proposed_sha256"]
                == hashlib.sha256(model_fixture.FILE_CONTENT.encode()).hexdigest()
            )
            key = hashlib.sha256(
                f"{user['tenant_id']}:{user['user_id']}:{run_id}".encode()
            ).hexdigest()
            assert frozen["workspace_key"] == key
            target = data_dir / "workspaces" / key / model_fixture.FILE_PATH
            assert not (data_dir / "workspaces").exists() and not target.exists()
            assert (
                client.get(
                    f"/api/v1/runs/{run_id}/file",
                    headers=headers,
                    params={"path": model_fixture.FILE_PATH},
                ).status_code
                == 404
            )
            body = {
                "approved": True,
                "call_id": approval["call_id"],
                "args_hash": approval["hash"],
                "baseline_hash": "0" * 64,
            }
            endpoint = f"/api/v1/runs/{run_id}/approval"
            assert client.post(endpoint, headers=headers, json=body).status_code == 409
            assert not target.exists() and not (data_dir / "workspaces").exists()
            unchanged = client.get(f"/api/v1/runs/{run_id}", headers=headers).json()
            assert unchanged["approval"] == approval and unchanged["status"] == "waiting_approval"
            body["baseline_hash"] = frozen["baseline_hash"]
            assert client.post(endpoint, headers=headers, json=body).status_code == 200
            done = wait_run(client, headers, run_id, {"completed", "failed", "interrupted"})
            assert done["status"] == "completed", done
            stored_events = verify_completed_feature(client, headers, run_id, target)
        verify_feature_ledger(data_dir, run_id)
        first_requests = fixture.snapshot()
        assert first_requests == {
            "requests": [{"phase": "write", "stream": True}, {"phase": "answer", "stream": True}],
            "errors": [],
        }, first_requests
        with running_service(
            command, root, data_dir, feature_environment, port, expected, "feature-service.log"
        ) as client:
            headers, _ = login_session(client, False, "package-feature-acceptance")
            assert verify_completed_feature(client, headers, run_id, target) == stored_events
            time.sleep(0.5)
            assert fixture.snapshot() == first_requests, "重启后重复调用了已完成任务的模型"
        verify_feature_ledger(data_dir, run_id)
        assert fixture.snapshot() == first_requests
    print("安装包功能验收通过：真实 SDK 流、公开增量、冻结差异审批、中文字节与重启不重放")
    print("本机确定性夹具禁用了额外风险分类和演化作业；未验证商业模型、生产延迟或生产风险分类")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--static", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--features", action="store_true", help="追加真实 SDK 流和文件差异审批验收")
    parser.add_argument(
        "--harness", action="store_true", help="追加 Plan、来源 Skill 与反馈修订专项"
    )
    args = parser.parse_args()
    if bool(args.wheel) == bool(args.setup):
        parser.error("只能选择 wheel 或 Setup 一种产物")
    expected = hashlib.sha256(args.static.read_bytes()).digest()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AEGIS_", "OPENAI_", "ANTHROPIC_", "AZURE_", "OLLAMA_"))
        and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="aegis-安装验收-") as temporary:
        root = Path(temporary)
        if args.wheel:
            # 应用及依赖安装到独立 venv，避免源码遮蔽或本机依赖掩盖缺失。
            subprocess.run(
                [sys.executable, "-m", "venv", str(root / "venv")],
                check=True,
                timeout=60,
            )
            python = root / ("venv/Scripts/python.exe" if os.name == "nt" else "venv/bin/python")
            subprocess.run(
                [str(python), "-m", "pip", "install", str(args.wheel.resolve())],
                cwd=root,
                env=environment,
                check=True,
                timeout=180,
            )
            probe = (
                subprocess.check_output(
                    [str(python), "-c", "import app; print(app.__file__)"],
                    cwd=root,
                    env=environment,
                    timeout=30,
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
        for marker in ("--migrate-threads", "--migrate-notifications", "--migrate-schedules"):
            subprocess.run(
                [*command, marker, "--help"],
                cwd=root,
                env=environment,
                capture_output=True,
                check=True,
                timeout=30,
            )
        for prepare in (True, False):
            with running_service(
                command, root, root / "data", environment, args.port, expected, "service.log"
            ) as client:
                verify_session(client, prepare)
            if prepare:
                # 服务已退出后执行真实迁移预览，验证安装包内包含迁移运行模块。
                subprocess.run(
                    [*command, "--migrate-schedules", "--data-dir", str(root / "data")],
                    cwd=root,
                    env=environment,
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
        print(
            "安装包验收通过：独立中文目录、静态校验、模板、资料快照、暂停计划、模型能力、登录、多会话、通知、迁移入口及重启持久化"
        )
        if args.features:
            verify_features(command, root, environment, args.port, expected)
        if args.harness:
            try:
                from .package_harness_acceptance import verify_harness
            except ImportError:
                from package_harness_acceptance import verify_harness
            verify_harness(command, root, environment, args.port, expected)


if __name__ == "__main__":
    main()
