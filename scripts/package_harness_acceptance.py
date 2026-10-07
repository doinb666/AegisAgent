"""从生产启动器经 HTTP 验收 Harness 新能力；--source 明确仅为源码研发验证。"""

import argparse
import hashlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import closing, contextmanager
from pathlib import Path

try:
    from . import package_acceptance as package
    from . import package_harness_fixture as fixture_module
except ImportError:
    import package_acceptance as package
    import package_harness_fixture as fixture_module


def wait_until(probe, label, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = probe()
        if result:
            return result
        time.sleep(0.1)
    raise AssertionError("专项超时：" + label)


def get_json(client, headers, endpoint):
    response = client.get(endpoint, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def assets(client, headers):
    return get_json(client, headers, "/api/v1/assets")


def submit(client, headers, message, key, mode="react"):
    response = client.post(
        "/api/v1/runs",
        headers={**headers, "Idempotency-Key": key},
        json={"message": message, "mode": mode},
    )
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    done = package.wait_run(
        client, headers, run_id, {"completed", "failed", "interrupted", "waiting_approval"}
    )
    assert done["status"] == "completed", done
    # 终态后的演化也应全部结束，再采集稳定事件。
    wait_until(
        lambda: any(
            event["type"] == "evolution" and event["data"]["state"] in {"done", "skipped"}
            for event in package.read_events(client, headers, run_id)
        ),
        "后台经验整理",
    )
    return run_id


def recalled(events):
    return next(event["data"]["assets"] for event in events if event["type"] == "assets_recalled")


def verify_plan(events):
    plans = [event["data"] for event in events if event["type"] == "plan"]
    assert len(plans) == 1, "安装包功能缺失：持久结构化 Plan 事件"
    nodes = plans[0]["nodes"]
    assert len(nodes) == 2 and nodes[0]["acceptance"] == {
        "output_kind": "tool_evidence",
        "required_tools": ["calculator"],
    }
    assert nodes[1]["inputs"]["from_nodes"] == [nodes[0]["id"]]
    accepted = [event["data"] for event in events if event["type"] == "node_accepted"]
    assert len(accepted) == 2 and {item["node_id"] for item in accepted} == {
        node["id"] for node in nodes
    }
    assert all(
        item["contract_satisfied"]
        and item["correctness_verified"] is False
        and item["business_success_verified"] is False
        for item in accepted
    )
    evidence = next(item for item in accepted if item["node_id"] == nodes[0]["id"])
    assert evidence["ledger_sources"], "节点缺少真实账本证据"
    results = [event["data"] for event in events if event["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["call_id"] == fixture_module.CALL_IDS["plan_tool"]
    assert results[0]["result"] == fixture_module.CALCULATOR_RESULT, results


def ledger_snapshot(data_dir, plan_id, source_id):
    with closing(
        sqlite3.connect((data_dir / "aegis.db").resolve().as_uri() + "?mode=ro", uri=True)
    ) as connection:
        connection.row_factory = sqlite3.Row
        # 完整行保留所有列名和值，后续新增账本列也参与重启比较。
        calls = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM harness_tool_calls ORDER BY run_id,call_id"
            )
        ]
        assert len(calls) == 2, calls
        assert {(row["run_id"], row["call_id"]) for row in calls} == {
            (plan_id, fixture_module.CALL_IDS["plan_tool"]),
            (source_id, fixture_module.CALL_IDS["tool"]),
        }
        assert all(
            row["name"] == "calculator"
            and row["status"] == "done"
            and json.loads(row["arguments"]) == {"expression": "2+3"}
            and json.loads(row["result"]) == fixture_module.CALCULATOR_RESULT
            for row in calls
        )
        row = connection.execute(
            "SELECT config FROM harness_runs WHERE id=?", (plan_id,)
        ).fetchone()
        plan = json.loads(row[0])["plan"]
        assert plan["status"] == "completed" and all(
            node["status"] == "completed" for node in plan["nodes"]
        )
        ledger_id = next(row["id"] for row in calls if row["run_id"] == plan_id)
        assert plan["nodes"][0]["tool_call_ids"] == [ledger_id]
        sources = plan["nodes"][0]["acceptance_result"]["ledger_sources"]
        assert len(sources) == 1 and sources[0]["id"] == ledger_id
        assert sources[0]["run_id"] == plan_id and sources[0]["name"] == "calculator"
        jobs = []
        for run_id, config in connection.execute("SELECT id,config FROM harness_runs ORDER BY id"):
            config = json.loads(config)
            evolution = config.get("evolution_state")
            revision = config.get("skill_revision", {}).get("state")
            assert evolution in {"done", "skipped"}, "经验整理仍处于未完成状态"
            assert revision in {None, "done", "skipped"}, "修订仍处于未完成状态"
            jobs.append((run_id, evolution, revision))
        return {"calls": calls, "plan": plan, "jobs": jobs}


@contextmanager
def failure_report(fixture):
    try:
        yield
    except Exception:
        print("失败时模型请求统计：" + fixture_module.encoded(fixture.snapshot()))
        raise


def verify_harness(command, root, environment, port, expected):
    data_dir = root / "Harness中文数据"
    print("验收静态SHA256：" + expected.hex())
    with fixture_module.LocalHarnessFixture() as fixture, failure_report(fixture):
        environment = {**environment, **fixture.environment()}
        with package.running_service(
            command, root, data_dir, environment, port, expected, "harness-service.log"
        ) as client:
            headers, _ = package.login_session(client, True, "harness-owner")
            outsider, _ = package.login_session(client, True, "harness-other")
            plan_id = submit(client, headers, fixture_module.PLAN_TASK, "harness-plan", "plan")
            verify_plan(package.read_events(client, headers, plan_id))
            home = client.get("/").text
            assert 'id="plan-progress"' in home and 'id="feedback-note"' in home, (
                "安装包首页缺少计划或反馈说明"
            )
            source_id = submit(client, headers, fixture_module.TOOL_TASK, "harness-source")
            candidates = [
                asset
                for asset in assets(client, headers)
                if asset["name"] == fixture_module.SKILL_NAME
                and asset["metadata"].get("source_run_id") == source_id
            ]
            assert len(candidates) == 1, "实际工具轨迹未提炼唯一来源 Skill"
            source = candidates[0]
            assert source["status"] == "draft" and source["metadata"]["extracted"] is True
            assert (
                source["metadata"]["tools"] == ["calculator"]
                and source["metadata"]["tool_evidence"]
            )
            assert source["content"] == fixture_module.SKILL_CONTENT
            before_id = submit(client, headers, fixture_module.RECALL_TASK, "harness-before")
            assert source["id"] not in {
                item["id"] for item in recalled(package.read_events(client, headers, before_id))
            }, "来源草稿被自动召回"
            response = client.post(
                f"/api/v1/assets/{source['id']}/state", headers=headers, json={"status": "active"}
            )
            assert response.status_code == 200, response.text
            source = get_json(client, headers, f"/api/v1/assets/{source['id']}")
            source_history = get_json(client, headers, f"/api/v1/assets/{source['id']}/history")
            recall_id = submit(client, headers, fixture_module.RECALL_TASK, "harness-recall")
            recall_events = package.read_events(client, headers, recall_id)
            expected_ref = {
                "id": source["id"],
                "version": source["version"],
                "source_run_id": source_id,
            }
            assert expected_ref in recalled(recall_events), "人工启用后新任务未真实召回来源版本"
            feedback_body = {
                "success": False,
                "note": "安装专项：召回方法未满足输入边界，需人工复核；失败因果未验证。",
            }
            response = client.post(
                f"/api/v1/runs/{recall_id}/feedback", headers=headers, json=feedback_body
            )
            assert response.status_code == 200, response.text

            def find_revision():
                return [
                    asset
                    for asset in assets(client, headers)
                    if asset["metadata"].get("revision_original_id") == source["id"]
                ]

            revisions = wait_until(find_revision, "失败反馈生成修订草稿")
            assert len(revisions) == 1
            revision = revisions[0]
            metadata = revision["metadata"]
            assert (
                revision["status"] == "draft"
                and revision["version"] == 1
                and revision["content"].endswith(fixture_module.REVISION_CONTENT)
            )
            assert (
                metadata["revision_original_version"] == source["version"]
                and metadata["source_run_id"] == recall_id
            )
            assert metadata["revision_original_source_run_id"] == source_id
            assert all(
                metadata[key] is False
                for key in ("repair_verified", "manual_verified", "user_verified")
            )
            assert get_json(client, headers, f"/api/v1/assets/{source['id']}") == source, (
                "失败反馈修改了原 Skill"
            )
            assert (
                get_json(client, headers, f"/api/v1/assets/{source['id']}/history")
                == source_history
            )
            after_id = submit(client, headers, fixture_module.AFTER_TASK, "harness-after")
            references = recalled(package.read_events(client, headers, after_id))
            assert expected_ref in references and revision["id"] not in {
                item["id"] for item in references
            }, "修订草稿自动召回"
            counts = fixture.snapshot()
            assert (
                client.post(
                    f"/api/v1/runs/{recall_id}/feedback", headers=headers, json=feedback_body
                ).status_code
                == 200
            )
            time.sleep(2.5)
            assert fixture.snapshot() == counts and find_revision() == revisions, (
                "重复反馈新增请求或草稿"
            )
            for endpoint in (
                f"/api/v1/runs/{recall_id}",
                f"/api/v1/runs/{recall_id}/events",
                f"/api/v1/assets/{source['id']}",
                f"/api/v1/assets/{revision['id']}",
                f"/api/v1/assets/{source['id']}/history",
            ):
                assert client.get(endpoint, headers=outsider).status_code in {403, 404}, (
                    "跨账号GET越权：" + endpoint
                )
            ids = [plan_id, source_id, before_id, recall_id, after_id]
            stored_runs = {
                run_id: get_json(client, headers, f"/api/v1/runs/{run_id}") for run_id in ids
            }
            stored_events = {run_id: package.read_events(client, headers, run_id) for run_id in ids}
            stored_assets = assets(client, headers)
        stored_ledger = ledger_snapshot(data_dir, plan_id, source_id)
        assert not counts["errors"], counts
        phases = [item["phase"] for item in counts["requests"]]
        assert (
            phases.count("revision") == 1 and phases.count("tool") == phases.count("plan_tool") == 1
        ), counts
        with package.running_service(
            command, root, data_dir, environment, port, expected, "harness-service.log"
        ) as client:
            headers, _ = package.login_session(client, False, "harness-owner")
            assert {
                run_id: get_json(client, headers, f"/api/v1/runs/{run_id}") for run_id in ids
            } == stored_runs
            assert {
                run_id: package.read_events(client, headers, run_id) for run_id in ids
            } == stored_events
            assert assets(client, headers) == stored_assets
            assert (
                get_json(client, headers, f"/api/v1/assets/{source['id']}/history")
                == source_history
            )
            time.sleep(2.5)
            assert fixture.snapshot() == counts, "重启增加模型请求"
        assert ledger_snapshot(data_dir, plan_id, source_id) == stored_ledger
        assert fixture.snapshot() == counts
        print(
            "Harness专项通过：真实calculator账本2条、Plan两节点、来源草稿、人工启用、召回、失败反馈与幂等修订、跨账号拒绝、重启不重放"
        )
        print("模型请求统计：" + fixture_module.encoded(counts))
        print("构建静态SHA256：" + expected.hex())
        print("本机协议替身仅验证协议与持久化，不代表真实供应商质量；风险分类关闭，演化开启。")


def extract_portable(path, target):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            resolved = (target / name).resolve()
            assert resolved.is_relative_to(target.resolve()), "便携ZIP包含路径遍历"
        archive.extractall(target)
    executables = list(target.rglob("AegisCode.exe"))
    assert len(executables) == 1, "便携ZIP必须包含唯一启动器"
    return [str(executables[0])]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--source", action="store_true", help="仅本机源码研发验证，不能充当安装包绿测"
    )
    mode.add_argument("--portable", type=Path)
    mode.add_argument("--wheel", type=Path)
    mode.add_argument("--setup", type=Path)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--port", type=int, help="默认使用随机本机端口；禁止占用8001/8002/8003")
    args = parser.parse_args()
    if args.port in {8001, 8002, 8003} or (args.port is not None and not 1 <= args.port <= 65535):
        parser.error("端口无效或属于现有预览服务")
    expected = hashlib.sha256(args.static.read_bytes()).digest()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AEGIS_", "OPENAI_", "ANTHROPIC_", "AZURE_", "OLLAMA_"))
        and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    with closing(socket.socket()) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = args.port or reservation.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="aegis-Harness安装专项-") as temporary:
        root = Path(temporary)
        if args.source:
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            command = [sys.executable, "-m", "app.launcher"]
            print("源码研发验证：生产launcher；本次结果不是安装包验收，独立中文cwd与数据库。")
        elif args.portable:
            command = extract_portable(args.portable.resolve(), root / "中文便携目录")
        elif args.setup:
            target = root / "中文安装目录"
            subprocess.run(
                [str(args.setup.resolve()), "--target", str(target)],
                cwd=root,
                env=environment,
                check=True,
                timeout=120,
            )
            command = [str(target / "AegisCode.exe")]
        else:
            subprocess.run(
                [sys.executable, "-m", "venv", str(root / "venv")], check=True, timeout=60
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
            assert Path(probe).is_relative_to(root / "venv"), "检测到源码遮蔽wheel"
            print("独立wheel实际import：" + probe)
            command = [str(python), "-m", "app.launcher"]
        verify_harness(command, root, environment, port, expected)


if __name__ == "__main__":
    main()
