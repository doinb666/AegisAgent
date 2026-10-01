"""本机独立验收账号：部署重启前写入，重启后验证原有资产和任务。"""

import argparse
import json
import time
from urllib.parse import urlparse

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    assert urlparse(args.url).hostname in {"127.0.0.1", "localhost"}, "只连接本机验收服务"
    credentials = {"username": args.account, "password": "restart-test-only-2026"}
    with httpx.Client(base_url=args.url + "/api/v1", timeout=15) as client:
        if args.prepare:
            registered = client.post("/auth/register", json=credentials)
            assert registered.status_code in {201, 409}, registered.text
        login = client.post("/auth/login", json=credentials)
        assert login.status_code == 200, login.text
        client.headers["Authorization"] = "Bearer " + login.json()["token"]
        if args.prepare:
            asset = client.post(
                "/assets",
                json={
                    "kind": "profile",
                    "name": "重启保留偏好",
                    "content": "使用中文",
                    "status": "active",
                },
            )
            assert asset.status_code == 201, asset.text
            uploaded = client.post(
                "/documents/upload",
                files={
                    "file": ("restart.md", "重启后仍可召回本人知识".encode(), "text/markdown"),
                },
            )
            assert uploaded.status_code == 201, uploaded.text
            run = client.post(
                "/runs",
                headers={"Idempotency-Key": "restart-preserved-task"},
                json={"message": "测试未配置模型时明确失败"},
            )
            assert run.status_code == 202, run.text
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                state = client.get("/runs/" + run.json()["id"]).json()
                if state["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.1)
            assert state["status"] == "failed", "验收服务必须不配置真实模型"
        bootstrap = client.get("/auth/me").json()["bootstrap"]
        assert any(asset["name"] == "重启保留偏好" for asset in bootstrap["memories"])
        assert any(asset["name"] == "restart.md" for asset in client.get("/documents").json())
        runs = client.get("/runs").json()
        assert any(
            run["message"] == "测试未配置模型时明确失败" and run["status"] == "failed"
            for run in runs
        )
        print(
            json.dumps(
                {"阶段": "重启前" if args.prepare else "重启后", "持久数据": "通过"},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
