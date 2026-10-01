"""本机安装入口的 Skill 持久化验收，不连接真实模型。"""

import argparse
import json
from urllib.parse import urlparse

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    assert urlparse(args.url).hostname in {"127.0.0.1", "localhost"}, "只连接本机验收服务"
    credentials = {"username": args.account, "password": "skill-restart-test-2026"}
    with httpx.Client(base_url=args.url + "/api/v1", timeout=15) as client:
        if args.prepare:
            assert client.post("/auth/register", json=credentials).status_code == 201
        login = client.post("/auth/login", json=credentials)
        assert login.status_code == 200
        client.headers["Authorization"] = "Bearer " + login.json()["token"]
        if args.prepare:
            imported = client.post(
                "/skills/import",
                json={
                    "document": (
                        "---\nname: restart-review\ndescription: 重启后复用检查清单\n"
                        "allowed-tools: [file_read]\n---\n先读取文件再检查边界。"
                    ),
                    "directory": "coding/review",
                    "resources": {"references/checklist.md": "检查空输入和异常路径"},
                },
            )
            assert imported.status_code == 201, imported.text
            asset = imported.json()
            assert (
                client.post(f"/assets/{asset['id']}/state", json={"status": "active"}).status_code
                == 200
            )
        assets = client.get("/assets", params={"kind": "skill"}).json()
        asset = next(a for a in assets if a["name"] == "restart-review")
        assert asset["status"] == "active"
        bundle = client.get(f"/skills/{asset['id']}/export")
        assert bundle.status_code == 200
        assert bundle.json()["directory"] == "coding/review"
        assert bundle.json()["resources"] == {"references/checklist.md": "检查空输入和异常路径"}
        response = client.get(
            f"/skills/{asset['id']}/resources", params={"path": "references/checklist.md"}
        )
        assert response.status_code == 200 and "空输入" in response.json()["content"]
        print(
            json.dumps(
                {"阶段": "重启前" if args.prepare else "重启后", "技能持久数据": "通过"},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
