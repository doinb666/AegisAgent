"""真实临时 Git 仓库、审批、父子工具账本与 HTTP 配置的集成验收。"""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.harness.settings import HarnessSettings
from app.main import create_app
from tests.test_harness_api import account, wait_run


class CollaborationModel:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        if not kwargs.get("tools"):
            return SimpleNamespace(
                content='{"steps":["读取","复核"]}', model_id="test", usage={}, raw={}
            )
        task = next(item["content"] for item in messages if item["role"] == "user")
        tools = {
            call["function"]["name"]
            for message in messages
            for call in message.get("tool_calls", [])
        }
        calls = []
        if "父整合" in task and "delegate" not in tools:
            name, arguments = (
                "delegate",
                {
                    "tasks": [
                        {
                            "id": "inspect",
                            "message": "NODE_A 读取代码",
                            "acceptance": {"required_tools": ["file_read"]},
                        },
                        {
                            "id": "review",
                            "message": "NODE_B 独立读取复核",
                            "depends_on": ["inspect"],
                            "acceptance": {"required_tools": ["file_read"]},
                        },
                    ]
                },
            )
        elif "父整合" not in task and "file_read" not in tools:
            name, arguments = "file_read", {"path": "src/说明.py"}
        else:
            name = None
        if name:
            calls = [
                {
                    "id": f"call-{self.calls}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
        return SimpleNamespace(
            content="" if calls else "已核对代码证据",
            model_id="test",
            usage={},
            raw={"choices": [{"message": {"tool_calls": calls}}]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("project_mode", ["fork", "worktree"])
@pytest.mark.parametrize("mode", ["react", "plan"])
async def test_configured_project_preparation_approval_and_dependent_children(
    tmp_path, project_mode, mode
):
    git = shutil.which("git") or "C:/Program Files/Git/cmd/git.exe"
    if not Path(git).is_file():
        pytest.skip("集成验收需要 Git")
    repo = tmp_path / "源仓库"
    (repo / "src").mkdir(parents=True)
    (repo / "src/说明.py").write_text("print('安全代码')\n", encoding="utf-8")
    for arguments in (
        ["init"],
        ["add", "."],
        [
            "-c",
            "user.name=验收",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "初始代码",
        ],
    ):
        subprocess.run([git, *arguments], cwd=repo, check=True, capture_output=True)
    model = CollaborationModel()
    app = create_app(
        HarnessSettings(
            data_dir=tmp_path / "data",
            repository_root=repo,
            git_executable=git,
            max_steps=16,
            risk_review_enabled=False,
            evolution_enabled=False,
        ),
        model,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        headers = await account(client, "集成账号")
        request = {
            "message": "父整合代码",
            "collaboration_mode": "team",
            "project_mode": project_mode,
            "mode": mode,
        }
        result = await client.post(
            "/api/v1/runs", headers={**headers, "Idempotency-Key": "integration"}, json=request
        )
        assert result.status_code == 202, result.text
        run_id = result.json()["id"]
        pending = await wait_run(
            client, headers, run_id, {"waiting_approval", "completed", "failed"}
        )
        assert pending["status"] == "waiting_approval", pending
        assert pending["approval"]["name"] == "project_prepare" and model.calls == 0
        assert pending["approval"]["arguments"]["mode"] == project_mode
        assert not list((tmp_path / "data").glob("workspaces/**/说明.py"))
        approved = await client.post(
            f"/api/v1/runs/{run_id}/approval",
            headers=headers,
            json={
                "approved": True,
                "call_id": pending["approval"]["call_id"],
                "args_hash": pending["approval"]["hash"],
            },
        )
        assert approved.status_code == 200
        completed = await wait_run(
            client,
            headers,
            run_id,
            {"completed", "failed", "interrupted"},
            timeout=20,
        )
        assert completed["status"] == "completed", completed
        events = await app.state.harness.events(
            await app.state.harness.authenticate(headers["Authorization"][7:]), run_id
        )
        event_types = [event["type"] for event in events]
        if mode == "plan":
            assert "plan" in event_types, event_types
            assert event_types.index("project_prepared") < event_types.index("plan")
        else:
            assert "plan" not in event_types
        acceptances = [
            event["data"] for event in events if event["type"] == "collaboration_acceptance"
        ]
        assert len(acceptances) == 2 and all(value["status"] == "verified" for value in acceptances)
        assert all(value["evidence"][0]["name"] == "file_read" for value in acceptances)
        capability = (await client.get("/api/v1/capabilities", headers=headers)).json()
        assert capability["collaboration"]["project_modes"] == ["fork", "worktree"]
        listed = (await client.get("/api/v1/runs", headers=headers)).json()
        assert next(run for run in listed if run["id"] == run_id)["project_mode"] == project_mode
