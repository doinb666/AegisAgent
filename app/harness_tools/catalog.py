"""工具目录与运行适配：权限由内核和工具双重核对。"""

import asyncio
import re
from copy import deepcopy

from jsonschema import validate

from app.core.tools.builtin.calculator import CalculatorTool
from app.harness_tools.mcp import MCPGateway
from app.harness_tools.project import ProjectManager
from app.harness_tools.sandbox import SandboxClient, bounded
from app.harness_tools.workspace import Workspace


def tool_schema(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string", "maxLength": 12000}
SCHEMAS = [
    tool_schema("calculator", "计算有限算术表达式", {"expression": STRING}, ["expression"]),
    tool_schema("knowledge_search", "在本人知识库中检索，返回证据块", {"query": STRING}, ["query"]),
    tool_schema(
        "skill_read",
        "只读本人已激活 Skill 正文或文本资源，不执行资源脚本",
        {
            "asset_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "path": {"type": "string", "minLength": 1, "maxLength": 128},
        },
        ["asset_id"],
    ),
    tool_schema(
        "artifact_read",
        "按 ID 与范围读取本人的外置工具结果",
        {
            "asset_id": STRING,
            "start": {"type": "integer", "minimum": 0},
            "length": {"type": "integer", "minimum": 1, "maximum": 8000},
        },
        ["asset_id"],
    ),
    tool_schema("file_read", "读取本任务工作区中的文本文件", {"path": STRING}, ["path"]),
    tool_schema(
        "file_write",
        "经用户审批写入本任务工作区",
        {"path": STRING, "content": STRING},
        ["path", "content"],
    ),
    tool_schema(
        "python_execute", "经审批在无网络 Docker 沙箱执行 Python", {"code": STRING}, ["code"]
    ),
    tool_schema("mcp_list", "查询运维批准的 MCP 服务工具目录", {"server": STRING}, ["server"]),
    tool_schema(
        "mcp_call",
        "经审批调用运维允许的外部 MCP 工具",
        {"server": STRING, "tool": STRING, "arguments": {"type": "object"}},
        ["server", "tool", "arguments"],
    ),
    tool_schema(
        "delegate",
        "主控委派最多两个只读子任务，并汇总结果",
        {
            "tasks": {"type": "array", "minItems": 1, "maxItems": 2, "items": STRING},
            "mode": {"type": "string", "enum": ["fork", "team"]},
        },
        ["tasks"],
    ),
    tool_schema(
        "project_prepare",
        "经审批从管理员绑定的仓库准备代码副本或Worktree",
        {"mode": {"type": "string", "enum": ["fork", "worktree"]}},
        ["mode"],
    ),
]


class HarnessTools:
    def __init__(self, settings):
        self.settings = settings
        self.service = None
        self.workspace = Workspace(settings.data_dir)
        self.mcp = MCPGateway(settings.mcp_servers_json)
        self.sandbox = SandboxClient(settings)
        self.knowledge = None
        self.projects = ProjectManager(settings, self.workspace)

    def catalog(self, principal):
        result = []
        for item in SCHEMAS:
            name = item["function"]["name"]
            if name.startswith("mcp_"):
                servers = self.mcp.authorized_servers(principal)
                if not servers:
                    continue
                item = deepcopy(item)
                item["function"]["parameters"]["properties"]["server"] = {
                    "type": "string",
                    "enum": servers,
                }
            if name == "delegate" and self.settings.max_child_runs < 1:
                continue
            if name == "project_prepare" and not self.settings.repository_root:
                continue
            if principal.role == "viewer" and self.requires_approval(name, {}):
                continue
            result.append(item)
        return result

    def requires_approval(self, name, arguments):
        return name in {"file_write", "python_execute", "mcp_call", "mcp_list", "project_prepare"}

    async def execute(self, name, arguments, principal, run_id):
        schema = next((s for s in self.catalog(principal) if s["function"]["name"] == name), None)
        if schema is None:
            raise PermissionError("没有使用此工具的权限")
        validate(arguments, schema["function"]["parameters"])
        if name == "calculator":
            return await CalculatorTool().execute(**arguments)
        if name == "knowledge_search":
            if self.knowledge is not None:
                return await self.knowledge.search(principal, arguments["query"])
            assets = await self.service.list_assets(principal, kind="document", status="active")
            tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", arguments["query"].lower()))
            ranked = sorted(
                assets, key=lambda a: sum(t in a["content"].lower() for t in tokens), reverse=True
            )
            return [
                {"id": a["id"], "source": a["name"], "content": a["content"][:2500]}
                for a in ranked[:5]
                if any(t in a["content"].lower() for t in tokens)
            ]
        if name == "artifact_read":
            asset = await self.service.get_asset(principal, arguments["asset_id"])
            start, length = arguments.get("start", 0), arguments.get("length", 3000)
            return {
                "id": asset["id"],
                "total": len(asset["content"]),
                "content": asset["content"][start : start + length],
            }
        if name == "skill_read":
            return await self.service.read_skill(principal, **arguments)
        if name == "file_read":
            return await self.workspace.read(principal, run_id, arguments["path"])
        if name == "file_write":
            return await self.workspace.write(principal, run_id, **arguments)
        if name == "python_execute":
            self.workspace.directory(principal, run_id)
            return await self.sandbox.execute(principal, run_id, arguments["code"])
        if name == "mcp_list":
            self.mcp.authorize(arguments["server"], principal)
            return await bounded(self.mcp.list_tools(arguments["server"]))
        if name == "mcp_call":
            self.mcp.authorize(arguments["server"], principal)
            return await bounded(self.mcp.call(**arguments))
        if name == "delegate":
            return await self._delegate(principal, run_id, arguments)
        if name == "project_prepare":
            return await self.projects.prepare(principal, run_id, arguments["mode"])
        raise ValueError("未知工具")

    async def _delegate(self, principal, run_id, arguments):
        parent = await self.service.get_run(principal, run_id)
        if parent.get("parent_run_id"):
            raise PermissionError("子 Agent 不允许递归委派")
        child_tools = ["calculator", "knowledge_search", "artifact_read", "skill_read"]
        children = []

        async def collect(child_id):
            for _ in range(200):
                parent_status = await self.service.get_run(principal, run_id)
                if parent_status["status"] in {"cancelled", "interrupted"}:
                    await self.service.cancel(principal, child_id)
                    return {"run_id": child_id, "status": "cancelled"}
                child = await self.service.get_run(principal, child_id)
                if child["status"] in {"completed", "failed", "cancelled", "interrupted"}:
                    return {
                        "run_id": child_id,
                        "status": child["status"],
                        "answer": (child.get("answer") or "")[:2000],
                        "error": child.get("error"),
                    }
                await asyncio.sleep(0.1)
            return {"run_id": child_id, "status": "queued", "message": "子任务继续运行，请按ID查询"}

        try:
            for index, message in enumerate(arguments["tasks"]):
                child = await self.service.create_run(
                    principal,
                    message,
                    f"{run_id}:child:{index}",
                    parent_run_id=run_id,
                    allowed_tools=child_tools,
                    max_steps=min(3, self.settings.max_steps),
                    session_id=parent["session_id"] if arguments.get("mode") == "fork" else None,
                )
                children.append(child["id"])
            return await asyncio.gather(*(collect(cid) for cid in children))
        finally:
            # 汇总超时或父执行被取消后，不留下继续运行的子任务。
            for child_id in children:
                child = await self.service.get_run(principal, child_id)
                if child["status"] in {"queued", "running", "waiting_approval"}:
                    await self.service.cancel(principal, child_id)
