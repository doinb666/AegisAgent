"""小规模只读协作图的无副作用规范化与校验。"""

from app.harness.errors import HarnessError


def parse_nodes(arguments, allowed_tools):
    tasks = arguments.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 2:
        raise HarnessError(422, "委派任务数量须为一至两个")
    if set(arguments) - {"tasks", "mode"} or arguments.get("mode", "team") not in {"fork", "team"}:
        raise HarnessError(422, "委派参数或模式无效")
    nodes = []
    for index, task in enumerate(tasks):
        if isinstance(task, str):
            task = {"id": str(index), "message": task}
        if not isinstance(task, dict) or set(task) - {"id", "message", "depends_on", "acceptance"}:
            raise HarnessError(422, "协作节点字段无效")
        identifier, message = task.get("id"), task.get("message")
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 64:
            raise HarnessError(422, "节点 ID 无效")
        if not isinstance(message, str) or not message.strip() or len(message) > 12000:
            raise HarnessError(422, "节点消息不能为空且最多12000字符")
        dependencies = task.get("depends_on", [])
        if (
            not isinstance(dependencies, list)
            or any(not isinstance(item, str) for item in dependencies)
            or len(set(dependencies)) != len(dependencies)
        ):
            raise HarnessError(422, "依赖节点列表无效")
        acceptance = task.get("acceptance", {})
        if not isinstance(acceptance, dict) or set(acceptance) - {"required_tools"}:
            raise HarnessError(422, "验收字段无效")
        required = acceptance.get("required_tools", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(item, str) for item in required)
            or len(set(required)) != len(required)
            or not set(required).issubset(allowed_tools)
        ):
            raise HarnessError(422, "必需工具必须来自父子只读权限交集")
        nodes.append(
            {
                "id": identifier,
                "message": message,
                "depends_on": dependencies,
                "acceptance": {"required_tools": required},
            }
        )
    identifiers = {node["id"] for node in nodes}
    if len(identifiers) != len(nodes):
        raise HarnessError(422, "节点 ID 重复")
    resolved = set()
    for node in nodes:
        if node["id"] in node["depends_on"] or not set(node["depends_on"]).issubset(identifiers):
            raise HarnessError(422, "节点自依赖或依赖不存在")
    while len(resolved) < len(nodes):
        ready = {
            node["id"]
            for node in nodes
            if node["id"] not in resolved and set(node["depends_on"]).issubset(resolved)
        }
        if not ready:
            raise HarnessError(422, "协作依赖包含环")
        resolved.update(ready)
    return nodes
