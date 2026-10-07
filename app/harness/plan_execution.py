"""持久计划的纯契约；不授予权限、不读写账本、不判定答案语义正确。"""

import json
import math
from collections.abc import Iterable
from typing import Any

from app.harness.errors import HarnessError

MAX_PLAN_CHARACTERS = 32768
MAX_OUTPUT_CHARACTERS = 131072
MAX_EVIDENCE = 16
MAX_SUMMARY_CHARACTERS = 2000
MAX_RESULT_DEPTH = 16
MAX_RESULT_ITEMS = 2048
MAX_IDENTIFIER_CHARACTERS = 256
FAILED_STATUSES = {"failed", "error", "rejected", "cancelled", "interrupted", "unknown"}


def _invalid(reason: str) -> None:
    raise HarnessError(422, reason)


def _exact_fields(value: Any, fields: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        _invalid(f"{label}字段缺失或包含未允许字段")
    return value


def _text(value: Any, limit: int, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _invalid(f"{label}必须为非空文本，且不超过{limit}字符")
    return value.strip()


def _unique_strings(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not _identifier(item) for item in value)
        or len(set(value)) != len(value)
    ):
        _invalid(f"{label}必须为唯一非空字符串列表，且每项最多256字符")
    return list(value)


def _json_object(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            _invalid("计划JSON包含重复字段")
        value[key] = item
    return value


def normalize_plan(value: str | dict, allowed_tools: Iterable[str], revision: int = 0) -> dict:
    """生成服务端顺序节点；依赖为零基索引，工具名最多256字符。"""
    if type(revision) is not int or not 0 <= revision <= 2**31 - 1:
        _invalid("计划修订号必须为非布尔的非负32位整数")
    if isinstance(value, str) and len(value) > MAX_PLAN_CHARACTERS:
        _invalid("原始计划JSON不得超过32768字符")
    try:
        if isinstance(value, str):
            value = json.loads(value, object_pairs_hook=_json_object)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise HarnessError(422, "计划必须为仅包含有限数值的合法JSON") from error
    if len(canonical) > MAX_PLAN_CHARACTERS:
        _invalid("规范计划JSON不得超过32768字符")
    _exact_fields(value, {"steps"}, "计划")
    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 8:
        _invalid("计划步骤必须为一至八项列表")
    legacy = all(isinstance(step, str) for step in steps)
    if not legacy and not all(isinstance(step, dict) for step in steps):
        _invalid("计划步骤必须全部为字符串或全部为结构化对象")
    try:
        permitted = set(allowed_tools)
    except TypeError as error:
        raise HarnessError(422, "当前工具权限集合无效") from error
    nodes = []
    for index, step in enumerate(steps):
        if legacy:
            objective = _text(step, 2000, "步骤目标")
            instruction = objective
            dependencies = [index - 1] if index else []
            output_kind, required = "answer", []
        else:
            _exact_fields(step, {"objective", "inputs", "acceptance"}, "步骤")
            objective = _text(step["objective"], 2000, "步骤目标")
            inputs = _exact_fields(step["inputs"], {"instruction", "from_steps"}, "步骤输入")
            instruction = _text(inputs["instruction"], 4000, "步骤指令")
            dependencies = inputs["from_steps"]
            if (
                not isinstance(dependencies, list)
                or any(type(item) is not int or not 0 <= item < index for item in dependencies)
                or len(set(dependencies)) != len(dependencies)
            ):
                _invalid("前序依赖必须为唯一的零基整数索引，且仅能引用此前步骤")
            acceptance = _exact_fields(
                step["acceptance"], {"output_kind", "required_tools"}, "验收契约"
            )
            output_kind = acceptance["output_kind"]
            if output_kind not in ("answer", "tool_evidence"):
                _invalid("输出类型必须为answer或tool_evidence")
            required = _unique_strings(acceptance["required_tools"], "必需工具")
            if not set(required).issubset(permitted):
                _invalid("必需工具必须属于当前服务端工具权限集合")
            if output_kind == "tool_evidence" and not required:
                _invalid("工具证据契约必须指定至少一个必需工具")
        nodes.append(
            {
                "id": f"n{revision}-{index + 1}",
                "objective": objective,
                "inputs": {
                    "instruction": instruction,
                    "from_nodes": [f"n{revision}-{item + 1}" for item in dependencies],
                },
                "acceptance": {"output_kind": output_kind, "required_tools": required},
                "status": "pending",
                "proposed_calls": [],
                "tool_call_ids": [],
                "output": None,
                "acceptance_result": None,
                "blocked_reason": None,
            }
        )
    return {
        "version": 1,
        "revision": revision,
        "replans_used": 0,
        "status": "active",
        "current_node_id": nodes[0]["id"],
        "nodes": nodes,
        "history": [],
        "budget_policy": "shared_actions_v1",
    }


def _failure_reason(value: dict) -> str | None:
    """只识别项目返回格式中的明确失败信号，不推断未知业务字段。"""
    status = value.get("status")
    if isinstance(status, str):
        if len(status) > MAX_IDENTIFIER_CHARACTERS:
            return "工具结果或子项状态文本超限，无法安全确认调用结果"
        if status.strip().lower() in FAILED_STATUSES:
            return "工具结果或子项报告失败、拒绝、中断或未知状态"
    if value.get("error") or value.get("errors"):
        return "工具结果或子项包含错误"
    if value.get("success") is False or value.get("ok") is False:
        return "工具结果或子项明确报告未成功"
    for field in ("exit_code", "returncode"):
        if field in value:
            code = value[field]
            if type(code) is not int:
                return "工具进程退出码必须为非布尔整数"
            if code != 0:
                return "工具进程报告非零退出码"
    if value.get("isError") is True or value.get("is_error") is True:
        return "MCP工具结果或子项明确报告执行失败"
    acceptance = value.get("acceptance")
    if isinstance(acceptance, dict) and "status" in acceptance:
        if acceptance["status"] != "verified":
            return "协作子项尚未通过独立账本验收"
    return None


def _inspect_result(result: Any) -> tuple[str | None, str]:
    """有界递归检查失败标志，同时生成摘要，避免保存无限工具正文。"""
    fragments = []
    remaining, inspected = MAX_SUMMARY_CHARACTERS, 0

    def append(text: str) -> None:
        nonlocal remaining
        if remaining:
            part = text[:remaining]
            fragments.append(part)
            remaining -= len(part)

    def visit(value: Any, depth: int) -> str | None:
        nonlocal inspected
        inspected += 1
        if depth > MAX_RESULT_DEPTH or inspected > MAX_RESULT_ITEMS:
            return "工具结果嵌套或条目超限，无法安全确认调用结果"
        if isinstance(value, dict):
            reason = _failure_reason(value)
            if reason:
                return reason
            append("{")
            for key, item in value.items():
                if not isinstance(key, str):
                    return "工具结果包含非JSON字段"
                append(key)
                append(": ")
                reason = visit(item, depth + 1)
                if reason:
                    return reason
                append("; ")
            append("}")
        elif isinstance(value, list):
            append("[")
            for item in value:
                reason = visit(item, depth + 1)
                if reason:
                    return reason
                append("; ")
            append("]")
        elif isinstance(value, str):
            append(value)
        elif value is None or type(value) in (bool, int, float):
            if isinstance(value, float) and not math.isfinite(value):
                return "工具结果包含非有限数值"
            try:
                append(str(value))
            except ValueError:
                return "工具结果数值超限"
        else:
            return "工具结果包含非JSON值"
        return None

    if result is None:
        return "已完成调用缺少已知结果", ""
    reason = visit(result, 0)
    return reason, "".join(fragments)


def _identifier(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) <= MAX_IDENTIFIER_CHARACTERS and bool(value.strip())
    )


def _bounded_call_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= MAX_EVIDENCE
        and all(_identifier(item) for item in value)
        and len(set(value)) == len(value)
    )


def _report(reason: str, sources: list[dict] | None = None, output: str | None = None) -> dict:
    sources = sources or []
    return {
        "contract_satisfied": output is not None,
        "correctness_verified": False,
        "business_success_verified": False,
        "reported_call_completed": bool(sources),
        "reason": reason,
        "output": output,
        "ledger_sources": sources,
    }


def accept_node(
    plan: dict,
    node_id: str,
    answer: str | None,
    evidence: list[dict],
    tenant_id: str,
    owner_id: str,
    run_id: str,
) -> dict:
    """仅校验节点结构契约与服务端账本证据；始终不突变输入计划。"""
    if not all(_identifier(value) for value in (node_id, tenant_id, owner_id, run_id)):
        return _report("节点或账本身份参数无效")
    nodes = plan.get("nodes") if isinstance(plan, dict) else None
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 8:
        return _report("可信计划节点结构无效")
    if any(not isinstance(node, dict) or not _identifier(node.get("id")) for node in nodes):
        return _report("可信计划节点ID无效")
    by_id = {node["id"]: node for node in nodes}
    if len(by_id) != len(nodes) or node_id not in by_id:
        return _report("节点ID未知或重复")
    node = by_id[node_id]
    inputs, acceptance = node.get("inputs"), node.get("acceptance")
    if not isinstance(inputs, dict) or not isinstance(acceptance, dict):
        return _report("可信节点输入或验收契约无效")
    dependencies = inputs.get("from_nodes")
    if not isinstance(dependencies, list):
        return _report("可信节点前序依赖无效")
    preceding = {item["id"] for item in nodes[: nodes.index(node)]}
    for dependency in dependencies:
        if not isinstance(dependency, str) or dependency not in preceding:
            return _report("前序依赖未知或并非此前节点")
        previous = by_id[dependency]
        previous_output = previous.get("output")
        if (
            previous.get("status") != "completed"
            or not isinstance(previous_output, str)
            or not previous_output.strip()
            or len(previous_output) > MAX_OUTPUT_CHARACTERS
        ):
            return _report("前序依赖尚未完成或缺少有界非空输出")
    output_kind, required = acceptance.get("output_kind"), acceptance.get("required_tools")
    if output_kind not in ("answer", "tool_evidence") or not isinstance(required, list):
        return _report("可信节点输出契约无效")
    if any(not _identifier(name) for name in required) or len(set(required)) != len(required):
        return _report("可信节点必需工具列表无效")
    if output_kind == "tool_evidence" and not required:
        return _report("工具证据契约缺少必需工具")
    if answer is not None and (not isinstance(answer, str) or len(answer) > MAX_OUTPUT_CHARACTERS):
        return _report("节点回答类型无效或超过131072字符")
    if output_kind == "answer" and (answer is None or not answer.strip()):
        return _report("回答契约要求非空回答")
    bound_ids = node.get("tool_call_ids")
    if (
        not _bounded_call_ids(bound_ids)
        or not isinstance(evidence, list)
        or len(evidence) != len(bound_ids)
    ):
        return _report("工具证据必须恰好覆盖最多16个唯一节点绑定调用")
    other_ids = set()
    for other in nodes:
        if other is node:
            continue
        other_calls = other.get("tool_call_ids")
        if not _bounded_call_ids(other_calls):
            return _report("其他节点工具调用绑定无效")
        other_ids.update(other_calls)
    if set(bound_ids) & other_ids:
        return _report("当前工具调用已绑定其他节点，不能重复作为本节点证据")
    sources, seen, covered = [], set(), set()
    fields = {"id", "name", "status", "result", "tenant_id", "owner_id", "run_id"}
    for item in evidence:
        if not isinstance(item, dict) or not fields.issubset(item):
            return _report("服务端账本记录缺少必需字段", sources)
        identifier, name = item["id"], item["name"]
        if (
            not _identifier(identifier)
            or not _identifier(name)
            or identifier in seen
            or identifier not in bound_ids
        ):
            return _report("账本调用ID重复、无效或未绑定当前节点", sources)
        if (item["tenant_id"], item["owner_id"], item["run_id"]) != (tenant_id, owner_id, run_id):
            return _report("账本证据不属于当前租户、所有者或运行", sources)
        if item["status"] != "done":
            return _report("工具调用尚未在账本中完成", sources)
        reason, summary = _inspect_result(item["result"])
        if reason:
            return _report(reason, sources)
        seen.add(identifier)
        covered.add(name)
        sources.append(
            {
                "id": identifier,
                "name": name,
                "tenant_id": tenant_id,
                "owner_id": owner_id,
                "run_id": run_id,
                "status": "done",
                "summary": summary,
                "reported_call_completed": True,
                "business_success_verified": False,
            }
        )
    if seen != set(bound_ids) or not set(required).issubset(covered):
        return _report("账本证据未完整覆盖节点绑定调用或必需工具", sources)
    if isinstance(answer, str) and answer.strip():
        output = answer.strip()
    else:
        output = "\n".join(
            f"{source['name']}（账本ID：{source['id']}）：{source['summary']}" for source in sources
        )
    if not output:
        return _report("节点缺少可用输出", sources)
    return _report("结构契约满足；答案及工具结果的业务语义正确性未验证", sources, output)
