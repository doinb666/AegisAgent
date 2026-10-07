"""持久计划的执行辅助；权限和工具副作用仍由既有 Worker 边界处理。"""

import asyncio
import json
from copy import deepcopy

from jsonschema import ValidationError, validate
from sqlalchemy import select

from app.infrastructure.llm.model_usage import safe_model_usage
from app.infrastructure.llm.stream_assembly import MAX_TOOLS

from .context import PLAN_PROMPT
from .errors import HarnessError
from .models import ToolCall, User
from .plan_execution import MAX_OUTPUT_CHARACTERS, accept_node, normalize_plan
from .security import canonical, digest

POLICY = "shared_actions_v1"


def bounded_final_answer(value):
    if not isinstance(value, str) or len(value) > MAX_OUTPUT_CHARACTERS or not value.strip():
        raise HarnessError(422, "最终摘要必须为非空文本，且不得超过131072字符")
    return value


def bounded_tool_calls(value):
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_TOOLS:
        raise HarnessError(422, "单轮工具调用必须为列表，且不得超过16项")
    if any(
        not isinstance(call, dict) or not isinstance(call.get("function"), dict) for call in value
    ):
        raise HarnessError(422, "单轮工具调用结构无效")
    return value


def bounded_fallback_answer(run):
    pending = PlanRuntime.validate_pending(run, "react")
    if not pending or pending.get("kind") != "answer" or run.messages[-1].get("tool_calls"):
        raise HarnessError(409, "降级模型答复与检查点不一致")
    return bounded_final_answer(run.messages[-1].get("content"))


def require_no_risk_deny(config, arg_hash):
    review = config.get("risk_reviews", {}).get(arg_hash)
    if isinstance(review, dict) and review.get("risk") == "deny":
        raise HarnessError(403, "已保存的独立风险审查拒绝该工具请求，禁止执行或复用")


def high_risk(executor, name, arguments):
    return any(
        term in name.lower()
        for term in ("shell", "exec", "write", "delete", "mcp", "commit", "push")
    ) or executor.requires_approval(name, arguments)


def shared(config):
    return config.get("budget_policy") == POLICY


def remaining(run):
    return run.config["max_steps"] - run.config.get("delegated_steps", 0) - run.step


def charge(run, cost=1, reserve=0):
    if remaining(run) < cost + reserve:
        raise HarnessError(409, "共享动作步数预算不足，操作未执行")
    run.step += cost


def current_node(config):
    plan = config.get("plan")
    if not plan or plan.get("status") != "active":
        return None
    return next((node for node in plan["nodes"] if node["id"] == plan["current_node_id"]), None)


def minimum_node_cost(node):
    acceptance = node["acceptance"]
    # 答复节点有必需工具时，工具轮之后还需要一轮答复模型。
    followup = acceptance["output_kind"] == "answer" and bool(acceptance["required_tools"])
    return 2 + len(acceptance["required_tools"]) + int(followup)


def minimum_cost(plan):
    return 1 + sum(
        minimum_node_cost(node) for node in plan["nodes"] if node["status"] != "completed"
    )


def reserve_cost(config, *, after_tools=False):
    plan = config.get("plan")
    if not plan:
        return 1
    active = current_node(config)
    reserved = 1
    for node in plan["nodes"]:
        if node["status"] == "completed":
            continue
        if node["status"] == "pending" and active and node["id"] != active["id"]:
            reserved += minimum_node_cost(node)
        else:
            reserved += 1
    if after_tools and active and active["acceptance"]["output_kind"] == "answer":
        reserved += 1
    return reserved


def bind_tool(run, identifier):
    plan = deepcopy(run.config.get("plan"))
    if not plan or plan.get("status") != "active":
        return
    node = next(item for item in plan["nodes"] if item["id"] == plan["current_node_id"])
    if identifier not in node["tool_call_ids"]:
        if len(node["tool_call_ids"]) >= MAX_TOOLS:
            raise HarnessError(422, "当前节点工具证据数量超限")
        if any(
            identifier in other["tool_call_ids"] for other in plan["nodes"] if other is not node
        ):
            raise HarnessError(409, "工具账本已绑定其他活动节点")
        node["tool_call_ids"].append(identifier)
    run.config = {**run.config, "plan": plan}


def tool_record(tool):
    return {
        key: getattr(tool, key)
        for key in ("id", "name", "status", "result", "tenant_id", "owner_id", "run_id")
    }


def known_success(tool):
    probe = normalize_plan(
        {
            "steps": [
                {
                    "objective": "复用源结果",
                    "inputs": {"instruction": "检查已知结果", "from_steps": []},
                    "acceptance": {"output_kind": "tool_evidence", "required_tools": [tool.name]},
                }
            ]
        },
        [tool.name],
    )
    probe["nodes"][0]["tool_call_ids"] = [tool.id]
    return accept_node(
        probe,
        probe["nodes"][0]["id"],
        None,
        [tool_record(tool)],
        tool.tenant_id,
        tool.owner_id,
        tool.run_id,
    )["contract_satisfied"]


async def reusable_tool(session, run, name, arguments, requires_approval=False):
    """只允许失败后缀的已知副作用结果作为明确标注的历史来源。"""
    plan = run.config.get("plan") or {}
    if not requires_approval and not any(
        term in name.lower()
        for term in ("shell", "exec", "write", "delete", "mcp", "commit", "push")
    ):
        return None
    source_ids = {
        identifier
        for archive in plan.get("history", [])
        for node in archive.get("nodes", [])
        for identifier in node.get("tool_call_ids", [])
    }
    if not source_ids:
        return None
    tools = (
        await session.scalars(
            select(ToolCall).where(
                ToolCall.id.in_(source_ids),
                ToolCall.run_id == run.id,
                ToolCall.tenant_id == run.tenant_id,
                ToolCall.owner_id == run.owner_id,
                ToolCall.name == name,
                ToolCall.status == "done",
            )
        )
    ).all()
    return next(
        (
            tool
            for tool in tools
            if canonical(tool.arguments) == canonical(arguments) and known_success(tool)
        ),
        None,
    )


async def current_tool_access(worker, run, principal, allowed, session=None):
    """计划工具门始终读取实际用户身份，长模型调用不能保留已撤销角色。"""
    from .errors import Principal

    if session is None:
        async with worker.store.sessions() as opened:
            return await current_tool_access(worker, run, principal, allowed, opened)
    user = await session.get(User, run.owner_id, populate_existing=True)
    if user is None or user.tenant_id != run.tenant_id:
        raise HarnessError(403, "计划工具执行身份已失效")
    principal = Principal(user.id, user.tenant_id, user.role)
    permitted = allowed & set(worker.service.accessible_tool_names(principal))
    if run.config.get("allowed_tools") is not None:
        permitted &= set(run.config["allowed_tools"])
    return principal, permitted


def validate_tool_schema(worker, principal, name, arguments):
    schema = next(
        (
            item
            for item in worker.service.tool_executor.catalog(principal)
            if item["function"]["name"] == name
        ),
        None,
    )
    if schema is None:
        raise HarnessError(403, "当前工具目录已撤销该工具权限")
    try:
        validate(arguments, schema["function"]["parameters"])
    except ValidationError as exc:
        raise HarnessError(422, "工具参数不符合当前权限目录或参数契约") from exc


async def verify_file_reuse(worker, principal, run_id, arguments):
    """核查当前目录边界，不生成新审批或刷新来源文件基线。"""
    from app.harness_tools.file_write import FileWriter

    executor = worker.service.tool_executor
    if not hasattr(executor, "workspace") or not hasattr(executor, "_authorize_file_write"):
        raise HarnessError(409, "当前文件工具不支持安全核查历史结果来源")
    await executor._authorize_file_write(principal, run_id)
    writer = FileWriter(executor.workspace)
    parts, _ = writer._arguments(arguments)
    await asyncio.to_thread(writer._snapshot, principal, run_id, parts)


class PlanRuntime:
    def __init__(self, worker, run_id, principal, catalog, allowed):
        self.worker = worker
        self.store = worker.store
        self.run_id = run_id
        self.principal = principal
        self.catalog = catalog
        self.allowed = allowed

    async def save_plan_response(self, response, purpose):
        """完整计划响应和归属一并保存；无效 JSON 也先保存已知结果。"""
        run = await self.worker.load(self.run_id)
        if purpose == "final":
            bounded_final_answer(response.content)
        message = {"role": "assistant", "content": response.content or ""}
        calls = (
            ((response.raw or {}).get("choices") or [{}])[0].get("message", {}).get("tool_calls")
        )
        try:
            calls = bounded_tool_calls(calls)
        except HarnessError:
            if purpose not in {"plan_initial", "replan"}:
                raise
            # 规划违规调用只记录已拒绝，不保存可进入执行队列的原生调用。
            message["tool_calls_rejected"] = True
            calls = []
        if calls:
            message["tool_calls"] = calls
        await self.worker.checkpoint(
            run.id,
            [*run.messages, message],
            run.step,
            "model",
            {
                "purpose": purpose,
                "tool_calls": calls or [],
                "model": response.model_id,
                "usage": safe_model_usage(getattr(response, "usage", None)),
                "model_parameters": run.config.get("model_parameters", {}),
            },
        )

    @staticmethod
    def validate_pending(run, purpose, node=None):
        marker = run.config.get("model_result_pending")
        if not marker:
            return None
        plan = run.config.get("plan") or {}
        index = marker.get("message_index")
        if (
            marker.get("purpose") != purpose
            or type(marker.get("step")) is not int
            or marker["step"] > run.step
            or (purpose != "node" and marker["step"] != run.step)
            or type(index) is not int
            or not 0 <= index < len(run.messages)
            or marker.get("plan_revision") != plan.get("revision")
            or marker.get("node_id") != (node["id"] if node else None)
            or run.messages[index].get("role") != "assistant"
            or (
                (marker.get("kind") == "answer" or purpose != "node")
                and index != len(run.messages) - 1
            )
            or (
                node is not None
                and node.get("model_result_identity") != marker
                and purpose == "node"
            )
        ):
            raise HarnessError(409, "已知模型结果归属与计划检查点不一致，禁止重新生成")
        return marker

    async def install_initial(self):
        run = await self.worker.load(self.run_id)
        if not run.config.get("model_result_pending"):
            budget = canonical(
                {
                    "规划后可用动作": max(0, remaining(run) - 1),
                    "每节点模型及验收最低动作": 2,
                    "每个新工具动作": 1,
                    "必要风险审查动作": 1,
                    "最终封口预留动作": 1,
                    "要求": "生成紧凑且可在预算内执行的计划，禁止添加预算字段。",
                }
            )
            response = await self.worker.model_call(
                run,
                [
                    *run.messages,
                    {
                        "role": "user",
                        "content": PLAN_PROMPT
                        + "\n允许工具："
                        + canonical(sorted(self.allowed))
                        + "\n服务端预算："
                        + budget,
                    },
                ],
                purpose="plan_initial",
            )
            await self.save_plan_response(response, "plan_initial")
        run = await self.worker.load(self.run_id)
        self.validate_pending(run, "plan_initial")
        try:
            if run.messages[-1].get("tool_calls") or run.messages[-1].get("tool_calls_rejected"):
                raise HarnessError(422, "规划阶段禁止工具调用")
            plan = normalize_plan(run.messages[-1]["content"], self.allowed)
            if minimum_cost(plan) > remaining(run):
                raise HarnessError(422, "计划最低动作成本超过剩余共享预算")
        except HarnessError as exc:
            async with self.worker.active_transaction(run.id) as (session, current):
                self.validate_pending(current, "plan_initial")
                current.config = {**current.config, "planned": True, "plan_fallback": True}
                current.config.pop("model_result_pending", None)
                # 无效规划原生调用不进入工具执行队列。
                current.messages = [
                    *current.messages[:-1],
                    {"role": "assistant", "content": "计划无效，继续有界执行。"},
                ]
                self.store.emit(
                    session, current, "plan_fallback", {"reason": exc.detail, "plan": None}
                )
            return
        async with self.worker.active_transaction(run.id) as (session, current):
            self.validate_pending(current, "plan_initial")
            current.config = {**current.config, "planned": True, "plan": plan}
            current.config.pop("model_result_pending", None)
            self.store.emit(session, current, "plan", plan)

    async def preflight(self, run, calls):
        """整批核查权限和新动作费用，首个副作用之前完成。"""
        calls = bounded_tool_calls(calls)
        cost = 0
        seen = set()
        async with self.worker.active_transaction(run.id) as (session, current):
            self.principal, permitted = await current_tool_access(
                self.worker, current, self.principal, self.allowed, session
            )
            if await self.store.unknown_tool(session, current):
                current.status, current.error = "interrupted", "工具结果未知，禁止重放或重规划"
                current.lease_owner, current.lease_until = None, None
                self.store.emit(session, current, "interrupted", {"reason": current.error})
                await self.store.stop_children(session, current)
                return False
            for call in calls:
                identifier, function = call.get("id"), call.get("function", {})
                name = function.get("name")
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier in seen
                    or not isinstance(name, str)
                    or name not in permitted
                ):
                    raise HarnessError(403, "工具批次包含无效调用ID或越权工具")
                seen.add(identifier)
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                except (ValueError, TypeError) as exc:
                    raise HarnessError(422, "工具参数必须是合法JSON对象") from exc
                if not isinstance(arguments, dict):
                    raise HarnessError(422, "工具参数必须是JSON对象")
                validate_tool_schema(self.worker, self.principal, name, arguments)
                requires_approval = high_risk(self.worker.service.tool_executor, name, arguments)
                if requires_approval and self.principal.role == "viewer":
                    raise HarnessError(403, "viewer不能执行或复用高风险工具")
                saved = await session.scalar(
                    select(ToolCall).where(
                        ToolCall.run_id == run.id,
                        ToolCall.tenant_id == run.tenant_id,
                        ToolCall.owner_id == run.owner_id,
                        ToolCall.call_id == identifier,
                    )
                )
                arg_hash = digest(
                    canonical({"call_id": identifier, "name": name, "arguments": arguments})
                )
                require_no_risk_deny(current.config, arg_hash)
                if saved and saved.arguments_hash != arg_hash:
                    raise HarnessError(409, "工具调用ID被用于不同参数")
                if saved and saved.status == "done":
                    plan = current.config.get("plan")
                    active = current_node(current.config)
                    if plan and any(
                        saved.id in item["tool_call_ids"]
                        for item in plan["nodes"]
                        if active is not None and item["id"] != active["id"]
                    ):
                        raise HarnessError(409, "工具账本已绑定其他活动节点")
                    continue
                if await reusable_tool(session, current, name, arguments, requires_approval):
                    continue
                cost += 1
                if requires_approval and self.worker.service.settings.risk_review_enabled:
                    approval = current.approval or {}
                    if not current.config.get("risk_reviews", {}).get(arg_hash) and not (
                        approval.get("call_id") == identifier and approval.get("hash") == arg_hash
                    ):
                        cost += 1
            if remaining(current) < cost + reserve_cost(current.config, after_tools=True):
                raise HarnessError(409, "整批工具及验收封口的共享步数预算不足，工具未执行")
            return True

    async def accept(self, run, answer):
        async with self.worker.active_transaction(run.id) as (session, current):
            plan = deepcopy(current.config["plan"])
            node = current_node(current.config)
            mutable = next(item for item in plan["nodes"] if item["id"] == node["id"])
            pending = current.config.get("model_result_pending")
            if pending:
                self.validate_pending(current, "node", node)
            evidence = (
                await session.scalars(
                    select(ToolCall).where(
                        ToolCall.id.in_(node["tool_call_ids"]),
                        ToolCall.run_id == current.id,
                        ToolCall.tenant_id == current.tenant_id,
                        ToolCall.owner_id == current.owner_id,
                    )
                )
            ).all()
            charge(current, reserve=1)
            if mutable["status"] != "accepting":
                mutable["status"] = "accepting"
                self.store.emit(
                    session,
                    current,
                    "node_accepting",
                    {"node_id": node["id"], "plan_revision": plan["revision"]},
                )
            report = accept_node(
                plan,
                node["id"],
                answer,
                [tool_record(tool) for tool in evidence],
                current.tenant_id,
                current.owner_id,
                current.id,
            )
            mutable["acceptance_result"] = report
            mutable["output"] = report["output"] if report["contract_satisfied"] else answer
            mutable["status"] = "completed" if report["contract_satisfied"] else "blocked"
            mutable["blocked_reason"] = None if report["contract_satisfied"] else report["reason"]
            if report["contract_satisfied"]:
                following = next(
                    (item for item in plan["nodes"] if item["status"] == "pending"), None
                )
                plan["current_node_id"] = following["id"] if following else None
            current.config = {**current.config, "plan": plan}
            current.config.pop("model_result_pending", None)
            self.store.emit(
                session,
                current,
                "node_accepted",
                {"node_id": node["id"], "plan_revision": plan["revision"], **report},
            )
            return report["contract_satisfied"]

    async def finalize(self, run):
        last = run.config["plan"]["nodes"][-1]
        if last["acceptance"]["output_kind"] == "answer":
            answer = last["output"]
            async with self.worker.active_transaction(run.id) as (session, current):
                if not all(
                    node["status"] == "completed" for node in current.config["plan"]["nodes"]
                ):
                    raise HarnessError(409, "计划尚未全部验收，禁止结束运行")
                charge(current)
                await self.finish_in_transaction(session, current, answer)
            return
        marker = run.config.get("model_result_pending")
        if not marker:
            outputs = [
                {
                    "node_id": node["id"],
                    "output": node["output"][:2000],
                    "business_success_verified": False,
                }
                for node in run.config["plan"]["nodes"]
            ]
            response = await self.worker.model_call(
                run,
                [
                    *run.messages,
                    {
                        "role": "user",
                        "content": (
                            "所有计划节点已完成结构验收。仅基于已知输出给出非空交付摘要，"
                            "禁止工具调用；业务语义仍未验证。"
                        )
                        + canonical({"有界已验收输出": outputs}),
                    },
                ],
                purpose="final",
            )
            await self.save_plan_response(response, "final")
        run = await self.worker.load(run.id)
        self.validate_pending(run, "final")
        async with self.worker.active_transaction(run.id) as (session, current):
            self.validate_pending(current, "final")
            if current.messages[-1].get("tool_calls"):
                raise HarnessError(422, "最终摘要不得调用工具")
            answer = bounded_final_answer(current.messages[-1].get("content"))
            if not all(node["status"] == "completed" for node in current.config["plan"]["nodes"]):
                raise HarnessError(409, "计划尚未全部验收，禁止结束运行")
            await self.finish_in_transaction(session, current, answer)

    async def finish_in_transaction(self, session, run, answer):
        """最终消费和终态同事务；不产生第二次预算收费。"""
        answer = bounded_final_answer(answer)
        plan = {**run.config["plan"], "status": "completed"}
        run.config = {**run.config, "plan": plan}
        run.config.pop("model_result_pending", None)
        run.status, run.answer, run.error = "completed", answer, None
        run.lease_owner, run.lease_until = None, None
        self.store.emit(session, run, "completed", {"answer": answer, "error": None})
        await self.store.stop_children(session, run)
        await self.worker.consolidate(run.id, session)

    async def repair(self, run):
        plan = run.config["plan"]
        node = current_node(run.config)
        if not node or node["status"] != "blocked":
            raise HarnessError(409, "不存在可重规划的已知契约失败")
        pending = run.config.get("model_result_pending") or {}
        if plan["replans_used"] >= 1 and pending.get("purpose") != "replan":
            await self.fallback(run, "第二次计划契约失败")
            return
        if not run.config.get("model_result_pending"):
            # 重规划登记和 replans_used 由 model_call 同事务写入。
            prefix = [
                {"id": item["id"], "output": item["output"][:2000]}
                for item in plan["nodes"]
                if item["status"] == "completed"
            ]
            prompt = (
                PLAN_PROMPT
                + "\n"
                + canonical(
                    {
                        "原目标": run.message[:4000],
                        "完成输出": prefix,
                        "失败原因": node["blocked_reason"],
                        "剩余预算": remaining(run),
                        "规则": "仅规划剩余后缀；from_steps为后缀内部零基索引，不改变完成前缀。",
                    }
                )
            )
            response = await self.worker.model_call(
                run,
                [*run.messages, {"role": "user", "content": prompt}],
                purpose="replan",
                node_id=node["id"],
            )
            await self.save_plan_response(response, "replan")
        run = await self.worker.load(run.id)
        self.validate_pending(run, "replan", current_node(run.config))
        try:
            if run.messages[-1].get("tool_calls") or run.messages[-1].get("tool_calls_rejected"):
                raise HarnessError(422, "重规划阶段禁止工具调用")
            replacement = normalize_plan(
                run.messages[-1]["content"],
                self.allowed,
                revision=run.config["plan"]["revision"] + 1,
            )
            completed = [
                item for item in run.config["plan"]["nodes"] if item["status"] == "completed"
            ]
            if len(completed) + len(replacement["nodes"]) > 8 or minimum_cost(
                replacement
            ) > remaining(run):
                raise HarnessError(422, "重规划节点或剩余动作预算超限")
        except HarnessError as exc:
            await self.fallback(run, exc.detail)
            return
        async with self.worker.active_transaction(run.id) as (session, current):
            self.validate_pending(current, "replan", current_node(current.config))
            old = current.config["plan"]
            suffix = [deepcopy(item) for item in old["nodes"] if item["status"] != "completed"]
            for item in suffix:
                item["status"] = "blocked"
            if completed:
                replacement["nodes"][0]["inputs"]["from_nodes"].append(completed[-1]["id"])
            replacement.update(
                nodes=completed + replacement["nodes"],
                replans_used=1,
                history=[*old["history"], {"revision": old["revision"], "nodes": suffix}][-2:],
            )
            current.config = {**current.config, "plan": replacement}
            current.config.pop("model_result_pending", None)
            self.store.emit(session, current, "plan_replanned", replacement)

    async def fallback(self, run, reason):
        async with self.worker.active_transaction(run.id) as (session, current):
            old = current.config["plan"]
            completed = [item for item in old["nodes"] if item["status"] == "completed"]
            suffix = [deepcopy(item) for item in old["nodes"] if item["status"] != "completed"]
            required = sorted(
                {name for item in suffix for name in item["acceptance"]["required_tools"]}
            )
            replacement = normalize_plan(
                {
                    "steps": [
                        {
                            "objective": "有界降级完成原目标",
                            "inputs": {
                                "instruction": "完成原目标并补齐必需工具；证据不足须明确说明。",
                                "from_steps": [],
                            },
                            "acceptance": {"output_kind": "answer", "required_tools": required},
                        }
                    ]
                },
                self.allowed,
                revision=old["revision"] + 1,
            )
            if remaining(current) < minimum_cost(replacement):
                raise HarnessError(409, "剩余共享步数预算不足以安全降级，运行失败")
            for item in suffix:
                item["status"] = "blocked"
            if completed:
                replacement["nodes"][0]["inputs"]["from_nodes"] = [completed[-1]["id"]]
            replacement.update(
                nodes=completed + replacement["nodes"],
                replans_used=1,
                fallback=True,
                history=[*old["history"], {"revision": old["revision"], "nodes": suffix}][-2:],
            )
            current.config = {**current.config, "plan": replacement}
            current.config.pop("model_result_pending", None)
            current.messages = [
                *current.messages[:-1],
                {"role": "assistant", "content": "计划契约失败，进入有界降级。"},
            ]
            self.store.emit(
                session, current, "plan_fallback", {"reason": reason, "plan": replacement}
            )

    async def execute(self):
        run = await self.worker.load(self.run_id)
        if not run.config.get("planned"):
            await self.install_initial()
        while True:
            run = await self.worker.load(self.run_id)
            plan = run.config.get("plan")
            if not plan:
                return False
            node = current_node(run.config)
            if node is None:
                await self.finalize(run)
                return True
            if node["status"] == "blocked":
                if plan.get("fallback"):
                    raise HarnessError(422, "降级节点仍未满足必需工具与输出契约")
                await self.repair(run)
                continue
            marker = run.config.get("model_result_pending")
            if marker:
                self.validate_pending(run, "node", node)
                if marker["kind"] == "answer":
                    await self.accept(run, run.messages[-1].get("content"))
                    continue
            calls = self.worker.pending_calls(run.messages)
            if calls:
                if not await self.preflight(run, calls):
                    return True
                messages = list(run.messages)
                for call in calls:
                    step = await self.worker.execute_tool(
                        run, self.principal, call, messages, run.step, self.allowed
                    )
                    if step is None:
                        return True
                    run = await self.worker.load(run.id)
                continue
            if marker:
                if node["acceptance"]["output_kind"] == "tool_evidence":
                    await self.accept(run, None)
                    continue
                async with self.worker.active_transaction(run.id) as (_, current):
                    self.validate_pending(current, "node", current_node(current.config))
                    current.config = {
                        key: value
                        for key, value in current.config.items()
                        if key != "model_result_pending"
                    }
            async with self.worker.active_transaction(run.id) as (session, current):
                plan = deepcopy(current.config["plan"])
                active = current_node(current.config)
                target = next(item for item in plan["nodes"] if item["id"] == active["id"])
                starting = target["status"] == "pending"
                target["status"] = "running"
                current.config = {**current.config, "plan": plan}
                if starting:
                    self.store.emit(
                        session,
                        current,
                        "node_started",
                        {"node_id": target["id"], "plan_revision": plan["revision"]},
                    )
            run = await self.worker.load(run.id)
            node = current_node(run.config)
            by_id = {item["id"]: item for item in run.config["plan"]["nodes"]}
            dependencies = []
            for identifier in node["inputs"]["from_nodes"]:
                predecessor = by_id.get(identifier)
                if (
                    not predecessor
                    or predecessor["status"] != "completed"
                    or not predecessor["output"]
                ):
                    raise HarnessError(409, "计划前序依赖尚未验收完成或缺少输出")
                dependencies.append({"node_id": identifier, "output": predecessor["output"][:4000]})
            prompt = canonical(
                {
                    "原始目标": run.message[:4000],
                    "当前节点": node["id"],
                    "目标": node["objective"],
                    "指令": node["inputs"]["instruction"],
                    "前序输出": dependencies,
                    "验收契约": node["acceptance"],
                    "说明": "只执行当前节点，答复不代表运行已经结束。",
                }
            )
            response = await self.worker.model_call(
                run,
                [*run.messages, {"role": "user", "content": prompt}],
                self.catalog,
                purpose="node",
                node_id=node["id"],
            )
            await self.save_plan_response(response, "node")
