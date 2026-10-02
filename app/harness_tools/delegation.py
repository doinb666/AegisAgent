"""只读子任务的预算分配、受限等待与完整证据归档。"""

import asyncio
from time import time as now

from app.harness.errors import HarnessError
from app.harness.security import canonical, digest
from app.harness.store import TERMINAL_STATUSES

READ_ONLY_TOOLS = ("calculator", "knowledge_search", "artifact_read", "skill_read")
sleep = asyncio.sleep


async def delegate(service, settings, principal, run_id, arguments):
    tasks = arguments["tasks"]
    try:
        context = await service.delegation_context(principal, run_id, len(tasks))
    except HarnessError as exc:
        # 创建前已知拒绝，不能被执行器误判为未知副作用。
        return {"status": "failed", "error": str(exc), "children": []}
    remaining_time = context["deadline"] - now()
    # 最多保留一次模型调用的期限；短任务至少预留一秒，通常保留剩余时间的20%。
    summary_seconds = min(settings.model_timeout_seconds, max(1, remaining_time * 0.2))
    wait_deadline = context["deadline"] - summary_seconds
    if wait_deadline <= now():
        return {"status": "failed", "error": "父运行时间预算不足，整批子运行未创建", "children": []}
    total_steps = min(context["remaining_steps"], 3 * len(tasks))
    base, extra = divmod(total_steps, len(tasks))
    child_tools = [name for name in READ_ONLY_TOOLS if name in context["allowed_tools"]]
    request_key = digest(canonical(arguments))
    children = []
    reasons = {}

    async def cancel_active_children():
        for child_id in children:
            child = await service.get_run(principal, child_id)
            if child["status"] not in TERMINAL_STATUSES:
                await service.cancel(principal, child_id)

    try:
        for index, message in enumerate(tasks):
            child = await service.create_run(
                principal,
                message,
                f"{run_id}:delegate:{request_key}:{index}",
                parent_run_id=run_id,
                allowed_tools=child_tools,
                max_steps=base + (index < extra),
                session_id=context["session_id"] if arguments.get("mode") == "fork" else None,
            )
            children.append(child["id"])
        pending = set(children)
        while pending:
            parent = await service.get_run(principal, run_id)
            reason = None
            if parent["status"] != "running":
                reason = "父运行已停止，委派终止"
            elif now() >= wait_deadline:
                reason = "委派等待预算耗尽，保留主控汇总时间"
            for child_id in tuple(pending):
                child = await service.get_run(principal, child_id)
                if child["status"] in TERMINAL_STATUSES:
                    pending.remove(child_id)
                elif reason:
                    await service.cancel(principal, child_id)
                    reasons[child_id] = reason
                    pending.remove(child_id)
            if pending:
                await sleep(min(0.1, max(0, wait_deadline - now())))

        results = []
        for child_id in children:
            # 取消提交完成后重新读取；报告真实终态而不是推测queued或cancelled。
            child = await service.get_run(principal, child_id)
            answer = child.get("answer") or ""
            result = {
                "run_id": child_id,
                "status": child["status"],
                "answer": answer[:2000],
                "total_chars": len(answer),
                "error": child.get("error"),
            }
            if child_id in reasons:
                result["message"] = reasons[child_id]
            if len(answer) > 2000:
                artifact = await service.put_artifact(
                    principal, "委派子任务完整回答", answer, child_id, request_key, "delegate"
                )
                result["artifact_id"] = artifact["id"]
            results.append(result)
        return results
    finally:
        # 创建中途失败、父取消或未知执行中断均不留下继续执行的子任务。
        await cancel_active_children()
