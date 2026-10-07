"""审批恢复复用冻结契约，文件 I/O 不占用长数据库事务。"""

from sqlalchemy import select

from app.harness_tools.file_write import contract_hash

from .errors import HarnessError
from .models import ToolCall
from .store import scope


def bound_contract(approval, call_id, args_hash):
    contract = (approval or {}).get("file_write")
    if not isinstance(contract, dict) or (
        contract.get("version") != 1
        or contract.get("call_id") != call_id
        or contract.get("args_hash") != args_hash
        or contract.get("baseline_hash") != contract_hash(contract)
    ):
        raise HarnessError(409, "文件写入审批缺少有效冻结基线，必须创建新调用")
    return contract


async def prepare_contract(worker, principal, run_id, call_id, args_hash, arguments):
    current = await worker.load(run_id)
    if current.parent_run_id or current.config.get("schedule_id") or principal.role == "viewer":
        raise HarnessError(403, "子任务、定时任务或 viewer 禁止文件写入")
    async with worker.store.sessions() as session:
        saved = await session.scalar(
            select(ToolCall).where(
                *scope(ToolCall, principal), ToolCall.run_id == run_id, ToolCall.call_id == call_id
            )
        )
    if saved is not None:
        # started 不得重新冻结，也不得重放；done 直接消费账本结果。
        if saved.arguments_hash != args_hash:
            raise HarnessError(409, "工具调用ID被用于不同参数")
        if saved.status in {"started", "done"}:
            return None
    approval = current.approval or {}
    if approval.get("call_id") == call_id and approval.get("hash") != args_hash:
        raise HarnessError(409, "已冻结的工具调用ID被用于不同参数，禁止刷新审批基线")
    if approval.get("call_id") == call_id and approval.get("hash") == args_hash:
        return bound_contract(approval, call_id, args_hash)
    return await worker.service.tool_executor.prepare_file_write(
        principal, run_id, arguments, call_id, args_hash
    )
