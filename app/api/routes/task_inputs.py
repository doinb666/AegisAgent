"""任务输入 HTTP 目录；所有入口只读且要求已认证身份。"""

from fastapi import APIRouter, Depends, Query, Request

from .harness import identity, service

router = APIRouter(tags=["任务输入"])


@router.get("/task-templates")
async def task_templates(request: Request, principal=Depends(identity)):
    return service(request).task_inputs.templates(principal)


@router.get("/documents/references")
async def document_references(
    request: Request,
    principal=Depends(identity),
    limit: int = Query(default=20, ge=1, le=50),
    before: str | None = None,
):
    return await service(request).task_inputs.document_references(principal, limit, before)


@router.get("/mcp/servers")
async def mcp_servers(request: Request, principal=Depends(identity)):
    return service(request).task_inputs.mcp_servers(principal)
