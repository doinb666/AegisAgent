"""私有项目会话API；沿用身份验证并严格限制参数。"""

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel, Field

from .harness import identity, service

router = APIRouter(tags=["项目会话"])


@router.get("/workspace/projects")
async def projects(
    request: Request,
    principal=Depends(identity),
    limit: int = Query(default=20, ge=1, le=50),
    before: str | None = Query(default=None, min_length=1, max_length=128),
):
    return await service(request).threads.projects(principal, limit, before)


class ThreadInput(BaseModel):
    model_config = {"extra": "forbid"}
    title: str = Field(min_length=1, max_length=200)
    project_id: str | None = Field(default=None, min_length=1, max_length=128)


class ThreadUpdate(BaseModel):
    model_config = {"extra": "forbid"}
    title: str | None = Field(default=None, min_length=1, max_length=200)
    archived: bool | None = Field(default=None, strict=True)


@router.post("/threads", status_code=201)
async def create_thread(
    body: ThreadInput,
    request: Request,
    principal=Depends(identity),
    idempotency_key: str | None = Header(default=None, min_length=1, max_length=256),
):
    return await service(request).threads.create(
        principal, body.title, body.project_id, idempotency_key
    )


@router.get("/threads")
async def list_threads(
    request: Request,
    principal=Depends(identity),
    project_id: str | None = Query(default=None, min_length=1, max_length=128),
    archived: bool = False,
    limit: int = Query(default=20, ge=1, le=50),
    before: str | None = Query(default=None, min_length=1, max_length=128),
):
    return await service(request).threads.list(principal, project_id, archived, limit, before)


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, request: Request, principal=Depends(identity)):
    return await service(request).threads.get(principal, thread_id)


@router.patch("/threads/{thread_id}")
async def update_thread(
    thread_id: str,
    body: ThreadUpdate,
    request: Request,
    principal=Depends(identity),
):
    return await service(request).threads.update(principal, thread_id, body.title, body.archived)
