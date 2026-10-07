"""定时计划 HTTP 边界；私有范围和权限由计划服务统一验证。"""

from fastapi import APIRouter, Depends, Header, Query, Request

from app.harness.schedule_schema import ScheduleInput, ScheduleStatus, ScheduleTransition

from .harness import identity, service

router = APIRouter(tags=["定时任务"])


@router.post("/schedules", status_code=201)
async def create_schedule(
    body: ScheduleInput,
    request: Request,
    principal=Depends(identity),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=256),
):
    return await service(request).schedules.create(principal, idempotency_key, **body.model_dump())


@router.get("/schedules")
async def schedules(
    request: Request,
    principal=Depends(identity),
    limit: int = Query(default=20, ge=1, le=50),
    before: str | None = Query(default=None, min_length=1, max_length=128),
    status: ScheduleStatus | None = None,
):
    return await service(request).schedules.list(principal, limit, before, status)


@router.get("/schedules/{identifier}")
async def schedule(identifier: str, request: Request, principal=Depends(identity)):
    return await service(request).schedules.get(principal, identifier)


@router.patch("/schedules/{identifier}")
async def transition_schedule(
    identifier: str,
    body: ScheduleTransition,
    request: Request,
    principal=Depends(identity),
):
    return await service(request).schedules.transition(
        principal, identifier, body.action, body.expected_version
    )
