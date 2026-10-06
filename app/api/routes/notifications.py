"""本人站内通知，不外发消息，不操作任务审批。"""

from fastapi import APIRouter, Depends, Path, Query, Request

from .harness import identity, service

router = APIRouter(tags=["站内通知"])


@router.get("/notifications")
async def notifications(
    request: Request,
    principal=Depends(identity),
    limit: int = Query(default=20, ge=1, le=50),
    before: int | None = Query(default=None, ge=1),
):
    return await service(request).notifications.list(principal, limit, before)


@router.get("/notifications/unread")
async def unread(request: Request, principal=Depends(identity)):
    return {"count": await service(request).notifications.unread(principal)}


@router.post("/notifications/{identifier}/read")
async def read_notification(
    request: Request, identifier: int = Path(ge=1), principal=Depends(identity)
):
    return await service(request).notifications.read(principal, identifier)
