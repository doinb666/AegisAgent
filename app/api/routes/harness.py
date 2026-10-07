"""AegisCode 同源 API：所有业务资源均经身份与所有权检查。"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.etl.isolated import parse_isolated
from app.harness.errors import HarnessError
from app.harness.schedules import SCHEDULE_TOOLS
from app.harness.task_inputs import validate_document_ids

router = APIRouter(tags=["AegisCode"])
TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


def service(request: Request):
    return request.app.state.harness


async def identity(request: Request, authorization: str = Header(default="")):
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "请先登录")
    return await service(request).authenticate(authorization[7:])


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=8, max_length=1024)


class Member(Credentials):
    role: Literal["operator", "viewer"] = "operator"


class RunInput(BaseModel):
    model_config = {"extra": "forbid"}

    message: str = Field(min_length=1, max_length=16000)
    session_id: str | None = Field(default=None, max_length=128)
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    mode: Literal["react", "plan", "reflection"] = "react"
    model: str | None = Field(default=None, max_length=128)
    collaboration_mode: Literal["fork", "team"] | None = None
    project_mode: Literal["fork", "worktree"] | None = None
    document_ids: list[str] | None = Field(default=None, max_length=3)

    @field_validator("document_ids", mode="before")
    @classmethod
    def documents(cls, value):
        try:
            return validate_document_ids(value)
        except HarnessError as exc:
            raise ValueError(exc.detail) from exc


class ApprovalInput(BaseModel):
    approved: bool
    call_id: str = Field(min_length=1, max_length=256)
    args_hash: str = Field(min_length=64, max_length=64)


class FeedbackInput(BaseModel):
    success: bool
    note: str = Field(default="", max_length=4000)


class AssetInput(BaseModel):
    kind: Literal[
        "profile", "preference", "constraint", "memory", "episodic", "skill", "procedure", "project"
    ]
    name: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=16000)
    metadata: dict = Field(default_factory=dict)
    status: Literal["draft", "active", "retired"] = "draft"
    expected_version: int | None = Field(default=None, ge=1, strict=True)


class TransitionInput(BaseModel):
    status: Literal["draft", "active", "retired"]


class RestoreInput(BaseModel):
    version: int = Field(ge=1)


class SkillImportInput(BaseModel):
    model_config = {"extra": "forbid"}

    document: str = Field(min_length=1, max_length=32768)
    directory: str = Field(default="", max_length=128)
    resources: dict[str, str] = Field(default_factory=dict, max_length=16)
    asset_id: str | None = Field(default=None, min_length=1, max_length=128)
    expected_version: int | None = Field(default=None, ge=1, strict=True)


@router.post("/auth/register", status_code=201)
async def register(body: Credentials, request: Request):
    return await service(request).register(body.username, body.password)


@router.post("/auth/login")
async def login(body: Credentials, request: Request):
    result = await service(request).login(body.username, body.password)
    principal = await service(request).authenticate(result["token"])
    return {**result, "bootstrap": await service(request).bootstrap(principal)}


@router.post("/auth/logout")
async def logout(request: Request, principal=Depends(identity), authorization: str = Header()):
    await service(request).logout(authorization[7:])
    return {"message": "已注销"}


@router.get("/auth/me")
async def me(request: Request, principal=Depends(identity)):
    return {
        "user_id": principal.user_id,
        "tenant_id": principal.tenant_id,
        "role": principal.role,
        "bootstrap": await service(request).bootstrap(principal),
    }


@router.post("/auth/members", status_code=201)
async def member(body: Member, request: Request, principal=Depends(identity)):
    return await service(request).create_user(principal, body.username, body.password, body.role)


@router.get("/capabilities")
async def capabilities(request: Request, principal=Depends(identity)):
    harness = service(request)
    models = getattr(harness.model_router, "_configs", [])
    routes = getattr(harness.model_router, "public_routes", lambda: [])()
    return {
        "name": "AegisCode",
        "models": sorted({model.model_id for model in models}),
        "model_routes": routes,
        "model_protocols": ["openai", "custom", "anthropic", "azure", "ollama"],
        "collaboration": {
            "modes": ["fork", "team"],
            "max_children": 2,
            "project_modes": ["fork", "worktree"] if harness.settings.repository_root else [],
        },
        "tools": [t["function"]["name"] for t in harness.tool_executor.catalog(principal)],
        "sandbox": bool(harness.settings.sandbox_url),
        "role": principal.role,
        "max_steps": harness.settings.max_steps,
        "threads": {"enabled": harness.store.threads_ready},
        "notifications": {"enabled": harness.store.notifications_ready},
        "task_inputs": {"enabled": True, "max_documents": 3},
        "schedules": {
            "enabled": harness.store.schedules_ready,
            "max_active": 100,
            "min_interval_seconds": 60,
            "max_interval_seconds": 31 * 86400,
            "readonly_tools": sorted(SCHEDULE_TOOLS),
        },
    }


@router.post("/runs", status_code=202)
async def create_run(
    body: RunInput,
    request: Request,
    principal=Depends(identity),
    idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=256),
):
    configured = getattr(service(request).model_router, "public_routes", lambda: [])()
    if (
        body.model
        and configured
        and body.model
        not in {value for route in configured for value in (route["id"], route["model"])}
    ):
        raise HTTPException(422, "模型未配置，请从可用来源中选择")
    return await service(request).create_run(
        principal, idempotency_key=idempotency_key, **body.model_dump()
    )


@router.get("/runs")
async def runs(
    request: Request,
    query: str | None = Query(default=None, max_length=200),
    status: Literal[
        "queued", "running", "waiting_approval", "completed", "failed", "cancelled", "interrupted"
    ]
    | None = None,
    limit: int = Query(default=100, ge=1, le=100),
    before: str | None = Query(default=None, max_length=128),
    principal=Depends(identity),
):
    return await service(request).list_runs(principal, query, status, limit, before)


@router.get("/workspace/overview")
async def workspace_overview(request: Request, principal=Depends(identity)):
    return await service(request).workspace_overview(principal)


@router.get("/runs/{run_id}/files")
async def run_files(run_id: str, request: Request, principal=Depends(identity)):
    return await service(request).list_run_files(principal, run_id)


@router.get("/runs/{run_id}/thread")
async def run_thread(
    run_id: str,
    request: Request,
    limit: int = Query(default=20, ge=1, le=50),
    before: str | None = Query(default=None, max_length=128),
    principal=Depends(identity),
):
    return await service(request).inspection.thread(principal, run_id, limit, before)


@router.get("/runs/{run_id}/file")
async def run_file(
    run_id: str,
    request: Request,
    path: str = Query(min_length=1, max_length=512),
    principal=Depends(identity),
):
    return await service(request).preview_run_file(principal, run_id, path)


@router.get("/runs/{run_id}")
async def run(run_id: str, request: Request, principal=Depends(identity)):
    return await service(request).get_run(principal, run_id)


@router.get("/runs/{run_id}/events")
async def events(
    run_id: str,
    request: Request,
    after: int = 0,
    principal=Depends(identity),
    last_event_id: str | None = Header(default=None),
):
    harness = service(request)
    await harness.get_run(principal, run_id)
    if last_event_id:
        try:
            after = max(after, int(last_event_id))
        except ValueError as exc:
            raise HTTPException(422, "事件游标无效") from exc
    if after < 0:
        raise HTTPException(422, "事件游标不能为负")

    async def stream():
        cursor = after
        idle = 0
        while not await request.is_disconnected():
            try:
                await identity(request, request.headers.get("authorization", ""))
                records = await harness.events(principal, run_id, after=cursor)
                for event in records:
                    cursor = event["id"]
                    payload = json.dumps(event["data"], ensure_ascii=False)
                    yield f"id: {cursor}\nevent: {event['type']}\ndata: {payload}\n\n"
                current = await harness.get_run(principal, run_id)
                if current["status"] in TERMINAL | {"waiting_approval"} and len(records) < 1000:
                    return
                idle += 1
                if idle % 10 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)
            except HarnessError:
                yield 'event: auth_expired\ndata: {"message":"登录已失效"}\n\n'
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{run_id}/approval")
async def approval(run_id: str, body: ApprovalInput, request: Request, principal=Depends(identity)):
    return await service(request).approve(
        principal,
        run_id,
        body.approved,
        expected_call_id=body.call_id,
        expected_args_hash=body.args_hash,
    )


@router.post("/runs/{run_id}/cancel")
async def cancel(run_id: str, request: Request, principal=Depends(identity)):
    return await service(request).cancel(principal, run_id)


@router.post("/runs/{run_id}/feedback")
async def feedback(run_id: str, body: FeedbackInput, request: Request, principal=Depends(identity)):
    return await service(request).feedback(principal, run_id, body.success, body.note)


@router.get("/assets")
async def assets(
    request: Request,
    kind: str | None = None,
    status: str | None = None,
    principal=Depends(identity),
):
    return await service(request).list_assets(principal, kind=kind, status=status)


@router.post("/skills/import", status_code=201)
async def import_skill(body: SkillImportInput, request: Request, principal=Depends(identity)):
    return await service(request).import_skill(principal, **body.model_dump())


@router.get("/skills/{asset_id}/export")
async def export_skill(asset_id: str, request: Request, principal=Depends(identity)):
    return await service(request).export_skill(principal, asset_id)


@router.get("/skills/{asset_id}/resources")
async def skill_resource(
    asset_id: str,
    request: Request,
    path: str = Query(min_length=1, max_length=128),
    principal=Depends(identity),
):
    return await service(request).read_skill(principal, asset_id, path, active_only=False)


@router.post("/assets", status_code=201)
async def create_asset(body: AssetInput, request: Request, principal=Depends(identity)):
    return await service(request).put_asset(principal, **body.model_dump())


@router.get("/assets/{asset_id}")
async def asset(asset_id: str, request: Request, principal=Depends(identity)):
    return await service(request).get_asset(principal, asset_id)


@router.put("/assets/{asset_id}")
async def update_asset(
    asset_id: str, body: AssetInput, request: Request, principal=Depends(identity)
):
    return await service(request).put_asset(principal, asset_id=asset_id, **body.model_dump())


@router.post("/assets/{asset_id}/state")
async def transition(
    asset_id: str, body: TransitionInput, request: Request, principal=Depends(identity)
):
    return await service(request).transition_asset(principal, asset_id, body.status)


@router.get("/assets/{asset_id}/history")
async def history(asset_id: str, request: Request, principal=Depends(identity)):
    return await service(request).asset_history(principal, asset_id)


@router.post("/assets/{asset_id}/restore")
async def restore(asset_id: str, body: RestoreInput, request: Request, principal=Depends(identity)):
    return await service(request).restore_asset(principal, asset_id, body.version)


@router.post("/documents/upload", status_code=201)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    principal=Depends(identity),
    idempotency_key: str | None = Header(default=None),
):
    harness = service(request)
    if principal.role == "viewer":
        raise HTTPException(403, "只读角色不能上传文档")
    filename = Path((file.filename or "document.txt").replace("\\", "/")).name
    if len(filename) > 256:
        raise HTTPException(422, "文件名最多256字符")
    if Path(filename).suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise HTTPException(415, "支持 TXT、Markdown 与 PDF")
    raw = await file.read(harness.settings.max_upload_bytes + 1)
    await file.close()
    if len(raw) > harness.settings.max_upload_bytes:
        raise HTTPException(413, "文档超过上传大小限制")
    mime_type = "application/pdf" if Path(filename).suffix.lower() == ".pdf" else "text/plain"

    try:
        async with asyncio.timeout(30):
            async with request.app.state.etl_limit:
                etl = await parse_isolated(raw, filename, mime_type)
    except TimeoutError as exc:
        raise HTTPException(504, "文档解析繁忙或超时，请稍后重试") from exc
    except Exception as exc:
        raise HTTPException(422, "文档解析失败") from exc
    if not etl.chunks:
        raise HTTPException(422, "没有抽取到文本，扫描PDF需要先OCR")
    document, reused = await harness.put_uploaded_document(
        principal,
        filename,
        etl.parsed.text,
        hashlib.sha256(raw).hexdigest(),
        idempotency_key,
    )
    indexing = None
    if reused and document["content"] != etl.parsed.text:
        indexing = {"warnings": ["原文档已修改，重试仅返回原ID，未覆盖当前正文或索引"]}
    elif harness.tool_executor.knowledge is not None and document["status"] == "active":
        try:
            indexing = await harness.tool_executor.knowledge.index(principal, document, etl.chunks)
        except Exception:
            # 已入库文档可用当前正文进行词法召回；重试使用同一ID，不重复创建。
            indexing = {
                "warnings": ["索引未完成，已保存文档；使用同一幂等键重试"],
                "status": "pending",
            }
    return {
        "document_id": document["id"],
        "filename": filename,
        "status": "ready" if document["status"] == "active" else document["status"],
        "reused": reused,
        "chunk_count": len(etl.chunks),
        "indexing": indexing,
    }


@router.get("/documents")
async def documents(request: Request, principal=Depends(identity)):
    return await service(request).list_assets(principal, kind="document", status="active")
