"""FastAPI 与 AegisCode 入口：持久化 Harness 管理生命周期。"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.limits import EntryLimits
from app.api.routes import harness, health
from app.config import get_settings
from app.harness.errors import HarnessError
from app.harness.evolution import EvolutionWorker
from app.harness.service import HarnessService
from app.harness.settings import HarnessSettings
from app.harness_tools.catalog import HarnessTools
from app.harness_tools.knowledge import KnowledgeService
from app.infrastructure.llm.model_router import build_harness_router


def create_app(harness_settings=None, model_router=None, tool_executor=None) -> FastAPI:
    load_dotenv(override=False)
    settings = get_settings()
    harness_settings = harness_settings or HarnessSettings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        model = model_router or build_harness_router(harness_settings)
        tools = tool_executor or HarnessTools(harness_settings)
        service = HarnessService(harness_settings, model, tools)
        tools.service = service
        if isinstance(tools, HarnessTools):
            tools.knowledge = KnowledgeService(harness_settings, service)
        application.state.harness = service
        application.state.etl_limit = asyncio.Semaphore(2)
        await service.initialize()
        evolution = EvolutionWorker(service)
        evolution.start()
        logger.info("AegisCode 启动，持久任务与身份隔离已启用")
        try:
            yield
        finally:
            await evolution.close()
            await service.close()
            if model and hasattr(model, "aclose"):
                await model.aclose()

    application = FastAPI(
        title="AegisCode",
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(harness.router, prefix=settings.api_prefix)
    application.add_middleware(
        EntryLimits,
        max_bytes=harness_settings.max_upload_bytes + 65536,
    )

    @application.exception_handler(HarnessError)
    async def harness_error(request: Request, error: HarnessError):
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})

    @application.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    web = Path(__file__).parent / "web"
    application.mount("/static", StaticFiles(directory=web), name="static")

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(web / "index.html")

    return application


app = create_app()
