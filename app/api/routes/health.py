# -*- coding: utf-8 -*-
"""健康检查：进程存活与依赖探测。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.infrastructure.database.session import get_async_session

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """轻量存活探针（不访问外部依赖）。"""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(
    session: AsyncSession = Depends(get_async_session),
) -> dict[str, Any]:
    """就绪探针：检查数据库连通性。"""
    settings = get_settings()
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        logger.warning("数据库就绪检查失败: {}", exc)
        db_ok = False

    return {
        "status": "ready" if db_ok else "degraded",
        "database": "up" if db_ok else "down",
        "app_env": settings.app_env,
    }
