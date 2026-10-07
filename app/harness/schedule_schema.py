"""计划请求统一校验；禁止携带路径和自行选择工具。"""

from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, field_validator

ScheduleStatus = Literal["active", "paused", "cancelled", "completed"]


def utc_iso(value):
    if value is None:
        return None
    return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=value)).isoformat()


class ScheduleInput(BaseModel):
    model_config = {"extra": "forbid"}

    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=32000)
    scheduled_at: AwareDatetime
    interval_seconds: int | None = Field(default=None, ge=60, le=31 * 86400, strict=True)
    mode: Literal["react", "plan", "reflection"] = "react"
    model: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("title", "message", "model")
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("内容不能为空白")
        return value

    @field_validator("scheduled_at", mode="before")
    @classmethod
    def explicit_timezone(cls, value):
        # 拒绝数字时间戳，确保公开请求显式说明时区。
        if not isinstance(value, (str, datetime)):
            raise ValueError("请提供带时区的 ISO8601 时间")
        # Pydantic 默认接受纯数字字符串时间戳；这里明确要求日历形式的 ISO8601。
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    @field_validator("scheduled_at")
    @classmethod
    def normalize_utc(cls, value):
        try:
            normalized = value.astimezone(UTC)
            utc_iso(normalized.timestamp())
        except (ValueError, OverflowError, OSError) as exc:
            raise ValueError("UTC 时间超出受支持的日历范围") from exc
        return normalized


class ScheduleTransition(BaseModel):
    model_config = {"extra": "forbid"}

    action: Literal["pause", "resume", "cancel"]
    expected_version: int = Field(ge=1, strict=True)
