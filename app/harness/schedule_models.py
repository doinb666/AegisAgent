"""私有计划与触发账本；租约只用于待处理触发，不复活终态。"""

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .models import Base


class Schedule(Base):
    __tablename__ = "harness_schedules"
    __table_args__ = (
        UniqueConstraint("tenant_id", "owner_id", "idempotency_key"),
        Index("ix_schedule_owner_page", "tenant_id", "owner_id", "created", "id"),
        Index("ix_schedule_due", "status", "next_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(36))
    owner_id: Mapped[str] = mapped_column(String(36))
    idempotency_key: Mapped[str] = mapped_column(String(256))
    payload_hash: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="active")
    version: Mapped[int] = mapped_column(Integer, default=1)
    next_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    interval_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str] = mapped_column(String(16))
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created: Mapped[float] = mapped_column(Float)


class Occurrence(Base):
    __tablename__ = "harness_schedule_occurrences"
    __table_args__ = (
        UniqueConstraint("schedule_id", "scheduled_at"),
        Index("ix_occurrence_claim", "status", "retry_at", "lease_until"),
        Index("ix_occurrence_schedule", "schedule_id", "scheduled_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    schedule_id: Mapped[str] = mapped_column(ForeignKey("harness_schedules.id"))
    tenant_id: Mapped[str] = mapped_column(String(36))
    owner_id: Mapped[str] = mapped_column(String(36))
    scheduled_at: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    retry_at: Mapped[float] = mapped_column(Float, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[float | None] = mapped_column(Float, nullable=True)
