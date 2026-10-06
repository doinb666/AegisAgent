"""站内通知读取；事件ID即唯一通知ID，不保留工具参数和回答。"""

import time

from sqlalchemy import func, select, update

from .errors import HarnessError
from .models import Notification
from .store import scope


def notification_dict(notification):
    return {
        "id": notification.event_id,
        **{
            field: getattr(notification, field)
            for field in (
                "run_id",
                "type",
                "title",
                "created",
                "read_at",
            )
        },
    }


class NotificationService:
    def __init__(self, store):
        self.store = store

    def require_ready(self):
        if not self.store.notifications_ready:
            raise HarnessError(503, "站内通知尚未启用，请先备份并运行通知迁移")

    @staticmethod
    def identifier(value):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise HarnessError(422, "通知标识无效")

    async def owned(self, session, principal, identifier):
        self.identifier(identifier)
        notification = await session.scalar(
            select(Notification).where(
                *scope(Notification, principal),
                Notification.event_id == identifier,
            )
        )
        if notification is None:
            raise HarnessError(404, "通知不存在或无权访问")
        return notification

    async def list(self, principal, limit=20, before=None):
        self.require_ready()
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50:
            raise HarnessError(422, "通知分页参数无效")
        async with self.store.sessions() as session:
            query = select(Notification).where(*scope(Notification, principal))
            if before is not None:
                await self.owned(session, principal, before)
                query = query.where(Notification.event_id < before)
            rows = (
                await session.scalars(query.order_by(Notification.event_id.desc()).limit(limit))
            ).all()
            return [notification_dict(row) for row in rows]

    async def unread(self, principal):
        self.require_ready()
        async with self.store.sessions() as session:
            return await session.scalar(
                select(func.count())
                .select_from(Notification)
                .where(
                    *scope(Notification, principal),
                    Notification.read_at.is_(None),
                )
            )

    async def read(self, principal, identifier):
        self.require_ready()
        async with self.store.transaction() as session:
            await self.owned(session, principal, identifier)
            await session.execute(
                update(Notification)
                .where(
                    *scope(Notification, principal),
                    Notification.event_id == identifier,
                    Notification.read_at.is_(None),
                )
                .values(read_at=time.time())
            )
            notification = await self.owned(session, principal, identifier)
            return notification_dict(notification)
