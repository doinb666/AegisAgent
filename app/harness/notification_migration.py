"""旧事件显式追加为站内通知；默认仅预览，执行必须先备份。"""

from sqlalchemy import exists, func, inspect, select

from .models import NOTIFICATION_TYPES, Event, Notification, Run
from .store import Store
from .thread_migration import create_backup


async def migrate_notifications(settings, apply=False, backup_path=None):
    store = Store(settings)
    try:
        async with store.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            if Event.__tablename__ not in tables:
                raise ValueError("请先安装并初始化应用")
            query = (
                select(
                    Event.id,
                    Event.tenant_id,
                    Event.owner_id,
                    Event.run_id,
                    Event.type,
                    Run.message,
                    Run.created,
                )
                .join(Run, Run.id == Event.run_id)
                .where(
                    Event.type.in_(NOTIFICATION_TYPES),
                    Run.parent_run_id.is_(None),
                    Event.tenant_id == Run.tenant_id,
                    Event.owner_id == Run.owner_id,
                )
            )
            if Notification.__tablename__ in tables:
                query = query.where(
                    ~exists(select(Notification.event_id).where(Notification.event_id == Event.id))
                )
            pending = await connection.scalar(select(func.count()).select_from(query.subquery()))
        result = {"applied": apply, "pending_notifications": pending, "created_notifications": 0}
        if not apply:
            return result
        await create_backup(store, backup_path)
        async with store.engine.begin() as connection:
            if store.engine.dialect.name == "sqlite":
                await connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                await connection.exec_driver_sql("SELECT pg_advisory_xact_lock(1650827634)")
            await connection.run_sync(
                lambda sync: Notification.__table__.create(sync, checkfirst=True)
            )
            # 重新加去重谓词，覆盖预览后另一实例已完成迁移的情况。
            query = query.where(
                ~exists(select(Notification.event_id).where(Notification.event_id == Event.id))
            )
            cursor = 0
            while True:
                rows = (
                    await connection.execute(
                        query.where(Event.id > cursor).order_by(Event.id).limit(200)
                    )
                ).all()
                if not rows:
                    break
                for row in rows:
                    await connection.execute(
                        Notification.__table__.insert().values(
                            event_id=row.id,
                            tenant_id=row.tenant_id,
                            owner_id=row.owner_id,
                            run_id=row.run_id,
                            type=row.type,
                            title=row.message[:160],
                            created=row.created,
                        )
                    )
                    result["created_notifications"] += 1
                cursor = rows[-1].id
        return result
    finally:
        await store.close()


def main(argv=None):
    import argparse
    import asyncio
    import json
    import os
    import sys
    from pathlib import Path

    from sqlalchemy.exc import SQLAlchemyError

    from .settings import HarnessSettings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args(argv)
    directory = args.data_dir.resolve() if args.data_dir else None
    backup = args.backup.resolve() if args.backup else None
    if getattr(sys, "frozen", False) and directory is None:
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AegisCode"
    if args.config:
        os.chdir(args.config.resolve())
    settings = HarnessSettings(**({"data_dir": directory} if directory else {}))
    try:
        result = asyncio.run(migrate_notifications(settings, args.apply, backup))
    except (ValueError, OSError, SQLAlchemyError):
        parser.error("通知迁移未完成，请检查数据库及备份；服务数据未删除")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
