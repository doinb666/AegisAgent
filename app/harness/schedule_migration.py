"""显式追加定时计划表；默认预览，执行前必须备份，不修改旧任务资产。"""

from sqlalchemy import inspect

from .models import Run
from .schedule_models import Occurrence, Schedule
from .store import Store
from .thread_migration import create_backup


async def migrate_schedules(settings, apply=False, backup_path=None):
    store = Store(settings)
    try:
        async with store.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            if Run.__tablename__ not in tables:
                raise ValueError("请先安装并初始化应用")
        missing = [
            table.name
            for table in (Schedule.__table__, Occurrence.__table__)
            if table.name not in tables
        ]
        result = {"applied": apply, "pending_tables": missing, "created_tables": []}
        if not apply:
            return result
        await create_backup(store, backup_path)
        async with store.engine.begin() as connection:
            if store.engine.dialect.name == "sqlite":
                await connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                await connection.exec_driver_sql("SELECT pg_advisory_xact_lock(1650827635)")
            current = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            for table in (Schedule.__table__, Occurrence.__table__):
                await connection.run_sync(
                    lambda sync, table=table: table.create(sync, checkfirst=True)
                )
                if table.name not in current:
                    result["created_tables"].append(table.name)
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
    parser.add_argument("--config", type=Path, help="含 .env 的配置目录")
    parser.add_argument("--apply", action="store_true", help="显式执行，默认只预览")
    parser.add_argument("--backup", type=Path, help="新 SQLite 备份路径，或已完成 pg_dump 的文件")
    args = parser.parse_args(argv)
    directory = args.data_dir.resolve() if args.data_dir else None
    backup = args.backup.resolve() if args.backup else None
    if getattr(sys, "frozen", False) and directory is None:
        directory = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AegisCode"
    if args.config:
        os.chdir(args.config.resolve())
    settings = HarnessSettings(**({"data_dir": directory} if directory else {}))
    try:
        result = asyncio.run(migrate_schedules(settings, args.apply, backup))
    except (ValueError, OSError, SQLAlchemyError):
        parser.error("定时任务迁移未完成，请检查数据库及备份；服务数据未删除")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
