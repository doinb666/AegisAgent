"""显式追加会话组织；SQLite在线备份，旧任务和资产不删除。"""

import asyncio
import sqlite3
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import and_, exists, func, inspect, or_, select, tuple_, update
from sqlalchemy.exc import SQLAlchemyError

from .models import RUN_SESSION_INDEX, Asset, Run, Thread, ThreadRun
from .security import canonical
from .store import Store


def sqlite_backup(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ValueError("旧SQLite数据库不存在")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb"):
        pass
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
        with sqlite3.connect(target) as backup:
            original.backup(backup)


async def migrate_threads(settings, apply=False, backup_path=None):
    store = Store(settings)
    try:
        async with store.engine.connect() as connection:
            tables = await connection.run_sync(lambda sync: set(inspect(sync).get_table_names()))
            if Run.__tablename__ not in tables:
                raise ValueError("未检测到旧任务表，请先安装并初始化应用")
            top_runs = await connection.scalar(
                select(func.count()).select_from(Run).where(Run.parent_run_id.is_(None))
            )
            groups = (
                select(Run.tenant_id, Run.owner_id, Run.session_id)
                .where(Run.parent_run_id.is_(None))
                .group_by(Run.tenant_id, Run.owner_id, Run.session_id)
            )
            session_groups = await connection.scalar(
                select(func.count()).select_from(groups.subquery())
            )
            existing_threads = (
                await connection.scalar(select(func.count()).select_from(Thread))
                if Thread.__tablename__ in tables
                else 0
            )
            existing_links = (
                await connection.scalar(select(func.count()).select_from(ThreadRun))
                if ThreadRun.__tablename__ in tables
                else 0
            )
            missing_groups = groups
            if Thread.__tablename__ in tables:
                missing_groups = missing_groups.where(
                    ~exists(
                        select(Thread.id).where(
                            Thread.tenant_id == Run.tenant_id,
                            Thread.owner_id == Run.owner_id,
                            Thread.session_id == Run.session_id,
                        )
                    )
                )
            pending_threads = await connection.scalar(
                select(func.count()).select_from(missing_groups.subquery())
            )
            missing_runs = select(func.count()).select_from(Run).where(Run.parent_run_id.is_(None))
            if ThreadRun.__tablename__ in tables:
                missing_runs = missing_runs.where(
                    ~exists(select(ThreadRun.run_id).where(ThreadRun.run_id == Run.id))
                )
            pending_links = await connection.scalar(missing_runs)
        result = {
            "applied": apply,
            "top_runs": top_runs,
            "session_groups": session_groups,
            "existing_threads": existing_threads,
            "existing_links": existing_links,
            "pending_threads": pending_threads,
            "pending_links": pending_links,
            "created_threads": 0,
            "linked_runs": 0,
        }
        if not apply:
            return result
        await create_backup(store, backup_path)
        async with store.engine.begin() as connection:
            if store.engine.dialect.name == "sqlite":
                await connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                await connection.exec_driver_sql("SELECT pg_advisory_xact_lock(1650827633)")
            for table in (Thread.__table__, ThreadRun.__table__):
                await connection.run_sync(
                    lambda sync, table=table: table.create(sync, checkfirst=True)
                )
            await connection.run_sync(lambda sync: RUN_SESSION_INDEX.create(sync, checkfirst=True))
            # 分组只读取标识，运行配置逐批200项读取，避免加载全部执行上下文。
            async for tenant_id, owner_id, session_id in grouped_sessions(connection, groups):
                private = (
                    Run.tenant_id == tenant_id,
                    Run.owner_id == owner_id,
                    Run.session_id == session_id,
                    Run.parent_run_id.is_(None),
                )
                thread = (
                    (
                        await connection.execute(
                            select(Thread.__table__).where(
                                Thread.tenant_id == tenant_id,
                                Thread.owner_id == owner_id,
                                Thread.session_id == session_id,
                            )
                        )
                    )
                    .mappings()
                    .first()
                )
                if thread is None:
                    earliest = (
                        await connection.execute(
                            select(Run.id, Run.message, Run.created)
                            .where(*private)
                            .order_by(Run.created, Run.id)
                            .limit(1)
                        )
                    ).one()
                    project = await connection.scalar(
                        select(Asset.id).where(
                            Asset.id == session_id,
                            Asset.kind == "project",
                            Asset.tenant_id == tenant_id,
                            Asset.owner_id == owner_id,
                        )
                    )
                    thread_id = str(
                        uuid5(NAMESPACE_URL, canonical([tenant_id, owner_id, session_id]))
                    )
                    await connection.execute(
                        Thread.__table__.insert().values(
                            id=thread_id,
                            tenant_id=tenant_id,
                            owner_id=owner_id,
                            session_id=session_id,
                            title=earliest.message.strip()[:200] or "历史会话",
                            archived=False,
                            project_asset_id=project,
                            created=earliest.created,
                        )
                    )
                    result["created_threads"] += 1
                else:
                    thread_id = thread["id"]
                cursor = None
                while True:
                    query = select(Run.id, Run.config, Run.created).where(*private)
                    if cursor:
                        query = query.where(
                            or_(
                                Run.created > cursor[0],
                                and_(Run.created == cursor[0], Run.id > cursor[1]),
                            )
                        )
                    rows = (
                        await connection.execute(query.order_by(Run.created, Run.id).limit(200))
                    ).all()
                    if not rows:
                        break
                    for run_id, config, created in rows:
                        relation = (
                            (
                                await connection.execute(
                                    select(ThreadRun.__table__).where(ThreadRun.run_id == run_id)
                                )
                            )
                            .mappings()
                            .first()
                        )
                        if relation is not None and (
                            relation["thread_id"] != thread_id
                            or relation["tenant_id"] != tenant_id
                            or relation["owner_id"] != owner_id
                        ):
                            raise ValueError("检测到冲突运行关联，迁移已回滚，请核对备份")
                        if relation is None:
                            await connection.execute(
                                ThreadRun.__table__.insert().values(
                                    run_id=run_id,
                                    thread_id=thread_id,
                                    tenant_id=tenant_id,
                                    owner_id=owner_id,
                                )
                            )
                            result["linked_runs"] += 1
                        if config.get("thread_id") != thread_id:
                            await connection.execute(
                                update(Run)
                                .where(Run.id == run_id)
                                .values(config={**config, "thread_id": thread_id})
                            )
                    cursor = (rows[-1].created, rows[-1].id)
        return result
    finally:
        await store.close()


async def grouped_sessions(connection, groups):
    """会话组也按游标分批，内存占用不随历史会话数量增长。"""
    columns = (Run.tenant_id, Run.owner_id, Run.session_id)
    cursor = None
    while True:
        query = groups.order_by(*columns).limit(200)
        if cursor is not None:
            query = query.where(tuple_(*columns) > tuple_(*cursor))
        rows = (await connection.execute(query)).all()
        if not rows:
            return
        for row in rows:
            yield row
        cursor = tuple(rows[-1])


async def create_backup(store, backup_path):
    """所有显式追加迁移共用备份门，拒绝覆盖已有SQLite备份。"""
    if backup_path is None:
        raise ValueError("执行迁移必须提供新的SQLite备份路径，或已完成的PostgreSQL备份文件")
    backup_path = Path(backup_path).resolve()
    if store.engine.dialect.name == "sqlite":
        await asyncio.to_thread(
            sqlite_backup, Path(store.engine.url.database).resolve(), backup_path
        )
        return
    if not backup_path.is_file() or backup_path.stat().st_size < 256:
        raise ValueError("请先完成PostgreSQL备份并提供非空备份文件")
    with backup_path.open("rb") as source:
        prefix = source.read(128)
    if not (prefix.startswith(b"PGDMP") or b"PostgreSQL database dump" in prefix):
        raise ValueError("备份文件须为pg_dump自定义格式或SQL格式")


def main(argv=None):
    """源码、wheel与EXE共用迁移入口，默认预览且不打印凭据。"""
    import argparse
    import json
    import os
    import sys

    from .settings import HarnessSettings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path, help="含.env的配置目录")
    parser.add_argument("--apply", action="store_true", help="显式执行，默认仅预览")
    parser.add_argument("--backup", type=Path, help="新SQLite备份路径，或已完成pg_dump的备份文件")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.resolve() if args.data_dir else None
    backup = args.backup.resolve() if args.backup else None
    if getattr(sys, "frozen", False) and data_dir is None:
        data_dir = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "AegisCode"
    if args.config:
        os.chdir(args.config.resolve())
    settings = HarnessSettings(**({"data_dir": data_dir} if data_dir else {}))
    try:
        result = asyncio.run(migrate_threads(settings, args.apply, backup))
    except (ValueError, OSError, SQLAlchemyError):
        parser.error("迁移未完成，请检查数据库、备份路径与文件格式；服务数据未删除")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
