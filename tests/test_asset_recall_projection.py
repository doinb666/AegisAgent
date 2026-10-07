"""真实 SQLite 行与 SQL 验证召回投影，不启动 HTTP 或后台服务。"""

import json
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import event, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.harness.assets import AssetService
from app.harness.errors import HarnessError, Principal
from app.harness.models import Asset
from app.harness.settings import HarnessSettings
from app.harness.store import Store


class RecordingSession(AsyncSession):
    async def execute(self, statement, *args, **kwargs):
        result = await super().execute(statement, *args, **kwargs)
        if statement.is_select:
            frozen = result.freeze()
            self.info["reads"].append((statement, frozen.data))
            if len(self.info["reads"]) == 1 and self.info.get("after_projection"):
                await self.info["after_projection"]()
            return frozen()
        return result


@pytest_asyncio.fixture
async def recall_db(tmp_path):
    store = Store(HarnessSettings(data_dir=tmp_path))
    await store.initialize()
    reads, sql = [], []
    store.sessions = async_sessionmaker(
        store.engine, class_=RecordingSession, expire_on_commit=False, info={"reads": reads}
    )

    @event.listens_for(store.engine.sync_engine, "before_cursor_execute")
    def capture_sql(connection, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            sql.append((statement, parameters))

    db = SimpleNamespace(
        store=store,
        service=AssetService(store),
        principal=Principal("owner", "tenant", "owner"),
        reads=reads,
        sql=sql,
    )
    try:
        yield db
    finally:
        await store.close()


async def insert_assets(db, *assets):
    async with db.store.sessions.begin() as session:
        for values in assets:
            session.add(
                Asset(
                    **{
                        "id": "match",
                        "tenant_id": "tenant",
                        "owner_id": "owner",
                        "kind": "skill",
                        "name": "reader",
                        "content": "正文秘密",
                        "status": "active",
                        "version": 1,
                        "attributes": {"description": "research"},
                        **values,
                    }
                )
            )


@pytest.mark.asyncio
async def test_first_query_returns_only_lightweight_columns_and_json_fields(recall_db):
    db = recall_db
    await insert_assets(
        db,
        {
            "attributes": {
                "description": "research",
                "resources": {"notes.md": "资源秘密"},
                "tools": [],
            }
        },
    )
    result = await db.service.recall(db.principal, "research", [])
    assert [asset["id"] for asset in result] == ["match"]
    first_sql = db.sql[0][0].split("FROM")[0]
    assert "harness_assets.content" not in first_sql
    assert "resources" not in first_sql
    statement, rows = db.reads[0]
    assert "content" not in statement.selected_columns.keys()
    assert "attributes" not in statement.selected_columns.keys()
    assert len(rows) == 1
    assert "正文秘密" not in repr(rows) and "资源秘密" not in repr(rows)
    assert result[0]["content"] == "正文秘密"
    assert result[0]["metadata"]["resource_paths"] == ["notes.md"]
    assert "resources" not in result[0]["metadata"]


@pytest.mark.asyncio
async def test_scan_shortlist_and_injection_have_500_32_8_limits(recall_db):
    db = recall_db
    await insert_assets(
        db,
        *(
            {
                "id": f"{index:04}",
                "name": f"research-{index:04}",
                "kind": "memory",
                "content": "research " + "长正文" * 10000,
            }
            for index in range(501)
        ),
    )
    result = await db.service.recall(db.principal, "research")
    assert len(db.reads[0][1]) == 500
    assert len(db.reads) == 2 and len(db.reads[1][1]) == 32
    assert [asset["id"] for asset in result] == [f"{index:04}" for index in range(8)]
    assert "长正文" not in repr(db.reads[0][1])


@pytest.mark.asyncio
async def test_empty_candidates_never_read_unrelated_long_bodies(recall_db):
    db = recall_db
    await insert_assets(
        db,
        *(
            {
                "id": str(index),
                "kind": "memory",
                "name": "unrelated",
                "attributes": {},
                "content": "research " + "无关正文" * 10000,
            }
            for index in range(80)
        ),
    )
    assert await db.service.recall(db.principal, "research") == []
    assert len(db.reads) == 1 and len(db.reads[0][1]) == 80
    assert "harness_assets.content" not in db.sql[0][0]


@pytest.mark.parametrize("query", ["reading", "文献证据", "", "missing"])
@pytest.mark.asyncio
async def test_directory_chinese_and_empty_query_preserve_priority(recall_db, query):
    db = recall_db
    await insert_assets(
        db,
        {"id": "skill", "attributes": {"description": "文献证据", "directory": "reading"}},
        *(
            {"id": kind, "kind": kind, "name": kind, "attributes": {}}
            for kind in ["constraint", "preference", "profile"]
        ),
    )
    result = await db.service.recall(db.principal, query)
    expected = ["constraint", "preference", "profile"]
    if query in ["reading", "文献证据"]:
        expected.append("skill")
    assert [asset["id"] for asset in result] == expected


@pytest.mark.asyncio
async def test_coarse_window_preserves_existing_score_order(recall_db):
    db = recall_db
    await insert_assets(
        db,
        {"id": "priority", "kind": "constraint", "attributes": {}},
        *({"id": f"skill-{i}", "attributes": {"description": "research"}} for i in range(40)),
    )
    result = await db.service.recall(db.principal, "research")
    assert "priority" not in [asset["id"] for asset in result]
    assert len(db.reads[1][1]) == 32


@pytest.mark.asyncio
async def test_tools_missing_null_and_invalid_values_are_distinct(recall_db):
    db = recall_db
    await insert_assets(
        db,
        {"id": "missing"},
        {
            "id": "empty",
            "attributes": {
                "description": "research",
                "tools": [],
            },
        },
        *(
            {"id": f"bad-{i}", "attributes": {"description": "research", "tools": value}}
            for i, value in enumerate([None, "calculator", {}, [1], [""], ["x"] * 33])
        ),
        {"id": "requires-tool", "attributes": {"description": "research", "tools": ["read"]}},
    )
    assert [asset["id"] for asset in await db.service.recall(db.principal, "research", [])] == [
        "empty",
        "missing",
    ]
    assert len(db.reads[1][1]) == 2
    projected = {row[0]: row._mapping for row in db.reads[0][1]}
    assert projected["missing"]["tools"] is None
    assert projected["bad-0"]["tools"] == "null"


@pytest.mark.asyncio
async def test_scope_status_and_excluded_kinds_are_filtered_in_sql(recall_db):
    db = recall_db
    await insert_assets(
        db,
        {},
        {"id": "peer", "owner_id": "peer"},
        {"id": "other-tenant", "tenant_id": "other"},
        {"id": "draft", "status": "draft"},
        {"id": "retired", "status": "retired"},
        *({"id": kind, "kind": kind} for kind in ["document", "artifact", "project"]),
    )
    assert [asset["id"] for asset in await db.service.recall(db.principal, "research")] == ["match"]
    assert all(len(rows) == 1 for _, rows in db.reads)
    for sql, parameters in db.sql:
        assert "tenant_id" in sql and "owner_id" in sql and "status" in sql
        assert "document" in parameters and "artifact" in parameters and "project" in parameters


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 2, "content": "新版秘密"},
        {"status": "retired", "content": "退役秘密"},
        {"owner_id": "peer", "content": "别人秘密"},
        {"tenant_id": "other", "content": "跨租户秘密"},
        {"kind": "document", "content": "资料秘密"},
        {"attributes": {"description": "research", "tools": ["root_shell"]}, "content": "危险正文"},
    ],
)
@pytest.mark.asyncio
async def test_changes_between_phases_cannot_inject_new_body(recall_db, changes):
    db = recall_db
    await insert_assets(db, {})

    async def change_candidate():
        async with db.store.sessions.begin() as session:
            await session.execute(update(Asset).where(Asset.id == "match").values(**changes))

    db.store.sessions.configure(info={"reads": db.reads, "after_projection": change_candidate})
    assert await db.service.recall(db.principal, "research", []) == []
    assert len(db.reads) == 2
    if "attributes" not in changes:
        assert len(db.reads[1][1]) == 0


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        "bad",
        {
            "description": "research",
            "resources": {
                "../escape": "秘密",
            },
        },
        {"description": "research", "resources": {"ok.md": 1}},
        {"description": "research", "description_extra": "x" * 9000},
    ],
)
@pytest.mark.asyncio
async def test_bad_legacy_metadata_or_resource_is_isolated(recall_db, metadata):
    db = recall_db
    await insert_assets(
        db, {"id": "good"}, {"id": "bad", "name": "research", "attributes": metadata}
    )
    assert [asset["id"] for asset in await db.service.recall(db.principal, "research")] == ["good"]


@pytest.mark.asyncio
async def test_invalid_shortlist_does_not_refill_beyond_32_bodies(recall_db):
    db = recall_db
    await insert_assets(
        db,
        *(
            {
                "id": f"{i:02}",
                "attributes": {
                    "description": "research",
                    "resources": {"../bad": "秘密"},
                },
            }
            for i in range(32)
        ),
        {"id": "good"},
    )
    assert await db.service.recall(db.principal, "research") == []
    assert len(db.reads[1][1]) == 32


@pytest.mark.asyncio
async def test_body_scoring_is_limited_to_first_3000_characters(recall_db):
    db = recall_db
    await insert_assets(
        db,
        {"id": "a", "kind": "memory", "content": "x" * 3000 + " research"},
        {"id": "b", "kind": "memory", "content": "research"},
    )
    assert [asset["id"] for asset in await db.service.recall(db.principal, "research")] == [
        "b",
        "a",
    ]


@pytest.mark.parametrize("field", ["tags", "examples", "boundaries", "trigger_conditions"])
@pytest.mark.asyncio
async def test_each_directory_descriptor_is_a_coarse_signal(recall_db, field):
    db = recall_db
    await insert_assets(db, {"attributes": {field: ["research"]}})
    assert [asset["id"] for asset in await db.service.recall(db.principal, "research")] == ["match"]


@pytest.mark.asyncio
async def test_user_verified_keeps_existing_fine_score_bonus(recall_db):
    db = recall_db
    await insert_assets(
        db,
        {"id": "a"},
        {
            "id": "z",
            "attributes": {
                "description": "research",
                "user_verified": True,
            },
        },
    )
    assert [asset["id"] for asset in await db.service.recall(db.principal, "research")] == [
        "z",
        "a",
    ]


@pytest.mark.asyncio
async def test_match_after_500_scan_window_remains_out_of_scope(recall_db):
    db = recall_db
    await insert_assets(
        db,
        *(
            {
                "id": f"{i:04}",
                "name": f"a-{i:04}",
                "attributes": {},
            }
            for i in range(500)
        ),
        {"id": "last", "name": "z-last"},
    )
    assert await db.service.recall(db.principal, "research") == []
    assert len(db.reads) == 1 and len(db.reads[0][1]) == 500


@pytest.mark.parametrize("dialect_name", ["sqlite", "postgresql"])
def test_projection_compiles_bound_fields_for_supported_dialects(dialect_name):
    from sqlalchemy.dialects import postgresql, sqlite

    from app.harness.asset_recall import PROJECTION_FIELDS, projection_query

    dialect = {"sqlite": sqlite.dialect, "postgresql": postgresql.dialect}[dialect_name]()
    statement = projection_query(Principal("owner", "tenant", "owner"))
    compiled = statement.compile(dialect=dialect)
    select_sql = str(compiled).split("FROM")[0]
    assert "harness_assets.content" not in select_sql and "resources" not in select_sql
    assert set(statement.selected_columns.keys()) == {
        "id",
        "kind",
        "name",
        "version",
        *PROJECTION_FIELDS,
    }
    assert "tenant" in compiled.params.values() and "owner" in compiled.params.values()
    assert "LIMIT" in str(compiled) and 500 in compiled.params.values()
    assert all(
        key in compiled.params.values() or f'$."{key}"' in compiled.params.values()
        for key in PROJECTION_FIELDS
    )


@pytest.mark.asyncio
async def test_json_boolean_descriptor_preserves_legacy_terms(recall_db):
    db = recall_db
    await insert_assets(db, {"kind": "memory", "attributes": {"tags": [True, False]}})
    # 历史普通资产的 JSON 描述字段没有 Skill frontmatter 类型约束。
    assert [asset["id"] for asset in await db.service.recall(db.principal, "true false")] == [
        "match"
    ]
    db.reads.clear()
    async with db.store.sessions.begin() as session:
        await session.execute(
            update(Asset).where(Asset.id == "match").values(attributes={"tags": True})
        )
    assert [asset["id"] for asset in await db.service.recall(db.principal, "true")] == ["match"]


@pytest.mark.parametrize(
    "value",
    [
        9223372036854775808,
        10**80,
        -(10**80),
        1.234567890123456e30,
        1e-9,
        None,
        True,
        False,
    ],
)
@pytest.mark.asyncio
async def test_scalar_json_projection_preserves_numeric_and_null_terms(recall_db, value):
    db = recall_db
    await insert_assets(db, {"kind": "memory", "attributes": {"tags": value}})
    result = await db.service.recall(db.principal, str(value))
    assert [asset["id"] for asset in result] == ["match"]
    decoded = json.loads(db.reads[0][1][0]._mapping["tags"])
    assert type(decoded) is type(value) and decoded == value


@pytest.mark.parametrize("version", [(3, 37, 2), None, ()])
@pytest.mark.asyncio
async def test_unsupported_sqlite_version_fails_before_query(recall_db, monkeypatch, version):
    db = recall_db
    monkeypatch.setattr(db.store.engine.dialect, "server_version_info", version)
    with pytest.raises(HarnessError) as error:
        await db.service.recall(db.principal, "research")
    assert error.value.status_code == 503
    assert "SQLite" in error.value.detail and "3.38" in error.value.detail
    assert "PostgreSQL" in error.value.detail
    assert not db.reads and not db.sql


@pytest.mark.asyncio
async def test_postgresql_dialect_is_not_blocked_by_sqlite_version_gate(recall_db, monkeypatch):
    """仅模拟方言分支，不作为真实 PostgreSQL 数据库运行验收。"""
    db = recall_db
    monkeypatch.setattr(db.store.engine.dialect, "name", "postgresql")
    monkeypatch.setattr(db.store.engine.dialect, "server_version_info", None)
    assert await db.service.recall(db.principal, "research") == []
    assert len(db.reads) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["tags", "extra"])
@pytest.mark.asyncio
async def test_nonfinite_legacy_json_is_guarded_in_every_projected_field(recall_db, value, field):
    """在当前 SQLite 验证严格 JSON 保护，未实际运行 SQLite 3.38～3.41。"""
    from app.harness.asset_recall import PROJECTION_FIELDS

    db = recall_db
    await insert_assets(
        db,
        {"id": "good"},
        {
            "id": "bad",
            "kind": "memory",
            "name": "research",
            "attributes": {
                "description": "research",
                field: value,
            },
        },
    )
    result = await db.service.recall(db.principal, "research")
    assert [asset["id"] for asset in result] == ["good"]
    projected = {row[0]: row._mapping for row in db.reads[0][1]}
    assert all(projected["bad"][key] is None for key in PROJECTION_FIELDS)
    assert len(db.reads[0][1]) == 2
    assert "json_valid" in db.sql[0][0].split("FROM")[0]
