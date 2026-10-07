"""有界词法召回：先读目录投影，再按原版本读取最多32条正文。"""

import json
import re

from sqlalchemy import Text, and_, literal, or_, select
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement

from .errors import HarnessError
from .models import Asset
from .skill_files import export_bundle, string_list, validate_metadata
from .store import asset_dict, scope

PROJECTION_FIELDS = (
    "description",
    "tags",
    "examples",
    "boundaries",
    "trigger_conditions",
    "directory",
    "tools",
    "user_verified",
)
DESCRIPTOR_FIELDS = PROJECTION_FIELDS[:6]
PRIORITY_KINDS = {"preference", "constraint", "profile"}
EXCLUDED_KINDS = ("document", "artifact", "project")
SCAN_LIMIT = 500
BODY_LIMIT = 32
RESULT_LIMIT = 8


class JsonFieldText(FunctionElement):
    """缺失返回 SQL NULL，显式 JSON null 返回文本 null，避免放宽工具权限。"""

    type = Text()
    inherit_cache = True


@compiles(JsonFieldText, "sqlite")
def sqlite_json_field(element, compiler, **kwargs):
    column, _key, path = element.clauses
    column_sql = compiler.process(column, **kwargs)
    path_sql = compiler.process(path, **kwargs)
    # SQLite 3.38 起的 -> 返回 JSON 文本，保留大整数与浮点精度及缺失/null区别。
    # 严格 JSON 校验在 CASE 内短路，旧版 SQLite 也不会解析 NaN/Infinity 旧行。
    return f"CASE WHEN json_valid({column_sql}) THEN ({column_sql} -> {path_sql}) END"


@compiles(JsonFieldText, "postgresql")
def postgresql_json_field(element, compiler, **kwargs):
    column, key, _path = element.clauses
    column_sql = compiler.process(column, **kwargs)
    key_sql = compiler.process(key, **kwargs)
    return f"CAST(({column_sql} -> {key_sql}) AS TEXT)"


def terms(text):
    words = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]+", text.lower()))
    for word in list(words):
        if re.search(r"[\u4e00-\u9fff]", word):
            words.update(word[index : index + 2] for index in range(len(word) - 1))
    return words


def eligible(principal):
    return (
        *scope(Asset, principal),
        Asset.status == "active",
        Asset.kind.not_in(EXCLUDED_KINDS),
    )


def projection_query(principal):
    """逐字段提取，绝不选择正文、整列元数据或资源内容。"""
    fields = [
        JsonFieldText(Asset.attributes, literal(key), literal(f'$."{key}"')).label(key)
        for key in PROJECTION_FIELDS
    ]
    return (
        select(Asset.id, Asset.kind, Asset.name, Asset.version, *fields)
        .where(*eligible(principal))
        .order_by(Asset.name, Asset.id)
        .limit(SCAN_LIMIT)
    )


def tools_allowed(metadata, allowed_tools):
    required = set(string_list(metadata.get("tools", []), "tools", 128))
    return allowed_tools is None or required.issubset(allowed_tools)


def coarse_candidates(rows, query_terms, allowed_tools):
    candidates = []
    for row in rows:
        try:
            metadata = {
                key: json.loads(row[key]) for key in PROJECTION_FIELDS if row[key] is not None
            }
            validate_metadata(metadata)
            if not tools_allowed(metadata, allowed_tools):
                continue
        except (HarnessError, ValueError):
            continue
        descriptor = terms(
            row["name"] + " " + " ".join(str(metadata.get(key, "")) for key in DESCRIPTOR_FIELDS)
        )
        score = len(query_terms & descriptor)
        if score or row["kind"] in PRIORITY_KINDS:
            candidates.append((score, row["id"], row["version"]))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[:BODY_LIMIT]


def validated_asset(row, allowed_tools):
    asset = asset_dict(row)
    if asset["kind"] == "skill":
        bundle = export_bundle(asset)
        metadata = {
            **{key: value for key, value in asset["metadata"].items() if key != "resources"},
            "resource_paths": list(bundle["resources"]),
        }
        asset = {**asset, "metadata": metadata}
    else:
        validate_metadata(asset["metadata"])
    if not tools_allowed(asset["metadata"], allowed_tools):
        return None
    return asset


async def recall(store, principal, query, allowed_tools=None):
    dialect = store.engine.dialect
    if dialect.name == "sqlite":
        version = dialect.server_version_info
        if not version or version < (3, 38, 0):
            raise HarnessError(
                503, "当前 SQLite 版本未知或不支持召回，请升级至 3.38.0 及以上，或使用 PostgreSQL"
            )
    allowed = None if allowed_tools is None else set(allowed_tools)
    query_terms = terms(query)
    async with store.sessions() as session:
        rows = (await session.execute(projection_query(principal))).mappings().all()
    shortlist = coarse_candidates(rows, query_terms, allowed)
    if not shortlist:
        return []
    versions = or_(
        *(
            and_(Asset.id == identifier, Asset.version == version)
            for _, identifier, version in shortlist
        )
    )
    async with store.sessions() as session:
        rows = (
            await session.scalars(
                select(Asset).where(*eligible(principal), versions).limit(BODY_LIMIT)
            )
        ).all()
    coarse_scores = {identifier: score for score, identifier, _ in shortlist}
    scored = []
    for row in rows:
        try:
            asset = validated_asset(row, allowed)
        except HarnessError:
            # 旧记录校验失败即隔离，不通过回填突破正文预算。
            continue
        if asset is None:
            continue
        score = coarse_scores[asset["id"]] * 3 + len(query_terms & terms(asset["content"][:3000]))
        score += 2 if asset["metadata"].get("user_verified") else 0
        score += 100 if asset["kind"] in PRIORITY_KINDS else 0
        scored.append((score, asset))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    return [asset for _, asset in scored[:RESULT_LIMIT]]
