"""版本资产与基于词法证据的有界召回。"""

import re

from sqlalchemy import select, update

from .errors import HarnessError
from .models import Asset, AssetVersion, User
from .security import canonical, digest
from .skill_files import SkillFileService, export_bundle, string_list, validate_metadata
from .store import asset_dict, scope, uid


def terms(text):
    words = set(re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fff]+", text.lower()))
    for word in list(words):
        if re.search(r"[\u4e00-\u9fff]", word):
            words.update(word[index : index + 2] for index in range(len(word) - 1))
    return words


class AssetService(SkillFileService):
    def __init__(self, store):
        self.store = store

    def snapshot(self, session, asset):
        session.add(
            AssetVersion(
                tenant_id=asset.tenant_id,
                owner_id=asset.owner_id,
                asset_id=asset.id,
                version=asset.version,
                snapshot=asset_dict(asset),
            )
        )

    async def list_assets(self, principal, kind=None, status=None):
        async with self.store.sessions() as session:
            query = select(Asset).where(*scope(Asset, principal))
            if kind:
                query = query.where(Asset.kind == kind)
            if status:
                query = query.where(Asset.status == status)
            assets = (await session.scalars(query.order_by(Asset.name).limit(500))).all()
            return [asset_dict(asset) for asset in assets]

    async def get_asset(self, principal, asset_id):
        async with self.store.sessions() as session:
            return asset_dict(await self.store.owned(session, Asset, asset_id, principal))

    async def put_uploaded_document(self, principal, name, content, raw_hash, request_key=None):
        """用户行锁保护上传去重；索引可重试，文档事实源只创建一次。"""
        if principal.role == "viewer":
            raise HarnessError(403, "viewer无权上传文档")
        payload_hash = digest(canonical({"name": name, "raw_hash": raw_hash}))
        request_key = request_key or payload_hash
        if not request_key or len(request_key) > 256:
            raise HarnessError(422, "上传幂等键最多256字符")
        async with self.store.write_lock, self.store.sessions.begin() as session:
            await session.execute(
                update(User)
                .where(
                    User.id == principal.user_id,
                    User.tenant_id == principal.tenant_id,
                )
                .values(role=User.role)
            )
            existing = await session.scalar(
                select(Asset).where(
                    *scope(Asset, principal),
                    Asset.kind == "document",
                    Asset.attributes["upload_key"].as_string() == request_key,
                )
            )
            if existing is not None:
                if existing.attributes.get("upload_request_hash") != payload_hash:
                    raise HarnessError(409, "上传幂等键已用于不同文档")
                return asset_dict(existing), True
            document = Asset(
                id=uid(),
                tenant_id=principal.tenant_id,
                owner_id=principal.user_id,
                kind="document",
                name=name,
                content=content,
                status="active",
                version=1,
                attributes={"upload_key": request_key, "upload_request_hash": payload_hash},
            )
            session.add(document)
            self.snapshot(session, document)
            return asset_dict(document), False

    async def put_asset(
        self,
        principal,
        kind,
        name,
        content,
        metadata=None,
        status="draft",
        asset_id=None,
        _internal=False,
        expected_version=None,
    ):
        if _internal and kind != "artifact":
            raise HarnessError(403, "内部写入入口仅用于工具证据")
        if principal.role == "viewer" and not _internal:
            raise HarnessError(403, "viewer无权修改资产")
        metadata = dict(validate_metadata({} if metadata is None else metadata))
        if "tools" in metadata:
            metadata["tools"] = string_list(metadata["tools"], "tools", 128)
        reserved = {
            "source_run_id",
            "tool_evidence",
            "manual_verified",
            "user_verified",
            "verification_source",
            "failure_reason",
            "call_id",
            "evolution_fingerprint",
            "extracted",
            "repair_verified",
            "edited",
            "upload_key",
            "upload_request_hash",
        }
        if not _internal and not asset_id and reserved.intersection(metadata or {}):
            raise HarnessError(422, "来源与验证元信息只能由服务端生成")
        if kind not in {
            "memory",
            "skill",
            "episodic",
            "preference",
            "constraint",
            "artifact",
            "profile",
            "procedure",
            "document",
            "project",
        }:
            raise HarnessError(422, "资产类型无效")
        if status not in {"draft", "active", "retired"} or not name.strip():
            raise HarnessError(422, "资产状态或名称无效")
        if len(name) > 256 or not isinstance(content, str):
            raise HarnessError(422, "资产名称过长或内容类型无效")
        if (
            kind == "document"
            and len(content.encode("utf-8")) > self.store.settings.max_upload_bytes
        ):
            raise HarnessError(413, "文档超出上传字节上限")
        async with self.store.write_lock, self.store.sessions.begin() as session:
            await session.execute(
                update(User)
                .where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
                .values(role=User.role)
            )
            if asset_id:
                asset = await self.store.owned(session, Asset, asset_id, principal)
                if expected_version is not None and asset.version != expected_version:
                    raise HarnessError(409, "资产版本已改变，请重新读取后重试")
                old_metadata = dict(validate_metadata(asset.attributes))
                if not _internal and any(
                    key not in old_metadata or metadata[key] != old_metadata[key]
                    for key in reserved.intersection(metadata or {})
                ):
                    raise HarnessError(422, "来源与验证元信息不能由客户端修改")
                metadata = {
                    **(metadata or {}),
                    **{
                        key: old_metadata[key]
                        for key in ("upload_key", "upload_request_hash")
                        if key in old_metadata
                    },
                }
                asset.version += 1
                if old_metadata.get("source_run_id") and not _internal:
                    old_tools = string_list(old_metadata.get("tools", []), "来源 tools", 128)
                    required_tools = string_list(
                        list(dict.fromkeys([*old_tools, *metadata.get("tools", [])])), "tools", 128
                    )
                    metadata = {
                        **(metadata or {}),
                        **{
                            key: value
                            for key, value in old_metadata.items()
                            if key in reserved | {"tools", "trigger_conditions", "failure_step"}
                        },
                        "edited": True,
                        "manual_verified": False,
                        "user_verified": False,
                        "repair_verified": False,
                        "tools": required_tools,
                    }
                    status = "draft"
            else:
                asset = Asset(
                    id=uid(), tenant_id=principal.tenant_id, owner_id=principal.user_id, version=1
                )
                session.add(asset)
            if kind == "skill":
                export_bundle(
                    {"id": asset.id, "name": name, "content": content, "metadata": metadata}
                )
            else:
                validate_metadata(metadata)
            asset.kind, asset.name, asset.content = kind, name, content
            asset.status, asset.attributes = status, metadata or {}
            self.snapshot(session, asset)
            return asset_dict(asset)

    async def put_artifact(self, principal, name, content, source_run_id, call_id, tool=None):
        """仅由执行器调用；HTTP不提供此受控证据写入接口。"""
        async with self.store.sessions() as session:
            from .models import Run

            await self.store.owned(session, Run, source_run_id, principal)
        return await self.put_asset(
            principal,
            "artifact",
            name,
            content,
            {"source_run_id": source_run_id, "call_id": call_id, "tool": tool},
            "active",
            _internal=True,
        )

    async def transition_asset(self, principal, asset_id, status):
        if principal.role == "viewer":
            raise HarnessError(403, "viewer无权修改资产状态")
        if status not in {"draft", "active", "retired"}:
            raise HarnessError(422, "资产状态无效")
        async with self.store.write_lock, self.store.sessions.begin() as session:
            await session.execute(
                update(User)
                .where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
                .values(role=User.role)
            )
            asset = await self.store.owned(session, Asset, asset_id, principal)
            asset.status, asset.version = status, asset.version + 1
            self.snapshot(session, asset)
            return asset_dict(asset)

    async def delete_asset(self, principal, asset_id):
        await self.transition_asset(principal, asset_id, "retired")

    async def restore_asset(self, principal, asset_id, version):
        if principal.role == "viewer":
            raise HarnessError(403, "viewer无权恢复资产版本")
        async with self.store.write_lock, self.store.sessions.begin() as session:
            await session.execute(
                update(User)
                .where(User.id == principal.user_id, User.tenant_id == principal.tenant_id)
                .values(role=User.role)
            )
            asset = await self.store.owned(session, Asset, asset_id, principal)
            historical = await session.scalar(
                select(AssetVersion).where(
                    *scope(AssetVersion, principal),
                    AssetVersion.asset_id == asset_id,
                    AssetVersion.version == version,
                )
            )
            if historical is None:
                raise HarnessError(404, "历史版本不存在或无权访问")
            snapshot = historical.snapshot
            for field in ("kind", "name", "content", "status"):
                setattr(asset, field, snapshot[field])
            asset.attributes = dict(snapshot.get("metadata", {}))
            asset.version += 1
            self.snapshot(session, asset)
            return asset_dict(asset)

    async def asset_history(self, principal, asset_id):
        async with self.store.sessions() as session:
            await self.store.owned(session, Asset, asset_id, principal)
            versions = await session.scalars(
                select(AssetVersion)
                .where(AssetVersion.asset_id == asset_id, *scope(AssetVersion, principal))
                .order_by(AssetVersion.version)
            )
            return [version.snapshot for version in versions]

    async def recall(self, principal, query, allowed_tools=None):
        async with self.store.sessions() as session:
            rows = (
                await session.scalars(
                    select(Asset)
                    .where(
                        *scope(Asset, principal),
                        Asset.status == "active",
                        Asset.kind.not_in(("document", "artifact", "project")),
                    )
                    .order_by(Asset.name, Asset.id)
                    .limit(500)
                )
            ).all()
            candidates = [asset_dict(row) for row in rows]
        query_terms = terms(query)
        shortlist = []
        for asset in candidates:
            try:
                if asset["kind"] == "skill":
                    bundle = export_bundle(asset)
                    metadata = {
                        **{
                            key: value
                            for key, value in asset["metadata"].items()
                            if key != "resources"
                        },
                        "resource_paths": list(bundle["resources"]),
                    }
                    asset = {**asset, "metadata": metadata}
                else:
                    metadata = validate_metadata(asset["metadata"])
                required_tools = set(string_list(metadata.get("tools", []), "tools", 128))
            except HarnessError:
                # 隔离旧记录与通用接口产生的异常元信息，避免挤掉其他有效候选。
                continue
            if allowed_tools is not None and not required_tools.issubset(set(allowed_tools)):
                continue
            descriptor = terms(
                asset["name"]
                + " "
                + str(metadata.get("tags", []))
                + " "
                + str(metadata.get("examples", []))
                + " "
                + str(metadata.get("boundaries", []))
                + " "
                + str(metadata.get("trigger_conditions", {}))
                + " "
                + str(metadata.get("description", ""))
                + " "
                + str(metadata.get("directory", ""))
            )
            score = len(query_terms & descriptor)
            if score or asset["kind"] in {"preference", "constraint", "profile"}:
                shortlist.append((score, asset))
        # 第一阶段只按目录元信息收敛候选；第二阶段才读取正文做有界词法精排。
        shortlist.sort(key=lambda item: (-item[0], item[1]["id"]))
        scored = []
        for coarse_score, asset in shortlist[:32]:
            score = coarse_score * 3 + len(query_terms & terms(asset["content"][:3000]))
            score += 2 if asset["metadata"].get("user_verified") else 0
            score += 100 if asset["kind"] in {"preference", "constraint", "profile"} else 0
            scored.append((score, asset))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        return [asset for _, asset in scored[:8]]
