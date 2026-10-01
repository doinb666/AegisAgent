"""知识分块、持久化恢复与租户/用户双隔离回归。"""

import asyncio

import pytest

from app.harness import HarnessError, HarnessService, HarnessSettings, Principal
from app.harness_tools.knowledge import KnowledgeService


async def setup_service(tmp_path, **options):
    settings = HarnessSettings(data_dir=tmp_path, database_url="", **options)
    service = HarnessService(settings)
    await service.store.initialize()
    user = await service.register("alice", "password123")
    principal = Principal(user["user_id"], user["tenant_id"], user["role"])
    return settings, service, principal


async def add_document(service, principal, name, chunks):
    return await service.put_asset(principal, "document", name, "\n".join(chunks), status="active")


@pytest.mark.asyncio
async def test_chunks_bm25_survive_service_restart(tmp_path):
    settings, service, principal = await setup_service(tmp_path)
    try:
        knowledge = KnowledgeService(settings, service)
        document = await add_document(
            service,
            principal,
            "运维手册",
            ["系统启动步骤与健康检查", "Milvus 索引支持知识检索与向量存储"],
        )
        await knowledge.index(
            principal, document, ["系统启动步骤与健康检查", "Milvus 索引支持知识检索与向量存储"]
        )
    finally:
        await service.close()
    restarted = HarnessService(settings)
    await restarted.store.initialize()
    try:
        result = await KnowledgeService(settings, restarted).search(principal, "Milvus 存储")
        assert result["backend"] == "bm25"
        assert len(result["results"]) == 1
        hit = result["results"][0]
        assert hit["document_id"] == document["id"]
        assert hit["chunk_index"] == 1
        assert hit["content"] == "Milvus 索引支持知识检索与向量存储"
        assert hit["score"] > 0
        assert not result["truncated"]
    finally:
        await restarted.close()


@pytest.mark.asyncio
async def test_scope_is_user_and_tenant_and_rejects_foreign_index(tmp_path):
    settings, service, owner = await setup_service(tmp_path)
    try:
        knowledge = KnowledgeService(settings, service)
        document = await add_document(service, owner, "私有手册", ["secretneedle"])
        await knowledge.index(owner, document, ["secretneedle"])
        member = await service.create_user(owner, "colleague", "password123", "operator")
        colleague = Principal(member["user_id"], member["tenant_id"], member["role"])
        foreign_tenant = Principal(owner.user_id, "foreign-tenant", owner.role)
        for outsider in (colleague, foreign_tenant):
            assert (await knowledge.search(outsider, "secretneedle"))["results"] == []
            with pytest.raises(HarnessError) as error:
                await knowledge.index(outsider, document, ["tampered"])
            assert error.value.status_code == 404
        assert (await knowledge.search(owner, "secretneedle"))["results"]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_retired_and_changed_documents_do_not_return_stale_chunks(tmp_path):
    settings, service, owner = await setup_service(tmp_path)
    try:
        knowledge = KnowledgeService(settings, service)
        document = await add_document(service, owner, "手册", ["oldneedle"])
        await knowledge.index(owner, document, ["oldneedle"])
        current = await service.get_asset(owner, document["id"])
        await service.put_asset(
            owner, "document", "手册", "newneedle", current["metadata"], "active", current["id"]
        )
        assert (await knowledge.search(owner, "oldneedle"))["results"] == []
        assert (await knowledge.search(owner, "newneedle"))["results"]
        await service.delete_asset(owner, document["id"])
        assert (await knowledge.search(owner, "newneedle"))["results"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_budgets_are_explicit_and_empty_query_is_safe(tmp_path):
    settings, service, owner = await setup_service(
        tmp_path, knowledge_max_documents=1, knowledge_max_chars=1000
    )
    try:
        knowledge = KnowledgeService(settings, service)
        for name in ("a", "b"):
            document = await add_document(service, owner, name, ["needle " * 300])
            await knowledge.index(owner, document, ["needle " * 300])
        result = await knowledge.search(owner, "needle")
        assert result["truncated"] and result["warnings"]
        assert all(len(hit["content"]) <= 1000 for hit in result["results"])
        assert (await knowledge.search(owner, " "))["results"] == []
        with pytest.raises(HarnessError):
            await knowledge.search(owner, "x" * 12001)
    finally:
        await service.close()


class FakeEmbedding:
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class ScopedVectors:
    def __init__(self):
        self.collections = {}
        self.searched = []

    async def create_collection(self, name, dim):
        self.collections.setdefault(name, [])

    async def insert(self, collection, vectors, metadata):
        self.collections[collection].extend(metadata)

    async def search(self, collection, query_vector, top_k=10):
        self.searched.append(collection)
        return [
            {"id": item["id"], "distance": 0.0}
            for item in self.collections.get(collection, [])[:top_k]
        ]


@pytest.mark.asyncio
async def test_vectors_use_private_collections_and_current_local_evidence(tmp_path):
    settings, service, owner = await setup_service(tmp_path, knowledge_backend="hybrid")
    try:
        member = await service.create_user(owner, "colleague", "password123", "operator")
        colleague = Principal(member["user_id"], member["tenant_id"], member["role"])
        vectors = ScopedVectors()
        knowledge = KnowledgeService(
            settings, service, vector_store=vectors, embedding_model=FakeEmbedding()
        )
        for principal, text in ((owner, "alice-only"), (colleague, "bob-only")):
            document = await add_document(service, principal, "手册", [text])
            await knowledge.index(principal, document, [text])
        assert len(vectors.collections) == 2
        result = await knowledge.search(owner, "semantic-query")
        assert result["backend"] == "hybrid"
        assert [hit["content"] for hit in result["results"]] == ["alice-only"]
        assert len(vectors.searched) == 1
        # 即使后端返回不属于当前范围的 ID，也只能关联本人的现有证据块。
        private_name = vectors.searched[0]
        foreign_items = next(
            items for name, items in vectors.collections.items() if name != private_name
        )
        vectors.collections[private_name] = foreign_items
        assert (await knowledge.search(owner, "semantic-query"))["results"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_vector_and_rerank_failures_fall_back_without_secret_details(tmp_path):
    settings, service, owner = await setup_service(tmp_path, knowledge_backend="hybrid")

    class FailedVectors(ScopedVectors):
        async def search(self, *args, **kwargs):
            raise RuntimeError("secret-password-must-not-appear")

    class FailedReranker:
        async def rerank(self, *args, **kwargs):
            raise RuntimeError("secret-password-must-not-appear")

    try:
        knowledge = KnowledgeService(
            settings,
            service,
            vector_store=FailedVectors(),
            embedding_model=FakeEmbedding(),
            reranker=FailedReranker(),
        )
        document = await add_document(service, owner, "手册", ["needle"])
        await knowledge.index(owner, document, ["needle"])
        result = await knowledge.search(owner, "needle")
        assert result["backend"] == "bm25"
        assert result["results"] and len(result["warnings"]) == 2
        assert "secret-password" not in str(result)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_hybrid_without_configuration_reports_downgrade_even_no_hits(tmp_path):
    settings, service, owner = await setup_service(tmp_path, knowledge_backend="hybrid")
    try:
        result = await KnowledgeService(settings, service).search(owner, "无匹配文档")
        assert result["backend"] == "bm25"
        assert result["warnings"] and result["results"] == []
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_keeps_keyword_evidence(tmp_path):
    settings, service, owner = await setup_service(
        tmp_path, knowledge_backend="hybrid", knowledge_timeout_seconds=0.01
    )

    class SlowVectors(ScopedVectors):
        async def search(self, *args, **kwargs):
            await asyncio.sleep(5)

    try:
        knowledge = KnowledgeService(
            settings, service, vector_store=SlowVectors(), embedding_model=FakeEmbedding()
        )
        document = await add_document(service, owner, "手册", ["needle"])
        await knowledge.index(owner, document, ["needle"])
        result = await asyncio.wait_for(knowledge.search(owner, "needle"), timeout=1)
        assert result["backend"] == "bm25" and result["results"]
        assert "超时" in "".join(result["warnings"])
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_configured_optional_backends_construct_without_network(tmp_path):
    settings, service, owner = await setup_service(
        tmp_path,
        knowledge_backend="hybrid",
        knowledge_embedding_model="local/embedding",
        knowledge_milvus_host="localhost",
        knowledge_milvus_password="never-print-me",
        knowledge_rerank_model="local/reranker",
    )
    try:
        knowledge = KnowledgeService(settings, service)
        assert knowledge.vector_enabled and knowledge.configuration_warning is None
        assert knowledge.embedding_model.model is None
        assert knowledge.vector_store._connected is False
        assert knowledge.reranker._model is None
        assert "never-print-me" not in repr(settings)
        assert knowledge.collection(owner).startswith("aegis_knowledge_")
    finally:
        await service.close()
