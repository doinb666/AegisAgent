"""Harness 独立配置：轻量个人版与企业持久化服务共用。"""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    database_url: str = ""
    auto_create_schema: bool = True
    registration_enabled: bool = True
    token_ttl_seconds: int = Field(default=28800, ge=60)
    max_concurrent_runs: int = Field(default=4, ge=1, le=64)
    max_child_runs: int = Field(default=2, ge=0, le=16)
    max_model_calls: int = Field(default=6, ge=1, le=64)
    max_user_runs: int = Field(default=4, ge=1, le=32)
    max_steps: int = Field(default=10, ge=1, le=100)
    run_timeout_seconds: int = Field(default=300, ge=5, le=3600)
    lease_seconds: int = Field(default=60, ge=10)
    model_timeout_seconds: int = Field(default=45, ge=1, le=300)
    context_chars: int = Field(default=24000, ge=4000, le=200000)
    tool_output_chars: int = Field(default=3000, ge=500, le=10000)
    max_upload_bytes: int = Field(default=10_000_000, ge=1024, le=10_000_000)
    models_json: str = "[]"
    mcp_servers_json: str = "[]"
    sandbox_url: str = ""
    sandbox_token: str = ""
    sandbox_image: str = "python:3.12-slim"
    sandbox_timeout_seconds: int = Field(default=30, ge=1, le=120)
    git_executable: str = "git"
    repository_root: Path | None = None
    risk_review_enabled: bool = True
    reflection_enabled: bool = True
    evolution_enabled: bool = True
    evolution_timeout_seconds: int = Field(default=30, ge=1, le=120)
    knowledge_backend: Literal["bm25", "hybrid"] = "bm25"
    knowledge_embedding_model: str = ""
    knowledge_embedding_device: str = ""
    knowledge_rerank_model: str = ""
    knowledge_milvus_host: str = ""
    knowledge_milvus_port: str = "19530"
    knowledge_milvus_user: str = ""
    knowledge_milvus_password: SecretStr = SecretStr("")
    knowledge_milvus_collection: str = "aegis_knowledge"
    knowledge_top_k: int = Field(default=5, ge=1, le=20)
    knowledge_max_documents: int = Field(default=64, ge=1, le=500)
    knowledge_max_chunks: int = Field(default=2048, ge=1, le=20000)
    knowledge_max_chars: int = Field(default=2_000_000, ge=1000, le=20_000_000)
    knowledge_timeout_seconds: float = Field(default=10, gt=0, le=120)

    @field_validator("repository_root", mode="before")
    @classmethod
    def optional_repository(cls, value):
        return None if value == "" else value

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return "sqlite+aiosqlite:///" + (self.data_dir.resolve() / "aegis.db").as_posix()
