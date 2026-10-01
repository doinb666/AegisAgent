"""API 请求与响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """单条对话消息。"""

    role: str = Field(description="角色：system | user | assistant")
    content: str = Field(description="文本内容")


class ChatRequest(BaseModel):
    """对话请求。"""

    messages: list[ChatMessage] = Field(min_length=1, description="OpenAI 风格消息列表")
    model: str | None = Field(default=None, description="优先使用的模型 ID")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    conversation_id: str | None = Field(default=None, description="可选会话 ID")


class ChatResponse(BaseModel):
    """非流式对话响应。"""

    id: str = Field(description="响应 ID")
    model: str = Field(description="实际使用的模型")
    content: str = Field(description="助手回复正文")
    trace_id: str | None = Field(default=None, description="链路追踪 ID")
    usage: dict[str, Any] | None = Field(default=None, description="Token 用量")


class DocumentUploadResponse(BaseModel):
    """文档上传响应。"""

    document_id: str
    filename: str
    status: str
    chunk_count: int = 0
    message: str = "ok"


class DocumentInfo(BaseModel):
    """文档列表项。"""

    id: str
    filename: str
    mime_type: str | None = None
    status: str
    created_at: str | None = None


class RetrievalResult(BaseModel):
    """检索单条结果。"""

    id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "vector"
