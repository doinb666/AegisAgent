"""旧接口与现有知识检索使用的 Pydantic 模型。"""

from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentUploadResponse,
    RetrievalResult,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "DocumentInfo",
    "DocumentUploadResponse",
    "RetrievalResult",
]
