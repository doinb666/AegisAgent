"""LLM 相关类型定义。"""

from enum import StrEnum


class ModelProvider(StrEnum):
    """模型提供方枚举（用于扩展路由策略）。"""

    OPENAI = "openai"
    AZURE = "azure"
    CUSTOM = "custom"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
