# -*- coding: utf-8 -*-
"""LLM 相关类型定义。"""

from enum import Enum


class ModelProvider(str, Enum):
    """模型提供方枚举（用于扩展路由策略）。"""

    OPENAI = "openai"
    AZURE = "azure"
    CUSTOM = "custom"
