# -*- coding: utf-8 -*-
"""应用级枚举定义（Agent、检索、消息与任务状态）。"""

from enum import Enum


class AgentMode(str, Enum):
    """Agent 推理模式。"""

    REACT = "react"
    PLAN_EXECUTE = "plan_execute"


class RetrievalMode(str, Enum):
    """检索模式：向量 / 关键词 / 混合。"""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class MessageRole(str, Enum):
    """对话消息角色。"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TaskStatus(str, Enum):
    """子任务或异步任务状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
