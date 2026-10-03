"""多模型路由器：优先级调度、加权选择与自动降级。"""

from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel

from app.infrastructure.llm.circuit_breaker import CircuitBreaker
from app.infrastructure.llm.types import ModelProvider

from .anthropic_adapter import AnthropicClient
from .model_config import parse_entries


class LLMResponse(BaseModel):
    """统一 LLM 响应结构。"""

    content: str = ""
    model_id: str = ""
    usage: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ModelConfig:
    """单路模型配置。"""

    model_id: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    provider: ModelProvider = ModelProvider.OPENAI
    priority: int = 0
    weight: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    api_version: str | None = None


class ModelRouter:
    """多模型路由器：支持优先级调度、负载均衡（同优先级加权随机）、自动降级。"""

    def __init__(
        self,
        model_configs: list[ModelConfig],
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        timeout: float = 45.0,
        max_concurrent_calls: int = 8,
    ) -> None:
        if not model_configs:
            raise ValueError("model_configs 不能为空")

        self._configs = sorted(model_configs, key=lambda c: c.priority)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent_calls))
        self._breakers: dict[str, CircuitBreaker] = {}
        for cfg in self._configs:
            route = self._route_key(cfg)
            self._breakers[route] = CircuitBreaker(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                name=f"llm:{route}",
            )
        self._clients: dict[str, Any] = {}
        for cfg in self._configs:
            kwargs: dict[str, Any] = {"api_key": cfg.api_key, "timeout": timeout, "max_retries": 0}
            if cfg.provider == ModelProvider.ANTHROPIC:
                client = AnthropicClient(cfg.api_key, cfg.base_url, timeout)
            elif cfg.provider == ModelProvider.AZURE:
                client = AsyncAzureOpenAI(
                    **kwargs, azure_endpoint=cfg.base_url, api_version=cfg.api_version
                )
            else:
                if cfg.base_url:
                    kwargs["base_url"] = cfg.base_url
                client = AsyncOpenAI(**kwargs)
            self._clients[self._route_key(cfg)] = client

    @staticmethod
    def _route_key(cfg: ModelConfig) -> str:
        """同名模型在不同端点具有独立连接与熔断状态。"""
        endpoint = hashlib.sha256((cfg.base_url or "openai").encode()).hexdigest()[:12]
        return f"{cfg.provider.value}:{endpoint}:{cfg.model_id}"

    def public_routes(self) -> list[dict]:
        """仅公开选择所需元信息，隐藏地址、凭证及自定义正文。"""
        return [
            {
                "id": self._public_id(cfg),
                "model": cfg.model_id,
                "label": cfg.label or f"{cfg.provider.value} · {cfg.model_id}",
                "provider": cfg.provider.value,
                "priority": cfg.priority,
            }
            for cfg in self._configs
        ]

    def _public_id(self, cfg: ModelConfig) -> str:
        return "route-" + hashlib.sha256(self._route_key(cfg).encode()).hexdigest()[:32]

    def _select_candidates(
        self,
        model_preference: str | None,
    ) -> list[ModelConfig]:
        """按优先级分组，在同优先级内按权重随机排序，形成候选列表。"""
        if model_preference:
            exact = [c for c in self._configs if self._public_id(c) == model_preference]
            if not exact:
                exact = [c for c in self._configs if c.model_id == model_preference]
            if exact:
                return exact + [c for c in self._configs if c not in exact]

        by_prio: dict[int, list[ModelConfig]] = {}
        for cfg in self._configs:
            by_prio.setdefault(cfg.priority, []).append(cfg)

        ordered: list[ModelConfig] = []
        for prio in sorted(by_prio.keys()):
            group = by_prio[prio]
            # 同优先级内按权重做随机排序（权重越大越容易被排到前面）
            scored = [(cfg, random.random() ** (1.0 / max(cfg.weight, 0.01))) for cfg in group]
            scored.sort(key=lambda x: -x[1])
            ordered.extend(cfg for cfg, _ in scored)
        return ordered or list(self._configs)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model_preference: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """智能路由到合适模型；失败时按候选顺序自动降级。"""
        async with self._semaphore:
            return await self._chat(messages, model_preference, **kwargs)

    async def _chat(self, messages, model_preference=None, **kwargs) -> LLMResponse:
        candidates = self._select_candidates(model_preference)

        for cfg in candidates:
            breaker = self._breakers[self._route_key(cfg)]
            try:
                return await breaker.call(
                    self._try_model,
                    self._route_key(cfg),
                    messages,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning("模型 [{}] 调用失败，尝试降级: {}", cfg.model_id, type(exc).__name__)

        raise RuntimeError("所有候选模型均不可用；请检查模型配置或稍后重试") from None

    async def _try_model(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        """尝试调用指定模型（经熔断器包装，不在此处重复熔断逻辑）。"""
        client = self._clients[model_id]
        config = next(c for c in self._configs if self._route_key(c) == model_id)
        params = {
            "temperature": 0.7,
            **config.extra,
            **kwargs,
            "model": config.model_id,
            "messages": messages,
        }
        if config.provider == ModelProvider.ANTHROPIC:
            raw = await client.create(params)
            return LLMResponse(
                content=raw["choices"][0]["message"]["content"],
                model_id=config.model_id,
                usage=raw["usage"],
                raw=raw,
            )
        resp = await client.chat.completions.create(**params)

        choice = resp.choices[0] if resp.choices else None
        content = (choice.message.content or "") if choice else ""
        if not content.strip() and not (choice and getattr(choice.message, "tool_calls", None)):
            raise ValueError("模型未返回可用文本或工具调用")
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            details = getattr(resp.usage, "prompt_tokens_details", None)
            if details:
                usage["cached_tokens"] = getattr(details, "cached_tokens", 0)

        return LLMResponse(
            content=content,
            model_id=config.model_id,
            usage=usage,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
        )

    async def aclose(self) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()))


def build_harness_router(settings) -> ModelRouter | None:
    """密钥只从环境读取；兼容原有 OPENAI 配置。"""
    from app.config import get_settings

    entries = parse_entries(settings.models_json)
    configs = []
    seen = set()
    for item in entries:
        config = ModelConfig(
            model_id=item.model,
            api_key=item.credentials(),
            provider=item.provider,
            base_url=item.base_url
            or ("http://127.0.0.1:11434/v1" if item.provider == ModelProvider.OLLAMA else None),
            priority=item.priority,
            weight=item.weight,
            extra=item.extra,
            label=item.label,
            api_version=item.api_version,
        )
        route = ModelRouter._route_key(config)
        if route in seen:
            raise ValueError("模型端点配置重复")
        seen.add(route)
        configs.append(config)
    if not configs:
        legacy = get_settings()
        if legacy.openai_api_key:
            configs.append(
                ModelConfig(
                    model_id=legacy.openai_model,
                    api_key=legacy.openai_api_key,
                    base_url=legacy.openai_api_base,
                )
            )
    return (
        ModelRouter(
            configs,
            timeout=settings.model_timeout_seconds,
            max_concurrent_calls=settings.max_model_calls,
        )
        if configs
        else None
    )
