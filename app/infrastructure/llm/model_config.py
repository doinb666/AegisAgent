"""管理员模型配置校验；端点与凭证不进入浏览器展示契约。"""

import json
import os
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .model_parameters import ModelParameterCapabilities
from .types import ModelProvider


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str = Field(min_length=1, max_length=128)
    provider: ModelProvider = ModelProvider.OPENAI
    label: str = Field(default="", max_length=80)
    base_url: str | None = Field(default=None, max_length=2048)
    api_key_env: str = Field(default="", pattern=r"^(?:[A-Za-z_][A-Za-z0-9_]*)?$")
    api_version: str | None = Field(default=None, min_length=1, max_length=80)
    priority: int = Field(default=0, strict=True, ge=0, le=1000)
    weight: float = Field(default=1, gt=0, le=10000, allow_inf_nan=False, strict=True)
    extra: dict = Field(default_factory=dict)
    parameters: ModelParameterCapabilities = Field(default_factory=ModelParameterCapabilities)

    @field_validator("base_url")
    @classmethod
    def valid_endpoint(cls, value):
        if value is None:
            return value
        parsed = urlsplit(value)
        if parsed.port == 0:
            raise ValueError("端口须为 1 至 65535")
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("端点须为不含凭证、查询或片段的 HTTP(S) 基础地址")
        return value.rstrip("/")

    @field_validator("extra")
    @classmethod
    def safe_parameters(cls, value):
        allowed = {
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "extra_body",
        }
        reserved = {
            "model",
            "messages",
            "tools",
            "tool_choice",
            "stream",
            "model_parameters",
            "max_output_tokens",
        }
        if "max_tokens" in value and "max_completion_tokens" in value:
            raise ValueError("不能同时配置两个输出上限字段")
        if set(value) - allowed or len(json.dumps(value, allow_nan=False)) > 8000:
            raise ValueError("自定义参数不在允许范围内或过大")
        extra_body = value.get("extra_body", {})
        if not isinstance(extra_body, dict) or (reserved | allowed).intersection(extra_body):
            raise ValueError("自定义正文不能覆盖任务、模型、工具或流式协议")
        for name, item in value.items():
            if name in {"max_tokens", "max_completion_tokens", "seed"}:
                if type(item) is not int or (name != "seed" and not 1 <= item <= 1_000_000):
                    raise ValueError("输出预算及 seed 须为有效整数")
            elif name in {"temperature", "top_p", "frequency_penalty", "presence_penalty"}:
                if type(item) not in {int, float}:
                    raise ValueError("采样参数须为有限数值")
            elif name == "stop":
                stops = [item] if isinstance(item, str) else item
                if (
                    not isinstance(stops, list)
                    or not 1 <= len(stops) <= 4
                    or any(not isinstance(stop, str) or not 1 <= len(stop) <= 512 for stop in stops)
                ):
                    raise ValueError("stop 须为非空字符串或最多四项字符串列表")
        return value

    @model_validator(mode="after")
    def required_provider_settings(self):
        self.parameters.validate_provider(self.provider)
        if not self.model.strip():
            raise ValueError("模型名不能为空")
        if self.provider == ModelProvider.AZURE and (not self.base_url or not self.api_version):
            raise ValueError("Azure 须配置资源地址与 api_version")
        if self.provider == ModelProvider.ANTHROPIC and set(self.extra) - {
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
        }:
            raise ValueError("Anthropic 仅支持温度、输出预算、top_p 和 stop 参数")
        return self

    def credentials(self):
        key = os.environ.get(self.api_key_env, "") if self.api_key_env else ""
        if self.provider == ModelProvider.OLLAMA and not self.api_key_env:
            return "ollama-local"
        if not key:
            raise ValueError("模型 api_key_env 对应的环境变量未配置")
        return key


def parse_entries(raw):
    try:
        values = json.loads(raw)
        if not isinstance(values, list) or len(values) > 32:
            raise ValueError("模型配置须为最多 32 条记录的数组")
        return [ModelEntry.model_validate(value) for value in values]
    except (ValidationError, ValueError, TypeError):
        # 原始输入可能错误地含有密钥，不在异常或日志中回显。
        raise ValueError("AEGIS_MODELS_JSON 无效，请核对协议、端点、环境变量名及参数") from None
