"""显式能力契约和任务参数校验；禁止根据模型名称猜测协议能力。"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import ModelProvider


class ModelParameterCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    temperature: bool = Field(default=False, strict=True)
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] | None = None
    max_output_tokens: int = Field(default=4096, strict=True, ge=1, le=32768)
    reasoning_efforts: list[str] = Field(default_factory=list, max_length=8, strict=True)

    @field_validator("reasoning_efforts")
    @classmethod
    def bounded_efforts(cls, values):
        if any(not value.strip() or len(value) > 32 for value in values):
            raise ValueError("推理级别须为最多32字符的非空字符串")
        if len(values) != len(set(values)):
            raise ValueError("推理级别不能重复")
        return values

    def validate_provider(self, provider):
        if provider == ModelProvider.ANTHROPIC and (
            self.reasoning_efforts or self.output_token_parameter == "max_completion_tokens"
        ):
            raise ValueError("Anthropic 不支持声明推理级别或 max_completion_tokens")


def normalize_model_parameters(value):
    """先校验请求形状，空配置保持旧幂等请求完全一致。"""
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - {
        "temperature",
        "max_output_tokens",
        "reasoning_effort",
    }:
        raise ValueError("任务模型参数仅允许温度、输出上限和推理级别")
    if "temperature" in value:
        temperature = value["temperature"]
        if (
            type(temperature) not in {int, float}
            or not 0 <= temperature <= 2
            or not math.isfinite(temperature)
        ):
            raise ValueError("温度须为0至2的有限数值")
    if "max_output_tokens" in value:
        limit = value["max_output_tokens"]
        if type(limit) is not int or not 1 <= limit <= 32768:
            raise ValueError("输出上限须为1至32768的整数")
    if "reasoning_effort" in value:
        effort = value["reasoning_effort"]
        if not isinstance(effort, str) or not effort.strip() or len(effort) > 32:
            raise ValueError("推理级别须为最多32字符的非空字符串")
    parameters = dict(value)
    if "temperature" in parameters:
        parameters["temperature"] = float(parameters["temperature"])
    return parameters


def validate_model_parameters(value, capabilities):
    """校验单路能力，也用于确定性验收替身，返回安全规范字段。"""
    parameters = normalize_model_parameters(value)
    contract = ModelParameterCapabilities.model_validate(capabilities or {})
    if "temperature" in parameters and not contract.temperature:
        raise ValueError("所选模型不支持温度覆盖")
    if "max_output_tokens" in parameters and (
        not contract.output_token_parameter
        or parameters["max_output_tokens"] > contract.max_output_tokens
    ):
        raise ValueError("输出上限超出所选模型声明能力")
    if (
        "reasoning_effort" in parameters
        and parameters["reasoning_effort"] not in contract.reasoning_efforts
    ):
        raise ValueError("所选模型不支持该推理级别")
    return parameters


def validate_model_selection(router, preference, parameters):
    """新建前检查显式选择；自动选择允许至少一路完整支持。"""
    parameters = normalize_model_parameters(parameters)
    routes = getattr(router, "public_routes", lambda: [])()
    selected = (
        [route for route in routes if preference in {route["id"], route["model"]}]
        if preference
        else routes
    )
    if preference and not selected:
        raise ValueError("模型未配置，请从可用来源中选择")
    if not parameters:
        return
    for route in selected:
        try:
            validate_model_parameters(parameters, route.get("parameters"))
            return
        except ValueError:
            continue
    raise ValueError("所选模型没有支持这些任务参数的能力声明")


def common_model_parameters(routes):
    contracts = [
        ModelParameterCapabilities.model_validate(route.get("parameters") or {}) for route in routes
    ]
    efforts = set(contracts[0].reasoning_efforts) if contracts else set()
    for contract in contracts[1:]:
        efforts.intersection_update(contract.reasoning_efforts)
    return {
        "temperature": bool(contracts) and all(contract.temperature for contract in contracts),
        "output_tokens": bool(contracts)
        and all(contract.output_token_parameter for contract in contracts),
        "max_output_tokens": min(
            (contract.max_output_tokens for contract in contracts), default=4096
        ),
        "reasoning_efforts": sorted(efforts),
    }
