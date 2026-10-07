"""有界组装兼容协议真实流；中间推理与工具参数不交给公开回调。"""

import json
import math

from .model_usage import safe_model_usage

MAX_TEXT_CHARS = 131072
MAX_TOOL_ARGUMENT_CHARS = 16384
MAX_TOTAL_ARGUMENT_CHARS = 131072
MAX_TOOLS = 16
MAX_CHUNKS = 65536


class StreamAssembly:
    def __init__(self):
        self.text = []
        self.text_chars = 0
        self.argument_chars = 0
        self.tools = {}
        self.chunks = 0
        self.finish_reason = None
        self.usage = None
        self.semantic_seen = False
        self.tools_announced = False

    async def consume(self, chunk, on_delta):
        self.chunks += 1
        if self.chunks > MAX_CHUNKS:
            raise ValueError("模型流片段数量超限")
        data = chunk if isinstance(chunk, dict) else chunk.model_dump()
        if data.get("usage") is not None:
            self.usage = safe_model_usage(data["usage"])
        choices = data.get("choices") or []
        if len(choices) > 1:
            raise ValueError("模型流只允许一个回答候选")
        for choice in choices:
            if type(choice.get("index")) is not int or choice["index"] != 0:
                raise ValueError("模型流回答索引无效")
            delta = choice.get("delta") or {}
            content = delta.get("content")
            tools = delta.get("tool_calls") or []
            if self.finish_reason and (content or tools or choice.get("finish_reason")):
                raise ValueError("模型流结束后继续返回语义片段")
            if content:
                if not isinstance(content, str):
                    raise ValueError("模型流仅支持文本输出")
                self.semantic_seen = True
                self.text_chars += len(content)
                if self.text_chars > MAX_TEXT_CHARS:
                    raise ValueError("模型流文本超限")
                self.text.append(content)
                await on_delta({"type": "content", "text": content})
            for tool in tools:
                # 出现工具结构即禁止换源，参数不完整也不能再拼接备用模型答案。
                self.semantic_seen = True
                if not self.tools_announced:
                    self.tools_announced = True
                    await on_delta({"type": "tools"})
                self._append_tool(tool)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    def _append_tool(self, delta):
        index = delta.get("index")
        if type(index) is not int or not 0 <= index < MAX_TOOLS:
            raise ValueError("模型流工具索引超限")
        if delta.get("type") not in {None, "function"}:
            raise ValueError("模型流工具类型无效")
        tool = self.tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
        function = delta.get("function") or {}
        for key, value in (
            ("id", delta.get("id")),
            ("name", function.get("name")),
            ("arguments", function.get("arguments")),
        ):
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError("模型流工具字段须为字符串")
            # 部分兼容服务重复完整标识，参数片段仍严格按顺序追加。
            if key != "arguments" and tool[key] == value:
                continue
            combined_length = len(tool[key]) + len(value)
            if key == "arguments":
                if (
                    combined_length > MAX_TOOL_ARGUMENT_CHARS
                    or self.argument_chars + len(value) > MAX_TOTAL_ARGUMENT_CHARS
                ):
                    raise ValueError("模型流工具参数超限")
                self.argument_chars += len(value)
            elif combined_length > 256:
                raise ValueError("模型流工具标识超限")
            tool[key] += value

    def complete(self):
        if self.finish_reason not in {"stop", "tool_calls"}:
            raise ValueError("模型流未合法结束，禁止执行不完整结果")
        if bool(self.tools) != (self.finish_reason == "tool_calls"):
            raise ValueError("模型流结束类型与工具结果不一致")
        calls, ids = [], set()
        for index in sorted(self.tools):
            tool = self.tools[index]
            if not tool["id"].strip() or not tool["name"].strip() or tool["id"] in ids:
                raise ValueError("模型流工具标识不完整或重复")
            ids.add(tool["id"])
            try:
                arguments = json.loads(
                    tool["arguments"],
                    parse_constant=self._invalid_constant,
                    parse_float=self._finite_float,
                )
            except (ValueError, TypeError, RecursionError):
                raise ValueError("模型流工具参数不是完整 JSON") from None
            if not isinstance(arguments, dict):
                raise ValueError("模型流工具参数须为 JSON 对象")
            calls.append(
                {
                    "id": tool["id"],
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": tool["arguments"]},
                }
            )
        content = "".join(self.text)
        if not content.strip() and not calls:
            raise ValueError("模型流未返回可用文本或工具调用")
        return content, {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content, "tool_calls": calls},
                    "finish_reason": self.finish_reason,
                }
            ]
        }

    @staticmethod
    def _invalid_constant(value):
        raise ValueError("JSON 参数不能包含非有限数值")

    @staticmethod
    def _finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON 参数不能包含非有限数值")
        return parsed
