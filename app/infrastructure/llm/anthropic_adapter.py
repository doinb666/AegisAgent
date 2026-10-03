"""将原生 Messages 协议转换为内核统一的文本与工具调用结构。"""

import json

import httpx


class AnthropicClient:
    def __init__(self, api_key, base_url, timeout):
        self.client = httpx.AsyncClient(
            base_url=(base_url or "https://api.anthropic.com/v1").rstrip("/") + "/",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=timeout,
            follow_redirects=False,
        )

    @staticmethod
    def messages(source):
        system, messages = [], []
        for message in source:
            role, content = message["role"], message.get("content") or ""
            if role in {"system", "developer"}:
                system.append(content)
                continue
            if role == "tool":
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["tool_call_id"],
                        "content": content,
                    }
                ]
                role = "user"
            else:
                if role not in {"user", "assistant"} or not isinstance(content, str):
                    raise ValueError("当前原生消息适配只支持文本任务")
                blocks = [{"type": "text", "text": content}] if content else []
                for call in message.get("tool_calls", []):
                    function = call["function"]
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call["id"],
                            "name": function["name"],
                            "input": json.loads(function["arguments"]),
                        }
                    )
            if not blocks:
                continue
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})
        return "\n\n".join(system), messages

    async def create(self, params):
        system, messages = self.messages(params["messages"])
        body = {
            "model": params["model"],
            "messages": messages,
            "max_tokens": params.get("max_tokens", 4096),
        }
        if system:
            body["system"] = system
        for name in ("temperature", "top_p"):
            if name in params:
                body[name] = params[name]
        if "stop" in params:
            stop = params["stop"]
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if params.get("tools"):
            body["tools"] = [
                {
                    "name": item["function"]["name"],
                    "description": item["function"].get("description", ""),
                    "input_schema": item["function"]["parameters"],
                }
                for item in params["tools"]
            ]
            choice = params.get("tool_choice", "auto")
            if isinstance(choice, dict):
                body["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
            else:
                body["tool_choice"] = {"type": "any" if choice == "required" else choice}
        response = await self.client.post("messages", json=body)
        response.raise_for_status()
        data = response.json()
        content, calls = [], []
        for block in data["content"]:
            if block["type"] == "text":
                content.append(block["text"])
            elif block["type"] == "tool_use":
                calls.append(
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block["input"], ensure_ascii=False),
                        },
                    }
                )
            else:
                raise ValueError("原生模型返回尚未支持的内容类型")
        if not "\n".join(content).strip() and not calls:
            raise ValueError("原生模型未返回可用文本或工具调用")
        usage = data.get("usage", {})
        cached = usage.get("cache_read_input_tokens", 0)
        prompt = usage.get("input_tokens", 0) + cached + usage.get("cache_creation_input_tokens", 0)
        output = usage.get("output_tokens", 0)
        return {
            "choices": [{"message": {"content": "\n".join(content), "tool_calls": calls}}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": output,
                "total_tokens": prompt + output,
                "cached_tokens": cached,
            },
        }

    async def close(self):
        await self.client.aclose()
