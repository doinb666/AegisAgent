"""跨完整响应、流组装及持久事件复用的有界用量契约。"""


def safe_model_usage(usage):
    """只保留四个标准字段及有界非负整数，避免供应商数据旁路。"""
    if not isinstance(usage, dict):
        return None
    result = {
        key: value
        for key, value in usage.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"}
        and type(value) is int
        and 0 <= value <= 10**12
    }
    details = usage.get("prompt_tokens_details")
    if "cached_tokens" not in result and isinstance(details, dict):
        cached = details.get("cached_tokens")
        if type(cached) is int and 0 <= cached <= 10**12:
            result["cached_tokens"] = cached
    return result
