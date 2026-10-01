"""公开错误与身份契约。"""

from dataclasses import dataclass


class HarnessError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class Principal:
    user_id: str
    tenant_id: str
    role: str
