"""密码与不透明令牌；数据库不保存明文凭证。"""

import hashlib
import hmac
import json
import secrets


def password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310000)
    return salt + ":" + digest.hex()


def password_matches(password: str, stored: str) -> bool:
    salt, expected = stored.split(":", 1)
    actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 310000)
    return hmac.compare_digest(actual.hex(), expected)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
