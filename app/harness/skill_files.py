"""受限 SKILL.md 与文本资源；只保存数据库，不访问或执行宿主文件。"""

import json
import re

import yaml
from yaml.events import AliasEvent

from .errors import HarnessError

MAX_DOCUMENT_BYTES = 32768
MAX_BODY_CHARS = 16000
MAX_RESOURCE_BYTES = 16000
MAX_TOTAL_RESOURCE_BYTES = 65536
MAX_METADATA_BYTES = 512 * 1024
MAX_SKILL_METADATA_BYTES = 8192
FRONTMATTER_FIELDS = {"name", "description", "tags", "allowed-tools"}
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
YAML_TAGS = {"tag:yaml.org,2002:str", "tag:yaml.org,2002:seq", "tag:yaml.org,2002:map"}


def invalid(detail: str) -> None:
    raise HarnessError(422, detail)


def text_bytes(text: str, limit: int, label: str) -> int:
    if not isinstance(text, str) or "\x00" in text:
        invalid(f"{label}必须是无 NUL 的文本")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise HarnessError(422, f"{label}必须是有效 UTF-8 文本") from exc
    if size > limit:
        invalid(f"{label}超出字节上限")
    return size


def relative_path(path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(path, str):
        invalid("目录或资源路径必须是文本")
    if allow_empty and path == "":
        return path
    if not path or len(path) > 128 or "\\" in path or ":" in path:
        invalid("目录或资源路径须为最长128字符的相对路径")
    parts = path.split("/")
    if len(parts) > 4 or any(
        not part or part.startswith(".") or part != part.strip() for part in parts
    ):
        invalid("目录或资源路径最多4层，禁止隐藏项和路径跳转")
    if any(ord(char) < 32 or ord(char) == 127 for char in path):
        invalid("目录或资源路径不能包含控制字符")
    return path


class SkillLoader(yaml.SafeLoader):
    """在 YAML 构造前限制节点、深度、标签、锚与别名。"""

    def __init__(self, stream):
        super().__init__(stream)
        self.node_count = 0
        self.node_depth = 0

    def compose_node(self, parent, index):
        event = self.peek_event()
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            invalid("frontmatter 不允许 YAML 锚或别名")
        if event.tag is not None and event.tag not in YAML_TAGS:
            invalid("frontmatter 不允许非基础 YAML 标签")
        self.node_count += 1
        self.node_depth += 1
        if self.node_count > 160 or self.node_depth > 3:
            invalid("frontmatter 节点过多或层级过深")
        try:
            node = super().compose_node(parent, index)
            if node.tag not in YAML_TAGS:
                invalid("frontmatter 仅支持文本和文本列表")
            return node
        finally:
            self.node_depth -= 1

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
                invalid("frontmatter 字段名必须是文本")
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                invalid("frontmatter 不允许重复字段")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def string_list(value, label: str, item_limit: int) -> list[str]:
    if not isinstance(value, list) or len(value) > 32:
        invalid(f"{label}须为最多32项的文本列表")
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > item_limit:
            invalid(f"{label}列表项无效或过长")
        text_bytes(item, item_limit * 4, label)
    return list(dict.fromkeys(value))


def metadata_bytes(value, limit: int, label: str) -> int:
    serialized = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return text_bytes(serialized, limit, label)


def validate_metadata(metadata: dict, *, skill: bool = False) -> dict:
    if not isinstance(metadata, dict):
        invalid("资产元数据必须是 JSON 对象")
    try:
        metadata_bytes(metadata, MAX_METADATA_BYTES, "资产元数据 JSON")
        if skill:
            non_resource = {key: value for key, value in metadata.items() if key != "resources"}
            metadata_bytes(non_resource, MAX_SKILL_METADATA_BYTES, "Skill 非资源元数据 JSON")
    except (TypeError, ValueError, RecursionError) as exc:
        raise HarnessError(422, "资产元数据必须可序列化为有界 JSON") from exc
    return metadata


def validate_frontmatter(data: dict) -> dict:
    if not isinstance(data, dict) or set(data) - FRONTMATTER_FIELDS:
        invalid("frontmatter 仅支持 name、description、tags、allowed-tools")
    name, description = data.get("name"), data.get("description")
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or not NAME_PATTERN.fullmatch(name):
        invalid("Skill name须为1至64位小写字母数字与连字符，不以连字符开头或结尾")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        invalid("Skill description须为非空且最长1024字符的文本")
    text_bytes(description, 4096, "description")
    tools = data.get("allowed-tools", [])
    # 兼容生态中的空格分隔工具名；仅声明依赖，不增加任何执行权限。
    if isinstance(tools, str):
        if len(tools) > 4096:
            invalid("allowed-tools 字符串过长")
        tools = tools.split()
    return {
        "name": name,
        "description": description,
        "tags": string_list(data.get("tags", []), "tags", 64),
        "allowed-tools": string_list(tools, "allowed-tools", 128),
    }


def parse_document(document: str) -> tuple[dict, str]:
    text_bytes(document, MAX_DOCUMENT_BYTES, "SKILL.md")
    match = re.match(r"\A---\r?\n(.*?)^---[ \t]*(?:\r?\n|\Z)", document, re.MULTILINE | re.DOTALL)
    if match is None:
        invalid("SKILL.md 必须以 YAML frontmatter 开始")
    try:
        frontmatter = yaml.load(match.group(1), Loader=SkillLoader)
    except yaml.YAMLError as exc:
        raise HarnessError(422, "SKILL.md frontmatter 格式无效") from exc
    body = document[match.end() :]
    if len(body) > MAX_BODY_CHARS:
        invalid("Skill 正文最多16000字符")
    return validate_frontmatter(frontmatter), body


def validate_resources(resources: dict) -> dict[str, str]:
    if not isinstance(resources, dict) or len(resources) > 16:
        invalid("Skill 最多附带16个文本资源")
    total = 0
    paths = set()
    for path, content in resources.items():
        relative_path(path)
        folded = path.casefold()
        if path.split("/")[-1].casefold() == "skill.md" or folded in paths:
            invalid("资源不能覆盖 SKILL.md 或使用大小写重名路径")
        paths.add(folded)
        total += text_bytes(content, MAX_RESOURCE_BYTES, "资源")
        if total > MAX_TOTAL_RESOURCE_BYTES:
            invalid("Skill 文本资源总量最多64KiB")
    return dict(resources)


def serialize_document(frontmatter: dict, body: str) -> str:
    if not isinstance(body, str) or len(body) > MAX_BODY_CHARS:
        invalid("Skill 正文最多16000字符")
    document = (
        "---\n" + yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False) + "---\n" + body
    )
    text_bytes(document, MAX_DOCUMENT_BYTES, "SKILL.md")
    return document


def export_bundle(asset: dict) -> dict:
    """每次读取再次验证可被通用资产 API 修改的元数据。"""
    metadata = validate_metadata(asset["metadata"], skill=True)
    name = asset["name"]
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or not NAME_PATTERN.fullmatch(name):
        # 旧版生成资产允许中文名称；导出以资产 ID 生成稳定的生态兼容名称。
        identifier = re.sub(r"[^a-z0-9]", "", str(asset["id"]).lower())[:58] or "legacy"
        name = "skill-" + identifier
    frontmatter = validate_frontmatter(
        {
            "name": name,
            "description": metadata.get("description", asset["name"]),
            "tags": metadata.get("tags", []),
            "allowed-tools": metadata.get("tools", []),
        }
    )
    return {
        "document": serialize_document(frontmatter, asset["content"]),
        "directory": relative_path(metadata.get("directory", ""), allow_empty=True),
        "resources": validate_resources(metadata.get("resources", {})),
    }


class SkillFileService:
    async def import_skill(
        self,
        principal,
        document,
        directory="",
        resources=None,
        asset_id=None,
        expected_version=None,
    ):
        if principal.role == "viewer":
            raise HarnessError(403, "viewer无权导入或修改 Skill")
        if (asset_id is None) != (expected_version is None):
            invalid("更新 Skill 必须同时提供 asset_id 与 expected_version")
        if asset_id is not None and (
            not isinstance(asset_id, str) or not asset_id or len(asset_id) > 128
        ):
            invalid("asset_id须为非空且最长128字符的文本")
        if expected_version is not None and (
            not isinstance(expected_version, int)
            or isinstance(expected_version, bool)
            or expected_version < 1
        ):
            invalid("expected_version须为正整数")
        frontmatter, body = parse_document(document)
        metadata = {
            "description": frontmatter["description"],
            "tags": frontmatter["tags"],
            "tools": frontmatter["allowed-tools"],
            "directory": relative_path(directory, allow_empty=True),
            "resources": validate_resources(resources if resources is not None else {}),
        }
        if asset_id is not None:
            existing = await self.get_asset(principal, asset_id)
            if existing["kind"] != "skill":
                invalid("只能通过 Skill 导入更新 skill 类型资产")
            if existing["metadata"].get("source_run_id"):
                old_tools = string_list(existing["metadata"].get("tools", []), "tools", 128)
                metadata["tools"] = string_list(
                    list(dict.fromkeys([*old_tools, *metadata["tools"]])), "tools", 128
                )
        # 输入与规范化导出均需有界，保证成功导入后可再次往返。
        serialize_document({**frontmatter, "allowed-tools": metadata["tools"]}, body)
        return await self.put_asset(
            principal,
            "skill",
            frontmatter["name"],
            body,
            metadata,
            status="draft",
            asset_id=asset_id,
            expected_version=expected_version,
        )

    async def export_skill(self, principal, asset_id):
        asset = await self.get_asset(principal, asset_id)
        if asset["kind"] != "skill":
            raise HarnessError(404, "Skill 不存在或无权访问")
        return export_bundle(asset)

    async def read_skill(self, principal, asset_id, path=None, *, active_only=True):
        asset = await self.get_asset(principal, asset_id)
        if asset["kind"] != "skill" or active_only and asset["status"] != "active":
            raise HarnessError(404, "可用 Skill 不存在或无权访问")
        bundle = export_bundle(asset)
        if path is None:
            content = asset["content"]
        else:
            relative_path(path)
            if path not in bundle["resources"]:
                raise HarnessError(404, "Skill 资源不存在")
            content = bundle["resources"][path]
        return {"id": asset["id"], "name": asset["name"], "path": path, "content": content}
