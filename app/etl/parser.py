# -*- coding: utf-8 -*-
"""文档解析器：按 MIME 类型选择解析策略。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from loguru import logger


@dataclass
class ParsedDocument:
    """解析后的纯文本与元数据。"""

    text: str
    mime_type: str
    meta: dict[str, str]


class DocumentParser:
    """文档解析器：支持纯文本与 PDF（pypdf）。"""

    def __init__(self, max_chars: int = 500_000) -> None:
        self._max_chars = max_chars

    def parse_file(self, path: Path, mime_type: str | None = None) -> ParsedDocument:
        """从文件路径解析。"""
        mime = mime_type or "application/octet-stream"
        suffix = path.suffix.lower()
        if suffix == ".txt" or mime.startswith("text/"):
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(text=raw[: self._max_chars], mime_type=mime, meta={})

        if suffix == ".pdf" or mime == "application/pdf":
            return self._parse_pdf(path.read_bytes(), mime)

        logger.warning("未知类型，按二进制忽略解析 path={}", path)
        return ParsedDocument(text="", mime_type=mime, meta={"warning": "unsupported"})

    def parse_bytes(self, data: bytes, filename: str, mime_type: str | None) -> ParsedDocument:
        """从内存字节解析。"""
        mime = mime_type or "application/octet-stream"
        lower = filename.lower()
        if lower.endswith(".txt") or (mime.startswith("text/") and mime != "text/html"):
            text = data.decode("utf-8", errors="ignore")
            return ParsedDocument(text=text[: self._max_chars], mime_type=mime, meta={})

        if lower.endswith(".pdf") or mime == "application/pdf":
            return self._parse_pdf(data, mime)

        return ParsedDocument(text="", mime_type=mime, meta={"warning": "unsupported"})

    def _parse_pdf(self, data: bytes, mime: str) -> ParsedDocument:
        """使用 pypdf 抽取文本。"""
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            logger.exception("未安装 pypdf: {}", exc)
            raise

        import io

        reader = PdfReader(io.BytesIO(data))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception as exc:
                logger.warning("单页 PDF 抽取失败: {}", exc)

        text = "\n".join(parts)[: self._max_chars]
        return ParsedDocument(text=text, mime_type=mime, meta={"pages": str(len(reader.pages))})
