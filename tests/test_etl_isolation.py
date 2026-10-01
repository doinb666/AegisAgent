"""真实固定解析进程：文本、PDF、超时终止与内存上限。"""

import asyncio
import io
import subprocess
import sys

import pytest
from pypdf import PdfWriter

from app.etl.isolated import parse_isolated


@pytest.mark.asyncio
async def test_real_parse_process_handles_text_and_pdf():
    text = await parse_isolated("隔离解析中文文档".encode(), "notes.md", "text/plain")
    assert text.parsed.text == "隔离解析中文文档" and text.chunks
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    output = io.BytesIO()
    writer.write(output)
    pdf = await parse_isolated(output.getvalue(), "blank.pdf", "application/pdf")
    assert pdf.parsed.meta["pages"] == "1" and not pdf.chunks


@pytest.mark.asyncio
async def test_source_parse_works_from_configuration_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = await parse_isolated(b"config-directory", "config.txt", "text/plain")
    assert result.parsed.text == "config-directory"


@pytest.mark.asyncio
async def test_timeout_kills_real_parser_process(monkeypatch):
    import app.etl.isolated as isolated

    original = asyncio.create_subprocess_exec
    children = []

    async def recorded(*args, **kwargs):
        process = await original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(isolated.asyncio, "create_subprocess_exec", recorded)
    with pytest.raises(TimeoutError):
        await parse_isolated(b"hello", "test.txt", "text/plain", timeout=0.001)
    assert len(children) == 1 and children[0].returncode is not None


def test_real_worker_memory_budget_rejects_large_allocation():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.etl.worker import apply_limits; apply_limits(); bytes(800*1024*1024)",
        ],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert b"MemoryError" in result.stderr, result.stderr[-500:]
