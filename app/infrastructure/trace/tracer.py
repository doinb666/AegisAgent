# -*- coding: utf-8 -*-
"""全链路追踪：内存存储 Span 树与查询接口。"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class TraceSpan:
    """单次操作 Span。"""

    span_id: str
    trace_id: str
    operation: str
    parent_span_id: str | None
    start_time: float
    end_time: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class TraceRecord:
    """一次 Trace 的完整记录。"""

    trace_id: str
    spans: list[TraceSpan] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class Tracer:
    """全链路追踪器：记录 Agent 执行的每个步骤（进程内存储，可替换为导出到 OTLP）。"""

    def __init__(self, max_traces: int = 5000) -> None:
        self._max_traces = max_traces
        self._traces: dict[str, TraceRecord] = {}
        self._lock = threading.RLock()
        self._current_parent: dict[str, str | None] = {}

    def start_trace(self, trace_id: str, operation: str) -> TraceSpan:
        """开始新的根 Span（同 trace_id 下若已有记录则追加 Span）。"""
        with self._lock:
            rec = self._traces.get(trace_id)
            if rec is None:
                rec = TraceRecord(trace_id=trace_id)
                self._traces[trace_id] = rec
                self._trim_locked()

            span = TraceSpan(
                span_id=str(uuid.uuid4()),
                trace_id=trace_id,
                operation=operation,
                parent_span_id=None,
                start_time=time.perf_counter(),
            )
            rec.spans.append(span)
            self._current_parent[trace_id] = span.span_id
            logger.debug("trace={} span={} op={} 开始", trace_id, span.span_id, operation)
            return span

    def start_child_span(
        self,
        trace_id: str,
        operation: str,
        parent_span_id: str | None = None,
    ) -> TraceSpan:
        """在已有 Trace 下创建子 Span。"""
        with self._lock:
            rec = self._traces.get(trace_id)
            if rec is None:
                rec = TraceRecord(trace_id=trace_id)
                self._traces[trace_id] = rec
                self._trim_locked()

            parent = parent_span_id or self._current_parent.get(trace_id)
            span = TraceSpan(
                span_id=str(uuid.uuid4()),
                trace_id=trace_id,
                operation=operation,
                parent_span_id=parent,
                start_time=time.perf_counter(),
            )
            rec.spans.append(span)
            self._current_parent[trace_id] = span.span_id
            return span

    def end_span(
        self,
        span: TraceSpan,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """结束 Span 并记录结果或错误。"""
        with self._lock:
            rec = self._traces.get(span.trace_id)
            if rec is None:
                logger.warning("end_span 找不到 trace_id={}", span.trace_id)
                return
            span.end_time = time.perf_counter()
            span.result = result
            span.error = error
            logger.debug(
                "trace={} span={} op={} 结束 error={}",
                span.trace_id,
                span.span_id,
                span.operation,
                error,
            )

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """按 trace_id 获取完整追踪记录。"""
        with self._lock:
            rec = self._traces.get(trace_id)
            if rec is None:
                return None
            # 返回浅拷贝，避免外部修改内部列表结构
            return TraceRecord(
                trace_id=rec.trace_id,
                spans=list(rec.spans),
                created_at=rec.created_at,
            )

    def _trim_locked(self) -> None:
        """限制内存中 trace 数量。"""
        if len(self._traces) <= self._max_traces:
            return
        # 删除最旧的一批 key（简单按字典序，生产可改为按 created_at）
        excess = len(self._traces) - self._max_traces
        for key in list(self._traces.keys())[:excess]:
            del self._traces[key]
