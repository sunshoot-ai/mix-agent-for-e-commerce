from __future__ import annotations

import contextlib
import contextvars
import time
import uuid
from dataclasses import asdict
from typing import Any, AsyncIterator, Iterator

import structlog

from .events import emit_event
from .metrics_registry import default_registry
from .schemas import EventStatus, SpanKind, SpanRecord, TraceContext

logger = structlog.get_logger()

_current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "observability_trace", default=None
)
_current_span: contextvars.ContextVar[SpanRecord | None] = contextvars.ContextVar(
    "observability_span", default=None
)


def current_trace() -> TraceContext | None:
    return _current_trace.get()


def current_span() -> SpanRecord | None:
    return _current_span.get()


class ObservabilityTracer:
    def _make_trace_context(
        self,
        *,
        request_id: str = "",
        workflow: str = "",
        user_id: str = "",
        scene: str = "",
        experiment_id: str = "",
        experiment_group: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        return TraceContext(
            trace_id=str(uuid.uuid4()),
            request_id=request_id,
            workflow=workflow,
            user_id=user_id,
            scene=scene,
            experiment_id=experiment_id,
            experiment_group=experiment_group,
            metadata=dict(metadata or {}),
        )

    def start_trace(
        self,
        *,
        request_id: str = "",
        workflow: str = "",
        user_id: str = "",
        scene: str = "",
        experiment_id: str = "",
        experiment_group: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        ctx = self._make_trace_context(
            request_id=request_id,
            workflow=workflow,
            user_id=user_id,
            scene=scene,
            experiment_id=experiment_id,
            experiment_group=experiment_group,
            metadata=dict(metadata or {}),
        )
        _current_trace.set(ctx)
        self._emit_trace_start(ctx)
        return ctx

    def end_trace(self, status: EventStatus = EventStatus.END, **fields: Any) -> None:
        ctx = current_trace()
        if not ctx:
            return
        emit_event(
            "trace.end",
            trace_id=ctx.trace_id,
            request_id=ctx.request_id,
            component=ctx.workflow or "workflow",
            component_type="trace",
            status=status,
            user_id=ctx.user_id,
            scene=ctx.scene,
            experiment_id=ctx.experiment_id,
            experiment_group=ctx.experiment_group,
            attributes=dict(fields),
        )

    def clear_trace(self) -> None:
        _current_trace.set(None)
        _current_span.set(None)

    @contextlib.contextmanager
    def trace(
        self,
        *,
        request_id: str = "",
        workflow: str = "",
        user_id: str = "",
        scene: str = "",
        experiment_id: str = "",
        experiment_group: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[TraceContext]:
        ctx = self._make_trace_context(
            request_id=request_id,
            workflow=workflow,
            user_id=user_id,
            scene=scene,
            experiment_id=experiment_id,
            experiment_group=experiment_group,
            metadata=metadata,
        )
        token = _current_trace.set(ctx)
        self._emit_trace_start(ctx)
        status = EventStatus.END
        try:
            yield ctx
        except Exception:
            status = EventStatus.ERROR
            raise
        finally:
            self.end_trace(status=status)
            _current_trace.reset(token)

    @contextlib.asynccontextmanager
    async def async_trace(
        self,
        *,
        request_id: str = "",
        workflow: str = "",
        user_id: str = "",
        scene: str = "",
        experiment_id: str = "",
        experiment_group: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TraceContext]:
        with self.trace(
            request_id=request_id,
            workflow=workflow,
            user_id=user_id,
            scene=scene,
            experiment_id=experiment_id,
            experiment_group=experiment_group,
            metadata=metadata,
        ) as ctx:
            yield ctx

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.WORKFLOW,
        **attributes: Any,
    ) -> Iterator[SpanRecord]:
        ctx = current_trace()
        parent = current_span()
        span = SpanRecord(
            name=name,
            span_id=str(uuid.uuid4()),
            trace_id=ctx.trace_id if ctx else str(uuid.uuid4()),
            parent_span_id=parent.span_id if parent else None,
            kind=kind,
            status=EventStatus.START,
            start_ts=time.perf_counter(),
            attributes=dict(attributes),
        )
        token = _current_span.set(span)
        emit_event(
            "span.start",
            trace_id=span.trace_id,
            request_id=ctx.request_id if ctx else "",
            span_id=span.span_id,
            parent_span_id=span.parent_span_id or "",
            component=name,
            component_type=kind.value,
            status=EventStatus.START,
            user_id=ctx.user_id if ctx else "",
            scene=ctx.scene if ctx else "",
            experiment_id=ctx.experiment_id if ctx else "",
            experiment_group=ctx.experiment_group if ctx else "",
            attributes=dict(attributes),
        )
        try:
            yield span
        except Exception as exc:
            span.status = EventStatus.ERROR
            span.error = str(exc)
            raise
        finally:
            span.end_ts = time.perf_counter()
            span.duration_ms = (span.end_ts - span.start_ts) * 1000
            if span.status == EventStatus.START:
                span.status = EventStatus.OK
            _current_span.reset(token)
            default_registry.observe(
                "span_latency_ms",
                span.duration_ms,
                workflow=ctx.workflow if ctx else "",
                component=name,
                kind=kind.value,
            )
            if kind == SpanKind.WORKFLOW:
                default_registry.observe(
                    "workflow_latency_ms",
                    span.duration_ms,
                    workflow=ctx.workflow if ctx else "",
                    component=name,
                )
            elif kind == SpanKind.AGENT:
                default_registry.observe(
                    "agent_latency_ms",
                    span.duration_ms,
                    agent=str(attributes.get("agent", name)),
                )
            elif kind in {SpanKind.TOOL, SpanKind.FEATURE}:
                default_registry.observe(
                    "tool_latency_ms",
                    span.duration_ms,
                    tool=str(attributes.get("tool_name", name)),
                    kind=kind.value,
                )
            emit_event(
                "span.end",
                trace_id=span.trace_id,
                request_id=ctx.request_id if ctx else "",
                span_id=span.span_id,
                parent_span_id=span.parent_span_id or "",
                component=name,
                component_type=kind.value,
                status=span.status,
                latency_ms=span.duration_ms,
                user_id=ctx.user_id if ctx else "",
                scene=ctx.scene if ctx else "",
                experiment_id=ctx.experiment_id if ctx else "",
                experiment_group=ctx.experiment_group if ctx else "",
                error_message=span.error or "",
                attributes=asdict(span),
            )

    @contextlib.asynccontextmanager
    async def async_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.WORKFLOW,
        **attributes: Any,
    ) -> AsyncIterator[SpanRecord]:
        with self.span(name, kind=kind, **attributes) as span:
            yield span

    def _emit_trace_start(self, ctx: TraceContext) -> None:
        emit_event(
            "trace.start",
            trace_id=ctx.trace_id,
            request_id=ctx.request_id,
            component=ctx.workflow or "workflow",
            component_type="trace",
            user_id=ctx.user_id,
            scene=ctx.scene,
            experiment_id=ctx.experiment_id,
            experiment_group=ctx.experiment_group,
            attributes=dict(ctx.metadata),
        )


tracer = ObservabilityTracer()
