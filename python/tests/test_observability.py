from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.observability import (
    EventStatus,
    MetricRegistry,
    ObservabilityEventBuffer,
    SpanKind,
    TokenUsage,
    emit_event,
    extract_token_usage,
    get_event_buffer,
    tracer,
)


def test_metric_registry_snapshot():
    registry = MetricRegistry()
    registry.inc("workflow_requests_total", workflow="supervisor")
    registry.observe("workflow_latency_ms", 120.0, workflow="supervisor")
    registry.observe("workflow_latency_ms", 180.0, workflow="supervisor")

    snapshot = registry.snapshot()
    assert snapshot["workflow_requests_total"]["counter"] == 1.0
    assert snapshot["workflow_latency_ms"]["count"] == 2
    assert snapshot["workflow_latency_ms"]["p95"] >= 120.0


def test_event_buffer_and_emit():
    buffer = ObservabilityEventBuffer(max_events=10)
    event = emit_event(
        "span.test",
        trace_id="trace-1",
        status=EventStatus.OK,
        component="unit",
        component_type=SpanKind.AGENT.value,
    )
    buffer.emit(event)
    assert buffer.snapshot()[-1].event_name == "span.test"


def test_token_usage_extraction():
    usage = extract_token_usage(
        {
            "response_metadata": {"token_usage": {"prompt_tokens": 12, "completion_tokens": 8}},
            "content": "hello world",
        },
        model="gpt-test",
    )
    assert isinstance(usage, TokenUsage)
    assert usage.total_tokens == 20
    assert usage.prompt_tokens == 12


def test_tracer_span_records():
    trace = tracer.start_trace(request_id="req-1", workflow="supervisor", user_id="u1")
    with tracer.span("agent.run", kind=SpanKind.AGENT, agent="user_profile") as span:
        assert span.trace_id == trace.trace_id
    tracer.end_trace()
    assert get_event_buffer().snapshot()
