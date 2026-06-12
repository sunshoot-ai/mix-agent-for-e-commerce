from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SpanKind(str, Enum):
    WORKFLOW = "workflow"
    AGENT = "agent"
    TOOL = "tool"
    FEATURE = "feature"
    LLM = "llm"
    RETRY = "retry"
    FALLBACK = "fallback"


class EventStatus(str, Enum):
    START = "start"
    OK = "ok"
    ERROR = "error"
    RETRY = "retry"
    FALLBACK = "fallback"
    END = "end"


@dataclass(slots=True)
class TraceContext:
    trace_id: str
    request_id: str = ""
    workflow: str = ""
    user_id: str = ""
    scene: str = ""
    experiment_id: str = ""
    experiment_group: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SpanRecord:
    name: str
    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.WORKFLOW
    status: EventStatus = EventStatus.START
    start_ts: float = 0.0
    end_ts: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    estimated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceEvent:
    event_name: str
    trace_id: str
    timestamp: float
    request_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    component: str = ""
    component_type: str = ""
    status: EventStatus = EventStatus.OK
    latency_ms: float = 0.0
    error_code: str = ""
    error_message: str = ""
    retry_count: int = 0
    fallback_used: bool = False
    user_id: str = ""
    scene: str = ""
    experiment_id: str = ""
    experiment_group: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_name: str = ""
    tool_result_count: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricSample:
    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = 0.0
