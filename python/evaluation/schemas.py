from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.observability import TraceEvent


@dataclass(slots=True)
class WorkflowReplayRecord:
    trace_id: str
    request_id: str = ""
    workflow: str = ""
    events: list[TraceEvent] = field(default_factory=list)
    response: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationCase:
    name: str
    replay: WorkflowReplayRecord
    expected: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationMetric:
    name: str
    value: float
    passed: bool = True
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "passed": self.passed,
            "threshold": self.threshold,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class EvaluationResult:
    suite_name: str
    case_name: str
    passed: bool
    metrics: list[EvaluationMetric] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "case_name": self.case_name,
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "failures": list(self.failures),
            "metadata": dict(self.metadata),
        }
