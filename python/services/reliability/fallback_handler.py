from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .error_taxonomy import ClassifiedError, classify_exception
from .retry_policy import RetryAttempt


@dataclass(slots=True)
class FallbackRecord:
    component: str
    fallback_mode: str
    reason: str
    latency_ms: float
    error: dict[str, Any]
    retry_count: int = 0
    degraded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_fallback_metadata(
    *,
    component: str,
    exc: Exception,
    latency_ms: float,
    fallback_mode: str = "agent_result",
    attempts: list[RetryAttempt] | None = None,
    classified_error: ClassifiedError | None = None,
) -> dict[str, Any]:
    classified = classified_error or classify_exception(exc, source_component=component)
    retry_count = len(attempts or [])
    record = FallbackRecord(
        component=component,
        fallback_mode=fallback_mode,
        reason=classified.code.value,
        latency_ms=latency_ms,
        error=classified.to_dict(),
        retry_count=retry_count,
    )
    data = record.to_dict()
    data["attempts"] = [attempt.to_dict() for attempt in attempts or []]
    return data
