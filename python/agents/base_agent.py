from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import structlog

from models.schemas import AgentResult
from services.observability import EventStatus, SpanKind, default_registry, tracer
from services.reliability import (
    ErrorCode,
    RetryExhaustedError,
    RetryPolicy,
    build_fallback_metadata,
)

logger = structlog.get_logger()


class BaseAgent(ABC):
    """All agents inherit from this base class with retry, timeout, and fallback."""

    def __init__(self, name: str, timeout: float = 10.0, max_retries: int = 2):
        self.name = name
        self.timeout = timeout
        self.max_retries = max_retries
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> AgentResult:
        """Core logic implemented by each concrete agent."""

    async def run(self, **kwargs: Any) -> AgentResult:
        """Public entry: wraps _execute with timing, retries, and fallback."""
        start = time.perf_counter()
        self._call_count += 1

        async with tracer.async_span(
            "agent.run",
            kind=SpanKind.AGENT,
            agent=self.name,
            timeout_seconds=self.timeout,
            max_attempts=max(1, self.max_retries),
        ) as span:
            try:
                result, attempts = await self._retry_execute(**kwargs)
                result.latency_ms = (time.perf_counter() - start) * 1000
                result.data.setdefault("reliability", {})
                result.data["reliability"].update(
                    {
                        "retry_count": len(attempts),
                        "timeout_seconds": self.timeout,
                        "max_attempts": max(1, self.max_retries),
                    }
                )
                span.attributes.update(
                    {
                        "success": True,
                        "latency_ms": result.latency_ms,
                        "retry_count": len(attempts),
                    }
                )
                self._record_observability(result)
                logger.info(
                    "agent.success",
                    agent=self.name,
                    latency_ms=round(result.latency_ms, 1),
                    retry_count=len(attempts),
                )
                return result
            except Exception as exc:
                self._error_count += 1
                latency_ms = (time.perf_counter() - start) * 1000
                classified = getattr(exc, "classified_error", None)
                logger.error(
                    "agent.failed",
                    agent=self.name,
                    error=str(exc),
                    error_code=getattr(classified, "code", ""),
                )
                result = self._fallback(latency_ms, exc)
                span.status = EventStatus.FALLBACK
                span.error = result.error
                span.attributes.update(
                    {
                        "success": False,
                        "latency_ms": result.latency_ms,
                        "error_code": result.data["reliability"]["reason"],
                        "retry_count": result.data["reliability"]["retry_count"],
                    }
                )
                self._record_observability(result)
                return result

    async def _retry_execute(self, **kwargs: Any) -> tuple[AgentResult, list[Any]]:
        policy = RetryPolicy(
            max_attempts=max(1, self.max_retries),
            timeout_seconds=self.timeout,
            wait_initial_seconds=0.5,
            wait_max_seconds=4.0,
        )
        execution = await policy.execute(
            lambda: self._execute(**kwargs),
            component=self.name,
            timeout_code=ErrorCode.AGENT_TIMEOUT,
        )
        return execution.value, execution.attempts

    def _fallback(self, latency_ms: float, exc: Exception) -> AgentResult:
        """Return a degraded but valid result when the agent fails."""
        attempts = getattr(exc, "attempts", [])
        classified = getattr(exc, "classified_error", None)
        original = exc.original_error if isinstance(exc, RetryExhaustedError) else exc
        return AgentResult(
            agent_name=self.name,
            success=False,
            latency_ms=latency_ms,
            error=str(original),
            data={
                "reliability": build_fallback_metadata(
                    component=self.name,
                    exc=original,
                    latency_ms=latency_ms,
                    attempts=attempts,
                    classified_error=classified,
                )
            },
            confidence=0.0,
        )

    def _record_observability(self, result: AgentResult) -> None:
        reliability = result.data.get("reliability", {})
        status = "success" if result.success else "fallback"
        error_code = str(reliability.get("reason", ""))
        default_registry.inc(
            "agent_calls_total",
            agent=self.name,
            status=status,
            error_code=error_code,
        )
        default_registry.observe(
            "agent_result_latency_ms",
            result.latency_ms,
            agent=self.name,
            status=status,
        )
        retry_count = float(reliability.get("retry_count", 0) or 0)
        if retry_count:
            default_registry.inc("retry_attempts_total", retry_count, component=self.name)
        if not result.success:
            default_registry.inc(
                "agent_fallback_total",
                agent=self.name,
                error_code=error_code,
                fallback_mode=str(reliability.get("fallback_mode", "agent_result")),
            )

    @property
    def error_rate(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._error_count / self._call_count
