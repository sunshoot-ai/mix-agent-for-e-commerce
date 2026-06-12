from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable, Generic, TypeVar

from .error_taxonomy import ClassifiedError, ErrorCode, classify_exception

T = TypeVar("T")


@dataclass(slots=True)
class RetryAttempt:
    attempt: int
    latency_ms: float
    error: ClassifiedError

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["error"] = self.error.to_dict()
        return data


@dataclass(slots=True)
class RetryExecutionResult(Generic[T]):
    value: T
    attempts: list[RetryAttempt]


class RetryExhaustedError(Exception):
    def __init__(
        self,
        *,
        component: str,
        classified_error: ClassifiedError,
        attempts: list[RetryAttempt],
        original_error: Exception,
    ):
        super().__init__(classified_error.message)
        self.component = component
        self.classified_error = classified_error
        self.attempts = attempts
        self.original_error = original_error


@dataclass(slots=True)
class RetryPolicy:
    max_attempts: int = 2
    timeout_seconds: float | None = None
    wait_initial_seconds: float = 0.5
    wait_max_seconds: float = 4.0

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        component: str = "",
        timeout_code: ErrorCode = ErrorCode.TOOL_TIMEOUT,
    ) -> T:
        result = await self.execute(
            operation,
            component=component,
            timeout_code=timeout_code,
        )
        return result.value

    async def execute(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        component: str = "",
        timeout_code: ErrorCode = ErrorCode.TOOL_TIMEOUT,
    ) -> RetryExecutionResult[T]:
        attempts: list[RetryAttempt] = []
        max_attempts = max(1, self.max_attempts)

        for attempt_number in range(1, max_attempts + 1):
            start = time.perf_counter()
            try:
                value = await self._run_once(operation)
                return RetryExecutionResult(value=value, attempts=attempts)
            except Exception as exc:
                classified = classify_exception(
                    exc,
                    source_component=component,
                    timeout_code=timeout_code,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                attempts.append(
                    RetryAttempt(
                        attempt=attempt_number,
                        latency_ms=latency_ms,
                        error=classified,
                    )
                )

                if attempt_number >= max_attempts or not classified.retryable:
                    raise RetryExhaustedError(
                        component=component,
                        classified_error=classified,
                        attempts=attempts,
                        original_error=exc,
                    ) from exc

                await asyncio.sleep(self._wait_seconds(attempt_number))

        last_attempt = attempts[-1]
        raise RetryExhaustedError(
            component=component,
            classified_error=last_attempt.error,
            attempts=attempts,
            original_error=RuntimeError(last_attempt.error.message),
        )

    async def _run_once(self, operation: Callable[[], Awaitable[T]]) -> T:
        if self.timeout_seconds is None or self.timeout_seconds <= 0:
            return await operation()
        return await asyncio.wait_for(operation(), timeout=self.timeout_seconds)

    def _wait_seconds(self, attempt_number: int) -> float:
        return min(
            self.wait_max_seconds,
            self.wait_initial_seconds * (2 ** max(0, attempt_number - 1)),
        )
