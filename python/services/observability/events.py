from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from dataclasses import asdict
from threading import RLock
from typing import Any

import structlog

from .schemas import EventStatus, TraceEvent

logger = structlog.get_logger()


class ObservabilityEventBuffer:
    def __init__(self, max_events: int = 5000):
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._lock = RLock()

    def emit(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def drain(self) -> list[TraceEvent]:
        with self._lock:
            items = list(self._events)
            self._events.clear()
            return items


_event_buffer = ObservabilityEventBuffer()


def get_event_buffer() -> ObservabilityEventBuffer:
    return _event_buffer


def emit_event(
    event_name: str,
    *,
    trace_id: str,
    status: EventStatus = EventStatus.OK,
    timestamp: float | None = None,
    **fields: Any,
) -> TraceEvent:
    event = TraceEvent(
        event_name=event_name,
        trace_id=trace_id,
        timestamp=timestamp if timestamp is not None else time.time(),
        status=status,
        **fields,
    )
    _event_buffer.emit(event)
    if _should_log_event(event):
        logger.info("observability.event", **asdict(event))
    return event


def _should_log_event(event: TraceEvent) -> bool:
    enabled = os.getenv("ECOM_OBSERVABILITY_LOG_EVENTS", "false").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return False

    sample_rate_raw = os.getenv("ECOM_OBSERVABILITY_LOG_SAMPLE_RATE", "1.0")
    try:
        sample_rate = float(sample_rate_raw)
    except ValueError:
        sample_rate = 1.0

    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False

    key = f"{event.trace_id}:{event.span_id}:{event.event_name}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket <= sample_rate
