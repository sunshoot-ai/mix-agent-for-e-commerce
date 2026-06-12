from __future__ import annotations

from typing import Any

SEGMENT_PRIORITY = (
    "high_value",
    "high_intent",
    "active_viewer",
    "light_user",
    "new_user",
    "churn_risk",
)

UNKNOWN_SEGMENT = "unknown"


def get_primary_segment(segments: list[str] | tuple[Any, ...] | None) -> str:
    """Select one primary segment by fixed business priority."""
    try:
        if not segments:
            return UNKNOWN_SEGMENT
        normalized = {_normalize_segment(segment) for segment in segments}
        for candidate in SEGMENT_PRIORITY:
            if candidate in normalized:
                return candidate
        return UNKNOWN_SEGMENT
    except Exception:
        return UNKNOWN_SEGMENT


def _normalize_segment(segment: Any) -> str:
    value = getattr(segment, "value", segment)
    if value is None:
        return ""
    return str(value).strip().lower()
