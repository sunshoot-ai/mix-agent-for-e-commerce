from __future__ import annotations

from typing import Any

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None

from .schemas import TokenUsage


def extract_token_usage(result: Any, model: str = "") -> TokenUsage:
    payload = _as_mapping(result)
    metadata = _extract_metadata(payload)

    prompt_tokens = _first_int(
        metadata,
        "prompt_tokens",
        "input_tokens",
        "promptTokenCount",
    )
    completion_tokens = _first_int(
        metadata,
        "completion_tokens",
        "output_tokens",
        "completionTokenCount",
    )
    total_tokens = _first_int(
        metadata,
        "total_tokens",
        "token_count",
        "totalTokenCount",
    )

    if total_tokens == 0 and (prompt_tokens or completion_tokens):
        total_tokens = prompt_tokens + completion_tokens

    if total_tokens == 0 and payload:
        text = _extract_text(payload)
        total_tokens = estimate_token_count(text, model=model)

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=model or str(metadata.get("model", "")),
        estimated=total_tokens > 0 and prompt_tokens == 0 and completion_tokens == 0,
        metadata=dict(metadata),
    )


def estimate_token_count(text: str, model: str = "") -> int:
    if not text:
        return 0
    if tiktoken is not None:
        try:
            encoder = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
            return len(encoder.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except Exception:
            return {}
    return {}


def _extract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("usage", "response_metadata", "token_usage", "usage_metadata", "additional_kwargs"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            if key == "response_metadata":
                usage = nested.get("token_usage")
                if isinstance(usage, dict):
                    return usage
            return nested
    return {}


def _extract_text(payload: dict[str, Any]) -> str:
    for key in ("content", "text", "output_text"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _first_int(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0
