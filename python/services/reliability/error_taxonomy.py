from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from enum import Enum


class ErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    INVALID_MODEL_OUTPUT = "INVALID_MODEL_OUTPUT"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    WORKFLOW_TIMEOUT = "WORKFLOW_TIMEOUT"
    HALLUCINATED_TOOL = "HALLUCINATED_TOOL"
    ROUTING_FAILURE = "ROUTING_FAILURE"
    FEATURE_FETCH_FAILURE = "FEATURE_FETCH_FAILURE"
    FEATURE_STORE_UNAVAILABLE = "FEATURE_STORE_UNAVAILABLE"
    EXPERIMENT_ASSIGNMENT_FAILURE = "EXPERIMENT_ASSIGNMENT_FAILURE"
    METRIC_PERSISTENCE_FAILURE = "METRIC_PERSISTENCE_FAILURE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_AUTH_FAILURE = "LLM_AUTH_FAILURE"
    LLM_PROVIDER_FAILURE = "LLM_PROVIDER_FAILURE"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorCategory(str, Enum):
    INPUT = "input"
    MODEL = "model"
    TOOL = "tool"
    WORKFLOW = "workflow"
    FEATURE = "feature"
    EXPERIMENT = "experiment"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class ClassifiedError:
    code: ErrorCode
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    source_component: str = ""
    recoverable: bool = True
    retryable: bool = False
    safe_to_fallback: bool = True
    original_type: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        data = asdict(self)
        data["code"] = self.code.value
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        return data


def classify_exception(
    exc: Exception,
    *,
    source_component: str = "",
    timeout_code: ErrorCode = ErrorCode.TOOL_TIMEOUT,
) -> ClassifiedError:
    message = str(exc)
    text = message.lower()
    original_type = exc.__class__.__name__

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return _classified(
            timeout_code,
            message,
            source_component,
            original_type,
            retryable=True,
            severity=ErrorSeverity.WARNING,
        )

    if isinstance(exc, json.JSONDecodeError):
        return _classified(
            ErrorCode.INVALID_MODEL_OUTPUT,
            message,
            source_component,
            original_type,
            category=ErrorCategory.MODEL,
            recoverable=True,
            retryable=False,
        )

    if any(marker in text for marker in ("context length", "maximum context", "max tokens", "context overflow")):
        return _classified(
            ErrorCode.CONTEXT_OVERFLOW,
            message,
            source_component,
            original_type,
            category=ErrorCategory.MODEL,
            recoverable=True,
            retryable=False,
            severity=ErrorSeverity.ERROR,
        )

    if "rate limit" in text or "ratelimit" in original_type.lower():
        return _classified(
            ErrorCode.LLM_RATE_LIMIT,
            message,
            source_component,
            original_type,
            category=ErrorCategory.MODEL,
            retryable=True,
        )

    if "auth" in text or "api key" in text or "unauthorized" in text:
        return _classified(
            ErrorCode.LLM_AUTH_FAILURE,
            message,
            source_component,
            original_type,
            category=ErrorCategory.MODEL,
            recoverable=False,
            retryable=False,
            safe_to_fallback=True,
            severity=ErrorSeverity.ERROR,
        )

    if "feast" in text and "unavailable" in text:
        return _classified(
            ErrorCode.FEATURE_STORE_UNAVAILABLE,
            message,
            source_component,
            original_type,
            category=ErrorCategory.FEATURE,
            retryable=True,
        )

    if "feature" in text or "feast" in text:
        return _classified(
            ErrorCode.FEATURE_FETCH_FAILURE,
            message,
            source_component,
            original_type,
            category=ErrorCategory.FEATURE,
            retryable=True,
        )

    if isinstance(exc, (ValueError, TypeError, KeyError)):
        return _classified(
            ErrorCode.INVALID_ARGUMENT,
            message,
            source_component,
            original_type,
            category=ErrorCategory.INPUT,
            recoverable=False,
            retryable=False,
            safe_to_fallback=True,
        )

    if any(marker in text for marker in ("connection", "unavailable", "temporarily", "503", "502")):
        return _classified(
            ErrorCode.DEPENDENCY_UNAVAILABLE,
            message,
            source_component,
            original_type,
            category=ErrorCategory.DEPENDENCY,
            retryable=True,
        )

    if any(marker in original_type.lower() for marker in ("openai", "llm", "api")):
        return _classified(
            ErrorCode.LLM_PROVIDER_FAILURE,
            message,
            source_component,
            original_type,
            category=ErrorCategory.MODEL,
            retryable=True,
        )

    return _classified(
        ErrorCode.UNKNOWN_ERROR,
        message,
        source_component,
        original_type,
        category=ErrorCategory.UNKNOWN,
        retryable=True,
    )


def _classified(
    code: ErrorCode,
    message: str,
    source_component: str,
    original_type: str,
    *,
    category: ErrorCategory | None = None,
    severity: ErrorSeverity = ErrorSeverity.WARNING,
    recoverable: bool = True,
    retryable: bool = False,
    safe_to_fallback: bool = True,
) -> ClassifiedError:
    return ClassifiedError(
        code=code,
        category=category or _category_for_code(code),
        severity=severity,
        message=message,
        source_component=source_component,
        recoverable=recoverable,
        retryable=retryable,
        safe_to_fallback=safe_to_fallback,
        original_type=original_type,
    )


def _category_for_code(code: ErrorCode) -> ErrorCategory:
    if code in {ErrorCode.INVALID_ARGUMENT}:
        return ErrorCategory.INPUT
    if code in {
        ErrorCode.INVALID_MODEL_OUTPUT,
        ErrorCode.CONTEXT_OVERFLOW,
        ErrorCode.LLM_TIMEOUT,
        ErrorCode.LLM_RATE_LIMIT,
        ErrorCode.LLM_AUTH_FAILURE,
        ErrorCode.LLM_PROVIDER_FAILURE,
    }:
        return ErrorCategory.MODEL
    if code in {ErrorCode.TOOL_TIMEOUT, ErrorCode.HALLUCINATED_TOOL}:
        return ErrorCategory.TOOL
    if code in {ErrorCode.AGENT_TIMEOUT, ErrorCode.WORKFLOW_TIMEOUT, ErrorCode.ROUTING_FAILURE}:
        return ErrorCategory.WORKFLOW
    if code in {ErrorCode.FEATURE_FETCH_FAILURE, ErrorCode.FEATURE_STORE_UNAVAILABLE}:
        return ErrorCategory.FEATURE
    if code in {ErrorCode.EXPERIMENT_ASSIGNMENT_FAILURE, ErrorCode.METRIC_PERSISTENCE_FAILURE}:
        return ErrorCategory.EXPERIMENT
    if code == ErrorCode.DEPENDENCY_UNAVAILABLE:
        return ErrorCategory.DEPENDENCY
    return ErrorCategory.UNKNOWN
