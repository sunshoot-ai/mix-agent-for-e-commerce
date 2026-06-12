from .error_taxonomy import (
    ClassifiedError,
    ErrorCategory,
    ErrorCode,
    ErrorSeverity,
    classify_exception,
)
from .fallback_handler import FallbackRecord, build_fallback_metadata
from .retry_policy import (
    RetryAttempt,
    RetryExecutionResult,
    RetryExhaustedError,
    RetryPolicy,
)

__all__ = [
    "ClassifiedError",
    "ErrorCategory",
    "ErrorCode",
    "ErrorSeverity",
    "FallbackRecord",
    "RetryAttempt",
    "RetryExecutionResult",
    "RetryExhaustedError",
    "RetryPolicy",
    "build_fallback_metadata",
    "classify_exception",
]
