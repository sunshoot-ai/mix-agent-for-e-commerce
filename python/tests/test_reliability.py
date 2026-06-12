from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.base_agent import BaseAgent
from models.schemas import AgentResult
from services.reliability import (
    ErrorCode,
    RetryExhaustedError,
    RetryPolicy,
    classify_exception,
)


class FailingAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="failing_agent", timeout=1.0, max_retries=1)

    async def _execute(self, **kwargs):
        raise ValueError("bad input")


class SlowAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="slow_agent", timeout=0.01, max_retries=1)

    async def _execute(self, **kwargs):
        await asyncio.sleep(0.05)
        return AgentResult(agent_name=self.name)


def test_classify_timeout():
    classified = classify_exception(
        asyncio.TimeoutError(),
        source_component="agent",
        timeout_code=ErrorCode.AGENT_TIMEOUT,
    )
    assert classified.code == ErrorCode.AGENT_TIMEOUT
    assert classified.retryable is True


def test_retry_policy_retries_retryable_error():
    async def scenario():
        calls = {"count": 0}

        async def operation():
            calls["count"] += 1
            raise TimeoutError("temporary timeout")

        policy = RetryPolicy(max_attempts=2, timeout_seconds=None, wait_initial_seconds=0.0)

        try:
            await policy.execute(
                operation,
                component="unit",
                timeout_code=ErrorCode.TOOL_TIMEOUT,
            )
        except RetryExhaustedError as exc:
            assert calls["count"] == 2
            assert len(exc.attempts) == 2
            assert exc.classified_error.code == ErrorCode.TOOL_TIMEOUT
        else:
            raise AssertionError("RetryExhaustedError was not raised")

    asyncio.run(scenario())


def test_base_agent_fallback_has_reliability_metadata():
    async def scenario():
        result = await FailingAgent().run()

        assert result.success is False
        assert result.error == "bad input"
        assert result.data["reliability"]["reason"] == ErrorCode.INVALID_ARGUMENT.value
        assert result.data["reliability"]["retry_count"] == 1

    asyncio.run(scenario())


def test_base_agent_timeout_is_enforced():
    async def scenario():
        result = await SlowAgent().run()

        assert result.success is False
        assert result.data["reliability"]["reason"] == ErrorCode.AGENT_TIMEOUT.value

    asyncio.run(scenario())


def test_retry_policy_no_retry_for_invalid_argument():
    calls = {"count": 0}

    async def operation():
        calls["count"] += 1
        raise ValueError("bad input")

    async def scenario():
        policy = RetryPolicy(max_attempts=3, timeout_seconds=None, wait_initial_seconds=0.0)

        try:
            await policy.execute(operation, component="unit")
        except RetryExhaustedError as exc:
            assert exc.classified_error.code == ErrorCode.INVALID_ARGUMENT
        else:
            raise AssertionError("RetryExhaustedError was not raised")

    asyncio.run(scenario())
    assert calls["count"] == 1
