from __future__ import annotations

from .schemas import EvaluationCase, EvaluationMetric, EvaluationResult


class ToolCallEvaluator:
    name = "tool_call"

    def __init__(self, allowed_tools: set[str] | None = None):
        self.allowed_tools = allowed_tools or set()

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        events = case.replay.events
        tool_events = [
            event for event in events
            if event.tool_name or event.component_type in {"tool", "feature"}
        ]
        tool_names = {
            event.tool_name or str(event.attributes.get("tool_name", ""))
            for event in tool_events
        }
        tool_names.discard("")

        hallucinated = sorted(tool_names - self.allowed_tools) if self.allowed_tools else []
        expected_tools = set(case.expected.get("tools", []))
        missing_expected = sorted(expected_tools - tool_names)

        failures: list[str] = []
        if hallucinated:
            failures.append(f"hallucinated tool calls: {hallucinated}")
        if missing_expected:
            failures.append(f"missing expected tool calls: {missing_expected}")

        selection_accuracy = 1.0
        if expected_tools:
            selection_accuracy = len(expected_tools & tool_names) / len(expected_tools)

        metrics = [
            EvaluationMetric(
                "hallucinated_tool_calls",
                float(len(hallucinated)),
                not hallucinated,
                details={"tools": hallucinated},
            ),
            EvaluationMetric(
                "tool_selection_accuracy",
                selection_accuracy,
                not missing_expected,
                threshold=1.0 if expected_tools else None,
                details={"observed_tools": sorted(tool_names), "missing": missing_expected},
            ),
            EvaluationMetric("tool_call_count", float(len(tool_events)), True),
        ]

        return EvaluationResult(
            suite_name=self.name,
            case_name=case.name,
            passed=not failures and all(metric.passed for metric in metrics),
            metrics=metrics,
            failures=failures,
        )
