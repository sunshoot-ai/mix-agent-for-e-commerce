from __future__ import annotations

from collections import Counter

from services.observability import EventStatus

from .schemas import EvaluationCase, EvaluationMetric, EvaluationResult


class WorkflowEvaluator:
    name = "workflow"

    def __init__(self, required_transitions: list[tuple[str, str]] | None = None):
        self.required_transitions = required_transitions or []

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        events = case.replay.events
        failures: list[str] = []
        metrics: list[EvaluationMetric] = []

        completed = any(event.event_name == "workflow.complete" for event in events)
        if not completed:
            failures.append("workflow did not emit workflow.complete")
        metrics.append(EvaluationMetric("workflow_completion_rate", 1.0 if completed else 0.0, completed))

        fallback_count = sum(1 for event in events if event.status == EventStatus.FALLBACK)
        error_count = sum(1 for event in events if event.status == EventStatus.ERROR)
        metrics.append(EvaluationMetric("workflow_fallback_count", float(fallback_count)))
        metrics.append(EvaluationMetric("workflow_error_count", float(error_count), error_count == 0))

        observed = {
            (event.attributes.get("from"), event.attributes.get("to"))
            for event in events
            if event.event_name == "workflow.transition"
        }
        missing = [edge for edge in self.required_transitions if edge not in observed]
        if missing:
            failures.append(f"missing workflow transitions: {missing}")
        metrics.append(
            EvaluationMetric(
                "workflow_transition_coverage",
                1.0 - (len(missing) / max(len(self.required_transitions), 1)),
                not missing,
                details={"missing": missing},
            )
        )

        by_component = Counter(event.component for event in events if event.component)
        metrics.append(
            EvaluationMetric(
                "workflow_event_count",
                float(len(events)),
                len(events) > 0,
                details={"by_component": dict(by_component)},
            )
        )

        return EvaluationResult(
            suite_name=self.name,
            case_name=case.name,
            passed=not failures and all(metric.passed for metric in metrics),
            metrics=metrics,
            failures=failures,
        )
