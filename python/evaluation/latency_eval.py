from __future__ import annotations

from .schemas import EvaluationCase, EvaluationMetric, EvaluationResult


class LatencyEvaluator:
    name = "latency"

    def __init__(
        self,
        workflow_latency_threshold_ms: float | None = None,
        agent_latency_threshold_ms: float | None = None,
    ):
        self.workflow_latency_threshold_ms = workflow_latency_threshold_ms
        self.agent_latency_threshold_ms = agent_latency_threshold_ms

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        events = case.replay.events
        workflow_latencies = [
            event.latency_ms for event in events
            if event.component_type == "workflow" and event.latency_ms > 0
        ]
        agent_latencies = [
            event.latency_ms for event in events
            if event.component_type == "agent" and event.latency_ms > 0
        ]

        workflow_p95 = _percentile(workflow_latencies, 0.95)
        agent_p95 = _percentile(agent_latencies, 0.95)

        workflow_passed = (
            self.workflow_latency_threshold_ms is None
            or workflow_p95 <= self.workflow_latency_threshold_ms
        )
        agent_passed = (
            self.agent_latency_threshold_ms is None
            or agent_p95 <= self.agent_latency_threshold_ms
        )

        failures: list[str] = []
        if not workflow_passed:
            failures.append(
                f"workflow p95 latency {workflow_p95:.1f}ms exceeds "
                f"{self.workflow_latency_threshold_ms:.1f}ms"
            )
        if not agent_passed:
            failures.append(
                f"agent p95 latency {agent_p95:.1f}ms exceeds "
                f"{self.agent_latency_threshold_ms:.1f}ms"
            )

        metrics = [
            EvaluationMetric(
                "workflow_latency_p95_ms",
                workflow_p95,
                workflow_passed,
                self.workflow_latency_threshold_ms,
            ),
            EvaluationMetric(
                "agent_latency_p95_ms",
                agent_p95,
                agent_passed,
                self.agent_latency_threshold_ms,
            ),
        ]

        return EvaluationResult(
            suite_name=self.name,
            case_name=case.name,
            passed=not failures and all(metric.passed for metric in metrics),
            metrics=metrics,
            failures=failures,
        )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[index])
