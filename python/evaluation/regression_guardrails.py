from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .schemas import EvaluationMetric, EvaluationResult

Comparison = Literal["<=", ">=", "=="]


@dataclass(slots=True)
class Guardrail:
    metric_name: str
    threshold: float
    comparison: Comparison = ">="

    def check(self, metric: EvaluationMetric) -> bool:
        if self.comparison == ">=":
            return metric.value >= self.threshold
        if self.comparison == "<=":
            return metric.value <= self.threshold
        return metric.value == self.threshold


class RegressionGuardrails:
    name = "regression_guardrails"

    def __init__(self, guardrails: list[Guardrail]):
        self.guardrails = guardrails

    def evaluate_results(self, results: list[EvaluationResult]) -> EvaluationResult:
        metric_lookup: dict[str, list[EvaluationMetric]] = {}
        for result in results:
            for metric in result.metrics:
                metric_lookup.setdefault(metric.name, []).append(metric)

        failures: list[str] = []
        guardrail_metrics: list[EvaluationMetric] = []
        for guardrail in self.guardrails:
            values = metric_lookup.get(guardrail.metric_name, [])
            if not values:
                failures.append(f"missing guardrail metric: {guardrail.metric_name}")
                guardrail_metrics.append(
                    EvaluationMetric(
                        f"guardrail_{guardrail.metric_name}",
                        0.0,
                        False,
                        guardrail.threshold,
                    )
                )
                continue

            aggregate_value = sum(metric.value for metric in values) / len(values)
            aggregate_metric = EvaluationMetric(
                f"guardrail_{guardrail.metric_name}",
                aggregate_value,
                True,
                guardrail.threshold,
                details={"comparison": guardrail.comparison},
            )
            aggregate_metric.passed = guardrail.check(aggregate_metric)
            if not aggregate_metric.passed:
                failures.append(
                    f"{guardrail.metric_name}={aggregate_value:.4f} failed "
                    f"{guardrail.comparison} {guardrail.threshold:.4f}"
                )
            guardrail_metrics.append(aggregate_metric)

        return EvaluationResult(
            suite_name=self.name,
            case_name="aggregate",
            passed=not failures,
            metrics=guardrail_metrics,
            failures=failures,
        )
