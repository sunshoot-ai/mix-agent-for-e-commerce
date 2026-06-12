from __future__ import annotations

from typing import Protocol

from .schemas import EvaluationCase, EvaluationMetric, EvaluationResult


class Evaluator(Protocol):
    name: str

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        ...


class EvaluationSuite:
    def __init__(self, name: str, evaluators: list[Evaluator]):
        self.name = name
        self.evaluators = evaluators

    def run(self, cases: list[EvaluationCase]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for case in cases:
            case_metrics: list[EvaluationMetric] = []
            failures: list[str] = []
            metadata = {"evaluators": []}

            for evaluator in self.evaluators:
                result = evaluator.evaluate(case)
                case_metrics.extend(result.metrics)
                failures.extend(result.failures)
                metadata["evaluators"].append(evaluator.name)

            results.append(
                EvaluationResult(
                    suite_name=self.name,
                    case_name=case.name,
                    passed=not failures and all(metric.passed for metric in case_metrics),
                    metrics=case_metrics,
                    failures=failures,
                    metadata=metadata,
                )
            )
        return results
