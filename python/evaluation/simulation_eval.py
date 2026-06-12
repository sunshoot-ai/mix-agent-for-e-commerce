from __future__ import annotations

from typing import Any

from .schemas import EvaluationMetric, EvaluationResult


class CtrSimulationEvaluator:
    name = "ctr_simulation"

    def __init__(
        self,
        *,
        min_relative_uplift_percent: float = 0.0,
        min_rounds: int = 1,
        require_positive_ci: bool = False,
    ):
        self.min_relative_uplift_percent = min_relative_uplift_percent
        self.min_rounds = min_rounds
        self.require_positive_ci = require_positive_ci

    def evaluate_result(self, result: dict[str, Any], case_name: str = "ctr_simulation") -> EvaluationResult:
        rounds = int(result.get("rounds", 0) or 0)
        baseline_ctr = float(result.get("baseline", {}).get("ctr", 0.0) or 0.0)
        thompson_ctr = float(result.get("thompson", {}).get("ctr", 0.0) or 0.0)
        relative_uplift = float(result.get("uplift", {}).get("relative_percent", 0.0) or 0.0)
        ci = result.get("uplift", {}).get("ctr_diff_95ci", [0.0, 0.0])
        ci_low = float(ci[0]) if isinstance(ci, list) and ci else 0.0
        multi_seed = result.get("multi_seed", {})
        seed_count = int(result.get("seed_count", 0) or 0)
        mean_uplift = float(multi_seed.get("mean_relative_uplift_percent", relative_uplift) or 0.0)

        metrics = [
            EvaluationMetric("simulation_rounds", float(rounds), rounds >= self.min_rounds, self.min_rounds),
            EvaluationMetric("baseline_ctr", baseline_ctr, True),
            EvaluationMetric("thompson_ctr", thompson_ctr, thompson_ctr >= 0.0),
            EvaluationMetric(
                "relative_uplift_percent",
                relative_uplift,
                relative_uplift >= self.min_relative_uplift_percent,
                self.min_relative_uplift_percent,
            ),
            EvaluationMetric(
                "ctr_diff_ci_low",
                ci_low,
                (ci_low > 0.0) if self.require_positive_ci else True,
                0.0 if self.require_positive_ci else None,
            ),
        ]
        if multi_seed:
            metrics.extend([
                EvaluationMetric("seed_count", float(seed_count), seed_count > 1, 2.0),
                EvaluationMetric(
                    "mean_relative_uplift_percent",
                    mean_uplift,
                    mean_uplift >= self.min_relative_uplift_percent,
                    self.min_relative_uplift_percent,
                ),
            ])

        failures: list[str] = []
        if rounds < self.min_rounds:
            failures.append(f"rounds {rounds} below required {self.min_rounds}")
        if relative_uplift < self.min_relative_uplift_percent:
            failures.append(
                f"relative uplift {relative_uplift:.4f}% below "
                f"{self.min_relative_uplift_percent:.4f}%"
            )
        if self.require_positive_ci and ci_low <= 0.0:
            failures.append(f"CTR uplift confidence interval lower bound {ci_low:.6f} is not positive")
        if multi_seed and mean_uplift < self.min_relative_uplift_percent:
            failures.append(
                f"mean relative uplift {mean_uplift:.4f}% below "
                f"{self.min_relative_uplift_percent:.4f}%"
            )

        return EvaluationResult(
            suite_name=self.name,
            case_name=case_name,
            passed=not failures and all(metric.passed for metric in metrics),
            metrics=metrics,
            failures=failures,
            metadata={
                "posterior": result.get("posterior", {}),
                "source": result.get("source", "public_dataset"),
                "multi_seed": multi_seed,
                "segments": result.get("segments", {}),
            },
        )


def evaluate_ctr_simulation_result(
    result: dict[str, Any],
    *,
    case_name: str = "ctr_simulation",
    min_relative_uplift_percent: float = 0.0,
    min_rounds: int = 1,
    require_positive_ci: bool = False,
) -> dict[str, Any]:
    evaluator = CtrSimulationEvaluator(
        min_relative_uplift_percent=min_relative_uplift_percent,
        min_rounds=min_rounds,
        require_positive_ci=require_positive_ci,
    )
    return evaluator.evaluate_result(result, case_name=case_name).to_dict()
