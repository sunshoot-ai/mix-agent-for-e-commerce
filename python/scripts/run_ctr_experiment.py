from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.experimentation.ab_test import (
    load_public_ecommerce_dataset,
    run_ctr_simulation,
    run_ctr_simulation_multi_seed,
)
from evaluation import evaluate_ctr_simulation_result, validate_ctr_artifact


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "artifacts" / "simulations" / "public_ecommerce_events_sample.csv"
    events = load_public_ecommerce_dataset(str(data_path))
    seeds = list(range(20))
    result = run_ctr_simulation(events=events, rounds=20000, seed=42, segmented=True)
    multi_seed = run_ctr_simulation_multi_seed(
        events=events,
        rounds=20000,
        seeds=seeds,
        segmented=True,
    )
    result["seed_count"] = multi_seed["seed_count"]
    result["seeds"] = multi_seed["seeds"]
    result["multi_seed"] = multi_seed["multi_seed"]
    result["per_seed"] = multi_seed["per_seed"]

    out_path = Path(__file__).resolve().parents[1] / "artifacts" / "simulations" / "ctr_simulation_result.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluation = evaluate_ctr_simulation_result(
        result,
        case_name="public_ctr_simulation",
        min_relative_uplift_percent=0.0,
        min_rounds=20000,
    )
    eval_path = out_path.with_name(f"{out_path.stem}_evaluation.json")
    eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    guardrail = validate_ctr_artifact(
        {"kind": "public", "result": result, "evaluation": evaluation},
        min_relative_uplift_percent=0.0,
        min_rounds=20000,
    )

    print("dataset_rows", len(events))
    print("baseline_ctr", round(result["baseline"]["ctr"], 6))
    print("thompson_ctr", round(result["thompson"]["ctr"], 6))
    print("uplift_percent", round(result["uplift"]["relative_percent"], 4))
    print(
        "mean_uplift_percent",
        round(result["multi_seed"]["mean_relative_uplift_percent"], 4),
    )
    print("result_file", out_path)
    print("evaluation_passed", evaluation["passed"])
    print("evaluation_file", eval_path)
    print("artifact_guardrail_passed", guardrail["passed"])
    if not guardrail["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
