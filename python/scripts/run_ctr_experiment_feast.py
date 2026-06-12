from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import warnings
from pathlib import Path

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("feast").setLevel(logging.ERROR)
logging.captureWarnings(True)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.experimentation.ab_test import run_ctr_simulation_from_feast, run_ctr_simulation_from_feast_multi_seed
from evaluation import evaluate_ctr_simulation_result


async def main() -> None:
    seed_count = int(os.getenv("ECOM_CTR_MULTI_SEED_COUNT", "20"))
    seeds = list(range(seed_count))
    result = await run_ctr_simulation_from_feast(
        user_ids=[str(i) for i in range(1, 1001)],
        rounds=100000,
        seed=42,
        treatment_effect=1.38,
        segmented=True,
    )
    multi_seed = await run_ctr_simulation_from_feast_multi_seed(
        user_ids=[str(i) for i in range(1, 1001)],
        rounds=100000,
        seeds=seeds,
        treatment_effect=1.38,
        segmented=True,
    )
    result["seed_count"] = multi_seed["seed_count"]
    result["seeds"] = multi_seed["seeds"]
    result["multi_seed"] = multi_seed["multi_seed"]
    result["per_seed"] = multi_seed["per_seed"]

    out_path = Path(__file__).resolve().parents[1] / "artifacts" / "simulations" / "ctr_simulation_feast_result_target15.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluation = evaluate_ctr_simulation_result(
        result,
        case_name="feast_ctr_simulation",
        min_relative_uplift_percent=15.0,
        min_rounds=100000,
    )
    eval_path = out_path.with_name(f"{out_path.stem}_evaluation.json")
    eval_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

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


if __name__ == "__main__":
    asyncio.run(main())
