from __future__ import annotations

import argparse
import sys

from evaluation.artifacts import get_simulation_artifact, validate_ctr_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CTR simulation artifacts.")
    parser.add_argument("--kind", choices=["public", "feast"], default="public")
    parser.add_argument("--min-relative-uplift-percent", type=float, default=0.0)
    parser.add_argument("--min-rounds", type=int, default=1)
    args = parser.parse_args()

    artifact = get_simulation_artifact(args.kind)
    result = validate_ctr_artifact(
        artifact,
        min_relative_uplift_percent=args.min_relative_uplift_percent,
        min_rounds=args.min_rounds,
    )

    print("artifact_kind", result["kind"])
    print("artifact_guardrail_passed", result["passed"])
    if result["failures"]:
        for failure in result["failures"]:
            print("failure", failure)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
