from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "simulations"

SIMULATION_ARTIFACTS: dict[str, tuple[str, str]] = {
    "public": ("ctr_simulation_result.json", "ctr_simulation_result_evaluation.json"),
    "feast": (
        "ctr_simulation_feast_result_target15.json",
        "ctr_simulation_feast_result_target15_evaluation.json",
    ),
}


def get_simulation_artifact(kind: str, data_dir: Path | None = None) -> dict[str, Any]:
    if kind not in SIMULATION_ARTIFACTS:
        raise KeyError(f"unknown simulation artifact kind: {kind}")

    root = data_dir or DATA_DIR
    result_name, evaluation_name = SIMULATION_ARTIFACTS[kind]
    result_path = root / result_name
    evaluation_path = root / evaluation_name

    result = _read_json_if_exists(result_path)
    evaluation = _read_json_if_exists(evaluation_path)

    return {
        "kind": kind,
        "available": result is not None and evaluation is not None,
        "result_file": str(result_path),
        "evaluation_file": str(evaluation_path),
        "result": result,
        "evaluation": evaluation,
    }


def get_simulation_artifact_status(data_dir: Path | None = None) -> dict[str, Any]:
    status = {}
    for kind in SIMULATION_ARTIFACTS:
        artifact = get_simulation_artifact(kind, data_dir=data_dir)
        status[kind] = {
            "available": artifact["available"],
            "result_file": artifact["result_file"],
            "evaluation_file": artifact["evaluation_file"],
        }
    return status


def validate_ctr_artifact(
    artifact: dict[str, Any],
    *,
    min_relative_uplift_percent: float = 0.0,
    min_rounds: int = 1,
) -> dict[str, Any]:
    failures: list[str] = []
    result = artifact.get("result")
    evaluation = artifact.get("evaluation")

    if not isinstance(result, dict):
        failures.append("missing result payload")
    else:
        _require_number(result, ["rounds"], failures)
        _require_number(result, ["baseline", "ctr"], failures)
        _require_number(result, ["thompson", "ctr"], failures)
        _require_number(result, ["uplift", "relative_percent"], failures)
        if float(result.get("rounds", 0) or 0) < min_rounds:
            failures.append(f"rounds below required minimum {min_rounds}")
        uplift = float(result.get("uplift", {}).get("relative_percent", 0.0) or 0.0)
        if uplift < min_relative_uplift_percent:
            failures.append(
                f"relative uplift below required minimum {min_relative_uplift_percent}"
            )

    if not isinstance(evaluation, dict):
        failures.append("missing evaluation payload")
    elif evaluation.get("passed") is not True:
        failures.append("evaluation did not pass")

    return {
        "kind": artifact.get("kind", "unknown"),
        "passed": not failures,
        "failures": failures,
    }


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _require_number(payload: dict[str, Any], path: list[str], failures: list[str]) -> None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            failures.append(f"missing numeric field: {'.'.join(path)}")
            return
        current = current[key]
    if not isinstance(current, (int, float)):
        failures.append(f"non-numeric field: {'.'.join(path)}")
