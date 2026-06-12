from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation.artifacts import get_simulation_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Render latest CTR simulation report.")
    parser.add_argument(
        "--output",
        default="../docs/simulation_report.md",
        help="Markdown report path, relative to the python directory by default.",
    )
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(), encoding="utf-8")
    print("report_file", output_path)


def render_report() -> str:
    lines = [
        "# CTR Simulation Report",
        "",
        "| Source | Available | Single uplift % | Multi-seed mean % | Seed count | Evaluation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for kind in ("public", "feast"):
        artifact = get_simulation_artifact(kind)
        result = artifact.get("result") or {}
        evaluation = artifact.get("evaluation") or {}
        lines.append(
            "| {kind} | {available} | {uplift} | {mean_uplift} | {seed_count} | {passed} |".format(
                kind=kind,
                available=str(artifact["available"]).lower(),
                uplift=_fmt(result.get("uplift", {}).get("relative_percent")),
                mean_uplift=_fmt(
                    result.get("multi_seed", {}).get("mean_relative_uplift_percent")
                ),
                seed_count=result.get("seed_count", ""),
                passed=evaluation.get("passed", ""),
            )
        )

    lines.extend(["", "## Segments", ""])
    for kind in ("public", "feast"):
        artifact = get_simulation_artifact(kind)
        result = artifact.get("result") or {}
        segments: dict[str, Any] = result.get("segments", {})
        lines.append(f"### {kind}")
        if not segments:
            lines.append("")
            lines.append("No segment metrics available.")
            lines.append("")
            continue
        lines.extend([
            "",
            "| Segment | Baseline CTR | Thompson CTR | Uplift % |",
            "| --- | ---: | ---: | ---: |",
        ])
        for segment, values in sorted(segments.items()):
            lines.append(
                "| {segment} | {baseline} | {thompson} | {uplift} |".format(
                    segment=segment,
                    baseline=_fmt(values.get("baseline", {}).get("ctr")),
                    thompson=_fmt(values.get("thompson", {}).get("ctr")),
                    uplift=_fmt(values.get("uplift", {}).get("relative_percent")),
                )
            )
        lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""


if __name__ == "__main__":
    main()
