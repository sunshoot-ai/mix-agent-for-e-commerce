from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.experimentation.simulation import run_simulation


def test_flywheel_simulation_updates_features_and_ab_state():
    result = asyncio.run(run_simulation(rounds=2))

    round1 = result["rounds"][0]
    round2 = result["rounds"][1]
    treatment = next(
        group
        for group in result["ab_experiments"]["rec_strategy"]["groups"]
        if group["name"] == "treatment_llm"
    )

    assert len(result["rounds"]) == 2
    assert round1["features_after"]["click_count_1h"] > round1["features_before"]["click_count_1h"]
    assert round2["features_before"] == round1["features_after"]
    assert round2["profile"]["recent_views"]
    assert round2["marketing_copies"]
    assert treatment["successes"] > 1
