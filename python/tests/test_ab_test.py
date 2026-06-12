"""A/B测试引擎单元测试"""

import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from services.experimentation.ab_test import (
    ABTestEngine,
    Experiment,
    ExperimentGroup,
    run_ctr_simulation,
    run_ctr_simulation_from_feast,
    run_ctr_simulation_multi_seed,
)


def test_consistent_assignment():
    """Same user always gets the same group."""
    engine = ABTestEngine()
    group1 = engine.assign("user_001")
    group2 = engine.assign("user_001")
    assert group1["group"] == group2["group"]


def test_distribution():
    """Check rough distribution balance across many users."""
    engine = ABTestEngine()
    counts: dict[str, int] = {}
    for i in range(1000):
        result = engine.assign(f"user_{i}")
        grp = result["group"]
        counts[grp] = counts.get(grp, 0) + 1

    for grp, count in counts.items():
        assert 300 < count < 700, f"Group {grp} has {count} users — too skewed"


def test_thompson_sampling():
    """Thompson sampling updates posterior correctly."""
    engine = ABTestEngine()
    for _ in range(100):
        engine.record_outcome("rec_strategy", "treatment_llm", True)
    for _ in range(100):
        engine.record_outcome("rec_strategy", "control", False)

    exp = engine.experiments["rec_strategy"]
    treatment = next(g for g in exp.groups if g.name == "treatment_llm")
    control = next(g for g in exp.groups if g.name == "control")
    assert treatment.successes > control.successes


def test_custom_experiment():
    engine = ABTestEngine()
    engine.register_experiment(
        Experiment(
            id="prompt_test",
            name="Prompt模板实验",
            groups=[
                ExperimentGroup(name="template_a", weight=30),
                ExperimentGroup(name="template_b", weight=70),
            ],
        )
    )
    result = engine.assign("user_999", "prompt_test")
    assert result["group"] in ("template_a", "template_b")


def test_metrics_recording():
    engine = ABTestEngine()
    engine.record_metric("rec_strategy", "control", "ctr", 0.05, "user_001")
    engine.record_metric("rec_strategy", "control", "ctr", 0.08, "user_002")
    engine.record_metric("rec_strategy", "treatment_llm", "ctr", 0.12, "user_003")

    stats = engine.get_stats("rec_strategy")
    assert "control" in stats
    assert stats["control"]["ctr"]["count"] == 2


def test_segmented_metric_recording():
    engine = ABTestEngine()
    engine.record_metric(
        "rec_strategy",
        "control",
        "ctr",
        0.05,
        "user_001",
        segment="high_value",
        scene="homepage",
        policy="bucket",
        trace_id="trace-1",
    )
    engine.record_metric(
        "rec_strategy",
        "treatment_llm",
        "ctr",
        0.08,
        "user_002",
        segment="high_value",
        scene="homepage",
        policy="thompson",
        trace_id="trace-2",
    )

    segmented = engine.get_segmented_stats("rec_strategy")
    assert segmented["high_value"]["control"]["ctr"]["count"] == 1
    assert segmented["high_value"]["treatment_llm"]["ctr"]["mean"] == 0.08


def test_segmented_thompson_keeps_independent_posteriors():
    engine = ABTestEngine()
    engine.record_segment_outcome("rec_strategy", "treatment_llm", True, "homepage")
    engine.record_segment_outcome("rec_strategy", "control", False, "detail")

    posteriors = engine.get_segment_posteriors("rec_strategy")

    assert posteriors["homepage"]["treatment_llm"]["successes"] == 2
    assert posteriors["detail"]["control"]["failures"] == 2
    assert engine.experiments["rec_strategy"].groups[0].successes == 1


def test_segmented_ctr_simulation_outputs_segments():
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "event_type": ["click", "view", "view", "click"],
        }
    )

    result = run_ctr_simulation(events, rounds=200, seed=1, segmented=True)

    assert result["segmented"] is True
    assert result["segments"]
    assert result["posterior"]


def test_multi_seed_ctr_simulation_summary():
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "event_type": ["click", "view", "view", "click"],
        }
    )

    result = run_ctr_simulation_multi_seed(events, seeds=[1, 2, 3], rounds=100)

    assert result["seed_count"] == 3
    assert "mean_relative_uplift_percent" in result["multi_seed"]
    assert len(result["per_seed"]) == 3


def test_feast_segmented_ctr_simulation_outputs_segments():
    import services.features.store as feature_store_module

    original_store = feature_store_module.FeastFeatureStore

    class FakeFeastFeatureStore:
        async def get_user_features(self, user_id: str):
            return {
                "view_count_1h": 1,
                "view_count_24h": 3,
                "view_count_7d": 5,
                "avg_order_amount": 1200,
                "rfm_score": 1500,
            }

    feature_store_module.FeastFeatureStore = FakeFeastFeatureStore
    try:
        result = asyncio.run(
            run_ctr_simulation_from_feast(
                user_ids=["1", "2", "3"],
                rounds=200,
                seed=1,
                segmented=True,
            )
        )
    finally:
        feature_store_module.FeastFeatureStore = original_store

    assert result["segmented"] is True
    assert result["segments"]
    assert "active_viewer" in result["posterior"]


if __name__ == "__main__":
    test_consistent_assignment()
    test_distribution()
    test_thompson_sampling()
    test_custom_experiment()
    test_metrics_recording()
    test_segmented_metric_recording()
    test_segmented_thompson_keeps_independent_posteriors()
    test_segmented_ctr_simulation_outputs_segments()
    test_multi_seed_ctr_simulation_summary()
    test_feast_segmented_ctr_simulation_outputs_segments()
    print("All A/B test engine tests passed!")
