from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.marketing_copy_agent import MarketingCopyAgent
from models.schemas import BehaviorEventType, FeedbackRequest, Product, UserProfile, UserSegment
from services.experimentation.ab_test import ABTestEngine
from services.features.store import FeatureStore
from services.experimentation.feedback import FeedbackService
from services.operations.metrics import MetricsCollector
from services.experimentation.simulation import run_simulation


def test_feedback_records_copy_style_independently_from_rec_strategy():
    async def run():
        ab_engine = ABTestEngine()
        service = FeedbackService(FeatureStore(), ab_engine, MetricsCollector())
        common = {
            "user_id": "u1",
            "request_id": "req1",
            "product_id": "P001",
            "event_type": BehaviorEventType.CLICK,
            "metadata": {"segment": "new_user"},
        }

        await service.record_event(
            FeedbackRequest(
                **common,
                experiment_id="rec_strategy",
                experiment_group="treatment_llm",
            )
        )
        await service.record_event(
            FeedbackRequest(
                **common,
                experiment_id="copy_style",
                experiment_group="casual",
            )
        )

        rec = ab_engine.experiments["rec_strategy"]
        copy = ab_engine.experiments["copy_style"]
        treatment = next(group for group in rec.groups if group.name == "treatment_llm")
        casual = next(group for group in copy.groups if group.name == "casual")

        assert treatment.successes == 2
        assert casual.successes == 2
        assert "new_user" in ab_engine.get_segment_posteriors("copy_style")
        assert ab_engine.get_stats("copy_style")["casual"]["click"]["count"] == 1

    asyncio.run(run())


def test_marketing_copy_cache_is_isolated_by_copy_style():
    async def run():
        store = FeatureStore()
        agent = MarketingCopyAgent(cache_store=store)
        profile = UserProfile(user_id="u1", segments=[UserSegment.ACTIVE])
        products = [
            Product(product_id="P001", name="测试商品", category="手机", price=1000),
        ]

        formal = await agent.run(
            user_profile=profile,
            products=products,
            fast_mode=True,
            copy_style="formal",
        )
        casual = await agent.run(
            user_profile=profile,
            products=products,
            fast_mode=True,
            copy_style="casual",
        )

        assert formal.copies[0]["copy"].startswith("品质推荐")
        assert casual.copies[0]["copy"].startswith("你可能会喜欢")

    asyncio.run(run())


def test_scaled_simulation_outputs_copy_style_segmented_metrics():
    result = asyncio.run(
        run_simulation(
            rounds=2,
            user_count=3,
            impressions_per_round=2,
            click_probability=1.0,
            purchase_probability=1.0,
            seed=1,
        )
    )

    assert result["simulation_config"]["user_count"] == 3
    assert "copy_style" in result["ab_experiments"]
    assert result["ab_experiments"]["copy_style"]["stats"]
    assert result["segmented_experiments"]["copy_style"]["posteriors"]
    assert result["rounds"][0]["copy_experiment"]["policy"] == "thompson_segmented"
    assert result["dedup_business_report"]["impression"] < result["business_metrics"]["impression"]["count"]
    assert "ctr" in result["dedup_business_report"]
    assert "cvr" in result["dedup_business_report"]
    assert result["business_metrics"]["impression"]["count"] >= 12
