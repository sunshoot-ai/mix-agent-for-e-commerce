from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import BehaviorEventType, FeedbackRequest
from services.experimentation.ab_test import ABTestEngine
from services.features.store import FeatureStore
from services.experimentation.feedback import FeedbackService
from services.operations.metrics import MetricsCollector


def test_feature_store_records_behavior_without_redis():
    async def run():
        store = FeatureStore()
        await store.record_behavior(
            "u1",
            "click",
            "P001",
            {"category": "手机", "amount": 199.0},
        )

        features = await store.get_user_features("u1")

        assert features["click_count_1h"] == 1
        assert features["recent_views"] == ["P001"]
        assert features["preferred_categories"] == ["手机"]

    asyncio.run(run())


def test_feedback_service_is_idempotent_and_updates_success_outcome():
    async def run():
        store = FeatureStore()
        ab_engine = ABTestEngine()
        metrics = MetricsCollector()
        service = FeedbackService(store, ab_engine, metrics)

        event = FeedbackRequest(
            user_id="u1",
            request_id="req-1",
            product_id="P001",
            event_type=BehaviorEventType.CLICK,
            experiment_id="rec_strategy",
            experiment_group="treatment_llm",
            metadata={"category": "手机"},
        )

        first = await service.record_event(event)
        duplicate = await service.record_event(event)

        exp = ab_engine.experiments["rec_strategy"]
        treatment = next(g for g in exp.groups if g.name == "treatment_llm")
        features = await store.get_user_features("u1")

        assert first.status == "recorded"
        assert duplicate.duplicate is True
        assert treatment.successes == 2
        assert features["click_count_1h"] == 1
        assert metrics.get_business_stats()["click"]["count"] == 1

    asyncio.run(run())


def test_feedback_service_ignores_non_positive_ab_outcome():
    async def run():
        ab_engine = ABTestEngine()
        service = FeedbackService(FeatureStore(), ab_engine, MetricsCollector())
        event = FeedbackRequest(
            user_id="u1",
            request_id="req-2",
            product_id="P002",
            event_type=BehaviorEventType.IMPRESSION,
            experiment_id="rec_strategy",
            experiment_group="control",
        )

        await service.record_event(event)

        exp = ab_engine.experiments["rec_strategy"]
        control = next(g for g in exp.groups if g.name == "control")
        assert control.successes == 1
        assert control.failures == 1

    asyncio.run(run())
