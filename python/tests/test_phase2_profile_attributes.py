from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage

from agents.user_profile_agent import UserProfileAgent
from models.schemas import BehaviorEventType, FeedbackRequest, UserProfile
from services.experimentation.ab_test import ABTestEngine
from services.features.store import FeatureStore
from services.experimentation.feedback import FeedbackService
from services.operations.metrics import MetricsCollector


class EmptyAttrsProfileLLM:
    async def ainvoke(self, messages):
        return AIMessage(
            content='{"segments":["active"],"preferred_categories":["手机"],"price_range":[0,9000],"rfm_score":{},"real_time_tags":{}}'
        )


def test_context_attrs_are_preserved_when_llm_omits_them():
    async def run():
        store = FeatureStore()
        agent = UserProfileAgent(feature_store=store)
        agent.llm = EmptyAttrsProfileLLM()

        result = await agent.run(
            user_id="phase2_context_user",
            context={"age": 28, "gender": "female", "city": "Shanghai"},
        )

        assert result.profile.age == 28
        assert result.profile.gender == "female"
        assert result.profile.city == "Shanghai"

    asyncio.run(run())


def test_feedback_metadata_attrs_feed_next_rule_profile():
    async def run():
        store = FeatureStore()
        feedback = FeedbackService(store, ABTestEngine(), MetricsCollector())
        await feedback.record_event(
            FeedbackRequest(
                user_id="phase2_event_user",
                request_id="req-attrs",
                product_id="P001",
                event_type=BehaviorEventType.CLICK,
                metadata={
                    "age": 31,
                    "gender": "male",
                    "city": "Beijing",
                    "category": "手机",
                },
            )
        )

        agent = UserProfileAgent(feature_store=store)
        agent.llm = EmptyAttrsProfileLLM()
        result = await agent.run(user_id="phase2_event_user", fast_mode=True)

        assert result.profile.age == 31
        assert result.profile.gender == "male"
        assert result.profile.city == "Beijing"
        assert result.profile.preferred_categories == ["手机"]

    asyncio.run(run())


def test_cached_profile_gets_missing_attrs_from_latest_context():
    async def run():
        store = FeatureStore()
        await store.cache_set(
            "user_profile:phase2_cached_user",
            UserProfile(user_id="phase2_cached_user").model_dump(mode="json"),
        )
        agent = UserProfileAgent(feature_store=store)
        agent.llm = EmptyAttrsProfileLLM()

        result = await agent.run(
            user_id="phase2_cached_user",
            context={"age": 35, "gender": "female", "city": "Shenzhen"},
            fast_mode=True,
        )
        cached = await store.cache_get("user_profile:phase2_cached_user")

        assert result.profile.age == 35
        assert cached["city"] == "Shenzhen"

    asyncio.run(run())
