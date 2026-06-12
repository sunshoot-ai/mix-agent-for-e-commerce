from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage

from models.schemas import RecommendationRequest
from orchestrator.supervisor import SupervisorOrchestrator
from services.features.store import FeatureStore


class SlowFailLLM:
    async def ainvoke(self, messages):
        await asyncio.sleep(0.05)
        raise RuntimeError("LLM should only run in background for fast homepage")


class FastRerankLLM:
    async def ainvoke(self, messages):
        return AIMessage(content='["P001","P002","P003"]')


def test_homepage_fast_path_returns_without_waiting_for_profile_or_copy_llm():
    async def run():
        feature_store = FeatureStore()
        supervisor = SupervisorOrchestrator(feature_store=feature_store)
        supervisor.user_profile_agent.llm = SlowFailLLM()
        supervisor.marketing_copy_agent.llm = SlowFailLLM()
        supervisor.product_rec_agent.llm = FastRerankLLM()

        response = await supervisor.recommend(
            RecommendationRequest(
                user_id="phase1_user",
                scene="homepage",
                num_items=3,
            )
        )

        profile_result = response.agent_results["user_profile"]
        copy_result = response.agent_results["marketing_copy"]

        assert response.products
        assert response.marketing_copies
        assert profile_result.success is True
        assert profile_result.data["mode"] == "rule_fallback"
        assert copy_result.success is True
        assert copy_result.data["mode"] == "template_fallback"
        assert response.total_latency_ms < 1000

    asyncio.run(run())


def test_standard_path_still_uses_llm_when_not_homepage():
    async def run():
        feature_store = FeatureStore()
        supervisor = SupervisorOrchestrator(feature_store=feature_store)
        supervisor.product_rec_agent.llm = FastRerankLLM()

        class ProfileLLM:
            async def ainvoke(self, messages):
                return AIMessage(
                    content='{"segments":["active"],"preferred_categories":["手机"],"price_range":[0,9000],"rfm_score":{},"real_time_tags":{}}'
                )

        class CopyLLM:
            async def ainvoke(self, messages):
                return AIMessage(content='[{"product_id":"P001","copy":"标准LLM文案"}]')

        supervisor.user_profile_agent.llm = ProfileLLM()
        supervisor.marketing_copy_agent.llm = CopyLLM()

        response = await supervisor.recommend(
            RecommendationRequest(
                user_id="phase1_standard_user",
                scene="detail",
                num_items=1,
            )
        )

        assert response.agent_results["user_profile"].data["mode"] == "llm"
        assert response.agent_results["marketing_copy"].data["mode"] == "llm"

    asyncio.run(run())
