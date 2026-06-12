from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage

from models.schemas import RecommendationRequest
from orchestrator.supervisor import SupervisorOrchestrator
from services.operations.inventory_alerts import InventoryAlertService


class NoopLLM:
    async def ainvoke(self, messages):
        return AIMessage(content='["P009","P015","P001"]')


def test_inventory_alert_service_records_warning_and_critical():
    service = InventoryAlertService()
    recorded = service.record_alerts(
        [
            {
                "product_id": "P009",
                "name": "低库存商品",
                "current_stock": 80,
                "level": "warning",
                "action": "plan_restock",
            },
            {
                "product_id": "P015",
                "name": "极低库存商品",
                "current_stock": 50,
                "level": "critical",
                "action": "urgent_restock",
            },
        ],
        request_id="req-1",
        scene="homepage",
    )

    assert len(recorded) == 2
    assert service.snapshot(level="warning")[0]["product_id"] == "P009"
    assert service.snapshot(level="critical")[0]["action"] == "urgent_restock"


def test_supervisor_records_inventory_alerts_without_blocking_response():
    async def run():
        alert_service = InventoryAlertService()
        supervisor = SupervisorOrchestrator(inventory_alert_service=alert_service)
        supervisor.user_profile_agent.llm = NoopLLM()
        supervisor.product_rec_agent.llm = NoopLLM()
        supervisor.marketing_copy_agent.llm = NoopLLM()

        response = await supervisor.recommend(
            RecommendationRequest(
                user_id="phase4_inventory_user",
                scene="homepage",
                num_items=10,
            )
        )
        inventory_result = response.agent_results["inventory"]

        assert response.products
        assert inventory_result.low_stock_alerts
        assert inventory_result.data["inventory_alerts"]["recorded_count"] >= 1
        assert alert_service.snapshot()

    asyncio.run(run())
