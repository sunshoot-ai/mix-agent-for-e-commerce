"""
Supervisor编排器 — 并行分发 + 聚合模式

                    ┌──────────────┐
                    │  Supervisor   │
                    └──────┬───────┘
           ┌───────┬───────┼───────┬────────┐
           ▼       ▼       ▼       ▼        │
      UserProfile  ProdRec  MktCopy  Inventory │
           │       │       │       │        │
           └───────┴───────┴───────┘        │
                    │                        │
                    ▼                        │
               Aggregator ◄─────────────────┘
                    │
                    ▼
              A/B Test Engine
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import structlog

from agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from models.schemas import (
    Product,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
)
from services.features.store import FeatureStore
from services.experimentation.ab_test import ABTestEngine
from services.operations.inventory_alerts import InventoryAlertService
from services.observability import (
    EventStatus,
    SpanKind,
    current_trace,
    default_registry,
    emit_event,
    tracer,
)
from services.user_profile.segment_selector import get_primary_segment

logger = structlog.get_logger()


class SupervisorOrchestrator:
    """Coordinates four agents in parallel-then-aggregate pattern."""

    def __init__(
        self,
        ab_engine: ABTestEngine | None = None,
        feature_store: FeatureStore | None = None,
        inventory_alert_service: InventoryAlertService | None = None,
    ):
        self.feature_store = feature_store or FeatureStore()
        self.user_profile_agent = UserProfileAgent(feature_store=self.feature_store)
        self.product_rec_agent = ProductRecAgent()
        self.marketing_copy_agent = MarketingCopyAgent(cache_store=self.feature_store)
        self.inventory_agent = InventoryAgent()
        self.ab_engine = ab_engine or ABTestEngine()
        self.inventory_alert_service = inventory_alert_service or InventoryAlertService()

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        async with tracer.async_trace(
            request_id=request_id,
            workflow="supervisor_recommendation",
            user_id=request.user_id,
            scene=request.scene,
        ) as trace_ctx:
            async with tracer.async_span(
                "workflow.supervisor.recommend",
                kind=SpanKind.WORKFLOW,
                scene=request.scene,
                num_items=request.num_items,
            ) as workflow_span:
                logger.info(
                    "supervisor.start",
                    request_id=request_id,
                    user_id=request.user_id,
                    scene=request.scene,
                )
                default_registry.inc(
                    "workflow_requests_total",
                    workflow="supervisor_recommendation",
                    scene=request.scene,
                    status="started",
                )

                experiment = self.ab_engine.assign(request.user_id)
                trace_ctx.experiment_id = "rec_strategy"
                trace_ctx.experiment_group = experiment.get("group", "control")
                fast_homepage = request.scene == "homepage"

                self._record_transition("start", "phase1_parallel")
                async with tracer.async_span("workflow.phase1_parallel", kind=SpanKind.WORKFLOW):
                    profile_result, rec_result = await asyncio.gather(
                        self.user_profile_agent.run(
                            user_id=request.user_id,
                            context=request.context,
                            fast_mode=fast_homepage,
                        ),
                        self.product_rec_agent.run(
                            user_profile=None,
                            num_items=request.num_items * 2,
                        ),
                    )

                user_profile: UserProfile | None = getattr(profile_result, "profile", None)
                raw_products: list[Product] = getattr(rec_result, "products", [])
                profile_segment = self._primary_segment(user_profile)
                copy_experiment = self.ab_engine.assign_thompson_segmented(
                    request.user_id,
                    experiment_id="copy_style",
                    segment=profile_segment,
                )

                self._record_transition("phase1_parallel", "phase2_parallel")
                async with tracer.async_span(
                    "workflow.phase2_parallel",
                    kind=SpanKind.WORKFLOW,
                    raw_product_count=len(raw_products),
                ):
                    rerank_task = self.product_rec_agent.run(
                        user_profile=user_profile,
                        num_items=request.num_items,
                    )
                    inventory_task = self.inventory_agent.run(products=raw_products)
                    rerank_result, inventory_result = await asyncio.gather(
                        rerank_task, inventory_task
                    )
                self._record_inventory_alerts(
                    inventory_result,
                    request_id=request_id,
                    scene=request.scene,
                )

                ranked_products: list[Product] = getattr(rerank_result, "products", raw_products)

                self._record_transition("phase2_parallel", "filter")
                async with tracer.async_span(
                    "workflow.filter_inventory",
                    kind=SpanKind.WORKFLOW,
                    ranked_product_count=len(ranked_products),
                ):
                    available_ids = set(getattr(inventory_result, "available_products", []))
                    final_products = [p for p in ranked_products if p.product_id in available_ids]
                    if not final_products:
                        final_products = ranked_products[:request.num_items]
                    final_products = final_products[:request.num_items]

                self._record_transition("filter", "marketing_copy")
                async with tracer.async_span(
                    "workflow.marketing_copy",
                    kind=SpanKind.WORKFLOW,
                    final_product_count=len(final_products),
                ):
                    copy_result = await self.marketing_copy_agent.run(
                        user_profile=user_profile,
                        products=final_products,
                        fast_mode=fast_homepage,
                        copy_style=copy_experiment.get("config", {}).get(
                            "style",
                            copy_experiment.get("group", ""),
                        ),
                    )
                copies = getattr(copy_result, "copies", [])
                copy_result.data.setdefault("experiments", {})
                copy_result.data["experiments"]["copy_style"] = {
                    "group": copy_experiment.get("group", "control"),
                    "config": copy_experiment.get("config", {}),
                    "segment": profile_segment,
                    "policy": "thompson_segmented",
                }

                total_latency = (time.perf_counter() - start) * 1000
                agent_results = {
                    "user_profile": profile_result,
                    "product_rec": rerank_result,
                    "marketing_copy": copy_result,
                    "inventory": inventory_result,
                }
                fallback_count = self._record_agent_summaries(agent_results)

                workflow_span.attributes.update(
                    {
                        "total_latency_ms": total_latency,
                        "product_count": len(final_products),
                        "copy_count": len(copies),
                        "fallback_count": fallback_count,
                        "experiment_group": experiment.get("group", "control"),
                        "copy_experiment_group": copy_experiment.get("group", "control"),
                        "performance_mode": "fast_homepage" if fast_homepage else "standard",
                    }
                )
                default_registry.observe(
                    "workflow_total_latency_ms",
                    total_latency,
                    workflow="supervisor_recommendation",
                    scene=request.scene,
                    experiment_group=experiment.get("group", "control"),
                )
                default_registry.inc(
                    "workflow_requests_total",
                    workflow="supervisor_recommendation",
                    scene=request.scene,
                    status="completed",
                )

                logger.info(
                    "supervisor.complete",
                    request_id=request_id,
                    total_latency_ms=round(total_latency, 1),
                    product_count=len(final_products),
                    copy_count=len(copies),
                    fallback_count=fallback_count,
                )

                return RecommendationResponse(
                    request_id=request_id,
                    user_id=request.user_id,
                    products=final_products,
                    marketing_copies=copies,
                    experiment_group=experiment.get("group", "control"),
                    agent_results=agent_results,
                    total_latency_ms=total_latency,
                )

    def _record_transition(self, from_node: str, to_node: str) -> None:
        ctx = current_trace()
        if not ctx:
            return
        default_registry.inc(
            "agent_transition_total",
            workflow=ctx.workflow,
            from_agent=from_node,
            to_agent=to_node,
        )
        emit_event(
            "workflow.transition",
            trace_id=ctx.trace_id,
            request_id=ctx.request_id,
            component=ctx.workflow,
            component_type="workflow",
            status=EventStatus.OK,
            user_id=ctx.user_id,
            scene=ctx.scene,
            experiment_id=ctx.experiment_id,
            experiment_group=ctx.experiment_group,
            attributes={"from": from_node, "to": to_node},
        )

    def _record_agent_summaries(self, agent_results: dict[str, Any]) -> int:
        fallback_count = 0
        for agent_name, result in agent_results.items():
            success = bool(getattr(result, "success", False))
            if not success:
                fallback_count += 1
            default_registry.inc(
                "workflow_agent_results_total",
                agent=agent_name,
                status="success" if success else "fallback",
            )
        return fallback_count

    def _record_inventory_alerts(
        self,
        inventory_result: Any,
        *,
        request_id: str,
        scene: str,
    ) -> None:
        low_stock_alerts = getattr(inventory_result, "low_stock_alerts", []) or []
        if not low_stock_alerts:
            return
        recorded = self.inventory_alert_service.record_alerts(
            low_stock_alerts,
            request_id=request_id,
            scene=scene,
        )
        inventory_result.data.setdefault("inventory_alerts", {})
        inventory_result.data["inventory_alerts"] = {
            "recorded_count": len(recorded),
            "events": recorded,
        }

    @staticmethod
    def _primary_segment(profile: UserProfile | None) -> str:
        if not profile or not profile.segments:
            return "unknown"
        return get_primary_segment(profile.segments)
