"""
LangGraph state graph for the multi-agent recommendation pipeline.

Visualises the DAG of agent execution:

  [start] -> fan_out -> {user_profile, product_recall}  (parallel)
          -> merge_phase1 -> {rerank, inventory}         (parallel)
          -> merge_phase2 -> marketing_copy
          -> aggregate -> [end]
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    InventoryAgent,
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from models.schemas import Product, UserProfile
from services.experimentation.ab_test import ABTestEngine
from services.observability import (
    EventStatus,
    SpanKind,
    current_trace,
    default_registry,
    emit_event,
    tracer,
)


class PipelineState(TypedDict, total=False):
    request_id: str
    trace_id: str
    user_id: str
    scene: str
    num_items: int
    context: dict[str, Any]
    experiment_group: str

    user_profile: UserProfile | None
    raw_products: list[Product]
    ranked_products: list[Product]
    available_ids: set[str]
    final_products: list[Product]
    marketing_copies: list[dict[str, str]]

    agent_results: dict[str, Any]
    total_latency_ms: float
    _start_time: float


user_profile_agent = UserProfileAgent()
product_rec_agent = ProductRecAgent()
marketing_copy_agent = MarketingCopyAgent()
inventory_agent = InventoryAgent()
ab_engine = ABTestEngine()


async def init_node(state: PipelineState) -> PipelineState:
    state["request_id"] = str(uuid.uuid4())
    state["_start_time"] = time.perf_counter()
    state["agent_results"] = {}
    trace_ctx = tracer.start_trace(
        request_id=state["request_id"],
        workflow="langgraph_recommendation",
        user_id=state["user_id"],
        scene=state.get("scene", ""),
    )
    state["trace_id"] = trace_ctx.trace_id
    async with tracer.async_span("graph.init", kind=SpanKind.WORKFLOW):
        exp = ab_engine.assign(state["user_id"])
        state["experiment_group"] = exp.get("group", "control")
        trace_ctx.experiment_id = "rec_strategy"
        trace_ctx.experiment_group = state["experiment_group"]
        default_registry.inc(
            "workflow_requests_total",
            workflow="langgraph_recommendation",
            scene=state.get("scene", ""),
            status="started",
        )
    return state


async def user_profile_node(state: PipelineState) -> PipelineState:
    async with tracer.async_span("graph.user_profile", kind=SpanKind.WORKFLOW):
        result = await user_profile_agent.run(
            user_id=state["user_id"],
            context=state.get("context", {}),
        )
        state["user_profile"] = getattr(result, "profile", None)
        state["agent_results"]["user_profile"] = result
    return state


async def product_recall_node(state: PipelineState) -> PipelineState:
    async with tracer.async_span("graph.product_recall", kind=SpanKind.WORKFLOW):
        result = await product_rec_agent.run(
            user_profile=None,
            num_items=state.get("num_items", 10) * 2,
        )
        state["raw_products"] = getattr(result, "products", [])
        state["agent_results"]["product_recall"] = result
    return state


async def parallel_phase1(state: PipelineState) -> PipelineState:
    """Run user_profile and product_recall in parallel."""
    _record_graph_transition(state, "init", "parallel_phase1")
    async with tracer.async_span("graph.parallel_phase1", kind=SpanKind.WORKFLOW):
        profile_state, recall_state = await asyncio.gather(
            user_profile_node(dict(state)),
            product_recall_node(dict(state)),
        )
        state.update(profile_state)
        state.update(recall_state)
    return state


async def rerank_node(state: PipelineState) -> PipelineState:
    async with tracer.async_span("graph.rerank", kind=SpanKind.WORKFLOW):
        result = await product_rec_agent.run(
            user_profile=state.get("user_profile"),
            num_items=state.get("num_items", 10),
        )
        state["ranked_products"] = getattr(result, "products", state.get("raw_products", []))
        state["agent_results"]["rerank"] = result
    return state


async def inventory_node(state: PipelineState) -> PipelineState:
    async with tracer.async_span("graph.inventory", kind=SpanKind.WORKFLOW):
        result = await inventory_agent.run(
            products=state.get("raw_products", []),
        )
        state["available_ids"] = set(getattr(result, "available_products", []))
        state["agent_results"]["inventory"] = result
    return state


async def parallel_phase2(state: PipelineState) -> PipelineState:
    """Run rerank and inventory in parallel."""
    _record_graph_transition(state, "parallel_phase1", "parallel_phase2")
    async with tracer.async_span("graph.parallel_phase2", kind=SpanKind.WORKFLOW):
        rerank_state, inv_state = await asyncio.gather(
            rerank_node(dict(state)),
            inventory_node(dict(state)),
        )
        state.update(rerank_state)
        state.update(inv_state)
    return state


async def filter_node(state: PipelineState) -> PipelineState:
    _record_graph_transition(state, "parallel_phase2", "filter")
    async with tracer.async_span("graph.filter", kind=SpanKind.WORKFLOW):
        ranked = state.get("ranked_products", [])
        avail = state.get("available_ids", set())
        num = state.get("num_items", 10)
        final = [p for p in ranked if p.product_id in avail]
        if not final:
            final = ranked
        state["final_products"] = final[:num]
    return state


async def marketing_copy_node(state: PipelineState) -> PipelineState:
    _record_graph_transition(state, "filter", "marketing_copy")
    async with tracer.async_span("graph.marketing_copy", kind=SpanKind.WORKFLOW):
        result = await marketing_copy_agent.run(
            user_profile=state.get("user_profile"),
            products=state.get("final_products", []),
        )
        state["marketing_copies"] = getattr(result, "copies", [])
        state["agent_results"]["marketing_copy"] = result
    return state


async def aggregate_node(state: PipelineState) -> PipelineState:
    _record_graph_transition(state, "marketing_copy", "aggregate")
    fallback_count = 0
    async with tracer.async_span("graph.aggregate", kind=SpanKind.WORKFLOW):
        state["total_latency_ms"] = (time.perf_counter() - state.get("_start_time", 0)) * 1000
        fallback_count = sum(
            1 for result in state.get("agent_results", {}).values()
            if not bool(getattr(result, "success", False))
        )
        default_registry.observe(
            "workflow_total_latency_ms",
            state["total_latency_ms"],
            workflow="langgraph_recommendation",
            scene=state.get("scene", ""),
            experiment_group=state.get("experiment_group", ""),
        )
        default_registry.inc(
            "workflow_requests_total",
            workflow="langgraph_recommendation",
            scene=state.get("scene", ""),
            status="completed",
        )
    current = current_trace()
    if current:
        emit_event(
            "workflow.complete",
            trace_id=current.trace_id,
            request_id=current.request_id,
            component=current.workflow,
            component_type="workflow",
            status=EventStatus.OK,
            latency_ms=state["total_latency_ms"],
            user_id=current.user_id,
            scene=current.scene,
            experiment_id=current.experiment_id,
            experiment_group=current.experiment_group,
            attributes={"fallback_count": fallback_count},
        )
        tracer.end_trace(status=EventStatus.END, fallback_count=fallback_count)
        tracer.clear_trace()
    return state


def _record_graph_transition(state: PipelineState, from_node: str, to_node: str) -> None:
    current = current_trace()
    trace_id = current.trace_id if current else state.get("trace_id", "")
    if not trace_id:
        return
    default_registry.inc(
        "agent_transition_total",
        workflow="langgraph_recommendation",
        from_agent=from_node,
        to_agent=to_node,
    )
    emit_event(
        "workflow.transition",
        trace_id=trace_id,
        request_id=state.get("request_id", ""),
        component="langgraph_recommendation",
        component_type="workflow",
        status=EventStatus.OK,
        user_id=state.get("user_id", ""),
        scene=state.get("scene", ""),
        experiment_id="rec_strategy",
        experiment_group=state.get("experiment_group", ""),
        attributes={"from": from_node, "to": to_node},
    )


def build_recommendation_graph() -> StateGraph:
    """Build and compile the LangGraph state graph."""
    graph = StateGraph(PipelineState)

    graph.add_node("init", init_node)
    graph.add_node("parallel_phase1", parallel_phase1)
    graph.add_node("parallel_phase2", parallel_phase2)
    graph.add_node("filter", filter_node)
    graph.add_node("marketing_copy", marketing_copy_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "parallel_phase1")
    graph.add_edge("parallel_phase1", "parallel_phase2")
    graph.add_edge("parallel_phase2", "filter")
    graph.add_edge("filter", "marketing_copy")
    graph.add_edge("marketing_copy", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()
