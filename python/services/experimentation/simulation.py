from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_core.messages import AIMessage

from agents.product_rec_agent import MOCK_PRODUCTS
from models.schemas import (
    BehaviorEventType,
    FeedbackRequest,
    RecommendationRequest,
    UserSegment,
)
from orchestrator.supervisor import SupervisorOrchestrator
from services.experimentation.ab_test import ABTestEngine
from services.features.store import FeatureStore
from services.experimentation.feedback import FeedbackService
from services.operations.metrics import MetricsCollector
from services.observability import default_registry
from services.user_profile.segment_selector import get_primary_segment


class FakeUserProfileLLM:
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        content = _message_content(messages[-1])
        data = _extract_behavior_data(content)
        view_count = int(data.get("view_count_24h", 0) or 0)
        purchase_count = int(data.get("purchase_count_7d", 0) or 0)
        avg_amount = float(data.get("avg_order_amount", 0) or 0)
        preferred_categories = data.get("preferred_categories") or ["手机", "耳机"]

        segments = [UserSegment.ACTIVE.value]
        if purchase_count >= 1 or avg_amount >= 1000:
            segments.insert(0, UserSegment.HIGH_VALUE.value)
        if view_count <= 1 and purchase_count == 0:
            segments.insert(0, UserSegment.NEW_USER.value)

        return AIMessage(
            content=json.dumps(
                {
                    "segments": segments,
                    "preferred_categories": preferred_categories,
                    "price_range": [0, 9000],
                    "rfm_score": data.get("rfm", {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}),
                    "recent_views": data.get("recent_views", []),
                    "recent_purchases": data.get("recent_purchases", []),
                    "real_time_tags": {
                        "活跃时段": "evening",
                        "偏好风格": "digital",
                    },
                },
                ensure_ascii=False,
            )
        )


class FakeProductRerankLLM:
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        content = _message_content(messages[-1])
        product_ids = []
        for product in MOCK_PRODUCTS:
            if product.product_id in content:
                product_ids.append(product.product_id)
        return AIMessage(content=json.dumps(product_ids, ensure_ascii=False))


class FakeMarketingCopyLLM:
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        content = _message_content(messages[-1])
        copies = []
        for product in MOCK_PRODUCTS:
            if f"ID:{product.product_id}" in content:
                copies.append(
                    {
                        "product_id": product.product_id,
                        "copy": f"{product.name}贴合你的近期偏好，适合现在加入购物清单。",
                    }
                )
        return AIMessage(content=json.dumps(copies, ensure_ascii=False))


async def run_simulation(
    rounds: int = 2,
    use_real_llm: bool = False,
    user_count: int = 1,
    impressions_per_round: int = 3,
    click_probability: float = 1.0,
    purchase_probability: float = 0.5,
    seed: int = 7,
) -> dict[str, Any]:
    rng = random.Random(seed)
    feature_store = FeatureStore()
    ab_engine = ABTestEngine()
    metrics = MetricsCollector()
    feedback = FeedbackService(feature_store, ab_engine, metrics)
    supervisor = SupervisorOrchestrator(
        ab_engine=ab_engine,
        feature_store=feature_store,
    )
    if not use_real_llm:
        supervisor.user_profile_agent.llm = FakeUserProfileLLM()
        supervisor.product_rec_agent.llm = FakeProductRerankLLM()
        supervisor.marketing_copy_agent.llm = FakeMarketingCopyLLM()

    user_ids = [f"u_flywheel_{idx + 1:03d}" for idx in range(user_count)]
    for user_id in user_ids:
        await _seed_initial_behavior(feature_store, user_id)

    snapshots = []
    for user_id in user_ids:
        for round_index in range(1, rounds + 1):
            before_features = await feature_store.get_user_features(user_id)
            response = await supervisor.recommend(
                RecommendationRequest(
                    user_id=user_id,
                    scene="homepage",
                    num_items=5,
                )
            )
            emitted_events = await _emit_feedback_for_response(
                feedback=feedback,
                response=response,
                round_index=round_index,
                impressions_per_round=impressions_per_round,
                click_probability=click_probability,
                purchase_probability=purchase_probability,
                rng=rng,
            )
            after_features = await feature_store.get_user_features(user_id)

            profile = response.agent_results["user_profile"].profile
            inventory = response.agent_results["inventory"]
            copy_result = response.agent_results["marketing_copy"]
            snapshots.append(
                {
                    "round": round_index,
                    "user_id": user_id,
                    "request_id": response.request_id,
                    "experiment_group": response.experiment_group,
                    "copy_experiment": copy_result.data.get("experiments", {}).get("copy_style", {}),
                    "total_latency_ms": response.total_latency_ms,
                    "agent_summary": {
                        name: {
                            "success": result.success,
                            "latency_ms": result.latency_ms,
                            "confidence": result.confidence,
                            "error": result.error,
                        }
                        for name, result in response.agent_results.items()
                    },
                    "features_before": before_features,
                    "profile": profile.model_dump(mode="json") if profile else None,
                    "products": [p.model_dump(mode="json") for p in response.products],
                    "inventory": {
                        "available_products": inventory.available_products,
                        "low_stock_alerts": inventory.low_stock_alerts,
                        "purchase_limits": inventory.purchase_limits,
                    },
                    "marketing_copies": copy_result.copies,
                    "feedback_events": emitted_events,
                    "features_after": after_features,
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_ids": user_ids,
        "llm_mode": "real" if use_real_llm else "fake",
        "simulation_config": {
            "rounds": rounds,
            "user_count": user_count,
            "impressions_per_round": impressions_per_round,
            "click_probability": click_probability,
            "purchase_probability": purchase_probability,
            "seed": seed,
        },
        "flow": [
            "behavior_events",
            "real_time_features",
            "user_profile",
            "recommendation_inventory_copy",
            "feedback_events",
            "ab_and_feature_update",
        ],
        "rounds": snapshots,
        "ab_experiments": _experiment_snapshot(ab_engine),
        "segmented_experiments": {
            exp_id: {
                "stats": ab_engine.get_segmented_stats(exp_id),
                "posteriors": ab_engine.get_segment_posteriors(exp_id),
            }
            for exp_id in ("rec_strategy", "copy_style")
        },
        "dedup_business_report": _dedup_business_report(snapshots),
        "business_metrics": metrics.get_business_stats(),
        "observability": default_registry.snapshot(),
    }


async def _seed_initial_behavior(feature_store: FeatureStore, user_id: str) -> None:
    seeds = [
        ("view", "P001", {"category": "手机"}),
        ("view", "P003", {"category": "耳机"}),
        ("click", "P003", {"category": "耳机"}),
    ]
    for behavior_type, product_id, metadata in seeds:
        await feature_store.record_behavior(user_id, behavior_type, product_id, metadata)


async def _emit_feedback_for_response(
    feedback: FeedbackService,
    response: Any,
    round_index: int,
    impressions_per_round: int,
    click_probability: float,
    purchase_probability: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    emitted = []
    for product in response.products[:impressions_per_round]:
        emitted.extend(
            await _record_all_experiments(
                feedback,
                response,
                product,
                BehaviorEventType.IMPRESSION,
                round_index,
            )
        )
    if response.products and rng.random() < click_probability:
        emitted.extend(
            await _record_all_experiments(
                feedback,
                response,
                response.products[0],
                BehaviorEventType.CLICK,
                round_index,
            )
        )
    if round_index >= 2 and len(response.products) > 1 and rng.random() < purchase_probability:
        emitted.extend(
            await _record_all_experiments(
                feedback,
                response,
                response.products[1],
                BehaviorEventType.PURCHASE,
                round_index,
                {"amount": response.products[1].price},
            )
        )
    return emitted


async def _record_all_experiments(
    feedback: FeedbackService,
    response: Any,
    product: Any,
    event_type: BehaviorEventType,
    round_index: int,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    profile = response.agent_results["user_profile"].profile
    segment = _primary_segment(profile)
    copy_experiment = response.agent_results["marketing_copy"].data.get("experiments", {}).get(
        "copy_style",
        {},
    )
    rec_event = await _record(
        feedback,
        response,
        product,
        event_type,
        round_index,
        experiment_id="rec_strategy",
        experiment_group=response.experiment_group,
        segment=segment,
        extra_metadata=extra_metadata,
    )
    copy_event = await _record(
        feedback,
        response,
        product,
        event_type,
        round_index,
        experiment_id="copy_style",
        experiment_group=copy_experiment.get("group", "control"),
        segment=segment,
        extra_metadata={
            "copy_style": copy_experiment.get("config", {}).get(
                "style",
                copy_experiment.get("group", ""),
            ),
            **(extra_metadata or {}),
        },
    )
    return [rec_event, copy_event]


async def _record(
    feedback: FeedbackService,
    response: Any,
    product: Any,
    event_type: BehaviorEventType,
    round_index: int,
    experiment_id: str,
    experiment_group: str,
    segment: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event = FeedbackRequest(
        user_id=response.user_id,
        request_id=response.request_id,
        product_id=product.product_id,
        event_type=event_type,
        scene="homepage",
        experiment_id=experiment_id,
        experiment_group=experiment_group,
        metadata={
            "round": round_index,
            "segment": segment,
            "category": product.category,
            "price": product.price,
            **(extra_metadata or {}),
        },
    )
    result = await feedback.record_event(event)
    return {
        "event": event.model_dump(mode="json"),
        "result": result.model_dump(mode="json"),
    }


def _primary_segment(profile: Any) -> str:
    if not profile or not getattr(profile, "segments", None):
        return "unknown"
    return get_primary_segment(profile.segments)


def _message_content(message: Any) -> str:
    content = getattr(message, "content", message)
    return str(content)


def _extract_behavior_data(content: str) -> dict[str, Any]:
    marker = "行为数据:"
    if marker not in content:
        return {}
    raw = content.split(marker, 1)[1].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _experiment_snapshot(ab_engine: ABTestEngine) -> dict[str, Any]:
    output = {}
    for exp_id, exp in ab_engine.experiments.items():
        output[exp_id] = {
            "name": exp.name,
            "groups": [
                {
                    "name": group.name,
                    "successes": group.successes,
                    "failures": group.failures,
                    "posterior_mean": group.successes / (group.successes + group.failures),
                }
                for group in exp.groups
            ],
            "stats": ab_engine.get_stats(exp_id),
        }
    return output


def _dedup_business_report(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[tuple[str, str, str, str]] = set()
    counts = {"impression": 0, "click": 0, "purchase": 0}
    by_copy_group: dict[str, dict[str, int]] = {}
    by_segment: dict[str, dict[str, int]] = {}

    for snapshot in snapshots:
        copy_group = str(snapshot.get("copy_experiment", {}).get("group", "unknown"))
        for event_item in snapshot.get("feedback_events", []):
            event = event_item.get("event", {})
            event_type = str(event.get("event_type", ""))
            if event_type not in counts:
                continue
            key = (
                str(event.get("request_id", "")),
                str(event.get("user_id", "")),
                str(event.get("product_id", "")),
                event_type,
            )
            if key in seen:
                continue
            seen.add(key)
            counts[event_type] += 1
            metadata = event.get("metadata", {}) or {}
            segment = str(metadata.get("segment", "unknown"))
            _inc_report_bucket(by_copy_group, copy_group, event_type)
            _inc_report_bucket(by_segment, segment, event_type)

    return {
        **counts,
        "ctr": _safe_ratio(counts["click"], counts["impression"]),
        "cvr": _safe_ratio(counts["purchase"], counts["click"]),
        "by_copy_group": _with_rates(by_copy_group),
        "by_segment": _with_rates(by_segment),
    }


def _inc_report_bucket(report: dict[str, dict[str, int]], key: str, event_type: str) -> None:
    bucket = report.setdefault(key, {"impression": 0, "click": 0, "purchase": 0})
    bucket[event_type] += 1


def _with_rates(report: dict[str, dict[str, int]]) -> dict[str, Any]:
    return {
        key: {
            **counts,
            "ctr": _safe_ratio(counts["click"], counts["impression"]),
            "cvr": _safe_ratio(counts["purchase"], counts["click"]),
        }
        for key, counts in report.items()
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the recommendation data flywheel.")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--users", type=int, default=1)
    parser.add_argument("--impressions-per-round", type=int, default=3)
    parser.add_argument("--click-probability", type=float, default=1.0)
    parser.add_argument("--purchase-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Use configured production LLM clients instead of deterministic fake LLMs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "simulations" / "flywheel_test_data.json",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_simulation(
            rounds=args.rounds,
            use_real_llm=args.real_llm,
            user_count=args.users,
            impressions_per_round=args.impressions_per_round,
            click_probability=args.click_probability,
            purchase_probability=args.purchase_probability,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(args.output))


if __name__ == "__main__":
    main()
