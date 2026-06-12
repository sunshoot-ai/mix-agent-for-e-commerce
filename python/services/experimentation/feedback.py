from __future__ import annotations

from datetime import datetime
from typing import Any

from models.schemas import BehaviorEvent, BehaviorEventType, FeedbackResponse
from services.experimentation.ab_test import ABTestEngine
from services.features.store import FeatureStore
from services.operations.metrics import MetricsCollector
from services.observability import default_registry


POSITIVE_OUTCOMES = {
    BehaviorEventType.CLICK,
    BehaviorEventType.PURCHASE,
}


class FeedbackService:
    """Ingest user behavior and feed the recommendation data flywheel."""

    def __init__(
        self,
        feature_store: FeatureStore,
        ab_engine: ABTestEngine,
        metrics_collector: MetricsCollector,
    ):
        self.feature_store = feature_store
        self.ab_engine = ab_engine
        self.metrics_collector = metrics_collector
        self._seen_event_ids: set[str] = set()

    async def record_event(self, event: BehaviorEvent) -> FeedbackResponse:
        event_id = self.build_event_id(event)
        if event_id in self._seen_event_ids:
            default_registry.inc(
                "feedback_events_total",
                event_type=event.event_type.value,
                status="duplicate",
            )
            return FeedbackResponse(status="duplicate", duplicate=True, event_id=event_id)

        self._seen_event_ids.add(event_id)
        await self._record_feature(event)
        self._record_experiment(event)
        self._record_metrics(event)

        default_registry.inc(
            "feedback_events_total",
            event_type=event.event_type.value,
            status="recorded",
        )
        return FeedbackResponse(status="recorded", duplicate=False, event_id=event_id)

    @staticmethod
    def build_event_id(event: BehaviorEvent) -> str:
        return ":".join(
            [
                event.experiment_id,
                event.request_id,
                event.user_id,
                event.product_id,
                event.event_type.value,
            ]
        )

    async def _record_feature(self, event: BehaviorEvent) -> None:
        metadata: dict[str, Any] = {
            **event.metadata,
            "request_id": event.request_id,
            "scene": event.scene,
            "experiment_id": event.experiment_id,
            "experiment_group": event.experiment_group,
            "ts": _datetime_to_epoch(event.timestamp),
        }
        amount = metadata.get("amount")
        if event.event_type == BehaviorEventType.PURCHASE and amount is not None:
            metadata["amount"] = float(amount)

        await self.feature_store.record_behavior(
            user_id=event.user_id,
            behavior_type=event.event_type.value,
            item_id=event.product_id,
            metadata=metadata,
        )

    def _record_experiment(self, event: BehaviorEvent) -> None:
        segment = str(event.metadata.get("segment", "") or "")
        self.ab_engine.record_metric(
            event.experiment_id,
            event.experiment_group,
            metric_name=event.event_type.value,
            value=1.0,
            user_id=event.user_id,
            segment=segment,
            scene=event.scene,
            trace_id=event.request_id,
        )
        if event.event_type not in POSITIVE_OUTCOMES:
            return
        self.ab_engine.record_outcome(
            event.experiment_id,
            event.experiment_group,
            success=True,
        )
        if segment:
            self.ab_engine.record_segment_outcome(
                event.experiment_id,
                event.experiment_group,
                success=True,
                segment=segment,
            )

    def _record_metrics(self, event: BehaviorEvent) -> None:
        payload = {
            "user_id": event.user_id,
            "request_id": event.request_id,
            "product_id": event.product_id,
            "scene": event.scene,
            "experiment_id": event.experiment_id,
            "experiment_group": event.experiment_group,
            **event.metadata,
        }
        self.metrics_collector.record_business_event(event.event_type.value, **payload)
        default_registry.inc(
            "business_events_total",
            event_type=event.event_type.value,
            scene=event.scene,
            experiment_group=event.experiment_group,
        )


def _datetime_to_epoch(value: datetime) -> float:
    return value.timestamp()
