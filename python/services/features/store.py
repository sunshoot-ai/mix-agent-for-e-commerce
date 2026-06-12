"""
实时特征存储服务
- Redis Sorted Set 存储用户行为序列 (score=timestamp)
- 滑动窗口计算实时特征 (1h/24h/7d)
- 离线+在线特征合并
- RFM模型计算
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog

from services.observability import EventStatus, SpanKind, default_registry, tracer
from services.reliability import classify_exception

logger = structlog.get_logger()


class FeatureStore:
    """Redis-backed real-time feature store for user behavior and profiles."""

    def __init__(self, redis_client: Any = None, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl
        self._memory_behaviors: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self._memory_profiles: dict[str, dict[str, Any]] = {}
        self._memory_cache: dict[str, tuple[Any, float]] = {}

    # ---------- generic cache ----------

    async def cache_get(self, key: str) -> Any | None:
        """Read JSON-compatible data from Redis or the local fallback cache."""
        now = time.time()
        if not self.redis:
            cached = self._memory_cache.get(key)
            if not cached:
                return None
            value, expires_at = cached
            if expires_at and expires_at <= now:
                self._memory_cache.pop(key, None)
                return None
            return value

        raw = await self.redis.get(key)
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw)

    async def cache_set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Write JSON-compatible data to Redis or the local fallback cache."""
        ttl_seconds = ttl if ttl is not None else self.ttl
        if not self.redis:
            expires_at = time.time() + ttl_seconds if ttl_seconds else 0.0
            self._memory_cache[key] = (value, expires_at)
            return

        await self.redis.set(
            key,
            json.dumps(value, ensure_ascii=False),
            ex=ttl_seconds,
        )

    # ---------- behavior tracking ----------

    async def record_behavior(
        self, user_id: str, behavior_type: str, item_id: str, metadata: dict | None = None
    ):
        """Append a behavior event to user's sorted set (score = timestamp)."""
        event_ts = float((metadata or {}).get("ts") or time.time())
        payload_data = {"item_id": item_id, "ts": event_ts, **(metadata or {})}
        if not self.redis:
            self._memory_behaviors[(user_id, behavior_type)].append(payload_data)
            return
        key = f"behavior:{user_id}:{behavior_type}"
        payload = json.dumps(payload_data, ensure_ascii=False)
        await self.redis.zadd(key, {payload: event_ts})
        await self.redis.expire(key, self.ttl)

    async def get_recent_behaviors(
        self, user_id: str, behavior_type: str, window_seconds: int = 3600
    ) -> list[dict]:
        """Retrieve behaviors within a sliding time window."""
        cutoff = time.time() - window_seconds
        if not self.redis:
            return [
                item
                for item in self._memory_behaviors.get((user_id, behavior_type), [])
                if float(item.get("ts", 0)) >= cutoff
            ]
        key = f"behavior:{user_id}:{behavior_type}"
        raw_items = await self.redis.zrangebyscore(key, cutoff, "+inf")
        return [json.loads(item) for item in raw_items]

    # ---------- real-time features ----------

    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        """Build aggregated feature vector from recent behaviors."""
        views_1h = await self.get_recent_behaviors(user_id, "view", 3600)
        views_24h = await self.get_recent_behaviors(user_id, "view", 86400)
        impressions_24h = await self.get_recent_behaviors(user_id, "impression", 86400)
        clicks_1h = await self.get_recent_behaviors(user_id, "click", 3600)
        clicks_24h = await self.get_recent_behaviors(user_id, "click", 86400)
        purchases_7d = await self.get_recent_behaviors(user_id, "purchase", 604800)

        recent_view_events = (views_24h + impressions_24h + clicks_24h)[-20:]
        recent_view_items = [v.get("item_id", "") for v in recent_view_events]
        recent_purchase_items = [p.get("item_id", "") for p in purchases_7d[-10:]]
        preferred_categories = _top_categories(clicks_24h + purchases_7d + views_24h)
        event_profile = _latest_profile_attrs(
            purchases_7d + clicks_24h + impressions_24h + views_24h
        )

        rfm = await self._compute_rfm(user_id, purchases_7d)

        profile_key = f"profile:{user_id}"
        offline_tags = {}
        if self.redis:
            raw = await self.redis.get(profile_key)
            if raw:
                offline_tags = json.loads(raw)
        else:
            offline_tags = self._memory_profiles.get(user_id, {})

        return {
            "user_id": user_id,
            "age": event_profile.get("age") or offline_tags.get("age"),
            "gender": event_profile.get("gender") or offline_tags.get("gender"),
            "city": event_profile.get("city") or offline_tags.get("city"),
            "view_count_1h": len(views_1h),
            "view_count_24h": len(views_24h),
            "click_count_1h": len(clicks_1h),
            "purchase_count_7d": len(purchases_7d),
            "recent_views": recent_view_items,
            "recent_purchases": recent_purchase_items,
            "preferred_categories": preferred_categories,
            "rfm": rfm,
            "offline_tags": offline_tags,
        }

    # ---------- RFM model ----------

    async def _compute_rfm(self, user_id: str, purchases: list[dict]) -> dict[str, float]:
        """
        Recency / Frequency / Monetary scoring (normalised 0-1).
        Without full data we use heuristics.
        """
        if not purchases:
            return {"recency": 0.0, "frequency": 0.0, "monetary": 0.0}

        now = time.time()
        latest_ts = max(p.get("ts", 0) for p in purchases)
        days_since = (now - latest_ts) / 86400

        recency = max(0.0, 1.0 - days_since / 30.0)
        frequency = min(1.0, len(purchases) / 10.0)
        avg_amount = sum(p.get("amount", 100) for p in purchases) / len(purchases)
        monetary = min(1.0, avg_amount / 1000.0)

        return {
            "recency": round(recency, 3),
            "frequency": round(frequency, 3),
            "monetary": round(monetary, 3),
        }

    # ---------- offline merge ----------

    async def merge_offline_tags(self, user_id: str, tags: dict[str, Any]):
        """Write offline (batch-computed) tags so the profile agent can read them."""
        if not self.redis:
            self._memory_profiles[user_id] = dict(tags)
            return
        key = f"profile:{user_id}"
        await self.redis.set(key, json.dumps(tags), ex=self.ttl)


class FeastFeatureStore:
    """Feast online feature store client."""

    def __init__(self, repo_path: str | None = None):
        default_repo = Path(__file__).resolve().parent / "feature_repo"
        self.repo_path = Path(repo_path) if repo_path else default_repo
        self.store: Any = None
        self._init_error: Exception | None = None
        try:
            from feast import FeatureStore as FeastStore  # lazy import
            self.store = FeastStore(repo_path=str(self.repo_path))
        except Exception as exc:
            self._init_error = exc
            logger.warning("feast.init_failed", repo_path=str(self.repo_path), error=str(exc))

    async def get_user_features(self, user_id: str) -> dict[str, Any]:
        async with tracer.async_span(
            "feature.feast.get_user_features",
            kind=SpanKind.FEATURE,
            store="feast",
            user_id=user_id,
        ) as span:
            try:
                if not self.store:
                    raise RuntimeError(
                        f"Feast feature store is unavailable: {self._init_error}"
                    )
                result = self.store.get_online_features(
                    features=[
                        "user_behavior_features:view_count_1h",
                        "user_behavior_features:view_count_24h",
                        "user_behavior_features:view_count_7d",
                        "user_behavior_features:avg_order_amount",
                        "user_behavior_features:rfm_score",
                        "user_behavior_features:recent_views",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()

                recent_views_raw = result.get("recent_views", [None])[0]
                recent_views = []
                if isinstance(recent_views_raw, str) and recent_views_raw:
                    recent_views = recent_views_raw.split(",")

                default_registry.inc("feature_fetch_total", store="feast", status="success")
                return {
                    "user_id": user_id,
                    "view_count_1h": result.get("view_count_1h", [0])[0],
                    "view_count_24h": result.get("view_count_24h", [0])[0],
                    "view_count_7d": result.get("view_count_7d", [0])[0],
                    "avg_order_amount": result.get("avg_order_amount", [0])[0],
                    "rfm_score": result.get("rfm_score", [0])[0],
                    "recent_views": recent_views,
                }
            except Exception as exc:
                classified = classify_exception(exc, source_component="feast")
                span.status = EventStatus.ERROR
                span.error = str(exc)
                span.attributes["error_code"] = classified.code.value
                default_registry.inc(
                    "feature_fetch_total",
                    store="feast",
                    status="error",
                    error_code=classified.code.value,
                )
                raise


def _top_categories(events: list[dict[str, Any]], limit: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for event in events:
        category = event.get("category") or event.get("product_category")
        if not category:
            continue
        counts[str(category)] = counts.get(str(category), 0) + 1
    return [
        category
        for category, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def _latest_profile_attrs(events: list[dict[str, Any]]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for event in sorted(events, key=lambda item: float(item.get("ts", 0))):
        for key in ("age", "gender", "city"):
            value = event.get(key)
            if value not in (None, ""):
                attrs[key] = value
    return attrs
