"""
用户画像Agent
- 实时特征提取：浏览/点击/购买/收藏行为 -> Redis Feature Store
- 用户分群：RFM模型 + 实时标签
- 画像合并：离线标签(T+1) + 在线标签(实时)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    AgentResult,
    UserProfile,
    UserProfileResult,
    UserSegment,
)

from services.features.store import FeastFeatureStore, FeatureStore
from services.observability import (
    EventStatus,
    SpanKind,
    default_registry,
    extract_token_usage,
    tracer,
)
from services.reliability import classify_exception
from services.user_profile.standards import build_standard_profile_fields

from .base_agent import BaseAgent

logger = structlog.get_logger()

PROFILE_CACHE_TTL_SECONDS = 1800

SYSTEM_PROMPT = """你是一个电商用户画像分析专家。根据用户的行为数据,分析用户特征并生成画像。

你需要输出以下JSON格式:
{
  "segments": ["new_user"|"active"|"high_value"|"price_sensitive"|"churn_risk"],
  "preferred_categories": ["类目1", "类目2"],
  "price_range": [最低价, 最高价],
  "rfm_score": {"recency": 0-1, "frequency": 0-1, "monetary": 0-1},
  "real_time_tags": {"活跃时段": "...", "偏好风格": "..."}
}

只输出JSON,不要其他内容。"""


class UserProfileAgent(BaseAgent):
    def __init__(self, feature_store: Any | None = None):
        settings = get_settings()
        self.model = settings.llm_model
        super().__init__(
            name="user_profile",
            timeout=settings.agent_timeout_user_profile,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=self.model,
            temperature=0.3,
            max_tokens=1024,
        )
        self.feature_store = feature_store or FeatureStore(ttl=settings.feature_ttl_seconds)

    async def _execute(self, **kwargs: Any) -> UserProfileResult:
        user_id: str = kwargs["user_id"]
        context: dict = kwargs.get("context", {})
        fast_mode: bool = bool(kwargs.get("fast_mode", False))

        behavior_data = await self._collect_behavior(user_id, context)
        cache_key = self._profile_cache_key(user_id)
        cached_profile = await self._get_cached_profile(cache_key)
        if cached_profile:
            cached_profile = self._merge_profile_attrs(cached_profile, behavior_data, context)
            if fast_mode:
                self._schedule_profile_refresh(user_id, behavior_data, cache_key, context)
            await self._set_cached_profile(cache_key, cached_profile)
            return UserProfileResult(
                success=True,
                profile=cached_profile,
                data={"cache_status": "hit", "mode": "fast" if fast_mode else "cached"},
                confidence=0.9,
            )

        if fast_mode:
            profile_data = self._build_rule_profile(user_id, behavior_data, context)
            await self._set_cached_profile(cache_key, profile_data)
            self._schedule_profile_refresh(user_id, behavior_data, cache_key, context)
            return UserProfileResult(
                success=True,
                profile=profile_data,
                data={"cache_status": "miss", "mode": "rule_fallback", "async_refresh": True},
                confidence=0.65,
            )

        profile_data, raw_analysis = await self._generate_profile_with_llm(
            user_id,
            behavior_data,
            context,
        )
        await self._set_cached_profile(cache_key, profile_data)

        return UserProfileResult(
            success=True,
            profile=profile_data,
            data={"raw_analysis": raw_analysis, "cache_status": "miss", "mode": "llm"},
            confidence=0.85,
        )

    async def _generate_profile_with_llm(
        self,
        user_id: str,
        behavior_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[UserProfile, str]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"用户ID: {user_id}\n行为数据: {json.dumps(behavior_data, ensure_ascii=False)}"
            ),
        ]
        async with tracer.async_span(
            "llm.user_profile",
            kind=SpanKind.LLM,
            agent=self.name,
            model=self.model,
        ) as span:
            response = await self.llm.ainvoke(messages)
            usage = extract_token_usage(response, model=self.model)
            span.attributes.update(
                {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_tokens": usage.estimated,
                }
            )
            default_registry.inc(
                "llm_total_tokens_total",
                usage.total_tokens,
                agent=self.name,
                model=self.model,
            )
        return self._parse_profile(
            user_id,
            response.content,
            behavior_data,
            context or {},
        ), response.content

    async def _collect_behavior(self, user_id: str, context: dict) -> dict:
        """Collect user behavior from feature store or context fallback."""
        if self.feature_store:
            async with tracer.async_span(
                "feature.user_profile.fetch",
                kind=SpanKind.FEATURE,
                agent=self.name,
                store="feast",
                user_id=user_id,
            ) as span:
                try:
                    result = await self.feature_store.get_user_features(user_id)
                    default_registry.inc("feature_fetch_total", store="feast", status="success")
                    return result
                except Exception as exc:
                    classified = classify_exception(exc, source_component="feast")
                    span.status = EventStatus.ERROR
                    span.error = str(exc)
                    span.attributes["error_code"] = classified.code.value
                    default_registry.inc(
                        "feature_fetch_total",
                        store="feast",
                        status="fallback",
                        error_code=classified.code.value,
                    )
        if context:
            return context
        # 降级返回空字典
        return {}

    def _build_rule_profile(
        self,
        user_id: str,
        behavior_data: dict[str, Any],
        context: dict[str, Any],
    ) -> UserProfile:
        purchase_count = int(behavior_data.get("purchase_count_7d", 0) or 0)
        click_count = int(behavior_data.get("click_count_1h", 0) or 0)
        view_count = int(behavior_data.get("view_count_24h", 0) or 0)

        segments = [UserSegment.ACTIVE]
        if view_count <= 1 and purchase_count == 0:
            segments.insert(0, UserSegment.NEW_USER)
        if purchase_count >= 2:
            segments.insert(0, UserSegment.HIGH_VALUE)
        if view_count >= 10 and click_count == 0:
            segments.insert(0, UserSegment.CHURN_RISK)

        attrs = self._resolve_profile_attrs(behavior_data, context)
        profile = UserProfile(
            user_id=user_id,
            age=attrs.get("age"),
            gender=attrs.get("gender"),
            city=attrs.get("city"),
            segments=segments,
            preferred_categories=behavior_data.get("preferred_categories", []),
            price_range=(0.0, 10000.0),
            recent_views=behavior_data.get("recent_views", []),
            recent_purchases=behavior_data.get("recent_purchases", []),
            rfm_score=behavior_data.get("rfm", {}),
            real_time_tags={"profile_source": "rule_fallback"},
        )
        return self._with_standard_profile_tags(profile)

    async def _get_cached_profile(self, cache_key: str) -> UserProfile | None:
        if not hasattr(self.feature_store, "cache_get"):
            return None
        raw = await self.feature_store.cache_get(cache_key)
        if not raw:
            return None
        try:
            return UserProfile.model_validate(raw)
        except Exception:
            return None

    async def _set_cached_profile(self, cache_key: str, profile: UserProfile) -> None:
        if not hasattr(self.feature_store, "cache_set"):
            return
        await self.feature_store.cache_set(
            cache_key,
            profile.model_dump(mode="json"),
            ttl=PROFILE_CACHE_TTL_SECONDS,
        )

    def _schedule_profile_refresh(
        self,
        user_id: str,
        behavior_data: dict[str, Any],
        cache_key: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        async def refresh() -> None:
            start = time.perf_counter()
            try:
                profile, _ = await self._generate_profile_with_llm(
                    user_id,
                    behavior_data,
                    context or {},
                )
                await self._set_cached_profile(cache_key, profile)
                default_registry.observe(
                    "async_refresh_latency_ms",
                    (time.perf_counter() - start) * 1000,
                    agent=self.name,
                    status="success",
                )
            except Exception as exc:
                logger.warning("profile.async_refresh_failed", user_id=user_id, error=str(exc))
                default_registry.inc(
                    "async_refresh_total",
                    agent=self.name,
                    status="failed",
                )

        asyncio.create_task(refresh())

    @staticmethod
    def _profile_cache_key(user_id: str) -> str:
        return f"user_profile:{user_id}"

    def _parse_profile(
        self,
        user_id: str,
        raw: str,
        behavior_data: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> UserProfile:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            data = {}

        segments = []
        for s in data.get("segments", ["active"]):
            try:
                segments.append(UserSegment(s))
            except ValueError:
                continue

        price_range_raw = data.get("price_range", [0, 10000])
        price_range = (
            float(price_range_raw[0]),
            float(price_range_raw[1]) if len(price_range_raw) > 1 else 10000.0,
        )

        behavior_data = behavior_data or {}
        context = context or {}
        preferred_categories = (
            data.get("preferred_categories")
            or behavior_data.get("preferred_categories")
            or []
        )
        rfm_score = data.get("rfm_score") or behavior_data.get("rfm") or {}
        attrs = self._resolve_profile_attrs(behavior_data, context, data)

        profile = UserProfile(
            user_id=user_id,
            age=attrs.get("age"),
            gender=attrs.get("gender"),
            city=attrs.get("city"),
            segments=segments or [UserSegment.ACTIVE],
            preferred_categories=preferred_categories,
            price_range=price_range,
            recent_views=data.get("recent_views", behavior_data.get("recent_views", [])),
            recent_purchases=data.get(
                "recent_purchases",
                behavior_data.get("recent_purchases", []),
            ),
            rfm_score=rfm_score,
            real_time_tags=data.get("real_time_tags", {}),
        )
        return self._with_standard_profile_tags(profile)

    @staticmethod
    def _resolve_profile_attrs(
        behavior_data: dict[str, Any],
        context: dict[str, Any],
        llm_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        offline_tags = behavior_data.get("offline_tags", {}) or {}
        llm_data = llm_data or {}
        attrs: dict[str, Any] = {}
        for key in ("age", "gender", "city"):
            attrs[key] = (
                context.get(key)
                or behavior_data.get(key)
                or offline_tags.get(key)
                or llm_data.get(key)
            )
        return attrs

    def _merge_profile_attrs(
        self,
        profile: UserProfile,
        behavior_data: dict[str, Any],
        context: dict[str, Any],
    ) -> UserProfile:
        attrs = self._resolve_profile_attrs(behavior_data, context)
        update = {
            key: value
            for key, value in attrs.items()
            if value not in (None, "") and getattr(profile, key) in (None, "")
        }
        if not update:
            return self._with_standard_profile_tags(profile)
        return self._with_standard_profile_tags(profile.model_copy(update=update))

    @staticmethod
    def _with_standard_profile_tags(profile: UserProfile) -> UserProfile:
        real_time_tags = dict(profile.real_time_tags or {})
        real_time_tags["standardized_profile"] = build_standard_profile_fields(
            age=profile.age,
            gender=profile.gender,
            city=profile.city,
        )
        return profile.model_copy(update={"real_time_tags": real_time_tags})
