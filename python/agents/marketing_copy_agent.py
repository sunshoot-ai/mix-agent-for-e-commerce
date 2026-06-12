"""
营销文案Agent
- Prompt模板引擎：基于用户画像动态选择模板(新客/老客/高价值)
- 个性化生成：调用MiniMax M2.7生成文案
- 合规校验：敏感词过滤 + 广告法合规检查
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    MarketingCopyResult,
    Product,
    UserProfile,
    UserSegment,
)
from services.observability import (
    SpanKind,
    default_registry,
    extract_token_usage,
    tracer,
)

from .base_agent import BaseAgent

COPY_CACHE_TTL_SECONDS = 3600

PROMPT_TEMPLATES = {
    UserSegment.NEW_USER: """你是电商营销文案专家。为新用户撰写欢迎+推荐文案。
风格要求：热情友好、突出新人专属优惠感、降低决策门槛。
每个商品生成一条文案(30-50字)。""",

    UserSegment.HIGH_VALUE: """你是电商营销文案专家。为高价值VIP用户撰写推荐文案。
风格要求：品质感、尊享感、突出商品高端属性和品牌价值。
每个商品生成一条文案(30-50字)。""",

    UserSegment.PRICE_SENSITIVE: """你是电商营销文案专家。为价格敏感用户撰写推荐文案。
风格要求：突出性价比、促销价格、限时优惠、省钱金额。
每个商品生成一条文案(30-50字)。""",

    UserSegment.ACTIVE: """你是电商营销文案专家。为活跃用户撰写推荐文案。
风格要求：突出商品亮点和使用场景,引发共鸣。
每个商品生成一条文案(30-50字)。""",

    UserSegment.CHURN_RISK: """你是电商营销文案专家。为即将流失的用户撰写召回文案。
风格要求：情感唤回、专属折扣、限时活动、制造紧迫感。
每个商品生成一条文案(30-50字)。""",
}

FORBIDDEN_WORDS = [
    "最好", "第一", "国家级", "全球首", "绝对", "100%",
    "永久", "万能", "祖传", "纯天然",
]

COPY_OUTPUT_INSTRUCTION = """
请以JSON数组格式输出,每个元素格式:
[{"product_id": "xxx", "copy": "文案内容"}]
只输出JSON,不要其他内容。"""


class MarketingCopyAgent(BaseAgent):
    def __init__(self, cache_store: Any | None = None):
        settings = get_settings()
        self.model = settings.llm_model
        super().__init__(
            name="marketing_copy",
            timeout=settings.agent_timeout_marketing_copy,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=self.model,
            temperature=0.9,
            max_tokens=2048,
        )
        self.cache_store = cache_store

    async def _execute(self, **kwargs: Any) -> MarketingCopyResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        products: list[Product] = kwargs.get("products", [])
        fast_mode: bool = bool(kwargs.get("fast_mode", False))
        copy_style: str = str(kwargs.get("copy_style", "") or "")

        if not products:
            return MarketingCopyResult(success=True, copies=[], confidence=1.0)

        template_key = self._select_template(user_profile)
        cache_key = self._copy_cache_key(user_profile, products, template_key, copy_style)
        cached_copies = await self._get_cached_copies(cache_key)
        if cached_copies:
            return MarketingCopyResult(
                success=True,
                copies=cached_copies,
                prompt_template_used=template_key.value,
                data={
                    "cache_status": "hit",
                    "mode": "fast" if fast_mode else "cached",
                    "copy_style": copy_style,
                },
                confidence=0.9,
            )

        if fast_mode:
            fallback_copies = self._template_copies(products, template_key, copy_style)
            self._schedule_copy_refresh(cache_key, user_profile, products, template_key, copy_style)
            return MarketingCopyResult(
                success=True,
                copies=fallback_copies,
                prompt_template_used=template_key.value,
                data={
                    "cache_status": "miss",
                    "mode": "template_fallback",
                    "async_refresh": True,
                    "copy_style": copy_style,
                },
                confidence=0.6,
            )

        copies, raw_response = await self._generate_copies_with_llm(
            products,
            template_key,
            copy_style,
        )
        await self._set_cached_copies(cache_key, copies)

        return MarketingCopyResult(
            success=True,
            copies=copies,
            prompt_template_used=template_key.value,
            data={
                "raw_response": raw_response,
                "cache_status": "miss",
                "mode": "llm",
                "copy_style": copy_style,
            },
            confidence=0.9,
        )

    async def _generate_copies_with_llm(
        self,
        products: list[Product],
        template_key: UserSegment,
        copy_style: str = "",
    ) -> tuple[list[dict[str, str]], str]:
        system_prompt = PROMPT_TEMPLATES[template_key] + self._style_instruction(copy_style)

        product_info = "\n".join(
            f"- ID:{p.product_id} 名称:{p.name} 类目:{p.category} 价格:¥{p.price} 标签:{','.join(p.tags)}"
            for p in products
        )

        messages = [
            SystemMessage(content=system_prompt + COPY_OUTPUT_INSTRUCTION),
            HumanMessage(content=f"商品列表:\n{product_info}"),
        ]
        async with tracer.async_span(
            "llm.marketing_copy.generate",
            kind=SpanKind.LLM,
            agent=self.name,
            model=self.model,
            product_count=len(products),
            prompt_template=template_key.value,
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

        copies = self._parse_copies(response.content)
        copies = [self._compliance_check(c) for c in copies]
        return copies, response.content

    def _select_template(self, profile: UserProfile | None) -> UserSegment:
        if not profile or not profile.segments:
            return UserSegment.ACTIVE
        priority = [
            UserSegment.NEW_USER,
            UserSegment.HIGH_VALUE,
            UserSegment.CHURN_RISK,
            UserSegment.PRICE_SENSITIVE,
            UserSegment.ACTIVE,
        ]
        for seg in priority:
            if seg in profile.segments:
                return seg
        return UserSegment.ACTIVE

    def _parse_copies(self, raw: str) -> list[dict[str, str]]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            default_registry.inc(
                "llm_output_parse_failure_total",
                agent=self.name,
                model=self.model,
            )
            return []

    def _compliance_check(self, copy_item: dict[str, str]) -> dict[str, str]:
        """Filter forbidden advertising words per Chinese Ad Law."""
        text = copy_item.get("copy", "")
        for word in FORBIDDEN_WORDS:
            text = re.sub(re.escape(word), "***", text)
        copy_item["copy"] = text
        return copy_item

    def _template_copies(
        self,
        products: list[Product],
        template_key: UserSegment,
        copy_style: str = "",
    ) -> list[dict[str, str]]:
        style_prefix = self._style_prefix(copy_style) or {
            UserSegment.NEW_USER: "新人推荐",
            UserSegment.HIGH_VALUE: "品质优选",
            UserSegment.PRICE_SENSITIVE: "高性价比",
            UserSegment.CHURN_RISK: "专属召回",
            UserSegment.ACTIVE: "猜你喜欢",
        }.get(template_key, "猜你喜欢")
        return [
            {
                "product_id": product.product_id,
                "copy": f"{style_prefix}：{product.name}，适合关注{product.category}的你。",
            }
            for product in products
        ]

    async def _get_cached_copies(self, cache_key: str) -> list[dict[str, str]] | None:
        if not self.cache_store or not hasattr(self.cache_store, "cache_get"):
            return None
        raw = await self.cache_store.cache_get(cache_key)
        if not isinstance(raw, list):
            return None
        return raw

    async def _set_cached_copies(
        self,
        cache_key: str,
        copies: list[dict[str, str]],
    ) -> None:
        if not self.cache_store or not hasattr(self.cache_store, "cache_set"):
            return
        await self.cache_store.cache_set(cache_key, copies, ttl=COPY_CACHE_TTL_SECONDS)

    def _schedule_copy_refresh(
        self,
        cache_key: str,
        user_profile: UserProfile | None,
        products: list[Product],
        template_key: UserSegment,
        copy_style: str = "",
    ) -> None:
        async def refresh() -> None:
            start = time.perf_counter()
            try:
                copies, _ = await self._generate_copies_with_llm(
                    products,
                    template_key,
                    copy_style,
                )
                await self._set_cached_copies(cache_key, copies)
                default_registry.observe(
                    "async_refresh_latency_ms",
                    (time.perf_counter() - start) * 1000,
                    agent=self.name,
                    status="success",
                )
            except Exception as exc:
                default_registry.inc(
                    "async_refresh_total",
                    agent=self.name,
                    status="failed",
                )

        asyncio.create_task(refresh())

    @staticmethod
    def _copy_cache_key(
        user_profile: UserProfile | None,
        products: list[Product],
        template_key: UserSegment,
        copy_style: str = "",
    ) -> str:
        user_id = user_profile.user_id if user_profile else "anonymous"
        product_ids = ",".join(product.product_id for product in products)
        style = copy_style or "default"
        return f"marketing_copy:{user_id}:{product_ids}:{template_key.value}:{style}"

    @staticmethod
    def _style_instruction(copy_style: str) -> str:
        if copy_style == "formal":
            return "\n额外风格：正式、克制、强调品质与可信度。"
        if copy_style == "casual":
            return "\n额外风格：轻松、口语化、强调使用场景和亲近感。"
        return ""

    @staticmethod
    def _style_prefix(copy_style: str) -> str:
        if copy_style == "formal":
            return "品质推荐"
        if copy_style == "casual":
            return "你可能会喜欢"
        return ""
