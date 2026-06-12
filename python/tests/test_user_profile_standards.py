from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.user_profile_agent import UserProfileAgent
from services.features.store import FeatureStore
from services.user_profile.standards import (
    build_standard_profile_fields,
    normalize_age,
    normalize_age_group,
    normalize_city,
    normalize_gender,
)


def test_normalize_gender():
    assert normalize_gender("男") == "male"
    assert normalize_gender("female") == "female"
    assert normalize_gender("") == "unknown"
    assert normalize_gender("bad-value") == "unknown"


def test_normalize_age_and_group():
    assert normalize_age("25") == 25
    assert normalize_age(120) == 120
    assert normalize_age(121) is None
    assert normalize_age("25.5") is None
    assert normalize_age_group(17) == "unknown"
    assert normalize_age_group(18) == "18-24"
    assert normalize_age_group(34) == "25-34"
    assert normalize_age_group(45) == "45-54"
    assert normalize_age_group(80) == "55+"


def test_normalize_city():
    assert normalize_city("Shanghai") == "上海"
    assert normalize_city("北京市") == "北京"
    assert normalize_city("杭州") == "杭州"
    assert normalize_city("") == "unknown"


def test_build_standard_profile_fields_never_raises():
    result = build_standard_profile_fields(age=object(), gender=object(), city=object())
    assert result == {
        "gender": "unknown",
        "age": None,
        "age_group": "unknown",
        "city": "unknown",
    }


def test_user_profile_agent_appends_standardized_tags():
    async def run():
        agent = UserProfileAgent(feature_store=FeatureStore())
        result = await agent.run(
            user_id="standard_user",
            context={"age": 29, "gender": "女", "city": "Shanghai"},
            fast_mode=True,
        )
        standardized = result.profile.real_time_tags["standardized_profile"]

        assert result.profile.gender == "女"
        assert result.profile.city == "Shanghai"
        assert standardized == {
            "gender": "female",
            "age": 29,
            "age_group": "25-34",
            "city": "上海",
        }

    asyncio.run(run())
