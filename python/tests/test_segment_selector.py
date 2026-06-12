from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.schemas import UserProfile, UserSegment
from orchestrator.supervisor import SupervisorOrchestrator
from services.experimentation.simulation import _primary_segment as simulation_primary_segment
from services.user_profile.segment_selector import get_primary_segment


def test_get_primary_segment_uses_business_priority():
    assert get_primary_segment(["new_user", "high_value", "churn_risk"]) == "high_value"
    assert get_primary_segment(["churn_risk", "light_user"]) == "light_user"
    assert get_primary_segment(["active_viewer", "high_intent"]) == "high_intent"


def test_get_primary_segment_handles_unknown_and_invalid_values():
    assert get_primary_segment([]) == "unknown"
    assert get_primary_segment(None) == "unknown"
    assert get_primary_segment(["bad", "", None]) == "unknown"


def test_supervisor_primary_segment_uses_selector_not_first_item():
    profile = UserProfile(
        user_id="u1",
        segments=[UserSegment.NEW_USER, UserSegment.HIGH_VALUE],
    )

    assert SupervisorOrchestrator._primary_segment(profile) == "high_value"


def test_simulation_primary_segment_uses_selector_not_first_item():
    profile = UserProfile(
        user_id="u1",
        segments=[UserSegment.CHURN_RISK, UserSegment.NEW_USER],
    )

    assert simulation_primary_segment(profile) == "new_user"
