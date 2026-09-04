from __future__ import annotations

import math

import pytest

from control.darts.reachability_screen import (
    OperationalCandidateAssessment,
    assess_operational_candidate,
    select_operational_action,
)
from scripts.darts_bench import ARMS


def candidate(
    *,
    state: bool,
    budget: bool,
    joint: bool,
    margin: float = 0.0,
) -> OperationalCandidateAssessment:
    return OperationalCandidateAssessment(
        state_ok=state,
        budget_ok=budget,
        joint_ok=joint,
        velocity_nullable=budget,
        displacement_ok=state,
        margin=margin,
    )


def test_assessment_separates_state_and_budget_limits():
    assessment = assess_operational_candidate(
        theta0=0.0,
        omega0=-0.3,
        theta_target=0.0,
        flight_time_s=2.0,
        pitch_accel_radps2=2.0,
        B_up=0.1,
        B_down=0.5,
        theta_half_width_rad=math.radians(35.0),
        omega_half_width_radps=0.0,
    )
    assert assessment.state_ok is True
    assert assessment.budget_ok is False
    assert assessment.joint_ok is False


def test_outcome_only_preserves_supported_preset():
    out = select_operational_action(
        table_arm="nose_up",
        strategy="outcome_only",
        natural=candidate(state=True, budget=True, joint=True),
        preset=candidate(state=False, budget=False, joint=False),
    )
    assert out.selected_arm == "nose_up"
    assert out.changed_from_outcome_only is False


@pytest.mark.parametrize("strategy", ("state_only", "budget_only", "joint"))
def test_reachability_strategy_selects_natural_out_preset_in(strategy):
    fields = {
        "state_only": "state",
        "budget_only": "budget",
        "joint": "joint",
    }
    selected = fields[strategy]

    def assessment(value: bool) -> OperationalCandidateAssessment:
        flags = {"state": True, "budget": True, "joint": True}
        flags[selected] = value
        if not value and selected in {"state", "budget"}:
            flags["joint"] = False
        return candidate(**flags)

    out = select_operational_action(
        table_arm="nose_up",
        strategy=strategy,
        natural=assessment(False),
        preset=assessment(True),
    )
    assert out.selected_arm == "nose_up"
    assert out.reason == f"{strategy}_natural_out_preset_in"


def test_natural_already_in_returns_neutral():
    out = select_operational_action(
        table_arm="nose_up",
        strategy="joint",
        natural=candidate(state=True, budget=True, joint=True),
        preset=candidate(state=True, budget=True, joint=True),
    )
    assert out.selected_arm == "neutral"
    assert out.changed_from_outcome_only is True


def test_preset_not_in_avoids_preset():
    out = select_operational_action(
        table_arm="nose_up",
        strategy="joint",
        natural=candidate(state=True, budget=True, joint=False),
        preset=candidate(state=True, budget=True, joint=False),
    )
    assert out.selected_arm == "avoid_preset"


def test_non_preset_outcome_is_not_overridden():
    out = select_operational_action(
        table_arm="neutral",
        strategy="joint",
        natural=candidate(state=False, budget=False, joint=False),
        preset=candidate(state=True, budget=True, joint=True),
    )
    assert out.selected_arm == "neutral"
    assert out.reason == "outcome_table_did_not_propose_preset"


def test_joint_invariant_rejects_inconsistent_payload():
    with pytest.raises(ValueError):
        candidate(state=True, budget=False, joint=True)


def test_arxiv_policy_arms_cover_all_operational_strategies():
    expected = {
        "arxiv_outcome_only": "outcome_only",
        "arxiv_state_only": "state_only",
        "arxiv_budget_only": "budget_only",
        "arxiv_joint": "joint",
    }
    assert {
        arm: ARMS[arm]["operational_strategy"]
        for arm in expected
    } == expected
    for arm in expected:
        assert ARMS[arm]["gate"] is True
        assert ARMS[arm]["dart_wheel_speed_cap_radps"] == 200.0
        assert ARMS[arm]["approach_wheel_spin_guard_radps"] == 200.0
