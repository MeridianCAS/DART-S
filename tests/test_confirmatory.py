from __future__ import annotations

import pytest

from control.darts.cluster_stats import MultiArmSessionRecord
from scripts.confirmatory_analysis import (
    interaction_analysis,
    paired_analysis,
)
from scripts.confirmatory_design import (
    BLOCKS,
    build_sessions,
    candidate_preflight,
)


def test_frozen_design_has_600_runs_and_eight_sessions_per_block():
    sessions = build_sessions()
    assert len(sessions) == 72
    assert sum(session["expected_runs"] for session in sessions) == 600
    assert {
        block: sum(session["block_id"] == block for session in sessions)
        for block in BLOCKS
    } == {block: 8 for block in BLOCKS}


def test_candidate_preflight_exposes_budget_not_state_flip():
    result = candidate_preflight(BLOCKS["boundary_a35_v15"])
    assert result["table_arm"] == "nose_up"
    decisions = result["strategy_counterfactuals"]
    assert decisions["outcome_only"]["selected_arm"] == "nose_up"
    assert decisions["state_only"]["selected_arm"] == "neutral"
    assert decisions["budget_only"]["selected_arm"] == "nose_up"
    assert decisions["joint"]["selected_arm"] == "nose_up"


def multi_arm_rows(arms):
    rows = []
    values = {
        "neutral": 0.0,
        "nose_up": 0.0,
        "dart_guarded": 0.0,
        "arxiv_joint": 1.0,
    }
    for session in range(8):
        for roll in range(3):
            for arm in arms:
                rows.append(MultiArmSessionRecord(
                    physical_session_id=f"s{session}",
                    roll_index=roll,
                    arm=arm,
                    value=values.get(arm, 1.0),
                    speed_mps=15.0,
                ))
    return rows


def test_paired_and_interaction_analysis_use_session_units():
    rows = multi_arm_rows(
        ("neutral", "nose_up", "dart_guarded", "arxiv_joint"))
    paired = paired_analysis(
        rows, baseline="dart_guarded", candidate="arxiv_joint")
    assert paired["design"]["n_sessions"] == 8
    assert paired["session_signflip"]["p_two_sided"] == pytest.approx(
        2 / 2 ** 8)
    assert paired["passed_unadjusted"] is True
    interaction = interaction_analysis(rows)
    assert interaction["session_signflip"]["estimate"] == pytest.approx(1.0)
    assert interaction["passed_unadjusted"] is True
