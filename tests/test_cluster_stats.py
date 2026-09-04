from __future__ import annotations

import pytest

from control.darts.cluster_stats import (
    MultiArmSessionRecord,
    PairedSessionRecord,
    audit_multi_arm_design,
    audit_completeness,
    audit_session_design,
    audit_speed_match,
    cluster_analysis_payload,
    exact_mcnemar_p,
    holm_adjust,
    intraclass_correlation,
    leave_one_session_out,
    paired_contrast,
    session_level_exact_signflip,
    two_level_cluster_bootstrap,
)


def records(effect=1.0, speed_gap=0.05):
    return tuple(
        PairedSessionRecord(
            physical_session_id=f"session-{session}",
            roll_index=roll,
            baseline_value=float(session + roll),
            candidate_value=float(session + roll + effect),
            baseline_speed_mps=15.0,
            candidate_speed_mps=15.0 + speed_gap,
        )
        for session in range(8)
        for roll in range(3)
    )


def test_session_design_requires_eight_by_three_unique_pairs():
    audit = audit_session_design(records())
    assert audit.passed is True
    assert audit.n_sessions == 8
    assert audit.rolls_per_session == (3,) * 8
    broken = audit_session_design(records()[:-1])
    assert broken.passed is False
    assert broken.missing_keys == (("session-7", 2),)


def test_two_level_bootstrap_resamples_sessions_and_rolls():
    result = two_level_cluster_bootstrap(
        records(effect=2.0), n_bootstrap=1000, seed=7)
    assert result.estimate == pytest.approx(2.0)
    assert result.ci_lo == pytest.approx(2.0)
    assert result.ci_hi == pytest.approx(2.0)
    assert result.n_sessions == 8


def test_cluster_bootstrap_weights_physical_sessions_equally():
    rows = (
        PairedSessionRecord("small", 0, 0.0, 0.0, 15.0, 15.0),
        *(
            PairedSessionRecord(
                "large", roll, 0.0, 2.0, 15.0, 15.0)
            for roll in range(3)
        ),
    )
    result = two_level_cluster_bootstrap(rows, n_bootstrap=100, seed=3)
    assert result.estimate == pytest.approx(1.0)


def test_session_exact_signflip_uses_session_as_inference_unit():
    result = session_level_exact_signflip(records(effect=1.0))
    assert result.n_sessions == 8
    assert result.estimate == pytest.approx(1.0)
    assert result.p_two_sided == pytest.approx(2 / 2 ** 8)


def test_icc_and_loso_report_cluster_dependence():
    rows = tuple(
        PairedSessionRecord(
            physical_session_id=f"session-{session}",
            roll_index=roll,
            baseline_value=0.0,
            candidate_value=float(session),
            baseline_speed_mps=15.0,
            candidate_speed_mps=15.0,
        )
        for session in range(8)
        for roll in range(3)
    )
    icc = intraclass_correlation(rows)
    assert icc.estimable is True
    assert icc.icc_1_1 == pytest.approx(1.0)
    assert icc.effective_n == pytest.approx(8.0)
    loso = leave_one_session_out(rows)
    assert loso.n_sessions == 8
    assert loso.direction_flip is False
    assert loso.min_estimate < loso.max_estimate


def test_speed_match_uses_timed_and_standard_thresholds():
    timed = audit_speed_match(records(speed_gap=0.15), mode="timed")
    standard = audit_speed_match(records(speed_gap=0.15), mode="standard")
    assert timed.passed is False
    assert standard.passed is True


def test_incomplete_trajectories_remain_in_denominator():
    rows = list(records())
    rows[0] = PairedSessionRecord(
        **{
            **rows[0].__dict__,
            "candidate_window_complete": False,
        }
    )
    audit = audit_completeness(rows)
    assert audit.n_pairs == 24
    assert audit.candidate_incomplete == 1
    assert audit.passed is False


def test_exact_mcnemar_and_integrated_payload():
    candidate_only, baseline_only, p = exact_mcnemar_p(
        [False] * 8,
        [True] * 8,
    )
    assert (candidate_only, baseline_only) == (8, 0)
    assert p == pytest.approx(2 / 2 ** 8)
    payload = cluster_analysis_payload(
        records(), speed_mode="timed", n_bootstrap=200, seed=1)
    assert payload["decision_ready"] is True
    assert payload["design"]["n_sessions"] == 8


def test_multi_arm_design_and_paired_contrast():
    arms = ("state_only", "budget_only", "joint")
    rows = tuple(
        MultiArmSessionRecord(
            physical_session_id=f"session-{session}",
            roll_index=roll,
            arm=arm,
            value=float(session + roll + arm_index),
            speed_mps=15.0 + 0.01 * arm_index,
        )
        for session in range(8)
        for roll in range(3)
        for arm_index, arm in enumerate(arms)
    )
    audit = audit_multi_arm_design(rows, expected_arms=arms)
    assert audit.passed
    pairs = paired_contrast(
        rows, baseline_arm="state_only", candidate_arm="joint")
    assert len(pairs) == 24
    assert all(pair.difference == 2.0 for pair in pairs)


def test_holm_adjust_is_monotone_in_sorted_order():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.2)
