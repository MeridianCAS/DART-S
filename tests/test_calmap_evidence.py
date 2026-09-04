from __future__ import annotations

import json
from dataclasses import replace

import pytest

from control.darts.calmap_evidence import (
    EVIDENCE_FINGERPRINT_VERSION,
    CatalogPromotionPlan,
    EvidenceChecks,
    EvidenceError,
    EvidenceFailure,
    EvidenceGrade,
    EvidenceIntegrity,
    EvidenceKey,
    EvidencePromotionError,
    EvidenceSessions,
    LocalEvidence,
    PromotionAction,
    ReplayAnchorEvidence,
    SessionIsolationError,
    assert_grade_authorized,
    assess_evidence_catalog,
    assess_local_evidence,
    audit_replay_anchors,
    audit_session_isolation,
    decide_promotion,
    local_evidence_fingerprint,
    plan_catalog_update,
    versioned_fingerprint,
)


MODEL_HASH = "a" * 64
ANCHOR_HASH_A = "b" * 64
ANCHOR_HASH_B = "c" * 64


def key(suffix: str = "static") -> EvidenceKey:
    return EvidenceKey(
        domain=f"smooth-{suffix}",
        action_kind="static_preset",
        schedule_id="static_full",
    )


def w2_sessions(prefix: str = "s") -> EvidenceSessions:
    return EvidenceSessions(
        fit_session_ids=(f"{prefix}-fit-2", f"{prefix}-fit-1"),
        calibration_session_ids=(f"{prefix}-cal-2", f"{prefix}-cal-1"),
    )


def s1_sessions(prefix: str = "s") -> EvidenceSessions:
    return replace(
        w2_sessions(prefix),
        holdout_session_ids=(f"{prefix}-holdout-1", f"{prefix}-holdout-2"),
        ood_session_ids=(f"{prefix}-ood-1",),
    )


def s2_sessions(prefix: str = "s") -> EvidenceSessions:
    return replace(
        s1_sessions(prefix),
        closed_loop_session_ids=(f"{prefix}-closed-1", f"{prefix}-closed-2"),
        anchor_session_ids=(f"{prefix}-anchor-eval-1", f"{prefix}-anchor-eval-2"),
    )


def replay_audit(
    *,
    second_count: int = 10,
    second_actual_hash: str = ANCHOR_HASH_B,
    forbidden_session_ids=(),
):
    return audit_replay_anchors(
        (
            ReplayAnchorEvidence(
                anchor_id="old-static",
                source_session_id="old-source-session-1",
                replay_count=10,
                expected_source_fingerprint=ANCHOR_HASH_A,
                actual_source_fingerprint=ANCHOR_HASH_A,
                time_integrity_ok=True,
            ),
            ReplayAnchorEvidence(
                anchor_id="old-timed",
                source_session_id="old-source-session-2",
                replay_count=second_count,
                expected_source_fingerprint=ANCHOR_HASH_B,
                actual_source_fingerprint=second_actual_hash,
                time_integrity_ok=True,
            ),
        ),
        required_anchor_ids=("old-static", "old-timed"),
        forbidden_session_ids=forbidden_session_ids,
    )


def evidence(
    grade: EvidenceGrade,
    *,
    evidence_key: EvidenceKey | None = None,
) -> LocalEvidence:
    sessions = EvidenceSessions()
    checks = EvidenceChecks()
    replay = None
    if grade is EvidenceGrade.W1:
        sessions = EvidenceSessions(fit_session_ids=("single-fit",))
    elif grade is EvidenceGrade.W2:
        sessions = w2_sessions()
        checks = EvidenceChecks(joint_tube_frozen=True)
    elif grade is EvidenceGrade.S1:
        sessions = s1_sessions()
        checks = EvidenceChecks(
            joint_tube_frozen=True,
            holdout_untouched=True,
            holdout_coverage_passed=True,
            ood_abstention_passed=True,
        )
    elif grade is EvidenceGrade.S2:
        sessions = s2_sessions()
        checks = EvidenceChecks(
            joint_tube_frozen=True,
            holdout_untouched=True,
            holdout_coverage_passed=True,
            ood_abstention_passed=True,
            paired_closed_loop_passed=True,
            anchor_cohort_passed=True,
        )
        replay = replay_audit(
            forbidden_session_ids=(
                sessions.calibration_session_ids
                + sessions.holdout_session_ids
                + sessions.ood_session_ids
                + sessions.closed_loop_session_ids
                + sessions.anchor_session_ids
            )
        )
    return LocalEvidence(
        key=evidence_key or key(),
        sessions=sessions,
        integrity=(
            EvidenceIntegrity(False, False, False)
            if grade is EvidenceGrade.W0
            else EvidenceIntegrity.complete()
        ),
        checks=checks,
        model_fingerprint=MODEL_HASH,
        replay_anchor_audit=replay,
    )


def test_evidence_key_is_exact_local_identity():
    assert key().token == "smooth-static/static_preset/static_full"
    assert key("static") != key("timed")
    with pytest.raises(EvidenceError, match="trimmed"):
        EvidenceKey(" smooth", "static_preset", "static_full")
    with pytest.raises(EvidenceError, match="non-empty"):
        EvidenceKey("smooth", "", "static_full")


def test_gate_types_are_fail_closed_instead_of_using_truthiness():
    with pytest.raises(EvidenceError, match="time_integrity_ok must be bool"):
        EvidenceIntegrity(True, True, "false")  # type: ignore[arg-type]
    with pytest.raises(EvidenceError, match="joint_tube_frozen must be bool"):
        EvidenceChecks(joint_tube_frozen=1)  # type: ignore[arg-type]
    with pytest.raises(EvidenceError, match="replay_count"):
        ReplayAnchorEvidence(
            "old",
            "source",
            "10",  # type: ignore[arg-type]
            ANCHOR_HASH_A,
            ANCHOR_HASH_A,
            True,
        )


@pytest.mark.parametrize(
    ("integrity", "sessions", "model_fingerprint", "reason"),
    [
        (
            EvidenceIntegrity(False, True, True),
            EvidenceSessions(fit_session_ids=("fit",)),
            MODEL_HASH,
            "full_state_incomplete",
        ),
        (
            EvidenceIntegrity(True, False, True),
            EvidenceSessions(fit_session_ids=("fit",)),
            MODEL_HASH,
            "session_metadata_incomplete",
        ),
        (
            EvidenceIntegrity(True, True, False),
            EvidenceSessions(fit_session_ids=("fit",)),
            MODEL_HASH,
            "time_integrity_failed",
        ),
        (
            EvidenceIntegrity(True, True, True, session_identity_matches=False),
            EvidenceSessions(fit_session_ids=("fit",)),
            MODEL_HASH,
            "params_protocol_session_mismatch",
        ),
        (
            EvidenceIntegrity.complete(),
            EvidenceSessions(),
            MODEL_HASH,
            "physical_session_ids_missing",
        ),
        (
            EvidenceIntegrity.complete(),
            EvidenceSessions(fit_session_ids=("fit",)),
            None,
            "model_fingerprint_missing",
        ),
        (
            EvidenceIntegrity(
                True,
                True,
                True,
                missing_fields=("decision.vz",),
            ),
            EvidenceSessions(fit_session_ids=("fit",)),
            MODEL_HASH,
            "full_state_incomplete(decision.vz)",
        ),
    ],
)
def test_missing_integrity_session_or_fingerprint_is_w0(
    integrity,
    sessions,
    model_fingerprint,
    reason,
):
    assessment = assess_local_evidence(LocalEvidence(
        key=key(),
        sessions=sessions,
        integrity=integrity,
        checks=EvidenceChecks(joint_tube_frozen=True),
        model_fingerprint=model_fingerprint,
    ))
    assert assessment.grade is EvidenceGrade.W0
    assert reason in assessment.blocking_reasons
    assert not assessment.candidate_eligible


def test_historical_or_single_session_is_capped_at_w1():
    single = assess_local_evidence(evidence(EvidenceGrade.W1))
    historical = assess_local_evidence(replace(
        evidence(EvidenceGrade.S2),
        checks=replace(
            evidence(EvidenceGrade.S2).checks,
            historical_or_discovery=True,
        ),
    ))
    assert single.grade is EvidenceGrade.W1
    assert "single_physical_session" in single.blocking_reasons
    assert historical.grade is EvidenceGrade.W1
    assert "historical_or_discovery_evidence" in historical.blocking_reasons


def test_independent_fit_cal_and_frozen_joint_tube_earn_w2():
    assessment = assess_local_evidence(evidence(EvidenceGrade.W2))
    no_tube = assess_local_evidence(replace(
        evidence(EvidenceGrade.W2),
        checks=EvidenceChecks(joint_tube_frozen=False),
    ))
    assert assessment.grade is EvidenceGrade.W2
    assert assessment.candidate_eligible
    assert not assessment.shadow_eligible
    assert no_tube.grade is EvidenceGrade.W1
    assert "joint_tube_not_frozen" in no_tube.blocking_reasons


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "fit_session_ids": ("same",),
            "calibration_session_ids": ("same",),
        },
        {
            "fit_session_ids": ("same",),
            "holdout_session_ids": ("same",),
        },
        {
            "calibration_session_ids": ("same",),
            "holdout_session_ids": ("same",),
        },
        {
            "holdout_session_ids": ("same",),
            "closed_loop_session_ids": ("same",),
        },
    ],
)
def test_session_roles_are_fail_closed_and_never_reused(overrides):
    with pytest.raises(SessionIsolationError, match="session leakage"):
        EvidenceSessions(**overrides)


def test_raw_session_audit_reports_overlap_without_granting_a_grade():
    audit = audit_session_isolation({
        "fit_session_ids": ("shared", "fit-only"),
        "calibration_session_ids": ("shared",),
        "holdout_session_ids": ("holdout",),
    })
    assert not audit.passed
    assert audit.session_count == 3
    assert audit.overlaps[0].session_id == "shared"
    assert audit.overlaps[0].roles == (
        "calibration_session_ids",
        "fit_session_ids",
    )


@pytest.mark.parametrize(
    ("checks", "sessions", "reason"),
    [
        (
            replace(
                evidence(EvidenceGrade.S1).checks,
                holdout_untouched=False,
            ),
            s1_sessions(),
            "holdout_not_untouched",
        ),
        (
            replace(
                evidence(EvidenceGrade.S1).checks,
                holdout_coverage_passed=False,
            ),
            s1_sessions(),
            "holdout_coverage_failed",
        ),
        (
            replace(
                evidence(EvidenceGrade.S1).checks,
                ood_abstention_passed=False,
            ),
            s1_sessions(),
            "ood_abstention_failed",
        ),
        (
            evidence(EvidenceGrade.S1).checks,
            replace(s1_sessions(), ood_session_ids=()),
            "ood_sessions_missing",
        ),
    ],
)
def test_s1_requires_untouched_holdout_coverage_and_ood(
    checks,
    sessions,
    reason,
):
    passed = assess_local_evidence(evidence(EvidenceGrade.S1))
    blocked = assess_local_evidence(replace(
        evidence(EvidenceGrade.S1),
        checks=checks,
        sessions=sessions,
    ))
    assert passed.grade is EvidenceGrade.S1
    assert passed.shadow_eligible
    assert blocked.grade is EvidenceGrade.W2
    assert reason in blocked.blocking_reasons


def test_s2_requires_independent_closed_loop_anchors_and_replay_audit():
    passed = assess_local_evidence(evidence(EvidenceGrade.S2))
    no_replay = assess_local_evidence(replace(
        evidence(EvidenceGrade.S2),
        replay_anchor_audit=None,
    ))
    anchor_failed = assess_local_evidence(replace(
        evidence(EvidenceGrade.S2),
        checks=replace(
            evidence(EvidenceGrade.S2).checks,
            anchor_cohort_passed=False,
        ),
    ))
    assert passed.grade is EvidenceGrade.S2
    assert passed.deploy_eligible
    assert no_replay.grade is EvidenceGrade.S1
    assert "replay_anchor_audit_missing" in no_replay.blocking_reasons
    assert anchor_failed.grade is EvidenceGrade.S1
    assert "anchor_cohort_failed" in anchor_failed.blocking_reasons


def test_replay_anchor_audit_checks_balance_lineage_and_eval_isolation():
    assert replay_audit().passed

    unbalanced = replay_audit(second_count=13)
    assert not unbalanced.passed
    assert any("replay_not_session_balanced" in reason for reason in unbalanced.reasons)

    mismatch = replay_audit(second_actual_hash=ANCHOR_HASH_A)
    assert not mismatch.passed
    assert "replay_fingerprint_mismatch(old-timed)" in mismatch.reasons

    leaked = replay_audit(forbidden_session_ids=("old-source-session-2",))
    assert not leaked.passed
    assert leaked.forbidden_session_overlaps == ("old-source-session-2",)


def test_replay_anchor_audit_fails_on_missing_or_unclean_anchor():
    record = ReplayAnchorEvidence(
        anchor_id="old-static",
        source_session_id="old-source-session-1",
        replay_count=10,
        expected_source_fingerprint=ANCHOR_HASH_A,
        actual_source_fingerprint=ANCHOR_HASH_A,
        time_integrity_ok=False,
    )
    audit = audit_replay_anchors(
        (record,),
        required_anchor_ids=("old-static", "old-timed"),
    )
    assert not audit.passed
    assert audit.missing_anchor_ids == ("old-timed",)
    assert "replay_time_integrity_failed(old-static)" in audit.reasons


def test_discovered_failure_can_downgrade_existing_s2():
    previous = assess_local_evidence(evidence(EvidenceGrade.S2))
    candidate = assess_local_evidence(replace(
        evidence(EvidenceGrade.S2),
        failures=(
            EvidenceFailure(
                failure_id="rough-anchor-regression",
                reason="anchor noninferiority failed",
                maximum_grade=EvidenceGrade.W2,
            ),
        ),
    ))
    decision = decide_promotion(previous, candidate)
    assert candidate.uncapped_grade is EvidenceGrade.S2
    assert candidate.grade is EvidenceGrade.W2
    assert decision.action is PromotionAction.DOWNGRADE
    assert decision.effective_grade is EvidenceGrade.W2
    assert not decision.deploy_eligible
    assert "support_may_contract" in decision.reason


def test_overclaim_fails_closed_while_removal_is_safe_contraction():
    w2 = assess_local_evidence(evidence(EvidenceGrade.W2))
    with pytest.raises(EvidencePromotionError, match="requested S2 but earned W2"):
        assert_grade_authorized(w2, EvidenceGrade.S2)

    removed = decide_promotion(w2, None)
    assert removed.action is PromotionAction.WITHDRAW
    assert removed.effective_grade is None
    assert not removed.support_active


def test_catalog_grades_are_local_and_support_can_shrink():
    deploy_key = key("deploy")
    weak_key = key("new-terrain")
    previous = assess_evidence_catalog((
        evidence(EvidenceGrade.S2, evidence_key=deploy_key),
        evidence(EvidenceGrade.W1, evidence_key=weak_key),
    ))
    assert {item.key: item.grade for item in previous} == {
        deploy_key: EvidenceGrade.S2,
        weak_key: EvidenceGrade.W1,
    }

    candidate = assess_evidence_catalog((
        evidence(EvidenceGrade.W2, evidence_key=weak_key),
    ))
    plan = plan_catalog_update(
        previous,
        candidate,
        catalog_version="catalog-2",
    )
    assert isinstance(plan, CatalogPromotionPlan)
    assert plan.active_keys == (weak_key,)
    assert plan.deploy_keys == ()
    assert plan.withdrawn_keys == (deploy_key,)
    assert {
        decision.key: decision.action for decision in plan.decisions
    } == {
        deploy_key: PromotionAction.WITHDRAW,
        weak_key: PromotionAction.PROMOTE,
    }


def test_duplicate_catalog_key_is_rejected():
    item = evidence(EvidenceGrade.W2)
    with pytest.raises(EvidenceError, match="duplicate EvidenceKey"):
        assess_evidence_catalog((item, item))


def test_versioned_fingerprint_is_canonical_and_version_sensitive():
    first = evidence(EvidenceGrade.W2)
    reordered = replace(
        first,
        sessions=EvidenceSessions(
            fit_session_ids=("s-fit-1", "s-fit-2"),
            calibration_session_ids=("s-cal-1", "s-cal-2"),
        ),
    )
    assert local_evidence_fingerprint(first) == local_evidence_fingerprint(reordered)

    changed = replace(
        first,
        checks=replace(first.checks, joint_tube_frozen=False),
    )
    assert local_evidence_fingerprint(first) != local_evidence_fingerprint(changed)

    next_version = local_evidence_fingerprint(
        first,
        fingerprint_version="darts-calmap-evidence-fingerprint/2",
    )
    assert next_version.digest != local_evidence_fingerprint(first).digest
    assert EVIDENCE_FINGERPRINT_VERSION in local_evidence_fingerprint(first).token


def test_fingerprint_and_assessment_are_json_ready():
    assessment = assess_local_evidence(evidence(EvidenceGrade.S2))
    encoded = json.dumps(assessment.as_dict(), sort_keys=True)
    assert '"grade": "S2"' in encoded
    assert assessment.evidence_fingerprint.digest in encoded

    generic = versioned_fingerprint({"key": key(), "grade": EvidenceGrade.S1})
    assert len(generic.digest) == 64
