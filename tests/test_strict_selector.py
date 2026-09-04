from __future__ import annotations

from dataclasses import replace

from control.darts.calibration_map import (
    ActionConditionedCalibrationMap,
    ActionConditionedCalibrationPoint,
    ActionConditionedDelta,
    FrozenResidualTube,
)
from control.darts.landing_capture import (
    CornerPrecontactInterval,
    LandingCaptureRiskModelV2,
    SimulatorLandingCaptureRegion,
    landing_capture_risk_fingerprint,
)
from control.darts.strict_selector import (
    AdverseRegimeAssessment,
    ArtifactHash,
    CandidateAction,
    LandingResourceTube,
    MeasuredDecisionState,
    OutcomeQualifierEvidence,
    SelectorAuthorityConfig,
    StrictSelectorContext,
    TakeoffStateTube,
    strict_select_action,
)


def delta(theta: float = 0.05, wheel: float = 0.0) -> ActionConditionedDelta:
    return ActionConditionedDelta(
        dtheta0_rad=theta,
        domega0_radps=0.0,
        dvx_mps=0.0,
        dvz_mps=0.0,
        dz_m=0.0,
        dwheels_radps=(wheel, wheel, wheel, wheel),
        dT_s=0.0,
    )


def residual(radius: float = 0.001) -> FrozenResidualTube:
    return FrozenResidualTube(
        residual_set_id="res-static-v1",
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
        radius=ActionConditionedDelta(
            radius,
            radius,
            radius,
            radius,
            radius,
            (radius, radius, radius, radius),
            radius,
        ),
        fit_session_ids=("fit-1",),
        calibration_session_ids=("cal-1",),
        holdout_session_ids=("holdout-1",),
        method="split_conformal",
        target_coverage=0.95,
        source_sha256="b" * 64,
    )


def calmap(action_delta=None) -> ActionConditionedCalibrationMap:
    point = ActionConditionedCalibrationPoint(
        angle_deg=35.0,
        speed_mps=15.0,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
        delta=action_delta or delta(),
        train_session_ids=("fit-1",),
        source_id="static-point",
    )
    return ActionConditionedCalibrationMap((point,), (residual(),))


def capture() -> SimulatorLandingCaptureRegion:
    return SimulatorLandingCaptureRegion(
        model_id="capture-test-v1",
        source_dataset_sha256="c" * 64,
        fit_session_ids=("capture-fit",),
        holdout_session_ids=("capture-holdout",),
        independent_state_injection=True,
        pitch_lo_rad=-0.20,
        pitch_hi_rad=0.20,
        pitch_rate_lo_radps=-0.20,
        pitch_rate_hi_radps=0.20,
        max_abs_roll_rad=0.10,
        max_abs_roll_rate_radps=0.20,
        normal_velocity_lo_mps=-4.0,
        normal_velocity_hi_mps=-0.2,
        min_front_remaining_stroke_m=0.02,
        min_rear_remaining_stroke_m=0.02,
        max_impact_energy_per_mass_jpkg=10.0,
        max_cfc10_peak_decel_mps2=40.0,
        allowed_contact_orders=("front_first", "four_wheel"),
    )


def takeoff() -> TakeoffStateTube:
    return TakeoffStateTube(
        theta0_rad=0.0,
        omega0_radps=0.0,
        vx_mps=15.0,
        vz_mps=3.0,
        z_m=2.0,
        wheels_forward_radps=(40.0, 40.0, 40.0, 40.0),
        T_s=1.0,
        theta0_radius_rad=0.005,
        omega0_radius_radps=0.005,
        vx_radius_mps=0.05,
        vz_radius_mps=0.05,
        z_radius_m=0.02,
        wheel_radius_radps=(1.0, 1.0, 1.0, 1.0),
        T_radius_s=0.01,
        source_id="neutral-predictor-v1",
        residual_set_id="neutral-residual-v1",
        conditioned_on_state_id="decision-state-1",
    )


def resource() -> LandingResourceTube:
    return LandingResourceTube(
        max_abs_roll_rad=0.04,
        max_abs_roll_rate_radps=0.08,
        normal_velocity_lo_mps=-2.0,
        normal_velocity_hi_mps=-1.0,
        min_front_remaining_stroke_m=0.04,
        min_rear_remaining_stroke_m=0.04,
        max_impact_energy_per_mass_jpkg=6.0,
        max_cfc10_peak_decel_mps2=25.0,
        possible_contact_orders=("front_first",),
        source_id="landing-resource-v1",
    )


def evidence(*, eligible: bool = True) -> OutcomeQualifierEvidence:
    return OutcomeQualifierEvidence(
        available=True,
        deploy_eligible=eligible,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
        benefit_score=1.0,
        reason="positive_holdout" if eligible else "no_holdout_gain",
        source_id="outcome-v1",
        source_sha256="d" * 64,
        fit_session_ids=("outcome-fit",),
        holdout_session_ids=("outcome-holdout",),
    )


def context(**overrides) -> StrictSelectorContext:
    values = {
        "method_fingerprint": "strict-v2-test",
        "phase": "final",
        "physical_session_id": "physical-session-1",
        "query_angle_deg": 35.0,
        "query_speed_mps": 15.0,
        "geometry_profile": "tabletop_fillet",
        "decision_state": MeasuredDecisionState(
            state_id="decision-state-1",
            theta_rad=0.0,
            omega_radps=0.0,
            vx_mps=15.0,
            vz_mps=0.0,
            z_m=0.3,
            wheels_raw_radps={
                "FL": -40.0,
                "FR": 40.0,
                "RL": -40.0,
                "RR": 40.0,
            },
            wheel_spin_convention_id="beamng-sbr4-forward-spin/v1",
            sample_step=100,
        ),
        "candidates": (
            CandidateAction("neutral", "neutral", "neutral", "neutral"),
            CandidateAction(
                "static", "static_preset", "static_full", "nose_up", priority=1),
        ),
        "neutral_takeoff_tube": takeoff(),
        "calibration_map": calmap(),
        "capture_region": capture(),
        "landing_resource_tubes": {
            "neutral": resource(),
            "static": resource(),
        },
        "outcome_evidence": {"static": evidence()},
        "adverse_regime": AdverseRegimeAssessment(
            roughness_known=True, curvature_known=True),
        "authority": SelectorAuthorityConfig(
            a_up_radps2=2.0,
            a_down_radps2=2.0,
            a_up_uncertainty_radps2=0.05,
            a_down_uncertainty_radps2=0.05,
            wheel_inertia_kgm2=1.2,
            wheel_omega_min_radps=-125.7,
            wheel_omega_max_radps=125.7,
            body_pitch_inertia_kgm2=2043.0,
        ),
        "artifact_hashes": (
            ArtifactHash("calmap", "a" * 64, "a" * 64),
            ArtifactHash("capture", "b" * 64, "b" * 64),
        ),
        "can_stop_before_lip": True,
    }
    values.update(overrides)
    return StrictSelectorContext(**values)


def audit(decision, candidate_id):
    return next(
        item for item in decision.candidate_audits
        if item.candidate.candidate_id == candidate_id)


def test_strict_selector_certifies_before_outcome_and_deploys_preset():
    decision = strict_select_action(context())
    assert decision.authorized is True
    assert decision.deployment_status == "deploy_preset"
    assert decision.selected_candidate_id == "static"
    assert audit(decision, "static").certified is True
    assert audit(decision, "static").outcome_qualifier_ok is True
    # Single-call wall time is load-sensitive; the preregistered hard gate is
    # enforced by the 3000-call p99 benchmark, not by one CI sample.
    assert 0.0 < decision.latency_ms < 50.0


def test_strict_selector_outcome_rejection_falls_back_to_certified_dart():
    decision = strict_select_action(context(
        outcome_evidence={"static": evidence(eligible=False)}))
    assert decision.authorized is True
    assert decision.deployment_status == "dart_fallback"
    assert decision.selected_candidate_id == "neutral"
    assert audit(decision, "static").certified is True
    assert audit(decision, "static").eligible is False


def test_strict_selector_never_lets_outcome_override_failed_certificate():
    bad_map = calmap(delta(theta=1.0))
    decision = strict_select_action(context(calibration_map=bad_map))
    preset = audit(decision, "static")
    assert preset.outcome_qualifier_ok is True
    assert preset.certified is False
    assert decision.deployment_status == "dart_fallback"
    assert decision.selected_candidate_id == "neutral"


def test_strict_selector_signed_wheel_domain_failure_falls_back():
    bad_map = calmap(delta(theta=0.05, wheel=100.0))
    decision = strict_select_action(context(calibration_map=bad_map))
    preset = audit(decision, "static")
    assert preset.certified is False
    assert "outside" in preset.capture_resource_reasons[0]
    assert decision.deployment_status == "dart_fallback"


def test_state_only_mode_does_not_consume_wheel_budget():
    bad_map = calmap(delta(theta=0.05, wheel=100.0))
    decision = strict_select_action(context(
        calibration_map=bad_map,
        certificate_mode="state_only",
    ))
    assert decision.selected_candidate_id == "static"
    assert decision.deployment_status == "deploy_preset"
    assert audit(decision, "static").wheel_budget is None


def test_budget_only_mode_ignores_terminal_angle_and_capture():
    bad_state_map = calmap(delta(theta=1.0))
    decision = strict_select_action(context(
        calibration_map=bad_state_map,
        certificate_mode="budget_only",
    ))
    assert decision.selected_candidate_id == "static"
    assert decision.deployment_status == "deploy_preset"
    assert audit(decision, "static").capture_resources_ok is True


def test_capture_on_off_is_explicitly_audited():
    unsafe_resource = replace(resource(), max_abs_roll_rad=1.0)
    blocked = strict_select_action(context(
        landing_resource_tubes={
            "neutral": unsafe_resource,
            "static": unsafe_resource,
        },
    ))
    assert blocked.deployment_status == "abort"
    bypassed = strict_select_action(context(
        landing_resource_tubes={
            "neutral": unsafe_resource,
            "static": unsafe_resource,
        },
        capture_required=False,
    ))
    assert bypassed.deployment_status == "deploy_preset"
    assert bypassed.capture_required is False


def risk_corners():
    return tuple(
        CornerPrecontactInterval(
            wheel_name=name,
            gap_lo_m=0.03,
            gap_hi_m=0.05,
            closing_speed_lo_mps=2.0,
            closing_speed_hi_mps=2.2,
            stroke_lo_m=0.04,
            stroke_hi_m=0.05,
        )
        for name in ("FL", "FR", "RL", "RR")
    )


def risk_model(threshold: float):
    return LandingCaptureRiskModelV2(
        model_id="capture-risk-v2",
        source_dataset_sha256="d" * 64,
        fit_session_ids=("fit",),
        calibration_session_ids=("cal",),
        holdout_session_ids=("holdout",),
        alpha=0.5,
        threshold=threshold,
    )


def test_capture_risk_v2_replaces_axis_aligned_membership():
    model = risk_model(1.0e9)
    decision = strict_select_action(context(
        capture_risk_model=model,
        capture_corner_tubes={
            "neutral": risk_corners(),
            "static": risk_corners(),
        },
    ))
    assert decision.deployment_status == "deploy_preset"
    assert decision.capture_model_fingerprint == (
        landing_capture_risk_fingerprint(model))
    assert audit(decision, "static").capture_risk_bounds is not None
    assert audit(decision, "static").capture_risk_threshold == model.threshold


def test_capture_risk_v2_fails_closed_above_threshold():
    decision = strict_select_action(context(
        capture_risk_model=risk_model(1.0e-6),
        capture_corner_tubes={
            "neutral": risk_corners(),
            "static": risk_corners(),
        },
    ))
    assert decision.deployment_status == "abort"
    assert "capture_risk_upper_not_below_threshold" in (
        audit(decision, "static").capture_resource_reasons)


class EvidenceMap:
    def __init__(self, base, grade):
        self.base = base
        self.grade = grade

    def query(self, *args, **kwargs):
        result = self.base.query(*args, **kwargs)
        return replace(
            result,
            evidence_grade=self.grade,
            backend="neural_v2",
        )


def test_progressive_mode_uses_evidence_and_treats_reachability_as_audit():
    unsafe_resource = replace(resource(), max_abs_roll_rad=1.0)
    decision = strict_select_action(context(
        calibration_map=EvidenceMap(calmap(), "S2"),
        landing_resource_tubes={
            "neutral": unsafe_resource,
            "static": unsafe_resource,
        },
        authorization_mode="progressive_evidence",
        minimum_evidence_grade="S2",
    ))
    assert decision.deployment_status == "deploy_preset"
    assert decision.authorization_mode == "progressive_evidence"
    assert "capture_and_reachability_audit_only" in (
        audit(decision, "static").capture_resource_reasons)


def test_progressive_mode_falls_back_on_low_grade_or_operational_guard():
    low_grade = strict_select_action(context(
        calibration_map=EvidenceMap(calmap(), "W2"),
        authorization_mode="progressive_evidence",
        minimum_evidence_grade="S2",
    ))
    assert low_grade.deployment_status == "dart_fallback"
    assert "evidence_grade_below_runtime_minimum" in (
        audit(low_grade, "static").capture_resource_reasons)

    high_wheel = strict_select_action(context(
        calibration_map=EvidenceMap(calmap(delta(wheel=180.0)), "S2"),
        authorization_mode="progressive_evidence",
        minimum_evidence_grade="S2",
    ))
    assert high_wheel.deployment_status == "dart_fallback"
    assert "operational_wheel_guard_exceeded" in (
        audit(high_wheel, "static").capture_resource_reasons)


def test_strict_selector_unknown_adverse_context_blocks_only_preset():
    decision = strict_select_action(context(
        adverse_regime=AdverseRegimeAssessment(
            roughness_known=False, curvature_known=True)))
    preset = audit(decision, "static")
    assert preset.certified is True
    assert preset.outcome_qualifier_ok is False
    assert "roughness_unknown" in preset.outcome_reasons
    assert decision.deployment_status == "dart_fallback"


def test_strict_selector_missing_capture_or_hash_mismatch_aborts():
    missing = strict_select_action(context(capture_region=None))
    assert missing.authorized is False
    assert missing.reason == "landing_capture_model_missing"
    mismatch = strict_select_action(context(artifact_hashes=(
        ArtifactHash("calmap", "a" * 64, "b" * 64),
    )))
    assert mismatch.authorized is False
    assert mismatch.reason == "artifact_hash_mismatch(calmap)"


def test_final_authorization_requires_measured_and_linked_decision_state():
    estimated = replace(
        context().decision_state,
        source="gate_table_median",
    )
    decision = strict_select_action(context(decision_state=estimated))
    assert decision.authorized is False
    assert decision.reason == "final_authorization_requires_measured_decision_state"
    unlinked = replace(takeoff(), conditioned_on_state_id="other-state")
    decision = strict_select_action(context(neutral_takeoff_tube=unlinked))
    assert decision.authorized is False
    assert decision.reason == "takeoff_prediction_decision_state_mismatch"


def test_strict_selector_aborts_when_neutral_and_preset_are_uncertified():
    bad_takeoff = replace(
        takeoff(),
        theta0_rad=1.0,
        T_s=0.2,
        T_radius_s=0.01,
    )
    decision = strict_select_action(context(neutral_takeoff_tube=bad_takeoff))
    assert decision.authorized is False
    assert decision.selected_candidate_id is None
    assert decision.deployment_status == "abort"
    assert decision.reason == "no_certified_action; stop_before_lip"


def test_proposal_phase_never_authorizes_an_action():
    decision = strict_select_action(context(phase="proposal"))
    assert decision.authorized is False
    assert decision.selected_candidate_id == "static"
    assert decision.deployment_status == "proposal_only"
    assert "would_deploy_preset" in decision.reason


def test_strict_selector_audit_is_json_ready():
    payload = strict_select_action(context()).as_dict()
    assert payload["schema_version"] == "darts-strict-selector/2"
    assert payload["candidate_audits"][1]["certificate"]["reachable"] is True
    assert payload["candidate_audits"][1]["wheel_budget"]["convention_id"]
    assert payload["capture_model_fingerprint"]
