"""Certificate-first pure selector for DART-S strict_v2.

The selector performs no file I/O and does not ask an empirical table to
recommend an arm. It enumerates a frozen action catalog, certifies every
action-conditioned state/budget tube, and only then applies independent
outcome evidence and adverse-regime vetoes.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Mapping

from control.darts.calibration_map import (
    ActionConditionedCalibrationMap,
    ActionConditionedDelta,
    ActionConditionedEstimate,
)
from control.darts.erosion import (
    ExactCertificateUncertainty,
    RobustExactPitchCertificate,
    robust_exact_pitch_certificate,
)
from control.darts.landing_capture import (
    CaptureRiskBounds,
    CornerPrecontactInterval,
    LandingCaptureRiskModelV2,
    SimulatorLandingCaptureRegion,
    landing_capture_fingerprint,
    landing_capture_risk_fingerprint,
)
from control.darts.reachability import PitchAuthority, PitchTerminalRegion
from control.darts.wheel_budget import (
    BEAMNG_SBR4_SPIN_CONVENTION_ID,
    BEAMNG_SBR4_WHEEL_ORDER,
    BudgetInputMissingError,
    BudgetOutOfDomainError,
    CertifiedWheelBudget,
    WheelRateInterval,
    certified_directional_rate_budgets,
    resolve_signed_budget_inputs,
)


STRICT_SELECTOR_SCHEMA_VERSION = "darts-strict-selector/2"
VALID_ACTION_KINDS = frozenset({"neutral", "static_preset", "timed_preset"})
VALID_PHASES = frozenset({"proposal", "final"})
VALID_CERTIFICATE_MODES = frozenset({"state_only", "budget_only", "joint"})
VALID_AUTHORIZATION_MODES = frozenset({
    "certificate_first",
    "progressive_evidence",
})


@dataclass(frozen=True)
class MeasuredDecisionState:
    state_id: str
    theta_rad: float
    omega_radps: float
    vx_mps: float
    vz_mps: float
    z_m: float
    wheels_raw_radps: Mapping[str, float]
    wheel_spin_convention_id: str
    sample_step: int
    roll_rad: float = 0.0
    roll_rate_radps: float = 0.0
    source: str = "measured_decision_state"

    def __post_init__(self):
        values = (
            self.theta_rad,
            self.omega_radps,
            self.vx_mps,
            self.vz_mps,
            self.z_m,
            self.roll_rad,
            self.roll_rate_radps,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("measured decision state must be finite")
        if not self.state_id or self.sample_step < 0:
            raise ValueError("decision-state ID and nonnegative sample step are required")

    def signed_wheel_inputs(self):
        return resolve_signed_budget_inputs({
            "wheel_spin_convention_id": self.wheel_spin_convention_id,
            "wheels0_radps": self.wheels_raw_radps,
        })


@dataclass(frozen=True)
class TakeoffStateTube:
    theta0_rad: float
    omega0_radps: float
    vx_mps: float
    vz_mps: float
    z_m: float
    wheels_forward_radps: tuple[float, float, float, float]
    T_s: float
    theta0_radius_rad: float
    omega0_radius_radps: float
    vx_radius_mps: float
    vz_radius_mps: float
    z_radius_m: float
    wheel_radius_radps: tuple[float, float, float, float]
    T_radius_s: float
    source_id: str
    residual_set_id: str
    conditioned_on_state_id: str

    def __post_init__(self):
        values = (
            self.theta0_rad,
            self.omega0_radps,
            self.vx_mps,
            self.vz_mps,
            self.z_m,
            *self.wheels_forward_radps,
            self.T_s,
            self.theta0_radius_rad,
            self.omega0_radius_radps,
            self.vx_radius_mps,
            self.vz_radius_mps,
            self.z_radius_m,
            *self.wheel_radius_radps,
            self.T_radius_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("takeoff-state tube values must be finite")
        radii = (
            self.theta0_radius_rad,
            self.omega0_radius_radps,
            self.vx_radius_mps,
            self.vz_radius_mps,
            self.z_radius_m,
            *self.wheel_radius_radps,
            self.T_radius_s,
        )
        if any(value < 0.0 for value in radii):
            raise ValueError("takeoff-state tube radii must be nonnegative")
        if self.T_s - self.T_radius_s <= 0.0:
            raise ValueError("takeoff-state tube must have positive minimum flight time")
        if not (
            self.source_id
            and self.residual_set_id
            and self.conditioned_on_state_id
        ):
            raise ValueError(
                "takeoff-state source, residual, and decision-state IDs are required")


@dataclass(frozen=True)
class LandingResourceTube:
    max_abs_roll_rad: float
    max_abs_roll_rate_radps: float
    normal_velocity_lo_mps: float
    normal_velocity_hi_mps: float
    min_front_remaining_stroke_m: float
    min_rear_remaining_stroke_m: float
    max_impact_energy_per_mass_jpkg: float
    max_cfc10_peak_decel_mps2: float
    possible_contact_orders: tuple[str, ...]
    source_id: str

    def __post_init__(self):
        values = (
            self.max_abs_roll_rad,
            self.max_abs_roll_rate_radps,
            self.normal_velocity_lo_mps,
            self.normal_velocity_hi_mps,
            self.min_front_remaining_stroke_m,
            self.min_rear_remaining_stroke_m,
            self.max_impact_energy_per_mass_jpkg,
            self.max_cfc10_peak_decel_mps2,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("landing-resource tube values must be finite")
        if self.normal_velocity_lo_mps > self.normal_velocity_hi_mps:
            raise ValueError("landing normal-velocity interval is reversed")
        if any(value < 0.0 for value in (
            self.max_abs_roll_rad,
            self.max_abs_roll_rate_radps,
            self.min_front_remaining_stroke_m,
            self.min_rear_remaining_stroke_m,
            self.max_impact_energy_per_mass_jpkg,
            self.max_cfc10_peak_decel_mps2,
        )):
            raise ValueError("landing resource magnitudes must be nonnegative")
        if not self.possible_contact_orders or not self.source_id:
            raise ValueError("landing contact-order support and source ID are required")

    def within(
        self,
        capture: SimulatorLandingCaptureRegion,
    ) -> tuple[bool, tuple[str, ...]]:
        reasons: list[str] = []
        if self.max_abs_roll_rad > capture.max_abs_roll_rad:
            reasons.append("roll_outside_capture")
        if self.max_abs_roll_rate_radps > capture.max_abs_roll_rate_radps:
            reasons.append("roll_rate_outside_capture")
        if self.normal_velocity_lo_mps < capture.normal_velocity_lo_mps:
            reasons.append("normal_velocity_below_capture")
        if self.normal_velocity_hi_mps > capture.normal_velocity_hi_mps:
            reasons.append("normal_velocity_above_capture")
        if (
            self.min_front_remaining_stroke_m
            < capture.min_front_remaining_stroke_m
        ):
            reasons.append("front_stroke_below_capture")
        if (
            self.min_rear_remaining_stroke_m
            < capture.min_rear_remaining_stroke_m
        ):
            reasons.append("rear_stroke_below_capture")
        if (
            self.max_impact_energy_per_mass_jpkg
            > capture.max_impact_energy_per_mass_jpkg
        ):
            reasons.append("impact_energy_above_capture")
        if (
            self.max_cfc10_peak_decel_mps2
            > capture.max_cfc10_peak_decel_mps2
        ):
            reasons.append("cfc10_load_above_capture")
        unsupported_orders = sorted(
            set(self.possible_contact_orders)
            - set(capture.allowed_contact_orders)
        )
        if unsupported_orders:
            reasons.append(
                f"unsupported_contact_orders({','.join(unsupported_orders)})")
        return not reasons, tuple(reasons)


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    action_kind: str
    schedule_id: str
    command_arm: str
    priority: int = 0

    def __post_init__(self):
        if self.action_kind not in VALID_ACTION_KINDS:
            raise ValueError(f"unsupported action_kind {self.action_kind!r}")
        if not self.candidate_id or not self.schedule_id or not self.command_arm:
            raise ValueError("candidate ID, schedule ID, and command arm are required")
        if self.action_kind == "neutral" and self.command_arm != "neutral":
            raise ValueError("neutral candidate must command neutral")


@dataclass(frozen=True)
class OutcomeQualifierEvidence:
    available: bool
    deploy_eligible: bool
    action_kind: str
    geometry_profile: str
    schedule_id: str
    benefit_score: float
    reason: str
    source_id: str
    source_sha256: str
    fit_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]

    def __post_init__(self):
        if not math.isfinite(float(self.benefit_score)):
            raise ValueError("outcome benefit_score must be finite")
        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("outcome source_sha256 must be a 64-character hex digest")
        if self.available:
            if not self.fit_session_ids or not self.holdout_session_ids:
                raise ValueError("available outcome evidence requires fit and holdout sessions")
            leaked = set(self.fit_session_ids) & set(self.holdout_session_ids)
            if leaked:
                raise ValueError(f"outcome evidence session leakage: {sorted(leaked)}")


@dataclass(frozen=True)
class AdverseRegimeAssessment:
    roughness_known: bool
    curvature_known: bool
    roughness_harm_veto: bool = False
    curvature_harm_veto: bool = False
    reasons: tuple[str, ...] = ()

    def preset_allowed(self) -> tuple[bool, tuple[str, ...]]:
        reasons = list(self.reasons)
        if not self.roughness_known:
            reasons.append("roughness_unknown")
        if not self.curvature_known:
            reasons.append("curvature_unknown")
        if self.roughness_harm_veto:
            reasons.append("roughness_harm_veto")
        if self.curvature_harm_veto:
            reasons.append("curvature_harm_veto")
        return not reasons, tuple(reasons)


@dataclass(frozen=True)
class SelectorAuthorityConfig:
    a_up_radps2: float
    a_down_radps2: float
    a_up_uncertainty_radps2: float
    a_down_uncertainty_radps2: float
    wheel_inertia_kgm2: float
    wheel_omega_min_radps: float
    wheel_omega_max_radps: float
    body_pitch_inertia_kgm2: float
    terminal_theta_uncertainty_rad: float = 0.0
    terminal_omega_uncertainty_radps: float = 0.0

    def __post_init__(self):
        positive = (
            self.a_up_radps2,
            self.a_down_radps2,
            self.wheel_inertia_kgm2,
            self.body_pitch_inertia_kgm2,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("selector authority magnitudes must be finite and positive")
        if not (
            math.isfinite(self.wheel_omega_min_radps)
            and math.isfinite(self.wheel_omega_max_radps)
            and self.wheel_omega_min_radps < self.wheel_omega_max_radps
        ):
            raise ValueError("wheel certificate envelope is invalid")
        uncertainties = (
            self.a_up_uncertainty_radps2,
            self.a_down_uncertainty_radps2,
            self.terminal_theta_uncertainty_rad,
            self.terminal_omega_uncertainty_radps,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in uncertainties):
            raise ValueError("selector authority uncertainties must be nonnegative")


@dataclass(frozen=True)
class ArtifactHash:
    name: str
    expected_sha256: str
    actual_sha256: str

    def __post_init__(self):
        for label, digest in (
            ("expected", self.expected_sha256),
            ("actual", self.actual_sha256),
        ):
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError(f"{label} artifact hash must be 64 lowercase hex characters")

    def matches(self) -> bool:
        return self.expected_sha256 == self.actual_sha256


@dataclass(frozen=True)
class StrictSelectorContext:
    method_fingerprint: str
    phase: str
    physical_session_id: str
    query_angle_deg: float
    query_speed_mps: float
    geometry_profile: str
    decision_state: MeasuredDecisionState
    candidates: tuple[CandidateAction, ...]
    neutral_takeoff_tube: TakeoffStateTube
    calibration_map: ActionConditionedCalibrationMap
    capture_region: SimulatorLandingCaptureRegion | None
    landing_resource_tubes: Mapping[str, LandingResourceTube]
    outcome_evidence: Mapping[str, OutcomeQualifierEvidence]
    adverse_regime: AdverseRegimeAssessment
    authority: SelectorAuthorityConfig
    artifact_hashes: tuple[ArtifactHash, ...]
    can_stop_before_lip: bool
    certificate_mode: str = "joint"
    capture_required: bool = True
    capture_risk_model: LandingCaptureRiskModelV2 | None = None
    capture_corner_tubes: Mapping[
        str, tuple[CornerPrecontactInterval, ...]
    ] = field(default_factory=dict)
    authorization_mode: str = "certificate_first"
    minimum_evidence_grade: str = "S2"
    operational_wheel_guard_radps: float = 200.0
    terrain_features: Mapping[str, float] = field(default_factory=dict)
    audit_terminal_region: PitchTerminalRegion | None = None

    def __post_init__(self):
        if self.phase not in VALID_PHASES:
            raise ValueError(f"phase must be one of {sorted(VALID_PHASES)}")
        if not self.method_fingerprint or not self.physical_session_id:
            raise ValueError("method fingerprint and physical session ID are required")
        if not (
            math.isfinite(self.query_angle_deg)
            and math.isfinite(self.query_speed_mps)
        ):
            raise ValueError("selector query must be finite")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        if sum(candidate.action_kind == "neutral" for candidate in self.candidates) != 1:
            raise ValueError("candidate catalog must contain exactly one neutral action")
        if not self.artifact_hashes:
            raise ValueError("strict selector requires artifact hashes")
        if self.certificate_mode not in VALID_CERTIFICATE_MODES:
            raise ValueError(
                f"certificate_mode must be one of "
                f"{sorted(VALID_CERTIFICATE_MODES)}")
        if self.authorization_mode not in VALID_AUTHORIZATION_MODES:
            raise ValueError(
                f"authorization_mode must be one of "
                f"{sorted(VALID_AUTHORIZATION_MODES)}")
        if (
            not math.isfinite(self.operational_wheel_guard_radps)
            or self.operational_wheel_guard_radps <= 0.0
        ):
            raise ValueError("operational wheel guard must be positive")


@dataclass(frozen=True)
class CandidateAudit:
    candidate: CandidateAction
    calibration: ActionConditionedEstimate | None
    predicted_takeoff_tube: TakeoffStateTube | None
    wheel_budget: CertifiedWheelBudget | None
    certificate: RobustExactPitchCertificate | None
    capture_resources_ok: bool
    capture_resource_reasons: tuple[str, ...]
    outcome_qualifier_ok: bool
    outcome_reasons: tuple[str, ...]
    certified: bool
    eligible: bool
    capture_risk_bounds: CaptureRiskBounds | None = None
    capture_risk_threshold: float | None = None
    takeoff_objective_lower: float | None = None
    takeoff_objective_upper: float | None = None
    takeoff_benefit_lcb: float | None = None


@dataclass(frozen=True)
class StrictSelectorDecision:
    schema_version: str
    method_fingerprint: str
    phase: str
    certificate_mode: str
    capture_required: bool
    authorization_mode: str
    authorized: bool
    selected_candidate_id: str | None
    selected_action_kind: str | None
    selected_command_arm: str | None
    deployment_status: str
    reason: str
    candidate_audits: tuple[CandidateAudit, ...]
    artifact_hashes: tuple[ArtifactHash, ...]
    capture_model_fingerprint: str
    physical_session_id: str
    latency_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _apply_delta(
    base: TakeoffStateTube,
    delta: ActionConditionedDelta,
    radius: ActionConditionedDelta,
    residual_set_id: str,
    source_id: str,
) -> TakeoffStateTube:
    return TakeoffStateTube(
        theta0_rad=base.theta0_rad + delta.dtheta0_rad,
        omega0_radps=base.omega0_radps + delta.domega0_radps,
        vx_mps=base.vx_mps + delta.dvx_mps,
        vz_mps=base.vz_mps + delta.dvz_mps,
        z_m=base.z_m + delta.dz_m,
        wheels_forward_radps=tuple(
            base_value + shift
            for base_value, shift in zip(
                base.wheels_forward_radps, delta.dwheels_radps)
        ),
        T_s=base.T_s + delta.dT_s,
        theta0_radius_rad=base.theta0_radius_rad + radius.dtheta0_rad,
        omega0_radius_radps=(
            base.omega0_radius_radps + radius.domega0_radps),
        vx_radius_mps=base.vx_radius_mps + radius.dvx_mps,
        vz_radius_mps=base.vz_radius_mps + radius.dvz_mps,
        z_radius_m=base.z_radius_m + radius.dz_m,
        wheel_radius_radps=tuple(
            base_value + extra
            for base_value, extra in zip(
                base.wheel_radius_radps, radius.dwheels_radps)
        ),
        T_radius_s=base.T_radius_s + radius.dT_s,
        source_id=source_id,
        residual_set_id=residual_set_id,
        conditioned_on_state_id=base.conditioned_on_state_id,
    )


def _candidate_takeoff_tube(
    context: StrictSelectorContext,
    candidate: CandidateAction,
) -> tuple[TakeoffStateTube | None, ActionConditionedEstimate | None]:
    if candidate.action_kind == "neutral":
        return context.neutral_takeoff_tube, None
    signed = context.decision_state.signed_wheel_inputs()
    context_features = {
        "decision_theta_rad": context.decision_state.theta_rad,
        "decision_omega_radps": context.decision_state.omega_radps,
        "decision_roll_rad": context.decision_state.roll_rad,
        "decision_roll_rate_radps": context.decision_state.roll_rate_radps,
        "decision_vx_mps": context.decision_state.vx_mps,
        "decision_vz_mps": context.decision_state.vz_mps,
        "decision_z_m": context.decision_state.z_m,
        **{
            f"decision_wheel_{name}_radps": value
            for name, value in zip(
                signed.wheel_names, signed.canonical_omega_radps)
        },
        **{
            str(name): float(value)
            for name, value in context.terrain_features.items()
        },
    }
    estimate = context.calibration_map.query(
        context.query_angle_deg,
        context.query_speed_mps,
        action_kind=candidate.action_kind,
        geometry_profile=context.geometry_profile,
        schedule_id=candidate.schedule_id,
        context_features=context_features,
    )
    if not estimate.in_domain or estimate.tube is None:
        return None, estimate
    return (
        _apply_delta(
            context.neutral_takeoff_tube,
            estimate.tube.nominal,
            estimate.tube.radius,
            estimate.tube.residual_set_id,
            source_id=(
                f"{context.neutral_takeoff_tube.source_id}+"
                f"{estimate.tube.residual_set_id}"
            ),
        ),
        estimate,
    )


def _wheel_budget(
    tube: TakeoffStateTube,
    config: SelectorAuthorityConfig,
) -> CertifiedWheelBudget:
    intervals = {
        name: WheelRateInterval(value - radius, value + radius)
        for name, value, radius in zip(
            BEAMNG_SBR4_WHEEL_ORDER,
            tube.wheels_forward_radps,
            tube.wheel_radius_radps,
        )
    }
    return certified_directional_rate_budgets(
        intervals,
        I_w=config.wheel_inertia_kgm2,
        omega_max=config.wheel_omega_max_radps,
        omega_min=config.wheel_omega_min_radps,
        J_y=config.body_pitch_inertia_kgm2,
        convention_id=BEAMNG_SBR4_SPIN_CONVENTION_ID,
    )


def _outcome_check(
    context: StrictSelectorContext,
    candidate: CandidateAction,
) -> tuple[bool, tuple[str, ...], float]:
    if candidate.action_kind == "neutral":
        return True, (), 0.0
    reasons: list[str] = []
    evidence = context.outcome_evidence.get(candidate.candidate_id)
    if evidence is None or not evidence.available:
        reasons.append("independent_outcome_evidence_missing")
        score = float("-inf")
    else:
        score = evidence.benefit_score
        if (
            evidence.action_kind != candidate.action_kind
            or evidence.geometry_profile != context.geometry_profile
            or evidence.schedule_id != candidate.schedule_id
        ):
            reasons.append("outcome_evidence_action_mismatch")
        if not evidence.deploy_eligible:
            reasons.append(f"outcome_qualifier_reject({evidence.reason})")
    adverse_ok, adverse_reasons = context.adverse_regime.preset_allowed()
    if not adverse_ok:
        reasons.extend(adverse_reasons)
    return not reasons, tuple(reasons), score


def _capture_model_fingerprint(context: StrictSelectorContext) -> str:
    if context.capture_risk_model is not None:
        return landing_capture_risk_fingerprint(context.capture_risk_model)
    if context.capture_region is not None:
        return landing_capture_fingerprint(context.capture_region)
    return "missing"


def _takeoff_objective_bounds(
    tube: TakeoffStateTube,
    *,
    ramp_angle_deg: float,
    wheel_guard_radps: float,
) -> tuple[float, float]:
    """Frozen robust takeoff-shaping objective; unrelated to reachability."""
    target_theta = math.radians(float(ramp_angle_deg))

    def error_bounds(value: float, radius: float, target: float, scale: float):
        center_error = abs(value - target)
        return (
            max(0.0, center_error - radius) / scale,
            (center_error + radius) / scale,
        )

    theta_lo, theta_hi = error_bounds(
        tube.theta0_rad,
        tube.theta0_radius_rad,
        target_theta,
        math.radians(10.0),
    )
    omega_lo, omega_hi = error_bounds(
        tube.omega0_radps,
        tube.omega0_radius_radps,
        0.0,
        math.radians(20.0),
    )
    wheel_lower = max(
        max(0.0, abs(value) - radius)
        for value, radius in zip(
            tube.wheels_forward_radps, tube.wheel_radius_radps)
    )
    wheel_upper = max(
        abs(value) + radius
        for value, radius in zip(
            tube.wheels_forward_radps, tube.wheel_radius_radps)
    )
    return (
        theta_lo + omega_lo + 0.1 * wheel_lower / wheel_guard_radps,
        theta_hi + omega_hi + 0.1 * wheel_upper / wheel_guard_radps,
    )


def _abort_decision(
    context: StrictSelectorContext,
    *,
    started_ns: int,
    reason: str,
) -> StrictSelectorDecision:
    return StrictSelectorDecision(
        schema_version=STRICT_SELECTOR_SCHEMA_VERSION,
        method_fingerprint=context.method_fingerprint,
        phase=context.phase,
        certificate_mode=context.certificate_mode,
        capture_required=context.capture_required,
        authorization_mode=context.authorization_mode,
        authorized=False,
        selected_candidate_id=None,
        selected_action_kind=None,
        selected_command_arm=None,
        deployment_status="abort",
        reason=reason,
        candidate_audits=(),
        artifact_hashes=context.artifact_hashes,
        capture_model_fingerprint=_capture_model_fingerprint(context),
        physical_session_id=context.physical_session_id,
        latency_ms=(time.perf_counter_ns() - started_ns) / 1e6,
    )


def strict_select_action(context: StrictSelectorContext) -> StrictSelectorDecision:
    """Return a deterministic proposal or final certificate-first authorization."""
    started_ns = time.perf_counter_ns()
    mismatched = sorted(
        artifact.name for artifact in context.artifact_hashes
        if not artifact.matches()
    )
    if mismatched:
        return _abort_decision(
            context,
            started_ns=started_ns,
            reason=f"artifact_hash_mismatch({','.join(mismatched)})",
        )
    if (
        context.capture_region is None
        and context.authorization_mode == "certificate_first"
    ):
        return _abort_decision(
            context,
            started_ns=started_ns,
            reason="landing_capture_model_missing",
        )
    try:
        context.decision_state.signed_wheel_inputs()
    except (BudgetInputMissingError, ValueError) as exc:
        return _abort_decision(
            context,
            started_ns=started_ns,
            reason=f"measured_decision_state_invalid({exc})",
        )
    if (
        context.phase == "final"
        and context.decision_state.source != "measured_decision_state"
    ):
        return _abort_decision(
            context,
            started_ns=started_ns,
            reason="final_authorization_requires_measured_decision_state",
        )
    if (
        context.neutral_takeoff_tube.conditioned_on_state_id
        != context.decision_state.state_id
    ):
        return _abort_decision(
            context,
            started_ns=started_ns,
            reason="takeoff_prediction_decision_state_mismatch",
        )

    audits: list[CandidateAudit] = []
    scores: dict[str, float] = {}
    for candidate in context.candidates:
        calibration = None
        predicted = None
        budget = None
        certificate = None
        capture_ok = False
        capture_reasons: tuple[str, ...] = ()
        capture_risk = None
        progressive_supported = False
        objective_lower = None
        objective_upper = None
        try:
            predicted, calibration = _candidate_takeoff_tube(context, candidate)
            if predicted is None:
                capture_reasons = (
                    f"calibration_unavailable({calibration.reason})"
                    if calibration is not None
                    else "takeoff_prediction_missing",
                )
            else:
                progressive = (
                    context.authorization_mode == "progressive_evidence")
                budget_error = None
                if context.certificate_mode != "state_only":
                    try:
                        budget = _wheel_budget(predicted, context.authority)
                    except (
                        BudgetInputMissingError,
                        BudgetOutOfDomainError,
                    ) as exc:
                        if not progressive:
                            raise
                        budget_error = str(exc)
                certificate_budget_up = (
                    float("inf")
                    if context.certificate_mode == "state_only" or budget is None
                    else budget.B_up
                )
                certificate_budget_down = (
                    float("inf")
                    if context.certificate_mode == "state_only" or budget is None
                    else budget.B_down
                )
                terminal_region = context.audit_terminal_region
                if terminal_region is None and context.capture_region is not None:
                    terminal_region = (
                        context.capture_region.pitch_terminal_region())
                if terminal_region is not None:
                    if context.certificate_mode == "budget_only":
                        terminal_region = PitchTerminalRegion(
                            theta_lo=-1.0e6,
                            theta_hi=1.0e6,
                            omega_lo=terminal_region.omega_lo,
                            omega_hi=terminal_region.omega_hi,
                        )
                    certificate = robust_exact_pitch_certificate(
                        predicted.theta0_rad,
                        predicted.omega0_radps,
                        predicted.T_s,
                        PitchAuthority(
                            a_up=context.authority.a_up_radps2,
                            a_down=context.authority.a_down_radps2,
                            B_up=certificate_budget_up,
                            B_down=certificate_budget_down,
                        ),
                        terminal_region,
                        uncertainty=ExactCertificateUncertainty(
                            theta0=predicted.theta0_radius_rad,
                            omega0=predicted.omega0_radius_radps,
                            T=predicted.T_radius_s,
                            a_up=context.authority.a_up_uncertainty_radps2,
                            a_down=context.authority.a_down_uncertainty_radps2,
                            terminal_theta=(
                                context.authority.terminal_theta_uncertainty_rad),
                            terminal_omega=(
                                context.authority.terminal_omega_uncertainty_radps),
                        ),
                    )
                if progressive:
                    guard_ok = all(
                        abs(value) + radius
                        <= context.operational_wheel_guard_radps
                        for value, radius in zip(
                            predicted.wheels_forward_radps,
                            predicted.wheel_radius_radps,
                        )
                    )
                    grade_ok = candidate.action_kind == "neutral"
                    if calibration is not None:
                        from control.darts.calmap_evidence import grade_at_least
                        grade_ok = (
                            calibration.evidence_grade is not None
                            and grade_at_least(
                                calibration.evidence_grade,
                                context.minimum_evidence_grade,
                            )
                        )
                    progressive_supported = guard_ok and grade_ok
                    objective_lower, objective_upper = (
                        _takeoff_objective_bounds(
                            predicted,
                            ramp_angle_deg=context.query_angle_deg,
                            wheel_guard_radps=(
                                context.operational_wheel_guard_radps),
                        )
                    )
                    reasons = ["capture_and_reachability_audit_only"]
                    if not guard_ok:
                        reasons.append("operational_wheel_guard_exceeded")
                    if not grade_ok:
                        reasons.append("evidence_grade_below_runtime_minimum")
                    if budget_error is not None:
                        reasons.append(
                            f"wheel_budget_audit_unavailable({budget_error})")
                    capture_ok = True
                    capture_reasons = tuple(reasons)
                elif (
                    not context.capture_required
                    or context.certificate_mode == "budget_only"
                ):
                    capture_ok = True
                    capture_reasons = ("capture_bypassed_by_preregistered_mode",)
                else:
                    resource = context.landing_resource_tubes.get(
                        candidate.candidate_id)
                    if resource is None:
                        capture_reasons = ("landing_resource_tube_missing",)
                    elif context.capture_risk_model is not None:
                        corners = context.capture_corner_tubes.get(
                            candidate.candidate_id)
                        if corners is None:
                            capture_reasons = (
                                "capture_corner_tube_missing",)
                        else:
                            capture_ok, capture_risk = (
                                context.capture_risk_model.evaluate(corners))
                            capture_reasons = (
                                ()
                                if capture_ok
                                else (
                                    "capture_risk_upper_not_below_threshold",
                                )
                            )
                    else:
                        capture_ok, capture_reasons = resource.within(
                            context.capture_region)
        except (
            BudgetInputMissingError,
            BudgetOutOfDomainError,
            ValueError,
        ) as exc:
            capture_reasons = (f"candidate_model_error({exc})",)

        if context.authorization_mode == "progressive_evidence":
            outcome_ok = True
            outcome_reasons = ("progressive_takeoff_objective",)
            score = (
                -objective_upper
                if objective_upper is not None else float("-inf"))
        else:
            outcome_ok, outcome_reasons, score = _outcome_check(
                context, candidate)
        scores[candidate.candidate_id] = score
        if context.authorization_mode == "progressive_evidence":
            certified = predicted is not None and progressive_supported
        else:
            certified = (
                predicted is not None
                and certificate is not None
                and certificate.reachable
                and capture_ok
                and (
                    context.certificate_mode == "state_only"
                    or budget is not None
                )
            )
        audits.append(CandidateAudit(
            candidate=candidate,
            calibration=calibration,
            predicted_takeoff_tube=predicted,
            wheel_budget=budget,
            certificate=certificate,
            capture_resources_ok=capture_ok,
            capture_resource_reasons=capture_reasons,
            outcome_qualifier_ok=outcome_ok,
            outcome_reasons=outcome_reasons,
            certified=certified,
            eligible=certified and outcome_ok,
            capture_risk_bounds=capture_risk,
            capture_risk_threshold=(
                context.capture_risk_model.threshold
                if context.capture_risk_model is not None else None
            ),
            takeoff_objective_lower=objective_lower,
            takeoff_objective_upper=objective_upper,
        ))

    if context.authorization_mode == "progressive_evidence":
        neutral_audit = next(
            audit for audit in audits
            if audit.candidate.action_kind == "neutral")
        neutral_lower = neutral_audit.takeoff_objective_lower
        revised = []
        for audit in audits:
            if audit.candidate.action_kind == "neutral":
                revised.append(audit)
                continue
            benefit_lcb = (
                neutral_lower - audit.takeoff_objective_upper
                if (
                    neutral_lower is not None
                    and audit.takeoff_objective_upper is not None
                )
                else None
            )
            positive = benefit_lcb is not None and benefit_lcb > 0.0
            revised.append(replace(
                audit,
                takeoff_benefit_lcb=benefit_lcb,
                eligible=audit.eligible and positive,
                outcome_reasons=(
                    audit.outcome_reasons
                    if positive
                    else audit.outcome_reasons
                    + ("takeoff_benefit_lcb_not_positive",)
                ),
            ))
        audits = revised

    preset_eligible = [
        audit for audit in audits
        if audit.candidate.action_kind != "neutral" and audit.eligible
    ]
    neutral = next(
        audit for audit in audits if audit.candidate.action_kind == "neutral")
    if preset_eligible:
        selected = max(
            preset_eligible,
            key=lambda audit: (
                scores[audit.candidate.candidate_id],
                audit.candidate.priority,
                audit.candidate.candidate_id,
            ),
        )
        status = "deploy_preset"
        reason = (
            "progressive_evidence_and_takeoff_objective_pass"
            if context.authorization_mode == "progressive_evidence"
            else "certificate_then_outcome_qualifier_pass"
        )
    elif neutral.certified:
        selected = neutral
        status = "dart_fallback"
        reason = (
            "no_qualified_preset; neural_evidence_fallback"
            if context.authorization_mode == "progressive_evidence"
            else "no_qualified_preset; neutral_certificate_pass"
        )
    else:
        selected = None
        status = "abort" if context.can_stop_before_lip else "abort_unavoidable"
        reason = (
            "no_certified_action; stop_before_lip"
            if context.can_stop_before_lip
            else "no_certified_action_after_commit_point"
        )

    final_authorized = context.phase == "final" and selected is not None
    if context.phase == "proposal":
        would_status = status
        status = "proposal_only"
        reason = f"proposal_only; would_{would_status}"
    return StrictSelectorDecision(
        schema_version=STRICT_SELECTOR_SCHEMA_VERSION,
        method_fingerprint=context.method_fingerprint,
        phase=context.phase,
        certificate_mode=context.certificate_mode,
        capture_required=context.capture_required,
        authorization_mode=context.authorization_mode,
        authorized=final_authorized,
        selected_candidate_id=(
            selected.candidate.candidate_id if selected is not None else None),
        selected_action_kind=(
            selected.candidate.action_kind if selected is not None else None),
        selected_command_arm=(
            selected.candidate.command_arm if selected is not None else None),
        deployment_status=status,
        reason=reason,
        candidate_audits=tuple(audits),
        artifact_hashes=context.artifact_hashes,
        capture_model_fingerprint=_capture_model_fingerprint(context),
        physical_session_id=context.physical_session_id,
        latency_ms=(time.perf_counter_ns() - started_ns) / 1e6,
    )


__all__ = [
    "STRICT_SELECTOR_SCHEMA_VERSION",
    "MeasuredDecisionState",
    "TakeoffStateTube",
    "LandingResourceTube",
    "CandidateAction",
    "OutcomeQualifierEvidence",
    "AdverseRegimeAssessment",
    "SelectorAuthorityConfig",
    "ArtifactHash",
    "StrictSelectorContext",
    "CandidateAudit",
    "StrictSelectorDecision",
    "strict_select_action",
]
