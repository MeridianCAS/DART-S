"""File-backed adapter from BeamNG runner state to the strict_v2 pure selector."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from control.darts.calibration_map import (
    ActionConditionedCalibrationMap,
    ActionConditionedDelta,
    load_action_conditioned_calibration_map,
)
from control.darts.landing_capture import (
    LandingCaptureRiskModelV2,
    SimulatorLandingCaptureRegion,
    load_landing_capture_region,
    load_landing_capture_risk_model_v2,
)
from control.darts.landing_projection import (
    CaptureCornerProjectionConfig,
    LandingProjectionConfig,
    project_capture_corner_intervals,
    project_landing_resources,
)
from control.darts.reachability import PitchTerminalRegion
from control.darts.strict_selector import (
    AdverseRegimeAssessment,
    ArtifactHash,
    CandidateAction,
    MeasuredDecisionState,
    OutcomeQualifierEvidence,
    SelectorAuthorityConfig,
    StrictSelectorContext,
    TakeoffStateTube,
    strict_select_action,
)
from control.darts.susp_precondition import (
    CURVATURE_N_MIN,
    lip_curvature_margin,
    rough_runup_veto_active,
)
from control.darts.system_outcome_selector import query_system_outcome_table
from control.darts.takeoff_predictor import (
    DecisionStateVector,
    DecisionToTakeoffMap,
    load_decision_to_takeoff_map,
)
from control.darts.wheel_budget import resolve_signed_budget_inputs


CATALOG_SCHEMA = "darts-strict-v2-catalog/1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog_fingerprint(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("method_fingerprint", None)
    return hashlib.sha256(json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StrictRuntimeBundle:
    catalog: dict
    catalog_path: Path
    artifact_hashes: tuple[ArtifactHash, ...]
    action_map: ActionConditionedCalibrationMap | None
    neutral_map: DecisionToTakeoffMap | None
    capture_region: SimulatorLandingCaptureRegion | None
    capture_risk_model: LandingCaptureRiskModelV2 | None
    blockers: tuple[str, ...]


@lru_cache(maxsize=8)
def load_strict_runtime_bundle(catalog_path_text: str) -> StrictRuntimeBundle:
    catalog_path = Path(catalog_path_text).resolve()
    blockers: list[str] = []
    if not catalog_path.exists():
        return StrictRuntimeBundle(
            {}, catalog_path, (), None, None, None, None,
            ("strict_catalog_missing",))
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return StrictRuntimeBundle(
            {}, catalog_path, (), None, None, None, None,
            ("strict_catalog_unreadable",))
    if catalog.get("schema_version") != CATALOG_SCHEMA:
        blockers.append("strict_catalog_schema_mismatch")
    if catalog.get("method_fingerprint") != _catalog_fingerprint(catalog):
        blockers.append("strict_catalog_fingerprint_mismatch")
    blockers.extend(str(reason) for reason in catalog.get("blockers") or [])
    if not catalog.get("deployment_eligible"):
        blockers.append("strict_catalog_not_deployment_eligible")

    artifact_hashes = []
    docs = {}
    root = (catalog_path.parents[1] if catalog_path.parent.name == "calibration"
            else Path.cwd())
    for name, spec in (catalog.get("artifacts") or {}).items():
        try:
            path = (root / str(spec["path"])).resolve()
            actual = _sha256(path)
            expected = str(spec["sha256"])
            artifact_hashes.append(ArtifactHash(name, expected, actual))
            if expected != actual:
                blockers.append(f"artifact_hash_mismatch({name})")
            docs[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
            blockers.append(f"artifact_unreadable({name}:{exc})")
    for name, spec in (catalog.get("code_hashes") or {}).items():
        try:
            path = (root / str(spec["path"])).resolve()
            actual = _sha256(path)
            expected = str(spec["sha256"])
            artifact_hashes.append(ArtifactHash(name, expected, actual))
            if expected != actual:
                blockers.append(f"code_hash_mismatch({name})")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            blockers.append(f"code_hash_unreadable({name}:{exc})")
    action_map = neutral_map = capture_region = capture_risk_model = None
    if not blockers:
        try:
            action_map = load_action_conditioned_calibration_map(
                docs["action_calmap"])
            neutral_map = load_decision_to_takeoff_map(
                docs["neutral_predictor"])
            if "landing_capture_model" in docs:
                capture_region = load_landing_capture_region(
                    docs["landing_capture_model"])
            if "capture_risk_model" in docs:
                capture_risk_model = load_landing_capture_risk_model_v2(
                    docs["capture_risk_model"])
        except (KeyError, TypeError, ValueError) as exc:
            blockers.append(f"strict_model_load_failed({exc})")
    return StrictRuntimeBundle(
        catalog=catalog,
        catalog_path=catalog_path,
        artifact_hashes=tuple(artifact_hashes),
        action_map=action_map,
        neutral_map=neutral_map,
        capture_region=capture_region,
        capture_risk_model=capture_risk_model,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def _blocked_result(bundle: StrictRuntimeBundle) -> dict:
    reason = ";".join(bundle.blockers) or "strict_runtime_unavailable"
    return {
        "arm": "avoid_preset",
        "action_type": "abort",
        "reason": reason,
        "release_timing": None,
        "timed_effect": None,
        "abstained": True,
        "deployment_status": "abort",
        "entry": None,
        "strict_selector": {
            "schema_version": "darts-strict-selector/2",
            "authorized": False,
            "deployment_status": "abort",
            "reason": reason,
            "method_fingerprint": bundle.catalog.get("method_fingerprint"),
            "blockers": list(bundle.blockers),
            "artifact_hashes": [
                {
                    "name": item.name,
                    "expected_sha256": item.expected_sha256,
                    "actual_sha256": item.actual_sha256,
                }
                for item in bundle.artifact_hashes
            ],
        },
    }


def _candidate_tube(
    neutral: TakeoffStateTube,
    candidate: CandidateAction,
    action_map: ActionConditionedCalibrationMap,
    *,
    angle_deg: float,
    speed_mps: float,
    geometry_profile: str,
) -> TakeoffStateTube | None:
    if candidate.action_kind == "neutral":
        return neutral
    estimate = action_map.query(
        angle_deg,
        speed_mps,
        action_kind=candidate.action_kind,
        geometry_profile=geometry_profile,
        schedule_id=candidate.schedule_id,
    )
    if not estimate.in_domain or estimate.tube is None:
        return None
    nominal = estimate.tube.nominal
    radius = estimate.tube.radius
    return TakeoffStateTube(
        theta0_rad=neutral.theta0_rad + nominal.dtheta0_rad,
        omega0_radps=neutral.omega0_radps + nominal.domega0_radps,
        vx_mps=neutral.vx_mps + nominal.dvx_mps,
        vz_mps=neutral.vz_mps + nominal.dvz_mps,
        z_m=neutral.z_m + nominal.dz_m,
        wheels_forward_radps=tuple(
            value + shift for value, shift in zip(
                neutral.wheels_forward_radps, nominal.dwheels_radps)),
        T_s=neutral.T_s + nominal.dT_s,
        theta0_radius_rad=neutral.theta0_radius_rad + radius.dtheta0_rad,
        omega0_radius_radps=(
            neutral.omega0_radius_radps + radius.domega0_radps),
        vx_radius_mps=neutral.vx_radius_mps + radius.dvx_mps,
        vz_radius_mps=neutral.vz_radius_mps + radius.dvz_mps,
        z_radius_m=neutral.z_radius_m + radius.dz_m,
        wheel_radius_radps=tuple(
            value + extra for value, extra in zip(
                neutral.wheel_radius_radps, radius.dwheels_radps)),
        T_radius_s=neutral.T_radius_s + radius.dT_s,
        source_id=f"{neutral.source_id}+{estimate.tube.residual_set_id}",
        residual_set_id=estimate.tube.residual_set_id,
        conditioned_on_state_id=neutral.conditioned_on_state_id,
    )


def strict_v2_runtime_select(
    *,
    catalog_path: str | Path,
    decision_state: dict,
    query_angle_deg: float,
    query_speed_mps: float,
    geometry_profile: str,
    roughness_context: dict,
    lip_radius_m: float | None,
    landing_slope_deg: float,
    landing_slope_uncertainty_deg: float,
    physical_session_id: str,
    can_stop_before_lip: bool,
    certificate_mode: str = "joint",
    capture_required: bool = True,
    authorization_mode: str = "certificate_first",
    minimum_evidence_grade: str = "S2",
    terrain_features: dict[str, float] | None = None,
    phase: str = "final",
) -> dict:
    bundle = load_strict_runtime_bundle(str(Path(catalog_path).resolve()))
    if bundle.blockers:
        return _blocked_result(bundle)
    assert bundle.action_map is not None
    assert bundle.neutral_map is not None
    if authorization_mode == "certificate_first":
        assert bundle.capture_region is not None
    catalog = bundle.catalog
    try:
        measured = MeasuredDecisionState(
            state_id=str(decision_state["state_id"]),
            theta_rad=math.radians(float(decision_state["pitch_deg"])),
            omega_radps=math.radians(float(decision_state["pitch_rate_dps"])),
            vx_mps=float(decision_state["v_mps"]),
            vz_mps=float(decision_state["vz_mps"]),
            z_m=float(decision_state["pz_m"]),
            wheels_raw_radps=dict(decision_state["wheels_radps"]),
            wheel_spin_convention_id=str(
                decision_state["wheel_spin_convention_id"]),
            sample_step=int(decision_state["i"]),
            roll_rad=math.radians(float(decision_state.get("roll_deg", 0.0))),
            roll_rate_radps=math.radians(float(
                decision_state.get("roll_rate_dps", 0.0))),
        )
        signed = measured.signed_wheel_inputs()
        decision_vector = DecisionStateVector(
            state_id=measured.state_id,
            theta_rad=measured.theta_rad,
            omega_radps=measured.omega_radps,
            vx_mps=measured.vx_mps,
            vz_mps=measured.vz_mps,
            z_m=measured.z_m,
            wheels_forward_radps=tuple(signed.canonical_omega_radps),
        )
        uncertainty_doc = catalog["decision_uncertainty"]
        decision_uncertainty = ActionConditionedDelta(
            dtheta0_rad=float(uncertainty_doc["theta_rad"]),
            domega0_radps=float(uncertainty_doc["omega_radps"]),
            dvx_mps=float(uncertainty_doc["vx_mps"]),
            dvz_mps=float(uncertainty_doc["vz_mps"]),
            dz_m=float(uncertainty_doc["z_m"]),
            dwheels_radps=tuple(
                float(value) for value in uncertainty_doc["wheels_radps"]),
            dT_s=0.0,
        )
        prediction = bundle.neutral_map.predict(
            decision_vector,
            angle_deg=query_angle_deg,
            speed_mps=query_speed_mps,
            geometry_profile=geometry_profile,
            decision_uncertainty=decision_uncertainty,
            context_features=dict(terrain_features or {}),
        )
        if not prediction.in_domain or prediction.radius is None:
            return _blocked_result(StrictRuntimeBundle(
                bundle.catalog,
                bundle.catalog_path,
                bundle.artifact_hashes,
                bundle.action_map,
                bundle.neutral_map,
                bundle.capture_region,
                bundle.capture_risk_model,
                (f"neutral_takeoff_prediction_failed({prediction.reason})",),
            ))
        neutral = TakeoffStateTube(
            theta0_rad=float(prediction.theta0_rad),
            omega0_radps=float(prediction.omega0_radps),
            vx_mps=float(prediction.vx_mps),
            vz_mps=float(prediction.vz_mps),
            z_m=float(prediction.z_m),
            wheels_forward_radps=tuple(prediction.wheels_forward_radps),
            T_s=float(prediction.T_s),
            theta0_radius_rad=prediction.radius.dtheta0_rad,
            omega0_radius_radps=prediction.radius.domega0_radps,
            vx_radius_mps=prediction.radius.dvx_mps,
            vz_radius_mps=prediction.radius.dvz_mps,
            z_radius_m=prediction.radius.dz_m,
            wheel_radius_radps=prediction.radius.dwheels_radps,
            T_radius_s=float(prediction.flight_time_radius_s),
            source_id=str(prediction.source_id),
            residual_set_id=str(prediction.residual_set_id),
            conditioned_on_state_id=measured.state_id,
        )
        candidates = tuple(
            CandidateAction(
                candidate_id=str(item["candidate_id"]),
                action_kind=str(item["action_kind"]),
                schedule_id=str(item["schedule_id"]),
                command_arm=str(item["command_arm"]),
                priority=int(item.get("priority", 0)),
            )
            for item in catalog["candidates"]
        )
        projection_config = LandingProjectionConfig(
            **catalog["landing_projection"])
        terminal = None
        audit_terminal = catalog.get("audit_terminal_region")
        if isinstance(audit_terminal, dict):
            terminal = PitchTerminalRegion(
                theta_lo=float(audit_terminal["theta_lo"]),
                theta_hi=float(audit_terminal["theta_hi"]),
                omega_lo=float(audit_terminal["omega_lo"]),
                omega_hi=float(audit_terminal["omega_hi"]),
            )
        elif bundle.capture_region is not None:
            terminal = bundle.capture_region.pitch_terminal_region()
        resource_tubes = {}
        capture_corner_tubes = {}
        for candidate in (
            candidates
            if terminal is not None and authorization_mode == "certificate_first"
            else ()
        ):
            predicted = _candidate_tube(
                neutral,
                candidate,
                bundle.action_map,
                angle_deg=query_angle_deg,
                speed_mps=query_speed_mps,
                geometry_profile=geometry_profile,
            )
            if predicted is not None:
                resource = project_landing_resources(
                    predicted,
                    decision_roll_rad=math.radians(
                        float(decision_state["roll_deg"])),
                    decision_roll_rate_radps=math.radians(
                        float(decision_state["roll_rate_dps"])),
                    roll_radius_rad=math.radians(0.5),
                    roll_rate_radius_radps=math.radians(1.0),
                    landing_slope_deg=landing_slope_deg,
                    landing_slope_uncertainty_deg=(
                        landing_slope_uncertainty_deg),
                    terminal=terminal,
                    config=projection_config,
                )
                resource_tubes[candidate.candidate_id] = resource
                if (
                    bundle.capture_risk_model is not None
                    and authorization_mode == "certificate_first"
                ):
                    capture_corner_tubes[candidate.candidate_id] = (
                        project_capture_corner_intervals(
                            resource,
                            terminal,
                            config=CaptureCornerProjectionConfig(),
                        )
                    )
        outcome_path = (
            bundle.catalog_path.parents[2]
            / catalog["artifacts"]["outcome_table"]["path"]
        ).resolve()
        outcome_evidence = {}
        for candidate in candidates:
            if candidate.action_kind == "neutral":
                continue
            result = query_system_outcome_table(
                query_angle_deg,
                query_speed_mps,
                table_path=outcome_path,
                geometry_family=geometry_profile,
                action_kind=candidate.action_kind,
                schedule_id=candidate.schedule_id,
            )
            entry = result.evidence or {}
            fit_sessions = tuple(entry.get("fit_session_ids") or ())
            holdout_sessions = tuple(entry.get("holdout_session_ids") or ())
            available = (
                result.available and bool(fit_sessions) and bool(holdout_sessions))
            outcome_evidence[candidate.candidate_id] = OutcomeQualifierEvidence(
                available=available,
                deploy_eligible=bool(result.deploy_preset) if available else False,
                action_kind=candidate.action_kind,
                geometry_profile=geometry_profile,
                schedule_id=candidate.schedule_id,
                benefit_score=float(
                    entry.get("discordant_system_only", 0)
                    - entry.get("discordant_dart_only", 0)),
                reason=result.reason,
                source_id=str(entry.get("cell_id", "missing")),
                source_sha256=(
                    catalog["artifacts"]["outcome_table"]["sha256"]),
                fit_session_ids=fit_sessions,
                holdout_session_ids=holdout_sessions,
            )
        tabletop = geometry_profile.startswith("tabletop")
        curvature_known = tabletop or (
            lip_radius_m is not None and lip_radius_m > 0.0)
        curvature_veto = (
            not tabletop
            and curvature_known
            and lip_curvature_margin(
                query_speed_mps, float(lip_radius_m), query_angle_deg)
            <= CURVATURE_N_MIN
        )
        authority = SelectorAuthorityConfig(**catalog["authority"])
        context = StrictSelectorContext(
            method_fingerprint=str(catalog["method_fingerprint"]),
            phase=phase,
            physical_session_id=physical_session_id,
            query_angle_deg=query_angle_deg,
            query_speed_mps=query_speed_mps,
            geometry_profile=geometry_profile,
            decision_state=measured,
            candidates=candidates,
            neutral_takeoff_tube=neutral,
            calibration_map=bundle.action_map,
            capture_region=bundle.capture_region,
            landing_resource_tubes=resource_tubes,
            outcome_evidence=outcome_evidence,
            adverse_regime=AdverseRegimeAssessment(
                roughness_known=isinstance(roughness_context, dict),
                curvature_known=curvature_known,
                roughness_harm_veto=rough_runup_veto_active(
                    roughness_context),
                curvature_harm_veto=curvature_veto,
            ),
            authority=authority,
            artifact_hashes=bundle.artifact_hashes,
            can_stop_before_lip=can_stop_before_lip,
            certificate_mode=certificate_mode,
            capture_required=capture_required,
            capture_risk_model=bundle.capture_risk_model,
            capture_corner_tubes=capture_corner_tubes,
            authorization_mode=authorization_mode,
            minimum_evidence_grade=minimum_evidence_grade,
            operational_wheel_guard_radps=float(
                catalog.get("operational_wheel_guard_radps", 200.0)),
            terrain_features=dict(terrain_features or {}),
            audit_terminal_region=terminal,
        )
        decision = strict_select_action(context)
    except (KeyError, TypeError, ValueError) as exc:
        return _blocked_result(StrictRuntimeBundle(
            bundle.catalog,
            bundle.catalog_path,
            bundle.artifact_hashes,
            bundle.action_map,
            bundle.neutral_map,
            bundle.capture_region,
            bundle.capture_risk_model,
            (f"strict_runtime_input_error({exc})",),
        ))

    selected_catalog = next(
        (
            item for item in catalog["candidates"]
            if item["candidate_id"] == decision.selected_candidate_id
        ),
        {},
    )
    arm = (
        decision.selected_command_arm
        if decision.selected_command_arm is not None else "avoid_preset")
    action_type = (
        decision.selected_action_kind
        if decision.selected_action_kind is not None else "abort")
    return {
        "arm": arm,
        "action_type": action_type,
        "reason": decision.reason,
        "release_timing": selected_catalog.get("release_timing"),
        "timed_effect": None,
        "abstained": decision.deployment_status != "deploy_preset",
        "deployment_status": decision.deployment_status,
        "entry": None,
        "strict_selector": decision.as_dict(),
    }


__all__ = [
    "StrictRuntimeBundle",
    "load_strict_runtime_bundle",
    "strict_v2_runtime_select",
]
