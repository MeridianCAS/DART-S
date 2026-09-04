"""Timing-conditioned DART-S selector compatibility and fail-closed tests."""
from __future__ import annotations

import json

from control.darts.susp_precondition import (
    SelectorUncertainty,
    recommend_arm,
    rough_runup_veto_active,
)
from scripts.darts_bench import timing_config_for_gate


def _entry():
    return {
        "angle": 35.0,
        "v": 11.5,
        "recommended_arm": "nose_up",
        "reason": "timing_conditioned_gain",
        "static_recommended_arm": "avoid_preset",
        "static_reason": "significant_harm",
        "release_timing": {
            "mode": "partial_release_return",
            "lead_sec": 0.35,
            "duration_sec": 0.12,
            "release_fraction": 0.5,
            "final_arm": "nose_up",
            "restore_before_takeoff_required": True,
            "max_query_distance": 0.25,
        },
        "timed_effect": {
            "d_theta0": 3.35,
            "d_omega_y0": 7.0,
            "d_wheel_w0": -5.0,
        },
    }


def _table(tmp_path, entry=None):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"entries": [entry or _entry()]}), encoding="utf-8")
    return path


def test_exact_timing_cell_returns_deployable_schedule(tmp_path):
    rec = recommend_arm(35.0, 11.5, table_path=_table(tmp_path))
    assert rec["arm"] == "nose_up"
    assert rec["action_type"] == "timed_preset"
    assert rec["release_timing"]["lead_sec"] == 0.35
    assert rec["timed_effect"]["d_omega_y0"] == 7.0
    assert rec["table_audit"]["sha256"]


def test_timing_does_not_extrapolate_beyond_calibrated_radius(tmp_path):
    rec = recommend_arm(35.0, 11.8, table_path=_table(tmp_path))
    assert rec["arm"] == "avoid_preset"
    assert rec["action_type"] == "abort"
    assert rec["release_timing"] is None
    assert "timing_query_too_far" in rec["reason"]


def test_invalid_timing_metadata_fails_closed(tmp_path):
    entry = _entry()
    entry["release_timing"]["duration_sec"] = -1.0
    rec = recommend_arm(35.0, 11.5, table_path=_table(tmp_path, entry))
    assert rec["arm"] == "neutral"
    assert rec["action_type"] == "pass_through"
    assert "invalid_release_timing" in rec["reason"]


def test_curvature_veto_clears_timing_action(tmp_path):
    rec = recommend_arm(
        35.0, 11.5, table_path=_table(tmp_path), lip_radius=10.0)
    assert rec["arm"] == "neutral"
    assert rec["curvature_veto"] is True
    assert rec["release_timing"] is None


def test_legacy_static_entry_remains_compatible(tmp_path):
    entry = {
        "angle": 20.0,
        "v": 13.0,
        "recommended_arm": "nose_up",
        "reason": "significant_gain",
    }
    rec = recommend_arm(20.0, 13.0, table_path=_table(tmp_path, entry))
    assert rec["arm"] == "nose_up"
    assert rec["action_type"] == "static_preset"
    assert rec["release_timing"] is None


def test_runner_builds_timing_only_for_final_timed_nose_up(tmp_path):
    rec = recommend_arm(35.0, 11.5, table_path=_table(tmp_path))
    cfg = timing_config_for_gate(rec, "nose_up")
    assert cfg is not None
    assert (cfg.lead_sec, cfg.duration_sec, cfg.release_fraction) == (0.35, 0.12, 0.5)
    assert timing_config_for_gate(rec, "neutral") is None
    static_rec = {"action_type": "static_preset", "release_timing": None}
    assert timing_config_for_gate(static_rec, "nose_up") is None


def test_rough_runup_reverse_domain_forces_pass_through(tmp_path):
    context = {"runup_bumps": 6, "bump_height_m": 0.08, "bump_spacing_m": 1.6}
    assert rough_runup_veto_active(context)
    rec = recommend_arm(
        35.0, 11.5, table_path=_table(tmp_path), roughness_context=context)
    assert rec["arm"] == "neutral"
    assert rec["action_type"] == "pass_through"
    assert rec["roughness_veto"] is True
    assert rec["release_timing"] is None


def test_nonreverse_washboard_spacing_does_not_trigger_rough_veto(tmp_path):
    context = {"runup_bumps": 6, "bump_height_m": 0.08, "bump_spacing_m": 1.3}
    assert not rough_runup_veto_active(context)
    rec = recommend_arm(
        35.0, 11.5, table_path=_table(tmp_path), roughness_context=context)
    assert rec["action_type"] == "timed_preset"


def test_uncertainty_crossing_two_actions_abstains_to_airborne_only(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"entries": [
        {
            "angle": 35.0,
            "v": 11.0,
            "recommended_arm": "nose_up",
            "reason": "gain",
            "safe_criterion_basis": "safety_v2",
        },
        {
            "angle": 35.0,
            "v": 12.0,
            "recommended_arm": "neutral",
            "reason": "no_gain",
            "safe_criterion_basis": "safety_v2",
        },
    ]}), encoding="utf-8")

    rec = recommend_arm(
        35.0,
        11.4,
        table_path=path,
        uncertainty=SelectorUncertainty(speed_half_width_mps=0.4),
    )

    assert rec["arm"] == "neutral"
    assert rec["abstained"] is True
    assert rec["action_type"] == "airborne_only_fallback"
    assert rec["deployment_status"] == "abstain_to_airborne_only"


def test_safety_claim_mode_rejects_legacy_touchdown_only_entry(tmp_path):
    entry = {
        "angle": 20.0,
        "v": 13.0,
        "recommended_arm": "nose_up",
        "reason": "significant_gain",
        "safe_criterion_basis": "touchdown_v1",
    }
    rec = recommend_arm(
        20.0,
        13.0,
        table_path=_table(tmp_path, entry),
        uncertainty=SelectorUncertainty(require_safety_v2_evidence=True),
    )
    assert rec["arm"] == "neutral"
    assert rec["abstained"] is True
    assert "safety_v2_evidence_required" in rec["reason"]


def test_unknown_geometry_family_abstains_when_calmap_is_required(tmp_path):
    entry = {
        "angle": 20.0,
        "v": 13.0,
        "recommended_arm": "nose_up",
        "reason": "significant_gain",
        "safe_criterion_basis": "safety_v2",
        "d_theta0": 3.0,
        "d_omega_y0": 5.0,
        "d_wheel_w0": -4.0,
    }
    rec = recommend_arm(
        20.0,
        13.0,
        table_path=_table(tmp_path, entry),
        geometry_family="unseen_lip_family",
        require_calibration_map=True,
    )
    assert rec["arm"] == "neutral"
    assert rec["action_type"] == "airborne_only_fallback"
    assert rec["calibration_map"]["in_domain"] is False


def test_system_safety_mode_requires_independent_system_evidence(tmp_path):
    gate_entry = {
        "angle": 35.0,
        "v": 15.0,
        "recommended_arm": "nose_up",
        "reason": "pass_rate_gain",
        "safe_criterion_basis": "safety_v2",
        "d_theta0": 3.0,
        "d_omega_y0": 10.0,
        "d_wheel_w0": -10.0,
    }
    system_path = tmp_path / "system.json"
    system_path.write_text(json.dumps({
        "endpoint": "post_touchdown_safety_v2",
        "data_role": "calibration_for_future_independent_evaluation",
        "max_query_distance": 0.25,
        "entries": [{
            "angle_deg": 35.0,
            "speed_mps": 15.0,
            "geometry_family": "tabletop_fillet",
            "preset_action": "static_preset",
            "deploy_eligible": True,
            "decision_reason": "independent_system_safety_gain",
        }],
    }), encoding="utf-8")

    rec = recommend_arm(
        35.0,
        15.0,
        table_path=_table(tmp_path, gate_entry),
        require_calibration_map=True,
        system_outcome_table_path=system_path,
        require_system_safety_evidence=True,
    )

    assert rec["arm"] == "nose_up"
    assert rec["system_outcome_evidence"]["deploy_preset"] is True


def test_uncertainty_corners_recheck_system_evidence_radius(tmp_path):
    gate_entry = {
        "angle": 35.0,
        "v": 15.0,
        "v0_neutral": 15.0,
        "geometry_family": "tabletop_fillet",
        "recommended_arm": "nose_up",
        "reason": "pass_rate_gain",
        "safe_criterion_basis": "safety_v2",
        "d_theta0": 3.0,
        "d_omega_y0": 10.0,
        "d_wheel_w0": -10.0,
    }
    system_path = tmp_path / "system_radius.json"
    system_path.write_text(json.dumps({
        "endpoint": "post_touchdown_safety_v2",
        "data_role": "calibration_for_future_independent_evaluation",
        "max_query_distance": 0.25,
        "entries": [{
            "angle_deg": 35.0,
            "speed_mps": 15.0,
            "geometry_family": "tabletop_fillet",
            "preset_action": "static_preset",
            "deploy_eligible": True,
        }],
    }), encoding="utf-8")

    rec = recommend_arm(
        35.0,
        15.0,
        table_path=_table(tmp_path, gate_entry),
        uncertainty=SelectorUncertainty(speed_half_width_mps=0.3),
        require_calibration_map=True,
        system_outcome_table_path=system_path,
        require_system_safety_evidence=True,
    )

    assert rec["arm"] == "neutral"
    assert rec["abstained"] is True
    assert "uncertainty_decision_conflict" in rec["reason"]
