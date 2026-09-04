import pytest
from dataclasses import asdict

from control.darts.calibration_map import (
    ActionCalibrationMap,
    ActionConditionedCalibrationMap,
    ActionConditionedCalibrationPoint,
    ActionConditionedDelta,
    FrozenResidualTube,
    action_conditioned_calibration_fingerprint,
    load_action_conditioned_calibration_map,
)


def _entries():
    return [
        {
            "angle": 20.0,
            "v": 10.0,
            "n": 30,
            "d_theta0": 2.0,
            "d_omega_y0": 4.0,
            "d_wheel_w0": -10.0,
        },
        {
            "angle": 20.0,
            "v": 12.0,
            "n": 30,
            "d_theta0": 4.0,
            "d_omega_y0": 8.0,
            "d_wheel_w0": -20.0,
        },
    ]


def test_calibration_map_interpolates_takeoff_shift_not_outcome():
    calmap = ActionCalibrationMap.from_gate_entries(_entries())
    estimate = calmap.query(
        20.0, 11.0, action_kind="static_preset")

    assert estimate.available is True
    assert estimate.in_domain is True
    assert estimate.neighbor_count == 2
    assert estimate.dtheta0_deg == 3.0
    assert estimate.domega_y0_dps == 6.0
    assert estimate.dwheel_radps == -15.0


def test_calibration_map_abstains_outside_family_or_support():
    calmap = ActionCalibrationMap.from_gate_entries(_entries())
    unknown = calmap.query(
        20.0,
        11.0,
        action_kind="static_preset",
        geometry_family="unseen_lip_family",
    )
    outside = calmap.query(
        35.0, 20.0, action_kind="static_preset")

    assert unknown.in_domain is False
    assert unknown.reason == "no_matching_action_or_geometry_family"
    assert outside.in_domain is False
    assert outside.reason == "outside_calibrated_support"


def test_calibration_map_uses_online_decision_speed_coordinate():
    entries = _entries()
    entries[0]["decision_speed_neutral_mps"] = 15.8
    entries[0]["v0_neutral"] = 14.9
    entries[1]["decision_speed_neutral_mps"] = 16.2
    entries[1]["v0_neutral"] = 15.1
    calmap = ActionCalibrationMap.from_gate_entries(entries)

    estimate = calmap.query(
        20.0, 16.0, action_kind="static_preset", max_support_distance=0.5)

    assert estimate.in_domain is True
    assert estimate.neighbor_count == 2


def _delta(scale: float) -> ActionConditionedDelta:
    return ActionConditionedDelta(
        dtheta0_rad=0.01 * scale,
        domega0_radps=0.02 * scale,
        dvx_mps=0.03 * scale,
        dvz_mps=0.04 * scale,
        dz_m=0.05 * scale,
        dwheels_radps=(
            1.0 * scale,
            2.0 * scale,
            3.0 * scale,
            4.0 * scale,
        ),
        dT_s=0.006 * scale,
    )


def _strict_points():
    return (
        ActionConditionedCalibrationPoint(
            angle_deg=20.0,
            speed_mps=10.0,
            action_kind="static_preset",
            geometry_profile="tabletop_fillet",
            schedule_id="static_full",
            delta=_delta(1.0),
            train_session_ids=("fit-1",),
            source_id="point-10",
        ),
        ActionConditionedCalibrationPoint(
            angle_deg=20.0,
            speed_mps=12.0,
            action_kind="static_preset",
            geometry_profile="tabletop_fillet",
            schedule_id="static_full",
            delta=_delta(3.0),
            train_session_ids=("fit-2",),
            source_id="point-12",
        ),
    )


def _strict_residual(**overrides):
    values = {
        "residual_set_id": "residual-static-v1",
        "action_kind": "static_preset",
        "geometry_profile": "tabletop_fillet",
        "schedule_id": "static_full",
        "radius": _delta(0.1),
        "fit_session_ids": ("fit-1", "fit-2"),
        "calibration_session_ids": ("cal-1", "cal-2"),
        "holdout_session_ids": ("holdout-1",),
        "method": "split_conformal",
        "target_coverage": 0.95,
        "source_sha256": "a" * 64,
    }
    values.update(overrides)
    return FrozenResidualTube(**values)


def test_action_conditioned_map_predicts_full_state_and_per_wheel_tube():
    calmap = ActionConditionedCalibrationMap(
        _strict_points(), (_strict_residual(),))
    estimate = calmap.query(
        20.0,
        11.0,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
    )
    assert estimate.available is True
    assert estimate.in_domain is True
    assert estimate.neighbor_count == 2
    assert estimate.tube is not None
    assert estimate.tube.nominal.dtheta0_rad == pytest.approx(0.02)
    assert estimate.tube.nominal.domega0_radps == pytest.approx(0.04)
    assert estimate.tube.nominal.dvx_mps == pytest.approx(0.06)
    assert estimate.tube.nominal.dvz_mps == pytest.approx(0.08)
    assert estimate.tube.nominal.dz_m == pytest.approx(0.10)
    assert estimate.tube.nominal.dwheels_radps == pytest.approx((2, 4, 6, 8))
    assert estimate.tube.nominal.dT_s == pytest.approx(0.012)
    assert estimate.tube.residual_set_id == "residual-static-v1"
    assert estimate.calibration_session_ids == ("cal-1", "cal-2")
    assert estimate.holdout_session_ids == ("holdout-1",)


def test_action_conditioned_map_requires_frozen_independent_residual_tube():
    missing = ActionConditionedCalibrationMap(_strict_points(), ())
    estimate = missing.query(
        20.0,
        11.0,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
    )
    assert estimate.in_domain is False
    assert estimate.reason == "frozen_residual_tube_missing"
    with pytest.raises(ValueError, match="disjoint"):
        _strict_residual(holdout_session_ids=("cal-1",))


def test_action_conditioned_map_never_substitutes_another_schedule():
    calmap = ActionConditionedCalibrationMap(
        _strict_points(), (_strict_residual(),))
    estimate = calmap.query(
        20.0,
        11.0,
        action_kind="timed_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="release-0p35",
    )
    assert estimate.available is False
    assert estimate.reason == "no_matching_action_geometry_schedule"


def test_action_conditioned_single_point_supports_only_exact_coordinate():
    point = _strict_points()[0]
    residual = _strict_residual(
        fit_session_ids=("fit-1",),
        calibration_session_ids=("cal-1",),
    )
    calmap = ActionConditionedCalibrationMap((point,), (residual,))
    exact = calmap.query(
        20.0,
        10.0,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
    )
    nearby = calmap.query(
        20.0,
        10.01,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
    )
    assert exact.in_domain is True
    assert nearby.in_domain is False
    assert nearby.reason == "outside_calibrated_support"


def test_action_conditioned_predictor_and_residual_sessions_must_match():
    residual = _strict_residual(fit_session_ids=("different-fit",))
    estimate = ActionConditionedCalibrationMap(
        _strict_points(), (residual,)).query(
            20.0,
            11.0,
            action_kind="static_preset",
            geometry_profile="tabletop_fillet",
            schedule_id="static_full",
        )
    assert estimate.in_domain is False
    assert estimate.reason == "predictor_residual_fit_session_mismatch"


def test_action_conditioned_calmap_loader_checks_declared_fingerprint():
    points = _strict_points()
    tubes = (_strict_residual(),)
    fingerprint = action_conditioned_calibration_fingerprint(points, tubes)
    payload = {
        "schema_version": "darts-action-conditioned-calmap/1",
        "points": [asdict(point) for point in points],
        "residual_tubes": [asdict(tube) for tube in tubes],
        "config": {
            "angle_scale_deg": 5.0,
            "speed_scale_mps": 1.0,
            "max_support_distance": 1.5,
            "k_neighbors": 4,
        },
        "model_fingerprint": fingerprint,
    }
    loaded = load_action_conditioned_calibration_map(
        payload, expected_fingerprint=fingerprint)
    assert loaded.query(
        20.0,
        11.0,
        action_kind="static_preset",
        geometry_profile="tabletop_fillet",
        schedule_id="static_full",
    ).in_domain
    with pytest.raises(ValueError, match="declared"):
        load_action_conditioned_calibration_map({
            **payload,
            "model_fingerprint": "0" * 64,
        })
