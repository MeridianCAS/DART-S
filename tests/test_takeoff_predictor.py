from __future__ import annotations

from dataclasses import asdict

import pytest

from control.darts.calibration_map import ActionConditionedDelta
from control.darts.takeoff_predictor import (
    DecisionStateVector,
    DecisionToTakeoffMap,
    NeutralResidualTube,
    NeutralTransitionPoint,
    load_decision_to_takeoff_map,
    neutral_predictor_fingerprint,
)


def delta(scale):
    return ActionConditionedDelta(
        0.1 * scale,
        0.2 * scale,
        0.3 * scale,
        0.4 * scale,
        0.5 * scale,
        (1 * scale, 2 * scale, 3 * scale, 4 * scale),
        0.0,
    )


def points():
    return (
        NeutralTransitionPoint(
            35.0, 10.0, "tabletop_fillet", delta(1.0), 1.0,
            ("fit-1",), "neutral-10"),
        NeutralTransitionPoint(
            35.0, 12.0, "tabletop_fillet", delta(3.0), 1.4,
            ("fit-2",), "neutral-12"),
    )


def residual():
    return NeutralResidualTube(
        residual_set_id="neutral-res-v1",
        geometry_profile="tabletop_fillet",
        radius=delta(0.1),
        flight_time_radius_s=0.05,
        fit_session_ids=("fit-1", "fit-2"),
        calibration_session_ids=("cal-1",),
        holdout_session_ids=("holdout-1",),
        method="tolerance_bound",
        source_sha256="a" * 64,
    )


def state():
    return DecisionStateVector(
        state_id="decision-1",
        theta_rad=1.0,
        omega_radps=2.0,
        vx_mps=3.0,
        vz_mps=4.0,
        z_m=5.0,
        wheels_forward_radps=(10.0, 20.0, 30.0, 40.0),
    )


def test_neutral_predictor_conditions_on_measured_decision_state():
    prediction = DecisionToTakeoffMap(points(), (residual(),)).predict(
        state(),
        angle_deg=35.0,
        speed_mps=11.0,
        geometry_profile="tabletop_fillet",
    )
    assert prediction.in_domain is True
    assert prediction.state_id == "decision-1"
    assert prediction.theta0_rad == pytest.approx(1.2)
    assert prediction.omega0_radps == pytest.approx(2.4)
    assert prediction.vx_mps == pytest.approx(3.6)
    assert prediction.vz_mps == pytest.approx(4.8)
    assert prediction.z_m == pytest.approx(6.0)
    assert prediction.wheels_forward_radps == pytest.approx((12, 24, 36, 48))
    assert prediction.T_s == pytest.approx(1.2)
    assert prediction.flight_time_radius_s == 0.05


def test_neutral_predictor_fails_closed_outside_support():
    prediction = DecisionToTakeoffMap(points(), (residual(),)).predict(
        state(),
        angle_deg=50.0,
        speed_mps=20.0,
        geometry_profile="tabletop_fillet",
    )
    assert prediction.in_domain is False
    assert prediction.reason == "neutral_outside_calibrated_support"


def test_neutral_predictor_loader_checks_hash():
    pts = points()
    tubes = (residual(),)
    config = {
        "angle_scale_deg": 5.0,
        "speed_scale_mps": 1.0,
        "max_support_distance": 1.5,
        "k_neighbors": 4,
    }
    fingerprint = neutral_predictor_fingerprint(pts, tubes, config=config)
    payload = {
        "schema_version": "darts-neutral-takeoff-predictor/1",
        "points": [asdict(point) for point in pts],
        "residual_tubes": [asdict(tube) for tube in tubes],
        "config": config,
        "model_fingerprint": fingerprint,
    }
    loaded = load_decision_to_takeoff_map(
        payload, expected_fingerprint=fingerprint)
    assert loaded.predict(
        state(),
        angle_deg=35.0,
        speed_mps=11.0,
        geometry_profile="tabletop_fillet",
    ).in_domain
