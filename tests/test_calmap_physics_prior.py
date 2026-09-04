from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from control.darts.calmap_physics_prior import (
    OUTPUT_DIM,
    OUTPUT_ORDER,
    CalMapOutput,
    CalMapPhysicsPrior,
    physics_prior_fingerprint,
    predict_physics_prior,
)
from control.darts.susp_precondition import SuspPresetEffect


def test_output_order_is_frozen_and_float64():
    assert OUTPUT_ORDER == (
        "theta",
        "omega",
        "vx",
        "vz",
        "z",
        "wFL",
        "wFR",
        "wRL",
        "wRR",
        "T",
    )
    assert OUTPUT_DIM == len(OUTPUT_ORDER)
    output = CalMapOutput.from_array(range(OUTPUT_DIM))
    vector = output.as_array()
    assert vector.dtype == np.float64
    assert vector.tolist() == list(range(OUTPUT_DIM))
    assert tuple(output.as_dict()) == OUTPUT_ORDER


def test_static_prior_uses_only_existing_takeoff_effect_channels():
    effect = SuspPresetEffect(
        dtheta0=0.12,
        domega_y0=-0.34,
        dtheta_land_tol=9.99,
    )
    prior = CalMapPhysicsPrior(static_effect=effect)
    prediction = prior.predict("static_preset")

    assert is_dataclass(prior)
    assert is_dataclass(prediction)
    assert is_dataclass(prediction.shift)
    assert prediction.action_kind == "static"
    assert prediction.source == "SuspPresetEffect.takeoff_channels"
    assert prediction.as_array().dtype == np.float64
    assert prediction.as_array() == pytest.approx(
        [0.12, -0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert "landing" not in prediction.shift.as_dict()
    assert set(field.name for field in fields(prediction.shift)) == set(
        OUTPUT_ORDER)
    assert any(
        "dtheta_land_tol is not a takeoff output"
        in assumption
        for assumption in prediction.assumptions
    )


def test_timed_and_static_without_validated_effect_use_zero_prior():
    for action in ("neutral", "static", "timed_preset"):
        prediction = predict_physics_prior(action)
        np.testing.assert_array_equal(
            prediction.as_array(),
            np.zeros(OUTPUT_DIM, dtype=np.float64),
        )
        assert prediction.source == "explicit_zero_prior"
        assert any(
            "no landing outcome" in assumption
            for assumption in prediction.assumptions
        )


def test_timed_prior_can_use_a_separate_susp_preset_effect():
    effect = SuspPresetEffect.from_deg(2.0, -3.0)
    prior = CalMapPhysicsPrior(
        static_effect=SuspPresetEffect.from_deg(1.0, 1.0),
        timed_effect=effect,
    )
    timed = prior.predict("timed")
    static = prior.predict("static")
    assert timed.shift.theta == pytest.approx(np.deg2rad(2.0))
    assert timed.shift.omega == pytest.approx(np.deg2rad(-3.0))
    assert static.shift.theta == pytest.approx(np.deg2rad(1.0))
    assert timed.shift.vx == 0.0
    assert timed.shift.wRR == 0.0
    assert timed.shift.T == 0.0


def test_prior_fingerprint_is_semantic_and_deterministic():
    first = CalMapPhysicsPrior(
        static_effect=SuspPresetEffect(0.1, 0.2, 0.0))
    same_takeoff_different_landing = CalMapPhysicsPrior(
        static_effect=SuspPresetEffect(0.1, 0.2, 5.0))
    changed_takeoff = CalMapPhysicsPrior(
        static_effect=SuspPresetEffect(0.1, 0.3, 0.0))

    assert first.fingerprint == physics_prior_fingerprint(first)
    assert len(first.fingerprint) == 64
    assert first.fingerprint == same_takeoff_different_landing.fingerprint
    assert first.fingerprint != changed_takeoff.fingerprint
    assert first.predict("static").prior_fingerprint == first.fingerprint


def test_prior_rejects_unknown_action_and_nonfinite_effect():
    with pytest.raises(ValueError, match="action_kind"):
        CalMapPhysicsPrior().predict("landing_optimizer")
    with pytest.raises(ValueError, match="finite"):
        CalMapPhysicsPrior(
            static_effect=SuspPresetEffect(float("nan"), 0.0, 0.0))
