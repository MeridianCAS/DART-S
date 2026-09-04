"""Tests for DART-S pre-lip cut speed retention."""
import math

import pytest

from control.darts.cut_speed_compensation import (
    pre_cut_speed_target,
    speed_retention_ready,
)


def test_pre_cut_target_adds_energy_and_calibrated_bias() -> None:
    target = pre_cut_speed_target(16.0, 1.0, 35.0, calibrated_bias_mps=0.4)
    expected = math.sqrt(16.0**2 + 2 * 9.81 * math.tan(math.radians(35.0))) + 0.4
    assert target == pytest.approx(expected)


def test_zero_cut_only_adds_bias() -> None:
    assert pre_cut_speed_target(13.0, 0.0, 35.0, 0.4) == pytest.approx(13.4)


def test_speed_retention_gate() -> None:
    assert speed_retention_ready(16.30, 16.40, tolerance_mps=0.15)
    assert not speed_retention_ready(16.20, 16.40, tolerance_mps=0.15)


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        pre_cut_speed_target(13.0, 1.0, 35.0, -0.1)
    with pytest.raises(ValueError):
        speed_retention_ready(13.0, 13.0, -0.1)


def test_deconfounding_arms_form_requested_factorial() -> None:
    from scripts.darts_bench import (
        ARMS,
        DART_S_SYSTEM_ARM,
        build_cut_speed_arm_specs,
        build_nocut_speed_arm_specs,
    )

    assert ARMS["dart_only"].get("lip_cut_m") is None
    assert ARMS["dart_cut_only"]["lip_cut_m"] == 1.0
    assert ARMS["dart_s_no_cut"].get("gate") is True
    assert ARMS["dart_s_no_cut"]["lip_cut_m"] == 0.0
    assert ARMS["dart_s_no_cut"].get("lip_cut_on_prepare") is None
    assert DART_S_SYSTEM_ARM == "dart_s_system"
    assert ARMS[DART_S_SYSTEM_ARM] == ARMS["dart_s_no_cut"]
    assert ARMS["dart_s_cut"]["lip_cut_on_prepare"] == 1.0

    specs, mapping = build_cut_speed_arm_specs([0.0, 0.4])
    assert mapping["b040"]["bias_mps"] == 0.4
    assert specs["dart_cut_b040"]["lip_cut_m"] == 1.0
    assert specs["dart_s_cut_b040"]["lip_cut_on_prepare"] == 1.0

    n_specs, n_map = build_nocut_speed_arm_specs([-0.1, -0.3])
    assert n_map["m030"]["offset_mps"] == -0.3
    assert n_specs["dart_s_no_cut_m010"]["entry_speed_offset_mps"] == -0.1
