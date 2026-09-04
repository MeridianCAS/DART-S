"""Unit tests for the DART-S lip-relative suspension phase intervention."""
from __future__ import annotations

import math

import pytest

from control.darts.susp_release_phase import (
    ReleasePulseConfig,
    pitch_mode_length,
    scaled_preset,
)


def test_release_pulse_trigger_and_restore() -> None:
    cfg = ReleasePulseConfig(
        lead_sec=1.0, duration_sec=0.12, release_fraction=0.5, dt_sec=0.01)

    assert not cfg.should_trigger(distance_to_lip_m=12.0, speed_mps=10.0)
    assert cfg.should_trigger(distance_to_lip_m=9.9, speed_mps=10.0)
    assert cfg.duration_steps == 12

    trigger = 100
    assert cfg.state(99, trigger) == "armed"
    assert cfg.state(100, trigger) == "partial_release"
    assert cfg.command_scale(111, trigger) == pytest.approx(0.5)
    assert cfg.state(112, trigger) == "restored"
    assert cfg.command_scale(112, trigger) == pytest.approx(1.0)


def test_release_pulse_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        ReleasePulseConfig(lead_sec=0.0)
    with pytest.raises(ValueError):
        ReleasePulseConfig(lead_sec=1.0, release_fraction=1.0)


def test_scaled_preset_preserves_direction_and_final_setpoint() -> None:
    base = (1.0, 1.0, -1.0, -1.0)
    assert scaled_preset(base, 0.5) == (0.5, 0.5, -0.5, -0.5)
    assert scaled_preset(base, 1.0) == base


def test_pitch_mode_length() -> None:
    sample = {
        "aspring_FL": 1.2,
        "aspring_FR": 1.0,
        "aspring_RL1": 0.8,
        "aspring_RR1": 0.6,
    }
    assert pitch_mode_length(sample) == pytest.approx(0.4)
    assert pitch_mode_length({"aspring_FL": math.nan}) is None
