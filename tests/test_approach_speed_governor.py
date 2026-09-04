"""Tests for the shared matched-speed approach governor."""
import pytest

from control.darts.approach_speed_governor import (
    ApproachSpeedGovernorConfig,
    symmetric_speed_command,
)


def test_full_drive_below_target() -> None:
    assert symmetric_speed_command(15.0, 16.0) == (1.0, 0.0)


def test_coast_inside_deadband() -> None:
    assert symmetric_speed_command(15.98, 16.0) == (0.0, 0.0)


def test_proportional_brake_above_target() -> None:
    throttle, brake = symmetric_speed_command(
        16.2, 16.0, ApproachSpeedGovernorConfig(deadband_mps=0.05, brake_gain=2.0))
    assert throttle == 0.0
    assert brake == pytest.approx(0.3)


def test_brake_saturates() -> None:
    assert symmetric_speed_command(18.0, 16.0)[1] == 1.0


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ApproachSpeedGovernorConfig(deadband_mps=-0.1)
