"""Pitch-axis DART rate-tracking law, independent of BeamNG APIs."""
from __future__ import annotations

from dataclasses import dataclass

from control.darts.wheel_budget import OPERATIONAL_SPIN_GUARD_RADPS


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)
class DARTPitchRateConfig:
    pitch_rate_kp: float = 2.0
    rate_max_dps: float = 25.0
    drive_gain: float = 0.4
    engage_air_steps: int = 6
    wheel_speed_cap_radps: float = OPERATIONAL_SPIN_GUARD_RADPS


@dataclass(frozen=True)
class DARTPitchCommand:
    wheel_factor: float
    throttle: float
    brake: float
    engaged: bool
    guard_limited: bool
    desired_rate_dps: float


def dart_pitch_rate_command(
    pitch_deg: float,
    pitch_rate_dps: float,
    target_pitch_deg: float,
    *,
    air_streak: int,
    contact_count: int,
    wheel_speed_abs_radps: float | None,
    config: DARTPitchRateConfig = DARTPitchRateConfig(),
) -> DARTPitchCommand:
    error_deg = float(target_pitch_deg) - float(pitch_deg)
    desired_rate = _clip(
        config.pitch_rate_kp * error_deg,
        -config.rate_max_dps,
        config.rate_max_dps,
    )
    guard_limited = (
        config.wheel_speed_cap_radps > 0
        and wheel_speed_abs_radps is not None
        and float(wheel_speed_abs_radps) > config.wheel_speed_cap_radps
    )
    speed_ok = not guard_limited
    engaged = (
        int(air_streak) >= int(config.engage_air_steps)
        and int(contact_count) == 0
        and speed_ok
    )
    factor = (
        _clip(config.drive_gain * (desired_rate - float(pitch_rate_dps)) / 100.0, -1.0, 1.0)
        if engaged else 0.0
    )
    return DARTPitchCommand(
        wheel_factor=factor,
        throttle=1.0 if engaged else 0.0,
        brake=0.0,
        engaged=engaged,
        guard_limited=guard_limited,
        desired_rate_dps=desired_rate,
    )


__all__ = [
    "DARTPitchRateConfig",
    "DARTPitchCommand",
    "dart_pitch_rate_command",
]
