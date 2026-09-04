"""Shared closed-loop approach-speed governor for matched-speed experiments."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ApproachSpeedGovernorConfig:
    deadband_mps: float = 0.05
    brake_gain: float = 2.0
    max_brake: float = 1.0

    def __post_init__(self) -> None:
        vals = (self.deadband_mps, self.brake_gain, self.max_brake)
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("governor parameters must be finite")
        if self.deadband_mps < 0 or self.brake_gain <= 0:
            raise ValueError("deadband must be >=0 and brake_gain >0")
        if not 0 < self.max_brake <= 1:
            raise ValueError("max_brake must be in (0,1]")


def symmetric_speed_command(
    measured_v_mps: float,
    target_v_mps: float,
    config: ApproachSpeedGovernorConfig | None = None,
) -> tuple[float, float]:
    """Full drive below target, proportional braking above target, coast inside."""
    cfg = config or ApproachSpeedGovernorConfig()
    measured = float(measured_v_mps)
    target = float(target_v_mps)
    if not math.isfinite(measured) or not math.isfinite(target):
        raise ValueError("speed inputs must be finite")
    error = target - measured
    if error > cfg.deadband_mps:
        return 1.0, 0.0
    if error < -cfg.deadband_mps:
        return 0.0, min(cfg.max_brake, cfg.brake_gain * (-error - cfg.deadband_mps))
    return 0.0, 0.0


__all__ = ["ApproachSpeedGovernorConfig", "symmetric_speed_command"]
