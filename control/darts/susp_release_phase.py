"""Lip-relative active-suspension release-phase utilities.

The DART-S phase-sweep experiment keeps the same final nose-up preset but
briefly moves the command toward neutral before the lip.  Sweeping the pulse
lead changes the measured suspension state at contact release without changing
the final setpoint.  This module contains only deterministic, simulator-free
logic so the intervention can be unit tested before BeamNG execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class ReleasePulseConfig:
    """A finite partial-release pulse that returns to the original preset."""

    lead_sec: float
    duration_sec: float = 0.12
    release_fraction: float = 0.50
    dt_sec: float = 0.01

    def __post_init__(self) -> None:
        if not math.isfinite(self.lead_sec) or self.lead_sec <= 0:
            raise ValueError("lead_sec must be finite and > 0")
        if not math.isfinite(self.duration_sec) or self.duration_sec <= 0:
            raise ValueError("duration_sec must be finite and > 0")
        if not 0 < self.release_fraction < 1:
            raise ValueError("release_fraction must be in (0, 1)")
        if not math.isfinite(self.dt_sec) or self.dt_sec <= 0:
            raise ValueError("dt_sec must be finite and > 0")

    @property
    def duration_steps(self) -> int:
        return max(1, int(round(self.duration_sec / self.dt_sec)))

    def time_to_lip(self, distance_to_lip_m: float, speed_mps: float) -> float:
        """Return a conservative longitudinal time-to-lip estimate."""
        if not math.isfinite(distance_to_lip_m) or not math.isfinite(speed_mps):
            return math.inf
        if speed_mps <= 0.1:
            return math.inf
        return max(0.0, distance_to_lip_m) / speed_mps

    def should_trigger(self, distance_to_lip_m: float, speed_mps: float) -> bool:
        return self.time_to_lip(distance_to_lip_m, speed_mps) <= self.lead_sec

    def state(self, step: int, trigger_step: Optional[int]) -> str:
        if trigger_step is None or step < trigger_step:
            return "armed"
        if step < trigger_step + self.duration_steps:
            return "partial_release"
        return "restored"

    def command_scale(self, step: int, trigger_step: Optional[int]) -> float:
        return 1.0 - self.release_fraction if self.state(step, trigger_step) == "partial_release" else 1.0


def scaled_preset(base: Sequence[float], scale: float) -> tuple[float, float, float, float]:
    """Scale a four-corner preset toward neutral, preserving its direction."""
    if len(base) != 4:
        raise ValueError("base preset must have four corner commands")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("scale must be in [0, 1]")
    return tuple(float(scale) * float(v) for v in base)  # type: ignore[return-value]


def pitch_mode_length(sample: Mapping[str, float]) -> Optional[float]:
    """Return front-minus-rear mean spring length for one stroke sample."""
    keys = ("aspring_FL", "aspring_FR", "aspring_RL1", "aspring_RR1")
    vals = [sample.get(k) for k in keys]
    if any(v is None or not math.isfinite(float(v)) for v in vals):
        return None
    fl, fr, rl, rr = (float(v) for v in vals)
    return 0.5 * (fl + fr - rl - rr)


__all__ = ["ReleasePulseConfig", "scaled_preset", "pitch_mode_length"]
