"""Geometry-seeded, calibratable speed retention for pre-lip throttle cuts."""
from __future__ import annotations

import math


def pre_cut_speed_target(
    requested_v_mps: float,
    cut_distance_m: float,
    ramp_angle_deg: float,
    calibrated_bias_mps: float = 0.0,
    g_mps2: float = 9.81,
) -> float:
    """Target speed at cut onset.

    The energy term compensates gravity along the remaining ramp; the
    calibrated bias captures rolling, curvature, and power-saturation
    residuals measured per operating cell.
    """
    v = float(requested_v_mps)
    cut = max(0.0, float(cut_distance_m))
    bias = float(calibrated_bias_mps)
    if not all(math.isfinite(x) for x in (v, cut, bias, ramp_angle_deg)):
        raise ValueError("speed-compensation inputs must be finite")
    if v < 0 or bias < 0:
        raise ValueError("requested speed and bias must be non-negative")
    dh = cut * math.tan(math.radians(float(ramp_angle_deg)))
    return math.sqrt(v * v + 2.0 * float(g_mps2) * max(0.0, dh)) + bias


def speed_retention_ready(
    measured_v_mps: float,
    target_v_mps: float,
    tolerance_mps: float = 0.15,
) -> bool:
    """Whether the vehicle has reached the calibrated pre-cut target."""
    measured = float(measured_v_mps)
    target = float(target_v_mps)
    tol = float(tolerance_mps)
    if not all(math.isfinite(x) for x in (measured, target, tol)) or tol < 0:
        raise ValueError("speed-retention inputs must be finite and tolerance >= 0")
    return measured >= target - tol


__all__ = ["pre_cut_speed_target", "speed_retention_ready"]
