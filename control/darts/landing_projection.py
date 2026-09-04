"""Analytic landing-resource projection used before full capture membership."""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from control.darts.landing_capture import CornerPrecontactInterval
from control.darts.reachability import PitchTerminalRegion
from control.darts.strict_selector import LandingResourceTube, TakeoffStateTube


@dataclass(frozen=True)
class LandingProjectionConfig:
    gravity_mps2: float = 9.81
    roll_accel_bound_radps2: float = 0.05
    min_front_remaining_stroke_m: float = 0.045
    min_rear_remaining_stroke_m: float = 0.055
    stroke_uncertainty_m: float = 0.005
    peak_to_average_decel_factor: float = 2.0
    contact_pitch_deadband_rad: float = math.radians(2.0)
    model_id: str = "ballistic-resource-projection/v1"

    def __post_init__(self):
        positive = (
            self.gravity_mps2,
            self.min_front_remaining_stroke_m,
            self.min_rear_remaining_stroke_m,
            self.peak_to_average_decel_factor,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("landing projection positive bounds are invalid")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.roll_accel_bound_radps2,
                self.stroke_uncertainty_m,
                self.contact_pitch_deadband_rad,
            )
        ):
            raise ValueError("landing projection uncertainties must be nonnegative")


@dataclass(frozen=True)
class CaptureCornerProjectionConfig:
    wheelbase_m: float = 2.60
    track_m: float = 1.58
    gap_uncertainty_m: float = 0.01
    closing_speed_uncertainty_mps: float = 0.10
    stroke_upper_margin_m: float = 0.01

    def __post_init__(self):
        if self.wheelbase_m <= 0.0 or self.track_m <= 0.0:
            raise ValueError("capture corner wheelbase/track must be positive")
        if any(value < 0.0 for value in (
            self.gap_uncertainty_m,
            self.closing_speed_uncertainty_mps,
            self.stroke_upper_margin_m,
        )):
            raise ValueError("capture corner uncertainties must be nonnegative")


def project_capture_corner_intervals(
    resource: LandingResourceTube,
    terminal: PitchTerminalRegion,
    *,
    config: CaptureCornerProjectionConfig = CaptureCornerProjectionConfig(),
) -> tuple[CornerPrecontactInterval, ...]:
    """Project aggregate landing resources to four robust precontact corners."""
    half_wheelbase = config.wheelbase_m / 2.0
    half_track = config.track_m / 2.0
    offsets = {
        "FL": (half_wheelbase, half_track),
        "FR": (half_wheelbase, -half_track),
        "RL": (-half_wheelbase, half_track),
        "RR": (-half_wheelbase, -half_track),
    }
    gaps = {name: [] for name in offsets}
    for pitch, roll in product(
        (terminal.theta_lo, terminal.theta_hi),
        (-resource.max_abs_roll_rad, resource.max_abs_roll_rad),
    ):
        heights = {
            name: pitch * x + roll * y
            for name, (x, y) in offsets.items()
        }
        reference = min(heights.values())
        for name, height in heights.items():
            gaps[name].append(max(0.0, height - reference))

    closing = {name: [] for name in offsets}
    for normal_velocity, pitch_rate, roll_rate in product(
        (
            resource.normal_velocity_lo_mps,
            resource.normal_velocity_hi_mps,
        ),
        (terminal.omega_lo, terminal.omega_hi),
        (
            -resource.max_abs_roll_rate_radps,
            resource.max_abs_roll_rate_radps,
        ),
    ):
        for name, (x, y) in offsets.items():
            closing[name].append(max(
                0.0,
                -normal_velocity + pitch_rate * x + roll_rate * y,
            ))

    corners = []
    for name in ("FL", "FR", "RL", "RR"):
        stroke_lo = (
            resource.min_front_remaining_stroke_m
            if name.startswith("F")
            else resource.min_rear_remaining_stroke_m
        )
        if stroke_lo <= 0.0:
            raise ValueError(f"{name} projected remaining stroke is nonpositive")
        corners.append(CornerPrecontactInterval(
            wheel_name=name,
            gap_lo_m=max(
                0.0, min(gaps[name]) - config.gap_uncertainty_m),
            gap_hi_m=max(gaps[name]) + config.gap_uncertainty_m,
            closing_speed_lo_mps=max(
                0.0,
                min(closing[name]) - config.closing_speed_uncertainty_mps,
            ),
            closing_speed_hi_mps=(
                max(closing[name]) + config.closing_speed_uncertainty_mps),
            stroke_lo_m=stroke_lo,
            stroke_hi_m=stroke_lo + config.stroke_upper_margin_m,
        ))
    return tuple(corners)


def _contact_orders(
    terminal: PitchTerminalRegion,
    deadband: float,
) -> tuple[str, ...]:
    orders = set()
    if terminal.theta_lo < -deadband:
        orders.add("front_first")
    if terminal.theta_hi > deadband:
        orders.add("rear_first")
    if terminal.theta_lo <= deadband and terminal.theta_hi >= -deadband:
        orders.add("four_wheel")
        orders.add("mixed_first")
    return tuple(sorted(orders))


def project_landing_resources(
    takeoff: TakeoffStateTube,
    *,
    decision_roll_rad: float,
    decision_roll_rate_radps: float,
    roll_radius_rad: float,
    roll_rate_radius_radps: float,
    landing_slope_deg: float,
    landing_slope_uncertainty_deg: float,
    terminal: PitchTerminalRegion,
    config: LandingProjectionConfig = LandingProjectionConfig(),
) -> LandingResourceTube:
    """Project a takeoff tube to conservative touchdown resource bounds."""
    numeric = (
        decision_roll_rad,
        decision_roll_rate_radps,
        roll_radius_rad,
        roll_rate_radius_radps,
        landing_slope_deg,
        landing_slope_uncertainty_deg,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("landing projection inputs must be finite")
    if (
        roll_radius_rad < 0.0
        or roll_rate_radius_radps < 0.0
        or landing_slope_uncertainty_deg < 0.0
    ):
        raise ValueError("landing projection input radii must be nonnegative")

    normal_velocities = []
    for vx, vz, flight_time, slope_deg in product(
        (
            takeoff.vx_mps - takeoff.vx_radius_mps,
            takeoff.vx_mps + takeoff.vx_radius_mps,
        ),
        (
            takeoff.vz_mps - takeoff.vz_radius_mps,
            takeoff.vz_mps + takeoff.vz_radius_mps,
        ),
        (
            takeoff.T_s - takeoff.T_radius_s,
            takeoff.T_s + takeoff.T_radius_s,
        ),
        (
            landing_slope_deg - landing_slope_uncertainty_deg,
            landing_slope_deg + landing_slope_uncertainty_deg,
        ),
    ):
        slope = math.radians(slope_deg)
        touchdown_vz = vz - config.gravity_mps2 * flight_time
        normal_velocities.append(
            -vx * math.sin(slope) + touchdown_vz * math.cos(slope))
    normal_lo = min(normal_velocities)
    normal_hi = max(normal_velocities)
    max_abs_normal = max(abs(normal_lo), abs(normal_hi))
    front_stroke = max(
        0.0,
        config.min_front_remaining_stroke_m - config.stroke_uncertainty_m,
    )
    rear_stroke = max(
        0.0,
        config.min_rear_remaining_stroke_m - config.stroke_uncertainty_m,
    )
    limiting_stroke = min(front_stroke, rear_stroke)
    peak_decel = (
        float("inf")
        if limiting_stroke <= 0.0
        else (
            config.peak_to_average_decel_factor
            * max_abs_normal * max_abs_normal
            / (2.0 * limiting_stroke)
        )
    )
    T_hi = takeoff.T_s + takeoff.T_radius_s
    roll_rate_hi = (
        abs(decision_roll_rate_radps)
        + roll_rate_radius_radps
        + config.roll_accel_bound_radps2 * T_hi
    )
    roll_hi = (
        abs(decision_roll_rad)
        + roll_radius_rad
        + (abs(decision_roll_rate_radps) + roll_rate_radius_radps) * T_hi
        + 0.5 * config.roll_accel_bound_radps2 * T_hi * T_hi
    )
    return LandingResourceTube(
        max_abs_roll_rad=roll_hi,
        max_abs_roll_rate_radps=roll_rate_hi,
        normal_velocity_lo_mps=normal_lo,
        normal_velocity_hi_mps=normal_hi,
        min_front_remaining_stroke_m=front_stroke,
        min_rear_remaining_stroke_m=rear_stroke,
        max_impact_energy_per_mass_jpkg=0.5 * max_abs_normal ** 2,
        max_cfc10_peak_decel_mps2=peak_decel,
        possible_contact_orders=_contact_orders(
            terminal, config.contact_pitch_deadband_rad),
        source_id=config.model_id,
    )


__all__ = [
    "LandingProjectionConfig",
    "CaptureCornerProjectionConfig",
    "project_landing_resources",
    "project_capture_corner_intervals",
]
