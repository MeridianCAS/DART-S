from __future__ import annotations

import math

import pytest

from control.darts.landing_projection import (
    CaptureCornerProjectionConfig,
    LandingProjectionConfig,
    project_capture_corner_intervals,
    project_landing_resources,
)
from control.darts.reachability import PitchTerminalRegion
from control.darts.strict_selector import TakeoffStateTube


def tube():
    return TakeoffStateTube(
        theta0_rad=0.0,
        omega0_radps=0.0,
        vx_mps=10.0,
        vz_mps=5.0,
        z_m=2.0,
        wheels_forward_radps=(40.0, 40.0, 40.0, 40.0),
        T_s=1.0,
        theta0_radius_rad=0.01,
        omega0_radius_radps=0.01,
        vx_radius_mps=0.1,
        vz_radius_mps=0.1,
        z_radius_m=0.1,
        wheel_radius_radps=(1.0, 1.0, 1.0, 1.0),
        T_radius_s=0.05,
        source_id="test",
        residual_set_id="test",
        conditioned_on_state_id="decision",
    )


def test_flat_landing_normal_velocity_uses_ballistic_corners():
    projected = project_landing_resources(
        tube(),
        decision_roll_rad=0.0,
        decision_roll_rate_radps=0.0,
        roll_radius_rad=0.01,
        roll_rate_radius_radps=0.01,
        landing_slope_deg=0.0,
        landing_slope_uncertainty_deg=0.0,
        terminal=PitchTerminalRegion(-0.1, 0.1, -0.1, 0.1),
    )
    expected_lo = (5.0 - 0.1) - 9.81 * (1.0 + 0.05)
    expected_hi = (5.0 + 0.1) - 9.81 * (1.0 - 0.05)
    assert projected.normal_velocity_lo_mps == pytest.approx(expected_lo)
    assert projected.normal_velocity_hi_mps == pytest.approx(expected_hi)
    assert set(projected.possible_contact_orders) == {
        "front_first", "rear_first", "four_wheel", "mixed_first"}


def test_downhill_slope_consumes_horizontal_speed_uncertainty():
    flat = project_landing_resources(
        tube(),
        decision_roll_rad=0.0,
        decision_roll_rate_radps=0.0,
        roll_radius_rad=0.0,
        roll_rate_radius_radps=0.0,
        landing_slope_deg=0.0,
        landing_slope_uncertainty_deg=0.0,
        terminal=PitchTerminalRegion(-0.1, 0.1, -0.1, 0.1),
    )
    slope = project_landing_resources(
        tube(),
        decision_roll_rad=0.0,
        decision_roll_rate_radps=0.0,
        roll_radius_rad=0.0,
        roll_rate_radius_radps=0.0,
        landing_slope_deg=-10.0,
        landing_slope_uncertainty_deg=0.5,
        terminal=PitchTerminalRegion(-0.1, 0.1, -0.1, 0.1),
    )
    assert slope.normal_velocity_hi_mps > flat.normal_velocity_hi_mps


def test_projection_load_bound_tightens_with_more_stroke():
    common = dict(
        takeoff=tube(),
        decision_roll_rad=0.0,
        decision_roll_rate_radps=0.0,
        roll_radius_rad=0.0,
        roll_rate_radius_radps=0.0,
        landing_slope_deg=0.0,
        landing_slope_uncertainty_deg=0.0,
        terminal=PitchTerminalRegion(-0.1, 0.1, -0.1, 0.1),
    )
    short = project_landing_resources(
        **common,
        config=LandingProjectionConfig(
            min_front_remaining_stroke_m=0.03,
            min_rear_remaining_stroke_m=0.03,
            stroke_uncertainty_m=0.0,
        ),
    )
    long = project_landing_resources(
        **common,
        config=LandingProjectionConfig(
            min_front_remaining_stroke_m=0.06,
            min_rear_remaining_stroke_m=0.06,
            stroke_uncertainty_m=0.0,
        ),
    )
    assert long.max_cfc10_peak_decel_mps2 < short.max_cfc10_peak_decel_mps2


def test_projection_roll_bound_includes_rate_and_acceleration():
    projected = project_landing_resources(
        tube(),
        decision_roll_rad=math.radians(2.0),
        decision_roll_rate_radps=math.radians(3.0),
        roll_radius_rad=math.radians(1.0),
        roll_rate_radius_radps=math.radians(1.0),
        landing_slope_deg=0.0,
        landing_slope_uncertainty_deg=0.0,
        terminal=PitchTerminalRegion(-0.1, 0.1, -0.1, 0.1),
    )
    assert projected.max_abs_roll_rad > math.radians(3.0)
    assert projected.max_abs_roll_rate_radps > math.radians(4.0)


def test_corner_projection_returns_four_robust_precontact_intervals():
    terminal = PitchTerminalRegion(-0.1, 0.1, -0.2, 0.2)
    resource = project_landing_resources(
        tube(),
        decision_roll_rad=0.0,
        decision_roll_rate_radps=0.0,
        roll_radius_rad=0.02,
        roll_rate_radius_radps=0.03,
        landing_slope_deg=0.0,
        landing_slope_uncertainty_deg=0.0,
        terminal=terminal,
    )
    corners = project_capture_corner_intervals(
        resource,
        terminal,
        config=CaptureCornerProjectionConfig(),
    )
    assert {corner.wheel_name for corner in corners} == {"FL", "FR", "RL", "RR"}
    assert all(corner.gap_lo_m >= 0.0 for corner in corners)
    assert all(
        corner.closing_speed_hi_mps >= corner.closing_speed_lo_mps
        for corner in corners
    )
    assert all(corner.stroke_lo_m > 0.0 for corner in corners)
