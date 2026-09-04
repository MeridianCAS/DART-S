"""Side-camera pose and tabletop fillet length used by ``scripts/darts_bench.py``.
"""
from __future__ import annotations

import math

def side_cam_pose(base_x, peak_x, R, *, run_up=30.0, side_dist=30.0, height=10.0):
    """Side camera pose (placed at -Y, looking toward +Y) framing run-up, ramp, flight and landing."""
    x_lo = base_x - run_up - 5.0
    x_hi = peak_x + R + 6.0
    mid_x = 0.5 * (x_lo + x_hi)
    span = x_hi - x_lo
    sd = max(side_dist, span * 0.7)
    cam = (mid_x, -sd, height)
    look_at = (mid_x, 0.0, 2.5)
    dx, dy, dz = look_at[0] - cam[0], look_at[1] - cam[1], look_at[2] - cam[2]
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    return cam, (dx / n, dy / n, dz / n), (x_lo, x_hi)

def set_side_cam(bng, base_x, peak_x, R, run_up=30.0):
    cam, d, _ = side_cam_pose(base_x, peak_x, R, run_up=run_up)
    try:
        bng.camera.set_free(cam, d)
        return True
    except Exception:
        return False

def fillet_len_for(angle, budget, lmax):
    """Fillet length = min(lmax, budget/tan(θ/2)), so fillet_rise stays within budget."""
    return min(lmax, budget / math.tan(math.radians(angle) / 2.0))
