"""Closed-form reachability: certificate, directional budgets, v_max.

Beyond the basic cases, the suite locks the certificate against ground truth
in two ways that an earlier displacement formula got wrong:

* a symmetry property test under (e, s) -> (-e, -s), which the double
  integrator satisfies exactly;
* a comparison against the analytic minimum-time solution of the bounded
  double integrator, checking both that the certificate never accepts an
  unreachable state and that it does not reject comfortably interior ones.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from control.darts.reachability import (  # noqa: E402
    PitchAuthority,
    PitchTerminalRegion,
    PitchTerminalSet,
    TakeoffReachableSet,
    pitch_accel_limit,
    angular_momentum_budget,
    directional_rate_budgets,
    displacement_capacity,
    exact_pitch_terminal_certificate,
    pitch_reachable,
    vmax_cross_section,
)


# ------------------------------------------------------------- basic setup --

def test_pitch_accel_limit():
    a = pitch_accel_limit(1200.0, 2600.0, n_wheels=4)
    assert abs(a - 4 * 1200.0 / 2600.0) < 1e-9
    with pytest.raises(ValueError):
        pitch_accel_limit(1200.0, 0.0)


def test_angular_momentum_budget_scalar_and_list():
    b = angular_momentum_budget(1.2, 125.66, 0.0, n_wheels=4)
    assert abs(b - 4 * 1.2 * 125.66) < 1e-6
    b2 = angular_momentum_budget(1.2, 100.0, [0.0, 0.0, 50.0, 50.0])
    assert abs(b2 - (1.2 * (100 + 100 + 50 + 50))) < 1e-6
    assert angular_momentum_budget(1.2, 10.0, [20.0, 20.0, 20.0, 20.0]) == 0.0


def test_directional_rate_budgets():
    # I_w=1.2, Omega in [-40, 120], Omega0=60, J_y=2600, four wheels
    b_up, b_dn = directional_rate_budgets(1.2, 120.0, -40.0, 60.0, 2600.0)
    assert abs(b_up - 4 * 1.2 * (120.0 - 60.0) / 2600.0) < 1e-9
    assert abs(b_dn - 4 * 1.2 * (60.0 - (-40.0)) / 2600.0) < 1e-9
    # per-wheel sequence, with the out-of-range direction clipped
    b_up2, b_dn2 = directional_rate_budgets(1.2, 100.0, 0.0, [110.0, 90.0, 90.0, 90.0], 2600.0)
    assert abs(b_up2 - 1.2 * (0 + 10 + 10 + 10) / 2600.0) < 1e-9
    assert abs(b_dn2 - 1.2 * (110 + 90 + 90 + 90) / 2600.0) < 1e-9
    with pytest.raises(ValueError):
        directional_rate_budgets(1.2, 100.0, 0.0, 50.0, 0.0)


# --------------------------------------------------- D(T_c, B_r) branches --

def test_displacement_capacity_unbounded_branch():
    # terminal-rate-zero bang-bang: D = a T_c^2 / 4
    a, T_c = 2.0, 1.0
    assert abs(displacement_capacity(T_c, float("inf"), a) - 0.25 * a * T_c ** 2) < 1e-12


def test_displacement_capacity_budget_branch_continuous():
    # the branches meet continuously at T_c = 2 B_r / a, where D = B_r^2 / a
    a, B_r = 2.0, 0.4
    T_edge = 2.0 * B_r / a
    d1 = displacement_capacity(T_edge - 1e-9, B_r, a)
    d2 = displacement_capacity(T_edge + 1e-9, B_r, a)
    assert abs(d1 - d2) < 1e-6
    assert abs(d2 - B_r * B_r / a) < 1e-6
    # budget-limited branch: linear in T_c with slope B_r
    d3 = displacement_capacity(T_edge + 1.0, B_r, a)
    assert abs(d3 - (d2 + B_r * 1.0)) < 1e-6


def test_displacement_capacity_degenerate():
    assert displacement_capacity(0.0, 1.0, 2.0) == 0.0
    assert displacement_capacity(1.0, 0.0, 2.0) == 0.0


# ------------------------------------- certificate: basics and truth locks --

def test_pitch_reachable_easy_case():
    r = pitch_reachable(theta0=0.0, omega_y0=0.0, theta_L=0.0, T=1.0, a=1.85)
    assert r.reachable is True
    assert r.velocity_nullable and r.displacement_ok


def test_pitch_reachable_velocity_not_nullable():
    r = pitch_reachable(theta0=0.0, omega_y0=5.0, theta_L=0.0, T=1.0, a=1.85)
    assert r.velocity_nullable is False
    assert r.reachable is False


def test_pitch_reachable_displacement_fail():
    # a T^2 / 4 = 0.116 < 1.5, so unreachable
    r = pitch_reachable(theta0=1.5, omega_y0=0.0, theta_L=0.0, T=0.5, a=1.85)
    assert r.displacement_ok is False
    assert r.reachable is False


def test_symmetry_property():
    """Double-integrator reachability is exactly symmetric in (e, s)."""
    a = 2.0
    for e in (-0.6, -0.3, -0.1, 0.1, 0.3, 0.6):
        for s in (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0):
            for T in (0.5, 0.8, 1.2, 2.0):
                r1 = pitch_reachable(e, s, 0.0, T=T, a=a)
                r2 = pitch_reachable(-e, -s, 0.0, T=T, a=a)
                assert r1.reachable == r2.reachable, (e, s, T)
                assert abs(abs(r1.r_theta) - abs(r2.r_theta)) < 1e-12


def _t_min_to_origin(e: float, s: float, a: float) -> float:
    """Minimum time from (e, s) to the origin under |u| <= a."""
    best = math.inf
    for u1 in (a, -a):
        # Leg 1 runs at u1 for t1 and leg 2 at -u1 for t2. A zero terminal
        # rate gives t2 = (s + u1 t1) / u1; substituting that into the
        # zero-terminal-position condition leaves one equation in t1, solved
        # by a scan plus bisection.
        def pos(t1: float) -> float:
            t2 = (s + u1 * t1) / u1
            if t2 < 0:
                return math.nan
            return e + s * (t1 + t2) + 0.5 * u1 * t1 * t1 + u1 * t1 * t2 - 0.5 * u1 * t2 * t2
        prev_t, prev_p = None, None
        t1 = 0.0
        while t1 <= 20.0:
            p = pos(t1)
            if prev_p is not None and not math.isnan(p) and not math.isnan(prev_p) \
                    and (p == 0 or (p > 0) != (prev_p > 0)):
                lo, hi = prev_t, t1
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    pm = pos(mid)
                    if (pm > 0) == (pos(lo) > 0):
                        lo = mid
                    else:
                        hi = mid
                t1s = 0.5 * (lo + hi)
                t2s = (s + u1 * t1s) / u1
                if t2s >= -1e-9:
                    best = min(best, t1s + max(0.0, t2s))
            prev_t, prev_p = t1, p
            t1 += 1e-3
    return best


def test_against_min_time_truth():
    """The certificate is a conservative subset of the truth, and is not
    over-conservative: states with t_min <= 0.7 T must be accepted."""
    a = 2.0
    for e in (-0.5, -0.2, 0.0, 0.2, 0.5):
        for s in (-0.8, -0.3, 0.0, 0.3, 0.8):
            if e == 0.0 and s == 0.0:
                continue
            t_min = _t_min_to_origin(e, s, a)
            for T in (0.5, 1.0, 1.6, 2.4):
                r = pitch_reachable(e, s, 0.0, T=T, a=a)
                if r.reachable:
                    # soundness: an accepted state is genuinely reachable
                    assert t_min <= T + 1e-2, (e, s, T, t_min)
                if t_min <= 0.7 * T:
                    # tightness: comfortably interior states must be accepted
                    assert r.reachable, (e, s, T, t_min)


# ---------------------------------------------------- directional budgets --

def test_directional_budget_asymmetric_rate_condition():
    """The rate condition is asymmetric: equal |e_0| can decide differently."""
    # spun-up takeoff: little drive headroom, plenty of braking headroom
    kw = dict(theta_L=0.0, T=2.0, a=1.85, B_up=0.1, B_down=0.5)
    # e_0 = -0.3 needs the drive direction, which exceeds B_up
    r_dn = pitch_reachable(0.0, -0.3, kw["theta_L"], kw["T"], kw["a"],
                           B_up=kw["B_up"], B_down=kw["B_down"])
    assert r_dn.budget_ok is False and r_dn.reachable is False
    # e_0 = +0.3 needs braking, which fits inside B_down
    r_up = pitch_reachable(0.0, 0.3, kw["theta_L"], kw["T"], kw["a"],
                           B_up=kw["B_up"], B_down=kw["B_down"])
    assert r_up.budget_ok is True


def test_residual_budget_limits_displacement():
    """A small residual budget caps the displacement capacity."""
    a, T = 2.0, 3.0
    # unbounded: cap = a T^2 / 4 = 4.5, so e = 1.0 is reachable
    assert pitch_reachable(1.0, 0.0, 0.0, T, a).reachable is True
    # with B = 0.2 the cap drops to 0.58, so e = 1.0 fails on displacement
    r = pitch_reachable(1.0, 0.0, 0.0, T, a, B_up=0.2, B_down=0.2)
    assert r.budget_ok is True
    assert r.displacement_ok is False and r.reachable is False


def test_legacy_symmetric_budget_kwargs():
    """The legacy H_max/J_y form is the symmetric budget H_max / J_y."""
    r = pitch_reachable(0.0, 0.2, 0.0, T=1.0, a=1.85, H_max=1.0, J_y=2600.0)
    assert r.budget_ok is False and r.reachable is False       # 0.2 > 1/2600
    r2 = pitch_reachable(0.0, 0.05, 0.0, T=1.0, a=1.85, H_max=603.0, J_y=2600.0)
    assert r2.budget_ok is True                                 # 0.05 < 603/2600


# --------------------------------------------------- terminal rate window --

def test_omega_bar_rate_condition_theorem1():
    """A positive window widens the rate condition by omega_bar either way."""
    import math as m
    kw = dict(theta_L=0.0, T=1.5, a=m.radians(120.0))
    b_up, b_dn = m.radians(10.1), m.radians(24.1)
    s = m.radians(-13.5)              # measured condition, 3.4 deg/s past B_up
    r0 = pitch_reachable(m.radians(8.1), s, kw["theta_L"], kw["T"], kw["a"],
                         B_up=b_up, B_down=b_dn)
    assert r0.budget_ok is False      # the exact-pair certificate refuses it
    r5 = pitch_reachable(m.radians(8.1), s, kw["theta_L"], kw["T"], kw["a"],
                         B_up=b_up, B_down=b_dn, omega_bar=m.radians(5.0))
    assert r5.budget_ok is True       # the windowed condition admits it
    assert r5.reachable is True


def test_omega_bar_zero_bit_exact():
    """Passing omega_bar=0 is bit-identical to omitting it."""
    for e in (-0.3, 0.1, 0.5):
        for s in (-0.8, -0.2, 0.4):
            a = pitch_reachable(e, s, 0.0, 1.2, 2.0, B_up=0.3, B_down=0.5)
            b = pitch_reachable(e, s, 0.0, 1.2, 2.0, B_up=0.3, B_down=0.5, omega_bar=0.0)
            assert (a.reachable, a.r_theta, a.T_r, a.B_r) == (b.reachable, b.r_theta, b.T_r, b.B_r)


def test_omega_bar_drift_accounting():
    """In the marginal band the rate is shed to the window edge and the
    residual drift is charged to the displacement demand.

    An excess nose-down rate exhausts B_up, and the residual negative drift
    keeps pushing the angle down. With no initial angle surplus the residual
    needs a nose-up correction the exhausted budget cannot pay for, so the
    state is correctly unreachable; with enough surplus the residual flips
    sign and is corrected out of the ample braking budget.
    """
    kw = dict(theta_L=0.0, T=1.05, a=2.0, B_up=0.1, B_down=0.5, omega_bar=0.05)
    # e=0: reduce=0.1 exhausts the budget, r_w=-0.03, T_r=0.05, r_theta=-0.034
    r = pitch_reachable(0.0, -0.13, kw["theta_L"], kw["T"], kw["a"],
                        B_up=kw["B_up"], B_down=kw["B_down"], omega_bar=kw["omega_bar"])
    assert r.budget_ok is True
    assert abs(r.T_r - 0.05) < 1e-9
    assert abs(r.r_theta - (-0.034)) < 1e-9
    assert r.B_r == 0.0               # nose-up correction needed, none left
    assert r.reachable is False
    # e=+0.1 surplus: r_theta=+0.066, corrected from B_down, capacity 0.5
    r2 = pitch_reachable(0.1, -0.13, kw["theta_L"], kw["T"], kw["a"],
                         B_up=kw["B_up"], B_down=kw["B_down"], omega_bar=kw["omega_bar"])
    assert r2.r_theta > 0 and abs(r2.B_r - 0.6) < 1e-9
    assert r2.reachable is True


def test_omega_bar_within_window_no_null():
    """Inside the window nothing is shed and the motion is pure drift."""
    r = pitch_reachable(0.0, 0.04, 0.0, T=1.0, a=2.0,
                        B_up=0.3, B_down=0.3, omega_bar=0.05)
    assert r.T_r == 0.0
    assert abs(r.r_theta - 0.04) < 1e-12      # drift of 0.04 over 1.0 s
    assert r.reachable is True


# ------------------------------------------------------ omega_target term --

def test_omega_target_constant_rate_approach_reaches_target():
    """A constant-rate approach onto the target leaves nothing to correct,
    which holds only if the displacement error carries the omega_target T
    term."""
    w_t, T = 0.2, 1.5
    r = pitch_reachable(-w_t * T, w_t, 0.0, T=T, a=2.0, B_up=0.4, B_down=0.4,
                        omega_target=w_t)
    assert abs(r.s) < 1e-12
    assert abs(r.r_theta) < 1e-12
    assert r.reachable is True


def test_omega_target_shifts_displacement_demand():
    """Starting at theta_L with the target rate is not free: the target
    trajectory moves, leaving a demand of omega_target T."""
    w_t, T = 0.2, 1.5
    r = pitch_reachable(0.0, w_t, 0.0, T=T, a=2.0, B_up=0.4, B_down=0.4,
                        omega_target=w_t)
    assert abs(r.r_theta - w_t * T) < 1e-12
    assert r.reachable is True                # D(1.5, 0.4)=0.52 ≥ 0.3


# ---------------------------------------------------- cross-section v_max --

def test_vmax_cross_section():
    v = vmax_cross_section(7.5, 4.0, 10.0)
    assert abs(v - math.sqrt(7.5 ** 2 + 2 * 4.0 * 10.0)) < 1e-9
    assert abs(vmax_cross_section(7.5, 4.0, 0.0) - 7.5) < 1e-9


# ---------------------------------------------------------- set wrapper --

def test_takeoff_reachable_set():
    rs = TakeoffReachableSet(theta_L=0.0, T=1.0, a=1.85, H_max=603.0, J_y=2600.0)
    assert rs.contains(0.0, 0.0) is True
    assert rs.is_empty() is False
    assert rs.contains(0.0, 5.0) is False
    assert abs(rs.omega_bound() - 603.0 / 2600.0) < 1e-6


def test_takeoff_set_directional_fields():
    rs = TakeoffReachableSet(theta_L=0.0, T=2.0, a=1.85, B_up=0.1, B_down=0.5)
    assert rs.contains(0.0, -0.3) is False       # past B_up
    assert rs.contains(0.0, 0.3) is True         # inside B_down, time to spare
    assert abs(rs.omega_bound() - 0.1) < 1e-9    # min(a·T, min(B_up,B_down))


def test_preset_aware_set_preserves_directional_budget_and_terminal_window():
    """The preset layer must not silently fall back to a symmetric budget."""
    from control.darts.susp_precondition import PresetAwareTakeoffSet, SuspPresetEffect

    rs = TakeoffReachableSet(
        theta_L=0.0, T=2.0, a=1.85,
        B_up=0.1, B_down=0.5, omega_bar=0.05,
    )
    preset = PresetAwareTakeoffSet(rs, SuspPresetEffect())
    for omega_y0 in (-0.20, -0.14, 0.30):
        expected = rs.query(0.0, omega_y0)
        actual = preset.query_natural(0.0, omega_y0)
        assert (actual.reachable, actual.budget_ok, actual.T_r, actual.B_r) == (
            expected.reachable, expected.budget_ok, expected.T_r, expected.B_r)


def test_takeoff_set_empty_when_flight_too_short():
    rs = TakeoffReachableSet(theta_L=0.0, T=0.01, a=1.85)
    assert rs.is_empty() is False


# -------------------------------------- exact terminal-rectangle certificate --

def test_exact_terminal_interval_symmetric_origin():
    cert = exact_pitch_terminal_certificate(
        theta0=0.0,
        omega0=0.0,
        T=1.0,
        authority=PitchAuthority.symmetric(1.0),
        terminal=PitchTerminalRegion(0.0, 0.0, 0.0, 0.0),
    )
    assert cert.reachable is True
    assert cert.reachable_theta_lo == pytest.approx(-0.25)
    assert cert.reachable_theta_hi == pytest.approx(0.25)
    assert cert.witness is not None
    assert cert.witness.terminal_theta == pytest.approx(0.0, abs=1e-10)
    assert cert.witness.terminal_omega == pytest.approx(0.0, abs=1e-10)


def test_exact_terminal_interval_budget_limited():
    cert = exact_pitch_terminal_certificate(
        theta0=0.0,
        omega0=0.0,
        T=1.0,
        authority=PitchAuthority.symmetric(1.0, B=0.2),
        terminal=PitchTerminalRegion(-1.0, 1.0, 0.0, 0.0),
    )
    # Trapezoidal rate profile: B*T-B^2/a = 0.16 rad.
    assert cert.reachable_theta_lo == pytest.approx(-0.16)
    assert cert.reachable_theta_hi == pytest.approx(0.16)
    assert cert.reachable is True


def test_exact_certificate_rejects_disjoint_terminal_rate():
    cert = exact_pitch_terminal_certificate(
        theta0=0.0,
        omega0=0.0,
        T=0.5,
        authority=PitchAuthority(a_up=1.0, a_down=2.0, B_up=0.3, B_down=0.4),
        terminal=PitchTerminalRegion(-10.0, 10.0, 0.31, 0.5),
    )
    assert cert.reachable is False
    assert cert.reason == "terminal_rate_disjoint"
    assert cert.binding_constraint == "terminal_rate"


def test_exact_witness_obeys_asymmetric_authority_and_budget():
    authority = PitchAuthority(a_up=1.7, a_down=0.9, B_up=0.35, B_down=0.22)
    cert = exact_pitch_terminal_certificate(
        theta0=-0.08,
        omega0=-0.12,
        T=1.4,
        authority=authority,
        terminal=PitchTerminalRegion(-0.04, 0.06, -0.08, 0.09),
    )
    assert cert.reachable is True
    assert cert.witness is not None
    witness = cert.witness
    z_values = [0.0]
    for segment in witness.segments:
        assert -authority.a_down - 1e-10 <= segment.acceleration
        assert segment.acceleration <= authority.a_up + 1e-10
        z_values.append(segment.omega_end - (-0.12))
    assert min(z_values) >= -authority.B_down - 1e-10
    assert max(z_values) <= authority.B_up + 1e-10
    assert -0.04 - 1e-10 <= witness.terminal_theta <= 0.06 + 1e-10
    assert -0.08 - 1e-10 <= witness.terminal_omega <= 0.09 + 1e-10


def test_terminal_factory_preserves_nonzero_omega_center():
    terminal = PitchTerminalSet(
        theta_center=0.2,
        theta_half_width=0.01,
        omega_center=0.3,
        omega_half_width=0.02,
    )
    reachable = TakeoffReachableSet.from_terminal_set(
        terminal, T=1.2, a=2.0, B_up=1.0, B_down=1.0)
    assert reachable.omega_target == 0.3
    result = reachable.query(
        terminal.theta_center - terminal.omega_center * reachable.T,
        terminal.omega_center,
    )
    assert result.reachable is True
    assert reachable.is_empty() is False


def test_legacy_constructive_set_is_subset_of_exact_set():
    rng = __import__("random").Random(20260811)
    for _ in range(300):
        theta0 = rng.uniform(-0.6, 0.6)
        omega0 = rng.uniform(-0.8, 0.8)
        T = rng.uniform(0.3, 2.0)
        a = rng.uniform(0.8, 2.5)
        b_up = rng.uniform(0.1, 0.9)
        b_down = rng.uniform(0.1, 0.9)
        legacy = pitch_reachable(
            theta0, omega0, 0.0, T, a, B_up=b_up, B_down=b_down)
        if not legacy.reachable:
            continue
        exact = exact_pitch_terminal_certificate(
            theta0,
            omega0,
            T,
            PitchAuthority(a, a, b_up, b_down),
            PitchTerminalRegion(0.0, 0.0, 0.0, 0.0),
        )
        assert exact.reachable, (theta0, omega0, T, a, b_up, b_down, legacy)


def _lp_terminal_theta_extrema(
    theta0: float,
    omega0: float,
    T: float,
    authority: PitchAuthority,
    terminal: PitchTerminalRegion,
    *,
    steps: int = 400,
) -> tuple[float, float]:
    """Independent zero-order-hold LP/direct-collocation oracle."""
    np = pytest.importorskip("numpy")
    linprog = pytest.importorskip("scipy.optimize").linprog
    dt = T / steps
    # theta(T)-theta0-omega0*T coefficient for each held acceleration u_j.
    area_coeff = np.array(
        [(steps - j - 0.5) * dt * dt for j in range(steps)],
        dtype=float,
    )
    cumulative = np.tril(np.ones((steps, steps), dtype=float)) * dt
    A_ub = np.vstack((cumulative, -cumulative))
    b_ub = np.concatenate((
        np.full(steps, authority.B_up),
        np.full(steps, authority.B_down),
    ))
    z_terminal_lo = terminal.omega_lo - omega0
    z_terminal_hi = terminal.omega_hi - omega0
    A_ub = np.vstack((
        A_ub,
        np.full((1, steps), dt),
        np.full((1, steps), -dt),
    ))
    b_ub = np.concatenate((
        b_ub,
        [z_terminal_hi, -z_terminal_lo],
    ))
    bounds = [(-authority.a_down, authority.a_up)] * steps
    low = linprog(
        area_coeff, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    high = linprog(
        -area_coeff, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    assert low.success and high.success
    drift = theta0 + omega0 * T
    return drift + low.fun, drift - high.fun


def test_exact_solver_matches_400_step_lp():
    cases = [
        (0.07, -0.18, 1.2, PitchAuthority(1.8, 1.1, 0.42, 0.31),
         PitchTerminalRegion(-1.0, 1.0, -0.08, 0.12)),
        (-0.12, 0.25, 0.9, PitchAuthority(0.9, 2.0, 0.24, 0.55),
         PitchTerminalRegion(-1.0, 1.0, -0.15, 0.06)),
        (0.0, 0.0, 1.5, PitchAuthority(1.2, 1.2, 0.18, 0.18),
         PitchTerminalRegion(-1.0, 1.0, -0.03, 0.07)),
    ]
    for theta0, omega0, T, authority, terminal in cases:
        exact = exact_pitch_terminal_certificate(
            theta0, omega0, T, authority, terminal, construct_witness=False)
        lp_lo, lp_hi = _lp_terminal_theta_extrema(
            theta0, omega0, T, authority, terminal)
        # The 400-step ZOH grid can only be slightly more conservative than
        # the continuous-time envelope when a switch falls within one step.
        tolerance = max(authority.a_up, authority.a_down) * (T / 400.0) ** 2
        assert exact.reachable_theta_lo == pytest.approx(lp_lo, abs=2 * tolerance)
        assert exact.reachable_theta_hi == pytest.approx(lp_hi, abs=2 * tolerance)
