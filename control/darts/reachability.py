"""Airborne backward reachability and the ``v_max`` cross-section bound.

Three closed-form pieces, all pure functions evaluated in microseconds:

* Sufficient pitch-axis certificate (Theorem 3, ``pitch_reachable``): can a
  bounded pitch acceleration ``|u| <= a`` steer the takeoff state
  ``(theta0, omega_y0)`` into the terminal set within the flight time ``T``?
  The null-then-correct construction nulls the rate error first and then
  spends the remaining time on the displacement capacity ``D(T_c, B_r)``.
* Directional rate budgets (Theorem 1, ``directional_rate_budgets``):
  ``B_up = sum_i I_i (Omega_max - Omega_i0) / J_y`` for the nose-up (drive)
  direction and ``B_down = sum_i I_i (Omega_i0 - Omega_min) / J_y`` for the
  nose-down (braking) direction.
* One-dimensional reduction (``vmax_cross_section``): the landing constraint
  propagated backwards to a speed ceiling at every cross-section of the
  approach.

``exact_pitch_terminal_certificate`` solves the same terminal-rectangle
question exactly (necessary and sufficient) for the bounded double
integrator; the sufficient certificate above is what runs online, and the
exact solver is the reference the numerical cross-validation checks against.

Legacy calls that pass only ``H_max``/``J_y`` still work: the symmetric
budget ``B_up = B_down = H_max / J_y`` is used. With no budget at all only
the time and displacement conditions remain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PitchReachability:
    """Pitch-axis reachability result and diagnostic margins."""

    reachable: bool
    # Rate condition: T_r = |e_0| / a <= T.
    velocity_nullable: bool
    # Displacement condition: |r_theta| <= D(T_c, B_r).
    displacement_ok: bool
    # Directional rate budget; True when no budget was supplied.
    budget_ok: bool
    # Diagnostics.
    e: float          # theta0 - theta_L
    s: float          # e_0 = omega_y0 - omega_target
    T_r: float        # time spent nulling the rate error
    r_theta: float    # angle error remaining after the nulling leg
    T_c: float        # correction time left, max(0, T - T_r)
    B_r: float        # residual budget in the correction direction
    theta_bar: float  # terminal pitch half-window (rad)
    r_theta_to_set: float  # residual measured to the nearest window edge
    margin: float     # D(T_c, B_r) - |r_theta|; reachable when >= 0


@dataclass(frozen=True)
class PitchTerminalSet:
    """Airborne pitch terminal set, separate from post-touchdown safety."""

    theta_center: float
    theta_half_width: float = 0.0
    omega_center: float = 0.0
    omega_half_width: float = 0.0

    def __post_init__(self):
        if self.theta_half_width < 0.0:
            raise ValueError("theta_half_width must be >= 0")
        if self.omega_half_width < 0.0:
            raise ValueError("omega_half_width must be >= 0")


@dataclass(frozen=True)
class PitchAuthority:
    """Asymmetric pitch-acceleration and signed rate-budget limits.

    ``B_up`` and ``B_down`` bound the cumulative body-rate change from the
    queried initial state, not the absolute body rate:
    ``-B_down <= omega(t)-omega0 <= B_up``.
    """

    a_up: float
    a_down: float
    B_up: float = float("inf")
    B_down: float = float("inf")

    def __post_init__(self):
        for name, value in (("a_up", self.a_up), ("a_down", self.a_down)):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name, value in (("B_up", self.B_up), ("B_down", self.B_down)):
            v = float(value)
            if math.isnan(v) or v < 0.0:
                raise ValueError(f"{name} must be >= 0")

    @classmethod
    def symmetric(
        cls,
        a: float,
        B: float = float("inf"),
    ) -> "PitchAuthority":
        return cls(a_up=a, a_down=a, B_up=B, B_down=B)


@dataclass(frozen=True)
class PitchTerminalRegion:
    """Closed pitch terminal rectangle used by the analytic solver."""

    theta_lo: float
    theta_hi: float
    omega_lo: float
    omega_hi: float

    def __post_init__(self):
        values = (
            self.theta_lo, self.theta_hi, self.omega_lo, self.omega_hi)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("terminal-region bounds must be finite")
        if self.theta_lo > self.theta_hi:
            raise ValueError("theta_lo must be <= theta_hi")
        if self.omega_lo > self.omega_hi:
            raise ValueError("omega_lo must be <= omega_hi")

    @classmethod
    def from_terminal_set(
        cls,
        terminal: PitchTerminalSet,
    ) -> "PitchTerminalRegion":
        return cls(
            theta_lo=terminal.theta_center - terminal.theta_half_width,
            theta_hi=terminal.theta_center + terminal.theta_half_width,
            omega_lo=terminal.omega_center - terminal.omega_half_width,
            omega_hi=terminal.omega_center + terminal.omega_half_width,
        )


@dataclass(frozen=True)
class ReachabilityWitnessSegment:
    """One constant-acceleration segment of a constructive witness."""

    t0: float
    t1: float
    acceleration: float
    omega_start: float
    omega_end: float
    theta_delta: float
    active_lower: str
    active_upper: str


@dataclass(frozen=True)
class ReachabilityWitness:
    """Piecewise-constant-acceleration trajectory entering the terminal set."""

    terminal_theta: float
    terminal_omega: float
    terminal_rate_increment: float
    relative_displacement: float
    blend: float
    segments: tuple[ReachabilityWitnessSegment, ...]
    construction: str = "exact_envelope_blend_v1"


@dataclass(frozen=True)
class ExactPitchCertificate:
    """Exact finite-horizon reachability result for the bounded double integrator."""

    reachable: bool
    reason: str
    reachable_theta_lo: float
    reachable_theta_hi: float
    reachable_omega_lo: float
    reachable_omega_hi: float
    theta_overlap_lo: Optional[float]
    theta_overlap_hi: Optional[float]
    omega_overlap_lo: Optional[float]
    omega_overlap_hi: Optional[float]
    theta_margin: float
    omega_margin: float
    overall_margin: float
    binding_constraint: str
    witness: Optional[ReachabilityWitness]


@dataclass(frozen=True)
class _AffinePiece:
    t0: float
    t1: float
    slope: float
    intercept: float
    label: str

    def value(self, t: float) -> float:
        return self.slope * t + self.intercept

    def integral(self) -> float:
        return (
            0.5 * self.slope * (self.t1 * self.t1 - self.t0 * self.t0)
            + self.intercept * (self.t1 - self.t0)
        )


def _piecewise_envelope(
    lines: tuple[tuple[float, float, str], ...],
    T: float,
    *,
    upper: bool,
) -> tuple[_AffinePiece, ...]:
    """Return the exact min/max envelope of affine lines on ``[0,T]``."""
    cuts = {0.0, float(T)}
    for i, (m0, c0, _) in enumerate(lines):
        for m1, c1, _ in lines[i + 1:]:
            if abs(m0 - m1) <= 1e-15:
                continue
            t = (c1 - c0) / (m0 - m1)
            if 0.0 < t < T:
                cuts.add(float(t))
    ordered = sorted(cuts)
    pieces: list[_AffinePiece] = []
    chooser = min if upper else max
    for t0, t1 in zip(ordered, ordered[1:]):
        if t1 - t0 <= 1e-15:
            continue
        tm = 0.5 * (t0 + t1)
        chosen = chooser(lines, key=lambda line: line[0] * tm + line[1])
        piece = _AffinePiece(t0, t1, chosen[0], chosen[1], chosen[2])
        if (
            pieces
            and pieces[-1].label == piece.label
            and abs(pieces[-1].t1 - piece.t0) <= 1e-12
        ):
            prev = pieces[-1]
            pieces[-1] = _AffinePiece(
                prev.t0, piece.t1, prev.slope, prev.intercept, prev.label)
        else:
            pieces.append(piece)
    return tuple(pieces)


def _rate_profile(
    z_terminal: float,
    T: float,
    authority: PitchAuthority,
    *,
    upper: bool,
) -> tuple[_AffinePiece, ...]:
    if upper:
        lines: list[tuple[float, float, str]] = [
            (authority.a_up, 0.0, "accel_up"),
            (-authority.a_down, z_terminal + authority.a_down * T,
             "terminal_brake"),
        ]
        if math.isfinite(authority.B_up):
            lines.append((0.0, authority.B_up, "budget_up"))
    else:
        lines = [
            (-authority.a_down, 0.0, "accel_down"),
            (authority.a_up, z_terminal - authority.a_up * T,
             "terminal_drive"),
        ]
        if math.isfinite(authority.B_down):
            lines.append((0.0, -authority.B_down, "budget_down"))
    return _piecewise_envelope(tuple(lines), T, upper=upper)


def _profile_integral(profile: tuple[_AffinePiece, ...]) -> float:
    return sum(piece.integral() for piece in profile)


def _profile_value(profile: tuple[_AffinePiece, ...], t: float) -> tuple[float, str]:
    for piece in profile:
        if piece.t0 - 1e-12 <= t <= piece.t1 + 1e-12:
            return piece.value(t), piece.label
    raise RuntimeError(f"profile does not cover t={t}")


def _signed_interval_margin(
    reachable_lo: float,
    reachable_hi: float,
    target_lo: float,
    target_hi: float,
) -> float:
    """Positive overlap depth, or negative separating distance."""
    lo = max(reachable_lo, target_lo)
    hi = min(reachable_hi, target_hi)
    if lo <= hi:
        return min(reachable_hi - target_lo, target_hi - reachable_lo)
    return -min(abs(target_lo - reachable_hi), abs(reachable_lo - target_hi))


def _first_z_with_upper_at_least(
    target: float,
    z_lo: float,
    z_hi: float,
    T: float,
    authority: PitchAuthority,
) -> float:
    if _profile_integral(_rate_profile(z_lo, T, authority, upper=True)) >= target:
        return z_lo
    lo, hi = z_lo, z_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = _profile_integral(_rate_profile(mid, T, authority, upper=True))
        if value >= target:
            hi = mid
        else:
            lo = mid
    return hi


def _last_z_with_lower_at_most(
    target: float,
    z_lo: float,
    z_hi: float,
    T: float,
    authority: PitchAuthority,
) -> float:
    if _profile_integral(_rate_profile(z_hi, T, authority, upper=False)) <= target:
        return z_hi
    lo, hi = z_lo, z_hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        value = _profile_integral(_rate_profile(mid, T, authority, upper=False))
        if value <= target:
            lo = mid
        else:
            hi = mid
    return lo


def _construct_witness(
    theta0: float,
    omega0: float,
    T: float,
    authority: PitchAuthority,
    z_terminal: float,
    terminal_theta: float,
) -> ReachabilityWitness:
    lower = _rate_profile(z_terminal, T, authority, upper=False)
    upper = _rate_profile(z_terminal, T, authority, upper=True)
    lower_area = _profile_integral(lower)
    upper_area = _profile_integral(upper)
    target_area = terminal_theta - theta0 - omega0 * T
    span = upper_area - lower_area
    if span <= 1e-14:
        blend = 0.0
    else:
        blend = min(1.0, max(0.0, (target_area - lower_area) / span))

    cuts = sorted(
        {0.0, T}
        | {piece.t0 for piece in lower}
        | {piece.t1 for piece in lower}
        | {piece.t0 for piece in upper}
        | {piece.t1 for piece in upper}
    )
    segments: list[ReachabilityWitnessSegment] = []
    theta_sum = theta0
    for t0, t1 in zip(cuts, cuts[1:]):
        if t1 - t0 <= 1e-14:
            continue
        lo0, lo_label = _profile_value(lower, t0)
        lo1, _ = _profile_value(lower, t1)
        hi0, hi_label = _profile_value(upper, t0)
        hi1, _ = _profile_value(upper, t1)
        z0 = (1.0 - blend) * lo0 + blend * hi0
        z1 = (1.0 - blend) * lo1 + blend * hi1
        dt = t1 - t0
        acceleration = (z1 - z0) / dt
        omega_start = omega0 + z0
        omega_end = omega0 + z1
        theta_delta = 0.5 * (omega_start + omega_end) * dt
        theta_sum += theta_delta
        segments.append(
            ReachabilityWitnessSegment(
                t0=t0,
                t1=t1,
                acceleration=acceleration,
                omega_start=omega_start,
                omega_end=omega_end,
                theta_delta=theta_delta,
                active_lower=lo_label,
                active_upper=hi_label,
            )
        )
    if abs(theta_sum - terminal_theta) > 1e-8:
        raise RuntimeError(
            "internal witness construction error: "
            f"theta={theta_sum}, requested={terminal_theta}")
    return ReachabilityWitness(
        terminal_theta=theta_sum,
        terminal_omega=omega0 + z_terminal,
        terminal_rate_increment=z_terminal,
        relative_displacement=target_area,
        blend=blend,
        segments=tuple(segments),
    )


def exact_pitch_terminal_certificate(
    theta0: float,
    omega0: float,
    T: float,
    authority: PitchAuthority,
    terminal: PitchTerminalRegion,
    *,
    construct_witness: bool = True,
) -> ExactPitchCertificate:
    """Solve exact terminal-rectangle reachability for a bounded double integrator.

    For a fixed terminal rate increment ``z_f``, the pointwise maximal profile is
    ``min(a_up*t, B_up, z_f+a_down*(T-t))`` and the minimal profile is its
    lower-envelope counterpart. Their analytic integrals give exact terminal
    angle bounds. Convexity makes the projection over the admissible terminal
    rate interval a closed interval; a convex blend of the two profiles yields
    a constructive piecewise-constant-acceleration witness.
    """
    if not math.isfinite(float(theta0)) or not math.isfinite(float(omega0)):
        raise ValueError("theta0 and omega0 must be finite")
    if not math.isfinite(float(T)) or T <= 0.0:
        raise ValueError("T must be finite and > 0")

    z_lo = max(
        terminal.omega_lo - omega0,
        -authority.a_down * T,
        -authority.B_down,
    )
    z_hi = min(
        terminal.omega_hi - omega0,
        authority.a_up * T,
        authority.B_up,
    )
    reachable_omega_lo = omega0 + max(
        -authority.a_down * T, -authority.B_down)
    reachable_omega_hi = omega0 + min(
        authority.a_up * T, authority.B_up)
    omega_margin = _signed_interval_margin(
        reachable_omega_lo, reachable_omega_hi,
        terminal.omega_lo, terminal.omega_hi)

    if z_lo > z_hi + 1e-12:
        return ExactPitchCertificate(
            reachable=False,
            reason="terminal_rate_disjoint",
            reachable_theta_lo=float("nan"),
            reachable_theta_hi=float("nan"),
            reachable_omega_lo=reachable_omega_lo,
            reachable_omega_hi=reachable_omega_hi,
            theta_overlap_lo=None,
            theta_overlap_hi=None,
            omega_overlap_lo=None,
            omega_overlap_hi=None,
            theta_margin=float("-inf"),
            omega_margin=omega_margin,
            overall_margin=omega_margin,
            binding_constraint="terminal_rate",
            witness=None,
        )

    z_lo = min(z_hi, z_lo)
    theta_reachable_lo = (
        theta0 + omega0 * T
        + _profile_integral(_rate_profile(z_lo, T, authority, upper=False))
    )
    theta_reachable_hi = (
        theta0 + omega0 * T
        + _profile_integral(_rate_profile(z_hi, T, authority, upper=True))
    )
    theta_overlap_lo = max(theta_reachable_lo, terminal.theta_lo)
    theta_overlap_hi = min(theta_reachable_hi, terminal.theta_hi)
    theta_margin = _signed_interval_margin(
        theta_reachable_lo, theta_reachable_hi,
        terminal.theta_lo, terminal.theta_hi)
    reachable = theta_overlap_lo <= theta_overlap_hi + 1e-12
    overall_margin = min(theta_margin, omega_margin)
    if not reachable:
        return ExactPitchCertificate(
            reachable=False,
            reason="terminal_angle_disjoint",
            reachable_theta_lo=theta_reachable_lo,
            reachable_theta_hi=theta_reachable_hi,
            reachable_omega_lo=reachable_omega_lo,
            reachable_omega_hi=reachable_omega_hi,
            theta_overlap_lo=None,
            theta_overlap_hi=None,
            omega_overlap_lo=omega0 + z_lo,
            omega_overlap_hi=omega0 + z_hi,
            theta_margin=theta_margin,
            omega_margin=omega_margin,
            overall_margin=overall_margin,
            binding_constraint="terminal_angle",
            witness=None,
        )

    witness = None
    if construct_witness:
        terminal_theta = min(
            theta_overlap_hi,
            max(theta_overlap_lo, 0.5 * (terminal.theta_lo + terminal.theta_hi)),
        )
        target_area = terminal_theta - theta0 - omega0 * T
        feasible_z_lo = _first_z_with_upper_at_least(
            target_area, z_lo, z_hi, T, authority)
        feasible_z_hi = _last_z_with_lower_at_most(
            target_area, z_lo, z_hi, T, authority)
        if feasible_z_lo > feasible_z_hi + 1e-9:
            raise RuntimeError(
                "internal terminal-rate search failed despite interval overlap")
        z_terminal = 0.5 * (feasible_z_lo + feasible_z_hi)
        witness = _construct_witness(
            theta0, omega0, T, authority, z_terminal, terminal_theta)

    binding = (
        "terminal_rate" if omega_margin < theta_margin else "terminal_angle")
    return ExactPitchCertificate(
        reachable=True,
        reason="reachable",
        reachable_theta_lo=theta_reachable_lo,
        reachable_theta_hi=theta_reachable_hi,
        reachable_omega_lo=reachable_omega_lo,
        reachable_omega_hi=reachable_omega_hi,
        theta_overlap_lo=theta_overlap_lo,
        theta_overlap_hi=theta_overlap_hi,
        omega_overlap_lo=omega0 + z_lo,
        omega_overlap_hi=omega0 + z_hi,
        theta_margin=theta_margin,
        omega_margin=omega_margin,
        overall_margin=overall_margin,
        binding_constraint=binding,
        witness=witness,
    )


def pitch_accel_limit(tau_max: float, J_y: float, n_wheels: int = 4) -> float:
    """Return the pitch-acceleration limit ``n * tau_max / J_y``."""
    if J_y <= 0:
        raise ValueError("J_y must be > 0")
    return n_wheels * tau_max / J_y


def angular_momentum_budget(
    I_w: float,
    omega_max: float,
    omega_now: float | list[float] | tuple[float, ...],
    n_wheels: int = 4,
) -> float:
    """Return the legacy drive-direction momentum budget in kg m^2/s.

    ``omega_now`` is either a scalar (all wheels alike) or one entry per
    wheel, in rad/s. New code should call :func:`directional_rate_budgets`.
    """
    if isinstance(omega_now, (list, tuple)):
        return sum(max(0.0, I_w * (omega_max - float(w))) for w in omega_now)
    return max(0.0, n_wheels * I_w * (omega_max - float(omega_now)))


def directional_rate_budgets(
    I_w: float,
    omega_max: float,
    omega_min: float,
    omega_now: float | list[float] | tuple[float, ...],
    J_y: float,
    n_wheels: int = 4,
) -> tuple[float, float]:
    """Return the Theorem 1 budgets ``(B_up, B_down)`` in rad/s of body rate.

    ``B_up`` uses the remaining drive-direction wheel-speed capacity and
    ``B_down`` the braking capacity, including reversal through zero when
    ``omega_min < 0``. Each direction is clipped at zero, so a saturated
    wheel set simply exhausts that direction's budget.
    """
    if J_y <= 0:
        raise ValueError("J_y must be > 0")
    if isinstance(omega_now, (list, tuple)):
        h_up = sum(max(0.0, I_w * (omega_max - float(w))) for w in omega_now)
        h_dn = sum(max(0.0, I_w * (float(w) - omega_min)) for w in omega_now)
    else:
        w = float(omega_now)
        h_up = max(0.0, n_wheels * I_w * (omega_max - w))
        h_dn = max(0.0, n_wheels * I_w * (w - omega_min))
    return h_up / J_y, h_dn / J_y


def displacement_capacity(T_c: float, B_r: float, a: float) -> float:
    """Return the terminal-rate-zero displacement capacity ``D(T_c, B_r)``.

    Largest angle a bang-bang manoeuvre can cover in time ``T_c`` under peak
    rate ``B_r`` and acceleration ``a``, ending at zero rate:
    ``D = a T_c^2 / 4`` while the budget is slack (``T_c < 2 B_r / a``) and
    ``D = B_r T_c - B_r^2 / a`` once the rate saturates into a trapezoid.
    """
    if T_c <= 0.0:
        return 0.0
    if B_r <= 0.0:
        return 0.0
    if math.isinf(B_r) or T_c < 2.0 * B_r / a:
        return 0.25 * a * T_c * T_c
    return B_r * T_c - B_r * B_r / a


def pitch_reachable(
    theta0: float,
    omega_y0: float,
    theta_L: float,
    T: float,
    a: float,
    *,
    H_max: Optional[float] = None,
    J_y: Optional[float] = None,
    B_up: Optional[float] = None,
    B_down: Optional[float] = None,
    omega_target: float = 0.0,
    omega_bar: float = 0.0,
    theta_bar: float = 0.0,
) -> PitchReachability:
    """Evaluate the constructive pitch-axis certificate of Theorem 3.

    Parameters
    ----------
    theta0, omega_y0 : takeoff pitch (rad) and pitch rate (rad/s).
    theta_L : target landing pitch (rad).
    T : flight time (s), must be > 0.
    a : pitch-acceleration limit (rad/s^2), see :func:`pitch_accel_limit`.
    H_max, J_y : legacy symmetric budget, equivalent to
        ``B_up = B_down = H_max / J_y``. Prefer the directional pair.
    B_up, B_down : directional rate budgets (rad/s) from Theorem 1; when
        given they override the ``H_max``/``J_y`` form.
    omega_target : target landing pitch rate (rad/s), zero by default.
    omega_bar : terminal rate half-window (rad/s), zero by default, which
        recovers the exact-pair certificate. A positive value switches the
        rate condition to ``-(B_up + omega_bar) <= e_0 <= B_down +
        omega_bar`` and the construction to null-to-window plus drift: the
        budget only has to absorb ``|e_0| - r_omega``, and the residual rate
        left inside the window drifts by ``r_omega * T_c`` over the
        remaining time, which the displacement condition accounts for.
        Superposition is exact for the double integrator, so sufficiency
        still holds.
    theta_bar : terminal pitch half-window (rad), zero by default. When
        positive the displacement condition measures the distance to the
        nearest edge of ``[theta_L - theta_bar, theta_L + theta_bar]``.

    Returns
    -------
    PitchReachability, ``reachable`` iff all three conditions hold.
    """
    if T <= 0:
        raise ValueError("T (flight time) must be > 0")
    if a <= 0:
        raise ValueError("a (pitch accel limit) must be > 0")
    if omega_bar < 0:
        raise ValueError("omega_bar must be >= 0")
    if theta_bar < 0:
        raise ValueError("theta_bar must be >= 0")
    # Displacement error is taken against the target trajectory
    # theta_L + omega_target * (t - T), so psi(0) = theta0 - theta_L +
    # omega_target * T. Unchanged when omega_target is zero.
    e = theta0 - theta_L + omega_target * T
    s = omega_y0 - omega_target          # e_0, the initial rate error

    # Budget resolution: explicit directional, then legacy symmetric, then
    # unbounded.
    if B_up is not None and B_down is not None:
        b_up, b_dn = max(0.0, float(B_up)), max(0.0, float(B_down))
    elif H_max is not None and J_y is not None and J_y > 0:
        b_sym = max(0.0, float(H_max) / float(J_y))
        b_up = b_dn = b_sym
    else:
        b_up = b_dn = float("inf")

    # Rate condition: a nose-up rate error (e_0 > 0) has to be absorbed in
    # the braking direction and a nose-down error in the drive direction,
    # which is the typical case on a steep lip.
    budget_ok = (-(b_up + omega_bar) - 1e-12) <= s <= (b_dn + omega_bar + 1e-12)

    # Null-to-window plan: how much rate to shed, and the signed residual
    # left inside the window.
    b_dir = b_dn if s > 0.0 else b_up            # budget on the shed side
    if abs(s) <= omega_bar + 1e-15:
        reduce = 0.0                              # already inside the window
    elif math.isinf(b_dir) or abs(s) <= b_dir:
        reduce = abs(s)                           # budget suffices: null it
    else:
        reduce = min(abs(s), max(b_dir, abs(s) - omega_bar))  # shed to the edge
    r_w = math.copysign(abs(s) - reduce, s) if s != 0.0 else 0.0   # |r_w| <= omega_bar
    T_r = reduce / a

    # Time condition.
    velocity_nullable = T_r <= T + 1e-12

    # Displacement demand: the nulling leg plus the residual drift. The
    # nulling leg covers sign(s) * (s^2 - r_w^2) / (2a), which reduces to
    # sign(s) * s^2 / (2a) when the window is closed; the drift r_w * T_c is
    # deterministic and enters exactly rather than as a worst case.
    T_c = max(0.0, T - T_r)
    null_disp = math.copysign((s * s - r_w * r_w) / (2.0 * a), s) if s != 0.0 else 0.0
    r_theta = e + null_disp + r_w * T_c
    r_theta_to_set = (
        math.copysign(max(0.0, abs(r_theta) - theta_bar), r_theta)
        if r_theta != 0.0 else 0.0
    )
    # Residual budget: what the shed leg leaves in each direction. Shedding
    # spends `reduce` out of b_dir and returns the same amount to the
    # opposite side, by conservation of wheel momentum. A body nose-up rate
    # increment is paid for out of the drive budget B_up, so a residual
    # above the target (r_theta > 0) is corrected nose-down and draws on
    # B_down, and a residual below it draws on B_up.
    if math.isinf(b_up):
        B_r = float("inf")
    else:
        if s > 0.0:
            b_up_after, b_dn_after = b_up + reduce, b_dn - reduce
        elif s < 0.0:
            b_up_after, b_dn_after = b_up - reduce, b_dn + reduce
        else:
            b_up_after, b_dn_after = b_up, b_dn
        B_r = max(0.0, b_dn_after if r_theta >= 0.0 else b_up_after)
    if not velocity_nullable:
        displacement_ok = False
        cap = 0.0
    else:
        cap = displacement_capacity(T_c, B_r, a)
        displacement_ok = abs(r_theta_to_set) <= cap + 1e-12
    margin = cap - abs(r_theta_to_set)

    reachable = velocity_nullable and displacement_ok and budget_ok
    return PitchReachability(
        reachable=reachable,
        velocity_nullable=velocity_nullable,
        displacement_ok=displacement_ok,
        budget_ok=budget_ok,
        e=e,
        s=s,
        T_r=T_r,
        r_theta=r_theta,
        T_c=T_c,
        B_r=B_r,
        theta_bar=theta_bar,
        r_theta_to_set=r_theta_to_set,
        margin=margin,
    )


def vmax_cross_section(v_crit: float, a_brake: float, d: float) -> float:
    """Return the cross-section speed ceiling ``sqrt(v_crit^2 + 2 a_brake d)``.

    Propagates the takeoff-side critical speed ``v_crit`` backwards to a
    ceiling ``d`` metres short of the lip: any speed at or below the ceiling
    can still be brought inside ``v_crit`` over the remaining braking
    distance at deceleration ``a_brake``.
    """
    if a_brake < 0 or d < 0 or v_crit < 0:
        raise ValueError("v_crit, a_brake, d must be >= 0")
    return math.sqrt(v_crit * v_crit + 2.0 * a_brake * d)


@dataclass(frozen=True)
class TakeoffReachableSet:
    """Query wrapper for a fixed pitch-axis takeoff certificate.

    Once the terminal set, flight time and actuator limits are fixed, any
    candidate takeoff state can be tested for membership in the feasible
    set. :meth:`is_empty` drives the go/no-go branch: if not even the
    aligned, zero-rate takeoff is reachable, the set is empty and the jump
    must be refused.
    """

    theta_L: float
    T: float
    a: float
    H_max: Optional[float] = None
    J_y: Optional[float] = None
    B_up: Optional[float] = None
    B_down: Optional[float] = None
    omega_target: float = 0.0
    omega_bar: float = 0.0
    theta_bar: float = 0.0

    @classmethod
    def from_terminal_set(
        cls,
        terminal: PitchTerminalSet,
        *,
        T: float,
        a: float,
        H_max: Optional[float] = None,
        J_y: Optional[float] = None,
        B_up: Optional[float] = None,
        B_down: Optional[float] = None,
    ) -> "TakeoffReachableSet":
        return cls(
            theta_L=terminal.theta_center,
            T=T,
            a=a,
            H_max=H_max,
            J_y=J_y,
            B_up=B_up,
            B_down=B_down,
            omega_target=terminal.omega_center,
            omega_bar=terminal.omega_half_width,
            theta_bar=terminal.theta_half_width,
        )

    def contains(self, theta0: float, omega_y0: float) -> bool:
        return self.query(theta0, omega_y0).reachable

    def query(self, theta0: float, omega_y0: float) -> PitchReachability:
        return pitch_reachable(
            theta0, omega_y0, self.theta_L, self.T, self.a,
            H_max=self.H_max, J_y=self.J_y,
            B_up=self.B_up, B_down=self.B_down,
            omega_target=self.omega_target,
            omega_bar=self.omega_bar,
            theta_bar=self.theta_bar,
        )

    def omega_bound(self) -> float:
        """Return a conservative scalar bound on the takeoff pitch rate."""
        bound = self.a * self.T
        if self.B_up is not None and self.B_down is not None:
            bound = min(bound, max(0.0, min(float(self.B_up), float(self.B_down))))
        elif self.H_max is not None and self.J_y is not None and self.J_y > 0:
            bound = min(bound, self.H_max / self.J_y)
        return bound

    def is_empty(self) -> bool:
        """Return whether even the ideal takeoff at the terminal rate fails."""
        ideal_theta0 = self.theta_L - self.omega_target * self.T
        return not self.contains(ideal_theta0, self.omega_target)


__all__ = [
    "PitchReachability",
    "PitchTerminalSet",
    "PitchAuthority",
    "PitchTerminalRegion",
    "ReachabilityWitnessSegment",
    "ReachabilityWitness",
    "ExactPitchCertificate",
    "TakeoffReachableSet",
    "pitch_accel_limit",
    "angular_momentum_budget",
    "directional_rate_budgets",
    "displacement_capacity",
    "pitch_reachable",
    "exact_pitch_terminal_certificate",
    "vmax_cross_section",
]
