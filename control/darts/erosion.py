"""Certificate erosion: shrink the certificate by the predictor residual.

The takeoff-state predictor has a residual (pooled measurements: about
4.0 deg/s in pitch rate, 2.1 deg in pitch and 0.085 s in flight time), so a
gate that consumes predicted states has to erode the certificate by
``rho = k sigma``, with ``k = 2`` by default. Passing the eroded test then
implies that every true state within ``|delta| <= rho`` still satisfies the
nominal certificate.

Sufficiency comes from a Lipschitz-buffered grid. The perturbation box
``(theta0 +/- rho_theta) x (omega0 +/- rho_omega) x (T +/- rho_T)`` is
sampled on a grid, every node is tested with :func:`pitch_reachable`, and
each node must clear the nominal test by at least the half-step Lipschitz
buffer. The certificate predicate is piecewise C1 and globally Lipschitz in
each input, with the conservative gradient bounds

* ``|d/d theta0| <= 1``, the linear term of ``r_theta``;
* ``|d/d omega0| <= 2 T_hi``, from the nulling displacement plus drift and
  the change in capacity induced by shifting ``B_r``;
* ``|d/d T| <= B_cap + omega_bar``, from the capacity slope in ``T_c`` and
  the drift rate.

So if every node passes with
``margin >= h_theta + 2 T_hi h_omega + (B_cap + omega_bar) h_T``, where the
``h`` are the per-axis half-steps, then every continuous point in the box
passes the nominal certificate. The two boolean conditions, budget
feasibility and ``T_r <= T``, are covered explicitly at the grid endpoints,
including the ``+/-rho`` faces. An empty erosion, meaning the nominal point
itself fails, routes to the existing no-go branch.

Two earlier constructions were discarded after Monte Carlo worst-case
sampling found counterexamples: a coarse grid with no buffer, and a frozen
control profile, whose directional gap let a perturbation flip the required
correction onto an already exhausted budget direction. The Monte Carlo check
is kept as the regression lock for this module.

Pure logic, no simulator or tensor dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

from control.darts.reachability import (
    ExactPitchCertificate,
    PitchAuthority,
    PitchTerminalRegion,
    exact_pitch_terminal_certificate,
    pitch_reachable,
)


@dataclass(frozen=True)
class ErosionRho:
    """Erosion radii per axis, in the certificate's own units."""

    theta0: float = 0.0
    omega: float = 0.0
    T: float = 0.0

    @classmethod
    def from_sigma(cls, sigma: dict, k: float = 2.0) -> "ErosionRho":
        """Build ``rho = k sigma`` from a predictor residual given in degrees."""
        return cls(
            theta0=k * math.radians(float(sigma.get("theta0_deg", 0.0))),
            omega=k * math.radians(float(sigma.get("omega_dps", 0.0))),
            T=k * float(sigma.get("T_s", 0.0)),
        )


@dataclass(frozen=True)
class ExactCertificateUncertainty:
    """Half widths consumed by the certificate-first robust authorization."""

    theta0: float = 0.0
    omega0: float = 0.0
    T: float = 0.0
    a_up: float = 0.0
    a_down: float = 0.0
    B_up: float = 0.0
    B_down: float = 0.0
    terminal_theta: float = 0.0
    terminal_omega: float = 0.0

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} uncertainty must be finite and >= 0")


@dataclass(frozen=True)
class RobustExactPitchCertificate:
    reachable: bool
    reason: str
    worst_margin: float
    effective_authority: PitchAuthority
    effective_terminal: PitchTerminalRegion | None
    T_min: float
    T_max: float
    terminal_coast_erosion: float
    corner_certificates: tuple[ExactPitchCertificate, ...]
    failed_corner_indices: tuple[int, ...]
    construction: str = "vertex_box_plus_min_time_coast/v1"


def _lower_bound(value: float, uncertainty: float) -> float:
    if math.isinf(value):
        return value
    return max(0.0, value - uncertainty)


def robust_exact_pitch_certificate(
    theta0: float,
    omega0: float,
    T: float,
    authority: PitchAuthority,
    terminal: PitchTerminalRegion,
    *,
    uncertainty: ExactCertificateUncertainty,
    construct_witness: bool = False,
) -> RobustExactPitchCertificate:
    """Sufficient robust certificate for a continuous action-state box.

    The method uses lower authority/budget bounds and erodes the terminal
    rectangle. Time uncertainty is handled constructively: every corner must
    reach the eroded set at ``T_min``; the witness then coasts at zero
    acceleration for up to ``T_max-T_min``. The additional terminal-angle
    erosion bounds that drift for every terminal rate in the retained window.

    For fixed time and authority the feasible initial-state set is convex.
    Therefore checking all ``(theta0, omega0)`` box vertices is sufficient for
    every interior initial state; no unsound corner-only claim is made over
    time.
    """
    if not all(math.isfinite(float(value)) for value in (theta0, omega0, T)):
        raise ValueError("theta0, omega0, and T must be finite")
    T_min = T - uncertainty.T
    T_max = T + uncertainty.T
    if T_min <= 0.0:
        fallback_authority = PitchAuthority(
            a_up=max(1e-15, authority.a_up),
            a_down=max(1e-15, authority.a_down),
            B_up=authority.B_up,
            B_down=authority.B_down,
        )
        return RobustExactPitchCertificate(
            reachable=False,
            reason="nonpositive_minimum_flight_time",
            worst_margin=float("-inf"),
            effective_authority=fallback_authority,
            effective_terminal=None,
            T_min=T_min,
            T_max=T_max,
            terminal_coast_erosion=float("inf"),
            corner_certificates=(),
            failed_corner_indices=(),
        )
    a_up = authority.a_up - uncertainty.a_up
    a_down = authority.a_down - uncertainty.a_down
    if a_up <= 0.0 or a_down <= 0.0:
        fallback_authority = PitchAuthority(
            a_up=max(1e-15, a_up),
            a_down=max(1e-15, a_down),
            B_up=_lower_bound(authority.B_up, uncertainty.B_up),
            B_down=_lower_bound(authority.B_down, uncertainty.B_down),
        )
        return RobustExactPitchCertificate(
            reachable=False,
            reason="nonpositive_robust_acceleration",
            worst_margin=float("-inf"),
            effective_authority=fallback_authority,
            effective_terminal=None,
            T_min=T_min,
            T_max=T_max,
            terminal_coast_erosion=float("inf"),
            corner_certificates=(),
            failed_corner_indices=(),
        )
    effective_authority = PitchAuthority(
        a_up=a_up,
        a_down=a_down,
        B_up=_lower_bound(authority.B_up, uncertainty.B_up),
        B_down=_lower_bound(authority.B_down, uncertainty.B_down),
    )
    omega_lo = terminal.omega_lo + uncertainty.terminal_omega
    omega_hi = terminal.omega_hi - uncertainty.terminal_omega
    if omega_lo > omega_hi:
        return RobustExactPitchCertificate(
            reachable=False,
            reason="terminal_rate_eroded_empty",
            worst_margin=float("-inf"),
            effective_authority=effective_authority,
            effective_terminal=None,
            T_min=T_min,
            T_max=T_max,
            terminal_coast_erosion=float("inf"),
            corner_certificates=(),
            failed_corner_indices=(),
        )
    coast_erosion = max(abs(omega_lo), abs(omega_hi)) * (T_max - T_min)
    theta_erosion = uncertainty.terminal_theta + coast_erosion
    theta_lo = terminal.theta_lo + theta_erosion
    theta_hi = terminal.theta_hi - theta_erosion
    if theta_lo > theta_hi:
        return RobustExactPitchCertificate(
            reachable=False,
            reason="terminal_angle_eroded_empty",
            worst_margin=float("-inf"),
            effective_authority=effective_authority,
            effective_terminal=None,
            T_min=T_min,
            T_max=T_max,
            terminal_coast_erosion=coast_erosion,
            corner_certificates=(),
            failed_corner_indices=(),
        )
    effective_terminal = PitchTerminalRegion(
        theta_lo=theta_lo,
        theta_hi=theta_hi,
        omega_lo=omega_lo,
        omega_hi=omega_hi,
    )

    theta_values = sorted({
        theta0 - uncertainty.theta0,
        theta0 + uncertainty.theta0,
    })
    omega_values = sorted({
        omega0 - uncertainty.omega0,
        omega0 + uncertainty.omega0,
    })
    certificates = tuple(
        exact_pitch_terminal_certificate(
            theta,
            omega,
            T_min,
            effective_authority,
            effective_terminal,
            construct_witness=construct_witness,
        )
        for theta, omega in product(theta_values, omega_values)
    )
    failed = tuple(
        index for index, certificate in enumerate(certificates)
        if not certificate.reachable
    )
    worst_margin = min(
        (certificate.overall_margin for certificate in certificates),
        default=float("-inf"),
    )
    return RobustExactPitchCertificate(
        reachable=not failed,
        reason="reachable" if not failed else "initial_state_corner_failed",
        worst_margin=worst_margin,
        effective_authority=effective_authority,
        effective_terminal=effective_terminal,
        T_min=T_min,
        T_max=T_max,
        terminal_coast_erosion=coast_erosion,
        corner_certificates=certificates,
        failed_corner_indices=failed,
    )


def _axis_grid(center: float, half: float, n: int) -> list[float]:
    if half <= 0.0:
        return [center]
    return [center - half + 2.0 * half * i / (n - 1) for i in range(n)]


def eroded_reachable(
    theta0: float,
    omega_y0: float,
    theta_L: float,
    T: float,
    a: float,
    *,
    rho: ErosionRho,
    grid: int = 13,
    omega_bar: float = 0.0,
    B_up: float | None = None,
    B_down: float | None = None,
    **kw,
) -> bool:
    """Eroded certificate: pass only if every buffered grid node passes."""
    if T - rho.T <= 0.0:
        return False
    n = max(3, int(grid))
    ths = _axis_grid(theta0, rho.theta0, n)
    oms = _axis_grid(omega_y0, rho.omega, n)
    Ts = _axis_grid(T, rho.T, n)
    T_hi = T + rho.T
    # Lipschitz buffer: per-axis half-step times the gradient bound.
    h_th = (rho.theta0 / (n - 1)) if rho.theta0 > 0 else 0.0
    h_om = (rho.omega / (n - 1)) if rho.omega > 0 else 0.0
    h_T = (rho.T / (n - 1)) if rho.T > 0 else 0.0
    b_cap = a * T_hi
    if B_up is not None and B_down is not None:
        b_cap = min(b_cap, max(float(B_up), float(B_down)) + omega_bar)
    buffer_need = h_th * 1.0 + h_om * 2.0 * T_hi + h_T * (b_cap + omega_bar)
    for th in ths:
        for om in oms:
            for T_e in Ts:
                r = pitch_reachable(th, om, theta_L, T_e, a,
                                    omega_bar=omega_bar,
                                    B_up=B_up, B_down=B_down, **kw)
                if not r.reachable or r.margin < buffer_need - 1e-12:
                    return False
    return True


__all__ = [
    "ErosionRho",
    "ExactCertificateUncertainty",
    "RobustExactPitchCertificate",
    "eroded_reachable",
    "robust_exact_pitch_certificate",
]
