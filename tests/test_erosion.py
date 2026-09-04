"""Erosion certificate; the acceptance gate is 1e4 Monte Carlo worst-case
samples with zero counterexamples."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.darts.erosion import (  # noqa: E402
    ErosionRho,
    ExactCertificateUncertainty,
    eroded_reachable,
    robust_exact_pitch_certificate,
)
from control.darts.reachability import (  # noqa: E402
    PitchAuthority,
    PitchTerminalRegion,
    exact_pitch_terminal_certificate,
    pitch_reachable,
)

# Pooled measured residual sigma, so rho = 2 sigma.
RHO = ErosionRho.from_sigma({"omega_dps": 4.0, "theta0_deg": 2.1, "T_s": 0.085}, k=2.0)
A = math.radians(120.0)
B_UP, B_DN = math.radians(10.0), math.radians(24.0)
OB = math.radians(15.0)
KW = dict(B_up=B_UP, B_down=B_DN, omega_bar=OB)


def test_eroded_subset_of_nominal():
    """An eroded pass implies a nominal pass, so the eroded set is a subset."""
    rng = random.Random(1)
    for _ in range(500):
        th = rng.uniform(-0.4, 0.5)
        om = rng.uniform(-0.9, 0.5)
        T = rng.uniform(0.4, 2.0)
        if eroded_reachable(th, om, math.radians(-12), T, A, rho=RHO, **KW):
            assert pitch_reachable(th, om, math.radians(-12), T, A, **KW).reachable


def test_mc_worstcase_zero_counterexample():
    """An eroded pass must survive every perturbation within rho.

    The finer grid of 17 nodes shrinks the buffer and therefore admits more
    states, which is the harder test of sufficiency. Sampling is biased
    towards the certificate boundary.
    """
    rng = random.Random(7)
    n_pass = 0
    n_cex = 0
    trials = 0
    while n_pass < 40 and trials < 12000:
        trials += 1
        th = rng.uniform(-0.3, 0.5)
        om = rng.uniform(-0.8, 0.4)
        T = rng.uniform(0.5, 1.8)
        if not eroded_reachable(th, om, math.radians(-12), T, A, rho=RHO,
                                grid=17, **KW):
            continue
        n_pass += 1
        for _ in range(250):                     # 40 passing states x 250 = 1e4
            dth = rng.uniform(-RHO.theta0, RHO.theta0)
            dom = rng.uniform(-RHO.omega, RHO.omega)
            dT = rng.uniform(-RHO.T, RHO.T)
            r = pitch_reachable(th + dth, om + dom, math.radians(-12),
                                max(0.05, T + dT), A, **KW)
            if not r.reachable:
                n_cex += 1
    assert n_pass >= 30, f"too few eroded passes to be meaningful ({n_pass})"
    assert n_cex == 0, f"{n_cex} counterexamples: erosion is not sufficient"


def test_erosion_strictly_tightens():
    """Boundary states pass nominally and fail eroded, so erosion bites."""
    found = False
    rng = random.Random(3)
    for _ in range(2000):
        th = rng.uniform(-0.3, 0.5)
        om = rng.uniform(-0.8, 0.4)
        T = rng.uniform(0.4, 1.5)
        nom = pitch_reachable(th, om, math.radians(-12), T, A, **KW).reachable
        ero = eroded_reachable(th, om, math.radians(-12), T, A, rho=RHO, **KW)
        if nom and not ero:
            found = True
            break
    assert found


def test_zero_rho_equals_nominal():
    rng = random.Random(5)
    z = ErosionRho()
    for _ in range(300):
        th = rng.uniform(-0.4, 0.5)
        om = rng.uniform(-0.9, 0.5)
        T = rng.uniform(0.3, 2.0)
        nom = pitch_reachable(th, om, math.radians(-12), T, A, **KW).reachable
        assert eroded_reachable(th, om, math.radians(-12), T, A, rho=z, **KW) == nom


def test_from_sigma():
    r = ErosionRho.from_sigma({"omega_dps": 4.0, "theta0_deg": 2.1, "T_s": 0.085}, k=2.0)
    assert abs(r.omega - math.radians(8.0)) < 1e-9
    assert abs(r.theta0 - math.radians(4.2)) < 1e-9
    assert abs(r.T - 0.17) < 1e-9


def test_robust_exact_zero_uncertainty_matches_nominal():
    authority = PitchAuthority(1.8, 1.2, 0.4, 0.3)
    terminal = PitchTerminalRegion(-0.1, 0.1, -0.08, 0.08)
    nominal = exact_pitch_terminal_certificate(
        -0.05, -0.1, 1.2, authority, terminal, construct_witness=False)
    robust = robust_exact_pitch_certificate(
        -0.05,
        -0.1,
        1.2,
        authority,
        terminal,
        uncertainty=ExactCertificateUncertainty(),
    )
    assert robust.reachable == nominal.reachable
    assert len(robust.corner_certificates) == 1
    assert robust.worst_margin == nominal.overall_margin
    assert robust.effective_terminal == terminal


def test_robust_exact_time_uncertainty_uses_min_time_and_coast_erosion():
    terminal = PitchTerminalRegion(-0.2, 0.2, -0.1, 0.1)
    robust = robust_exact_pitch_certificate(
        0.0,
        0.0,
        1.0,
        PitchAuthority.symmetric(2.0, 0.8),
        terminal,
        uncertainty=ExactCertificateUncertainty(T=0.1),
    )
    assert robust.T_min == 0.9
    assert robust.T_max == 1.1
    assert robust.terminal_coast_erosion == pytest.approx(0.02)
    assert robust.effective_terminal is not None
    assert robust.effective_terminal.theta_lo == pytest.approx(-0.18)
    assert robust.effective_terminal.theta_hi == pytest.approx(0.18)


def test_robust_exact_checks_all_initial_state_vertices():
    robust = robust_exact_pitch_certificate(
        theta0=0.0,
        omega0=0.0,
        T=1.2,
        authority=PitchAuthority.symmetric(2.0, 0.8),
        terminal=PitchTerminalRegion(-0.15, 0.15, -0.1, 0.1),
        uncertainty=ExactCertificateUncertainty(theta0=0.02, omega0=0.03),
    )
    assert len(robust.corner_certificates) == 4
    assert robust.reachable is True
    assert robust.failed_corner_indices == ()


def test_robust_exact_authority_and_terminal_erosion_fail_closed():
    terminal = PitchTerminalRegion(-0.05, 0.05, -0.02, 0.02)
    authority = PitchAuthority.symmetric(1.0, 0.2)
    acceleration_empty = robust_exact_pitch_certificate(
        0.0, 0.0, 1.0, authority, terminal,
        uncertainty=ExactCertificateUncertainty(a_up=1.0))
    assert acceleration_empty.reachable is False
    assert acceleration_empty.reason == "nonpositive_robust_acceleration"
    terminal_empty = robust_exact_pitch_certificate(
        0.0, 0.0, 1.0, authority, terminal,
        uncertainty=ExactCertificateUncertainty(terminal_omega=0.03))
    assert terminal_empty.reachable is False
    assert terminal_empty.reason == "terminal_rate_eroded_empty"


def test_robust_exact_monotonically_shrinks_on_nested_uncertainty():
    authority = PitchAuthority.symmetric(1.4, 0.45)
    terminal = PitchTerminalRegion(-0.1, 0.1, -0.08, 0.08)
    small = ExactCertificateUncertainty(
        theta0=0.005, omega0=0.005, T=0.005,
        a_up=0.01, a_down=0.01, B_up=0.005, B_down=0.005,
        terminal_theta=0.002, terminal_omega=0.002,
    )
    large = ExactCertificateUncertainty(
        theta0=0.03, omega0=0.03, T=0.03,
        a_up=0.05, a_down=0.05, B_up=0.02, B_down=0.02,
        terminal_theta=0.01, terminal_omega=0.01,
    )
    rng = random.Random(20260811)
    for _ in range(300):
        theta0 = rng.uniform(-0.2, 0.2)
        omega0 = rng.uniform(-0.25, 0.25)
        small_result = robust_exact_pitch_certificate(
            theta0, omega0, 1.0, authority, terminal, uncertainty=small)
        large_result = robust_exact_pitch_certificate(
            theta0, omega0, 1.0, authority, terminal, uncertainty=large)
        assert not (large_result.reachable and not small_result.reachable)
