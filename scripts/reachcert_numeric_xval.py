#!/usr/bin/env python3
"""Numerically cross-validate the DART-S exact pitch certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from control.darts.reachability import (  # noqa: E402
    PitchAuthority,
    PitchTerminalRegion,
    exact_pitch_terminal_certificate,
    pitch_reachable,
)


DEFAULT_OUT = (
    REPO
    / "data/derived"
    / "cert_numeric_xval.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lp_extrema(
    theta0: float,
    omega0: float,
    T: float,
    authority: PitchAuthority,
    terminal: PitchTerminalRegion,
    steps: int,
) -> tuple[bool, float | None, float | None]:
    dt = T / steps
    area_coeff = np.array(
        [(steps - index - 0.5) * dt * dt for index in range(steps)],
        dtype=float,
    )
    cumulative = np.tril(np.ones((steps, steps), dtype=float)) * dt
    A_ub = np.vstack((
        cumulative,
        -cumulative,
        np.full((1, steps), dt),
        np.full((1, steps), -dt),
    ))
    b_ub = np.concatenate((
        np.full(steps, authority.B_up),
        np.full(steps, authority.B_down),
        [terminal.omega_hi - omega0, omega0 - terminal.omega_lo],
    ))
    bounds = [(-authority.a_down, authority.a_up)] * steps
    low = linprog(
        area_coeff, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    high = linprog(
        -area_coeff, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not low.success or not high.success:
        return False, None, None
    drift = theta0 + omega0 * T
    return True, drift + float(low.fun), drift - float(high.fun)


def validate_witness(
    theta0: float,
    omega0: float,
    authority: PitchAuthority,
    terminal: PitchTerminalRegion,
    certificate,
) -> bool:
    witness = certificate.witness
    if witness is None:
        return not certificate.reachable
    z_values = [0.0]
    theta = theta0
    last_t = 0.0
    last_omega = omega0
    for segment in witness.segments:
        if abs(segment.t0 - last_t) > 1e-9:
            return False
        if abs(segment.omega_start - last_omega) > 1e-8:
            return False
        if not (
            -authority.a_down - 1e-9
            <= segment.acceleration
            <= authority.a_up + 1e-9
        ):
            return False
        z_values.append(segment.omega_end - omega0)
        theta += segment.theta_delta
        last_t = segment.t1
        last_omega = segment.omega_end
    return (
        min(z_values) >= -authority.B_down - 1e-8
        and max(z_values) <= authority.B_up + 1e-8
        and terminal.theta_lo - 1e-8
        <= theta
        <= terminal.theta_hi + 1e-8
        and terminal.omega_lo - 1e-8
        <= last_omega
        <= terminal.omega_hi + 1e-8
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lp-cases", type=int, default=64)
    parser.add_argument("--lp-steps", type=int, default=400)
    parser.add_argument("--legacy-cases", type=int, default=5000)
    parser.add_argument("--latency-cases", type=int, default=3000)
    args = parser.parse_args()

    rng = random.Random(20260811)
    lp_rows: list[dict] = []
    false_accepts = 0
    bound_tolerance_violations = 0
    max_bound_error = 0.0
    for case_index in range(args.lp_cases):
        theta0 = rng.uniform(-0.7, 0.7)
        omega0 = rng.uniform(-0.9, 0.9)
        T = rng.uniform(0.35, 2.0)
        authority = PitchAuthority(
            a_up=rng.uniform(0.6, 2.6),
            a_down=rng.uniform(0.6, 2.6),
            B_up=rng.uniform(0.08, 0.9),
            B_down=rng.uniform(0.08, 0.9),
        )
        omega_center = rng.uniform(-0.35, 0.35)
        omega_half = rng.uniform(0.0, 0.12)
        theta_center = rng.uniform(-0.6, 0.6)
        theta_half = rng.uniform(0.0, 0.15)
        terminal = PitchTerminalRegion(
            theta_center - theta_half,
            theta_center + theta_half,
            omega_center - omega_half,
            omega_center + omega_half,
        )
        exact = exact_pitch_terminal_certificate(
            theta0, omega0, T, authority, terminal)
        lp_feasible, lp_lo, lp_hi = lp_extrema(
            theta0, omega0, T, authority, terminal, args.lp_steps)
        tolerance = (
            3.0 * max(authority.a_up, authority.a_down)
            * (T / args.lp_steps) ** 2
        )
        if lp_feasible:
            assert lp_lo is not None and lp_hi is not None
            error = max(
                abs(exact.reachable_theta_lo - lp_lo),
                abs(exact.reachable_theta_hi - lp_hi),
            )
            max_bound_error = max(max_bound_error, error)
            bound_tolerance_violations += int(error > tolerance)
            lp_reachable = (
                lp_lo <= terminal.theta_hi + tolerance
                and lp_hi >= terminal.theta_lo - tolerance
            )
        else:
            error = None
            lp_reachable = False
        false_accept = exact.reachable and not lp_reachable
        false_accepts += int(false_accept)
        lp_rows.append({
            "case_index": case_index,
            "exact_reachable": exact.reachable,
            "lp_reachable": lp_reachable,
            "lp_feasible": lp_feasible,
            "false_accept": false_accept,
            "bound_error": error,
            "tolerance": tolerance,
        })

    legacy_passes = 0
    legacy_subset_failures = 0
    for _ in range(args.legacy_cases):
        theta0 = rng.uniform(-0.8, 0.8)
        omega0 = rng.uniform(-1.0, 1.0)
        T = rng.uniform(0.25, 2.2)
        a = rng.uniform(0.5, 2.8)
        b_up = rng.uniform(0.05, 1.0)
        b_down = rng.uniform(0.05, 1.0)
        legacy = pitch_reachable(
            theta0, omega0, 0.0, T, a, B_up=b_up, B_down=b_down)
        if not legacy.reachable:
            continue
        legacy_passes += 1
        exact = exact_pitch_terminal_certificate(
            theta0,
            omega0,
            T,
            PitchAuthority(a, a, b_up, b_down),
            PitchTerminalRegion(0.0, 0.0, 0.0, 0.0),
            construct_witness=False,
        )
        legacy_subset_failures += int(not exact.reachable)

    witness_cases = 0
    witness_failures = 0
    for row in lp_rows:
        case_index = int(row["case_index"])
        local = random.Random(20260811 + case_index)
        authority = PitchAuthority(
            local.uniform(0.8, 2.0),
            local.uniform(0.8, 2.0),
            local.uniform(0.2, 0.8),
            local.uniform(0.2, 0.8),
        )
        theta0 = local.uniform(-0.15, 0.15)
        omega0 = local.uniform(-0.2, 0.2)
        terminal = PitchTerminalRegion(-0.2, 0.2, -0.15, 0.15)
        certificate = exact_pitch_terminal_certificate(
            theta0, omega0, 1.2, authority, terminal)
        if certificate.reachable:
            witness_cases += 1
            witness_failures += int(not validate_witness(
                theta0, omega0, authority, terminal, certificate))

    latency_samples: list[float] = []
    authority = PitchAuthority(1.8, 1.2, 0.4, 0.3)
    terminal = PitchTerminalRegion(-0.2, 0.2, -0.15, 0.15)
    for _ in range(args.latency_cases):
        theta0 = rng.uniform(-0.3, 0.3)
        omega0 = rng.uniform(-0.4, 0.4)
        started = time.perf_counter_ns()
        exact_pitch_terminal_certificate(
            theta0, omega0, 1.2, authority, terminal)
        latency_samples.append((time.perf_counter_ns() - started) / 1e6)
    latency_samples.sort()
    p50 = latency_samples[len(latency_samples) // 2]
    p99 = latency_samples[min(
        len(latency_samples) - 1,
        math.ceil(0.99 * len(latency_samples)) - 1,
    )]

    decision = (
        "PASS"
        if false_accepts == 0
        and bound_tolerance_violations == 0
        and legacy_subset_failures == 0
        and witness_failures == 0
        and p99 < 10.0
        else "FAIL"
    )
    payload = {
        "schema_version": "darts-cert-numeric-xval/1",
        "canonical_id": "CERT_NUMERIC_XVAL_V1",
        "decision": decision,
        "_provenance": {
            "time_integrity_ok": True,
            "time_basis": "not_applicable_zero_beamng",
            "sources": [
                {"file": "control/darts/reachability.py"},
                {"file": "control/darts/wheel_budget.py"},
                {"file": "control/darts/erosion.py"},
            ],
        },
        "method_fingerprint": {
            "reachability_py_sha256": sha256(
                REPO / "control/darts/reachability.py"),
            "wheel_budget_py_sha256": sha256(
                REPO / "control/darts/wheel_budget.py"),
            "erosion_py_sha256": sha256(REPO / "control/darts/erosion.py"),
        },
        "lp_direct_collocation": {
            "cases": args.lp_cases,
            "steps": args.lp_steps,
            "false_accepts": false_accepts,
            "bound_tolerance_violations": bound_tolerance_violations,
            "max_bound_error_rad": max_bound_error,
            "rows": lp_rows,
        },
        "legacy_subset": {
            "draws": args.legacy_cases,
            "legacy_passes": legacy_passes,
            "subset_failures": legacy_subset_failures,
        },
        "witness_validation": {
            "reachable_cases": witness_cases,
            "failures": witness_failures,
        },
        "latency_ms": {
            "n": len(latency_samples),
            "p50": p50,
            "p99": p99,
            "max": max(latency_samples),
        },
        "scope": (
            "Continuous-time pitch double integrator with asymmetric acceleration "
            "and signed cumulative rate budgets; not a full landing/hardware "
            "safety certificate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[CERT-XVAL] {decision}: LP false_accepts={false_accepts}, "
        f"bound_tolerance_violations={bound_tolerance_violations}, "
        f"legacy_subset_failures={legacy_subset_failures}, "
        f"witness_failures={witness_failures}, p99={p99:.3f} ms")
    print(f"[CERT-XVAL] saved {args.out}")
    return 0 if decision == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
