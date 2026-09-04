"""Resolve measured wheel-spin inputs for DART directional budgets."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from control.darts.reachability import directional_rate_budgets


SUSTAINED_EFFECTIVE_ENVELOPE_RADPS = 125.7
OPERATIONAL_SPIN_GUARD_RADPS = 200.0
DRIVETRAIN_HARD_LIMIT_RADPS = 221.2


class BudgetInputMissingError(ValueError):
    pass


class BudgetConventionError(BudgetInputMissingError):
    pass


class BudgetOutOfDomainError(ValueError):
    pass


BEAMNG_SBR4_WHEEL_ORDER = ("FL", "FR", "RL", "RR")
BEAMNG_SBR4_FORWARD_POLARITY = {
    "FL": -1.0,
    "FR": 1.0,
    "RL": -1.0,
    "RR": 1.0,
}
BEAMNG_SBR4_SPIN_CONVENTION_ID = "beamng-sbr4-forward-spin/v1"


@dataclass(frozen=True)
class WheelAuthorityLimits:
    """Separate certificate, operational, and physical wheel-speed limits."""

    sustained_effective_envelope_radps: float = SUSTAINED_EFFECTIVE_ENVELOPE_RADPS
    operational_spin_guard_radps: float = OPERATIONAL_SPIN_GUARD_RADPS
    drivetrain_hard_limit_radps: float = DRIVETRAIN_HARD_LIMIT_RADPS

    def __post_init__(self):
        sustained = float(self.sustained_effective_envelope_radps)
        guard = float(self.operational_spin_guard_radps)
        hard = float(self.drivetrain_hard_limit_radps)
        ordered = (
            0.0 < sustained <= hard
            and (guard <= 0.0 or sustained <= guard <= hard)
        )
        if not ordered:
            raise ValueError(
                "wheel limits must satisfy 0 < sustained envelope <= hard "
                "and, when enabled, sustained <= operational guard <= hard")


@dataclass(frozen=True)
class BudgetInputs:
    omega0_radps: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class SignedBudgetInputs:
    """Raw and canonical forward-spin values with an explicit sign convention."""

    wheel_names: tuple[str, ...]
    raw_omega_radps: tuple[float, ...]
    canonical_omega_radps: tuple[float, ...]
    source: str
    convention_id: str


@dataclass(frozen=True)
class WheelRateInterval:
    lower_radps: float
    upper_radps: float

    def __post_init__(self):
        if not (
            math.isfinite(float(self.lower_radps))
            and math.isfinite(float(self.upper_radps))
        ):
            raise ValueError("wheel-rate interval bounds must be finite")
        if self.lower_radps > self.upper_radps:
            raise ValueError("wheel-rate interval lower must be <= upper")


@dataclass(frozen=True)
class CertifiedWheelBudget:
    """Worst-case directional body-rate budgets over signed wheel intervals."""

    B_up: float
    B_down: float
    per_wheel_up: tuple[float, ...]
    per_wheel_down: tuple[float, ...]
    wheel_names: tuple[str, ...]
    convention_id: str


def _clean_wheels(values: Iterable[float]) -> tuple[float, ...]:
    # BeamNG left/right wheel signs differ; all paper runs approach forward.
    # Store raw values separately, but use forward-spin magnitudes in Theorem 1.
    return tuple(abs(float(v)) for v in values)


def resolve_signed_budget_inputs(
    takeoff_state: Mapping[str, object],
    *,
    wheel_order: tuple[str, ...] = BEAMNG_SBR4_WHEEL_ORDER,
    forward_polarity: Mapping[str, float] = BEAMNG_SBR4_FORWARD_POLARITY,
    convention_id: str = BEAMNG_SBR4_SPIN_CONVENTION_ID,
    require_declared_convention: bool = True,
) -> SignedBudgetInputs:
    """Resolve strict per-wheel inputs without discarding angular-velocity sign.

    BeamNG reports opposite raw signs on the two axle directions. The explicit
    polarity map converts each raw value to a common *forward-spin* coordinate.
    Strict callers must record the convention ID in the measured state; missing
    wheels, unknown names, non-finite values, or a convention mismatch fail
    closed.
    """
    declared = takeoff_state.get("wheel_spin_convention_id")
    if require_declared_convention and declared != convention_id:
        raise BudgetConventionError(
            "signed wheel budget requires wheel_spin_convention_id="
            f"{convention_id!r}, got {declared!r}")
    wheels = takeoff_state.get("wheels0_radps")
    if not isinstance(wheels, Mapping):
        raise BudgetInputMissingError("signed wheel budget requires wheels0_radps mapping")
    actual = {str(name) for name in wheels}
    expected = set(wheel_order)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BudgetInputMissingError(
            f"signed wheel set mismatch: missing={missing}, extra={extra}")
    raw: list[float] = []
    canonical: list[float] = []
    for name in wheel_order:
        if name not in forward_polarity:
            raise BudgetConventionError(f"missing polarity for wheel {name}")
        polarity = float(forward_polarity[name])
        if polarity not in (-1.0, 1.0):
            raise BudgetConventionError(
                f"wheel polarity must be -1 or +1: {name}={polarity}")
        value = float(wheels[name])
        if not math.isfinite(value):
            raise BudgetInputMissingError(f"non-finite wheel speed: {name}={value}")
        raw.append(value)
        canonical.append(polarity * value)
    return SignedBudgetInputs(
        wheel_names=tuple(wheel_order),
        raw_omega_radps=tuple(raw),
        canonical_omega_radps=tuple(canonical),
        source="measured_per_wheel_signed",
        convention_id=convention_id,
    )


def canonical_wheel_intervals(
    inputs: SignedBudgetInputs,
    *,
    uncertainty_radps: float | Mapping[str, float] = 0.0,
) -> dict[str, WheelRateInterval]:
    intervals: dict[str, WheelRateInterval] = {}
    for name, value in zip(inputs.wheel_names, inputs.canonical_omega_radps):
        uncertainty = (
            float(uncertainty_radps[name])
            if isinstance(uncertainty_radps, Mapping)
            else float(uncertainty_radps)
        )
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError(f"wheel uncertainty must be finite and >= 0: {name}")
        intervals[name] = WheelRateInterval(
            lower_radps=value - uncertainty,
            upper_radps=value + uncertainty,
        )
    return intervals


def certified_directional_rate_budgets(
    intervals: Mapping[str, WheelRateInterval],
    *,
    I_w: float | Mapping[str, float],
    omega_max: float | Mapping[str, float],
    omega_min: float | Mapping[str, float],
    J_y: float,
    wheel_order: tuple[str, ...] = BEAMNG_SBR4_WHEEL_ORDER,
    convention_id: str = BEAMNG_SBR4_SPIN_CONVENTION_ID,
) -> CertifiedWheelBudget:
    """Return worst-case signed directional budgets for a wheel-state tube."""
    if not math.isfinite(float(J_y)) or J_y <= 0.0:
        raise ValueError("J_y must be finite and > 0")
    actual = {str(name) for name in intervals}
    expected = set(wheel_order)
    if actual != expected:
        raise BudgetInputMissingError(
            "wheel interval set mismatch: "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")

    def value_for(value: float | Mapping[str, float], name: str) -> float:
        return float(value[name]) if isinstance(value, Mapping) else float(value)

    up: list[float] = []
    down: list[float] = []
    for name in wheel_order:
        interval = intervals[name]
        inertia = value_for(I_w, name)
        upper_limit = value_for(omega_max, name)
        lower_limit = value_for(omega_min, name)
        if not all(math.isfinite(v) for v in (inertia, upper_limit, lower_limit)):
            raise ValueError(f"non-finite wheel authority parameter for {name}")
        if inertia <= 0.0 or lower_limit >= upper_limit:
            raise ValueError(f"invalid wheel authority parameter for {name}")
        if (
            interval.lower_radps < lower_limit - 1e-12
            or interval.upper_radps > upper_limit + 1e-12
        ):
            raise BudgetOutOfDomainError(
                f"{name} interval [{interval.lower_radps}, {interval.upper_radps}] "
                f"outside [{lower_limit}, {upper_limit}]")
        # Worst-case headroom over the complete action-conditioned wheel tube.
        up.append(inertia * (upper_limit - interval.upper_radps) / J_y)
        down.append(inertia * (interval.lower_radps - lower_limit) / J_y)
    return CertifiedWheelBudget(
        B_up=sum(up),
        B_down=sum(down),
        per_wheel_up=tuple(up),
        per_wheel_down=tuple(down),
        wheel_names=tuple(wheel_order),
        convention_id=convention_id,
    )


def resolve_budget_inputs(
    takeoff_state: dict,
    *,
    v_fallback_mps: float | None,
    wheel_r_m: float,
    require_measured: bool = False,
) -> BudgetInputs:
    per_wheel = takeoff_state.get("wheels0_radps")
    if isinstance(per_wheel, dict) and per_wheel:
        vals = _clean_wheels(per_wheel.values())
        if vals:
            return BudgetInputs(vals, "measured_per_wheel")
    if takeoff_state.get("wheel_w0_radps") is not None:
        w = abs(float(takeoff_state["wheel_w0_radps"]))
        return BudgetInputs((w, w, w, w), "measured_abs_mean")
    if require_measured:
        raise BudgetInputMissingError("measured wheel speed required for directional budget")
    if v_fallback_mps is None or wheel_r_m <= 0:
        raise BudgetInputMissingError("A5 fallback requires v0 and positive wheel radius")
    w = max(0.0, float(v_fallback_mps)) / float(wheel_r_m)
    return BudgetInputs((w, w, w, w), "A5_v_over_r")


def budgets_from_takeoff_state(
    takeoff_state: dict,
    *,
    I_w: float,
    omega_max: float,
    omega_min: float,
    J_y: float,
    wheel_r_m: float,
    v_fallback_mps: float | None = None,
    require_measured: bool = False,
) -> tuple[float, float, BudgetInputs]:
    inputs = resolve_budget_inputs(
        takeoff_state,
        v_fallback_mps=v_fallback_mps,
        wheel_r_m=wheel_r_m,
        require_measured=require_measured,
    )
    b_up, b_down = directional_rate_budgets(
        I_w, omega_max, omega_min, list(inputs.omega0_radps), J_y)
    return b_up, b_down, inputs


def wheel_authority_audit(
    values: Iterable[float],
    limits: WheelAuthorityLimits = WheelAuthorityLimits(),
) -> dict:
    """Report three non-interchangeable wheel-speed margins."""
    cleaned = _clean_wheels(values)
    max_abs = max(cleaned) if cleaned else None
    if max_abs is None:
        return {
            "available": False,
            "max_abs_radps": None,
            "above_sustained_envelope": None,
            "above_operational_guard": None,
            "above_drivetrain_hard_limit": None,
        }
    return {
        "available": True,
        "max_abs_radps": max_abs,
        "sustained_effective_envelope_radps": (
            limits.sustained_effective_envelope_radps),
        "operational_spin_guard_radps": limits.operational_spin_guard_radps,
        "operational_guard_enabled": limits.operational_spin_guard_radps > 0.0,
        "drivetrain_hard_limit_radps": limits.drivetrain_hard_limit_radps,
        "above_sustained_envelope": (
            max_abs > limits.sustained_effective_envelope_radps),
        "above_operational_guard": (
            max_abs > limits.operational_spin_guard_radps
            if limits.operational_spin_guard_radps > 0.0 else None),
        "above_drivetrain_hard_limit": (
            max_abs > limits.drivetrain_hard_limit_radps),
        "margin_to_operational_guard_radps": (
            limits.operational_spin_guard_radps - max_abs
            if limits.operational_spin_guard_radps > 0.0 else None),
        "margin_to_drivetrain_hard_limit_radps": (
            limits.drivetrain_hard_limit_radps - max_abs),
    }


__all__ = [
    "SUSTAINED_EFFECTIVE_ENVELOPE_RADPS",
    "OPERATIONAL_SPIN_GUARD_RADPS",
    "DRIVETRAIN_HARD_LIMIT_RADPS",
    "BEAMNG_SBR4_WHEEL_ORDER",
    "BEAMNG_SBR4_FORWARD_POLARITY",
    "BEAMNG_SBR4_SPIN_CONVENTION_ID",
    "BudgetInputMissingError",
    "BudgetConventionError",
    "BudgetOutOfDomainError",
    "BudgetInputs",
    "SignedBudgetInputs",
    "WheelRateInterval",
    "CertifiedWheelBudget",
    "WheelAuthorityLimits",
    "resolve_budget_inputs",
    "resolve_signed_budget_inputs",
    "canonical_wheel_intervals",
    "certified_directional_rate_budgets",
    "budgets_from_takeoff_state",
    "wheel_authority_audit",
]
