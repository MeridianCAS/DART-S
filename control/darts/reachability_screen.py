"""Operational interval-reachability screen for the DART-S arXiv stack.

This module separates three objects that were historically mixed in the
BeamNG runner:

* outcome qualification selects a locally supported suspension action;
* interval reachability screens a natural-out / preset-in transition;
* exact-pair reachability remains a diagnostic audit.

All functions are pure and simulator-independent.  The legacy ``require_flip``
branch and the confirmatory policy ablations consume this shared logic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from control.darts.reachability import PitchReachability, pitch_reachable


VALID_OPERATIONAL_STRATEGIES = frozenset({
    "outcome_only",
    "state_only",
    "budget_only",
    "joint",
})
VALID_TABLE_ARMS = frozenset({
    "neutral",
    "nose_up",
    "nose_down",
    "avoid_preset",
})


@dataclass(frozen=True)
class OperationalCandidateAssessment:
    """Reachability factors for one natural or preset candidate."""

    state_ok: bool
    budget_ok: bool
    joint_ok: bool
    velocity_nullable: bool
    displacement_ok: bool
    margin: float

    def __post_init__(self) -> None:
        if self.joint_ok and not (
            self.state_ok and self.budget_ok and self.velocity_nullable
        ):
            raise ValueError("joint_ok requires state, budget, and time feasibility")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OperationalReachabilityDecision:
    """One support-qualified operational selector decision."""

    schema_version: str
    strategy: str
    table_arm: str
    selected_arm: str
    reason: str
    changed_from_outcome_only: bool
    natural: OperationalCandidateAssessment
    preset: OperationalCandidateAssessment

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["natural"] = self.natural.as_dict()
        payload["preset"] = self.preset.as_dict()
        return payload


def assess_operational_candidate(
    *,
    theta0: float,
    omega0: float,
    theta_target: float,
    flight_time_s: float,
    pitch_accel_radps2: float,
    B_up: float,
    B_down: float,
    theta_half_width_rad: float,
    omega_half_width_radps: float,
) -> OperationalCandidateAssessment:
    """Evaluate state-only, budget-only, and joint interval feasibility.

    ``state_ok`` evaluates the same terminal interval with infinite wheel
    budgets. ``budget_ok`` retains the directional rate budget and finite-time
    rate condition while intentionally ignoring terminal-angle displacement.
    ``joint_ok`` is the full interval-reachability result.
    """
    kwargs = {
        "theta0": theta0,
        "omega_y0": omega0,
        "theta_L": theta_target,
        "T": flight_time_s,
        "a": pitch_accel_radps2,
        "theta_bar": theta_half_width_rad,
        "omega_bar": omega_half_width_radps,
    }
    state_result = pitch_reachable(**kwargs)
    joint_result = pitch_reachable(
        **kwargs,
        B_up=B_up,
        B_down=B_down,
    )
    return OperationalCandidateAssessment(
        state_ok=bool(state_result.reachable),
        budget_ok=bool(
            joint_result.budget_ok and joint_result.velocity_nullable),
        joint_ok=bool(joint_result.reachable),
        velocity_nullable=bool(joint_result.velocity_nullable),
        displacement_ok=bool(joint_result.displacement_ok),
        margin=float(joint_result.margin),
    )


def _strategy_ok(
    assessment: OperationalCandidateAssessment,
    strategy: str,
) -> bool:
    if strategy == "state_only":
        return assessment.state_ok
    if strategy == "budget_only":
        return assessment.budget_ok
    if strategy == "joint":
        return assessment.joint_ok
    if strategy == "outcome_only":
        return True
    raise ValueError(f"unsupported operational strategy {strategy!r}")


def select_operational_action(
    *,
    table_arm: str,
    strategy: str,
    natural: OperationalCandidateAssessment,
    preset: OperationalCandidateAssessment,
) -> OperationalReachabilityDecision:
    """Combine a local outcome-table action with a reachability strategy."""
    table_arm = str(table_arm)
    strategy = str(strategy)
    if table_arm not in VALID_TABLE_ARMS:
        raise ValueError(f"unsupported outcome-table arm {table_arm!r}")
    if strategy not in VALID_OPERATIONAL_STRATEGIES:
        raise ValueError(f"unsupported operational strategy {strategy!r}")

    selected = table_arm
    if table_arm != "nose_up":
        reason = "outcome_table_did_not_propose_preset"
    elif strategy == "outcome_only":
        reason = "outcome_qualified_preset"
    else:
        natural_ok = _strategy_ok(natural, strategy)
        preset_ok = _strategy_ok(preset, strategy)
        if not natural_ok and preset_ok:
            selected = "nose_up"
            reason = f"{strategy}_natural_out_preset_in"
        elif natural_ok:
            selected = "neutral"
            reason = f"{strategy}_natural_already_in"
        else:
            selected = "avoid_preset"
            reason = f"{strategy}_preset_not_in"

    return OperationalReachabilityDecision(
        schema_version="darts-operational-reachability-selector/1",
        strategy=strategy,
        table_arm=table_arm,
        selected_arm=selected,
        reason=reason,
        changed_from_outcome_only=selected != table_arm,
        natural=natural,
        preset=preset,
    )


def assessment_from_pitch_result(
    result: PitchReachability,
    *,
    state_ok: bool,
) -> OperationalCandidateAssessment:
    """Adapt an existing joint result when a separate state audit exists."""
    return OperationalCandidateAssessment(
        state_ok=bool(state_ok),
        budget_ok=bool(result.budget_ok and result.velocity_nullable),
        joint_ok=bool(result.reachable),
        velocity_nullable=bool(result.velocity_nullable),
        displacement_ok=bool(result.displacement_ok),
        margin=float(result.margin),
    )


__all__ = [
    "VALID_OPERATIONAL_STRATEGIES",
    "OperationalCandidateAssessment",
    "OperationalReachabilityDecision",
    "assess_operational_candidate",
    "assessment_from_pitch_result",
    "select_operational_action",
]
