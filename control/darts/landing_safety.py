"""Versioned landing-safety criteria shared by DART-S runners and analyzers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LandingSafetyCriterion:
    pitch_limit_deg: float = 35.0
    roll_limit_deg: float = 30.0
    tumble_pitch_deg: float = 85.0
    tumble_roll_deg: float = 80.0
    inclusive: bool = True
    require_no_tumble: bool = True
    require_post_touchdown_stable: bool = True

    def angle_within(self, value: Optional[float], limit: float) -> bool:
        if value is None:
            return False
        return abs(float(value)) <= limit if self.inclusive else abs(float(value)) < limit


PAPER_SAFETY_V2 = LandingSafetyCriterion()


def touchdown_attitude_ok(run: dict, criterion: LandingSafetyCriterion = PAPER_SAFETY_V2) -> bool:
    return (
        bool(run.get("took_off"))
        and criterion.angle_within(run.get("land_pitch_deg"), criterion.pitch_limit_deg)
        and criterion.angle_within(run.get("land_roll_deg"), criterion.roll_limit_deg)
    )


def paper_safe_v2(
    run: dict,
    criterion: LandingSafetyCriterion = PAPER_SAFETY_V2,
    *,
    missing_is_false: bool = True,
) -> Optional[bool]:
    """Full paper criterion: touchdown attitude + known no-tumble + stable window.

    Legacy reports lack ``tumbled`` and ``post_touchdown``.  They return
    ``None`` unless ``missing_is_false`` requests fail-closed ``False``.
    """
    if not touchdown_attitude_ok(run, criterion):
        return False
    if criterion.require_no_tumble:
        tumbled = run.get("tumbled")
        if tumbled is None:
            return False if missing_is_false else None
        if bool(tumbled):
            return False
    if criterion.require_post_touchdown_stable:
        post = run.get("post_touchdown") or {}
        stable = post.get("stable")
        if stable is None:
            return False if missing_is_false else None
        if not bool(stable):
            return False
    return True


def safety_schema(criterion: LandingSafetyCriterion = PAPER_SAFETY_V2) -> dict:
    return {
        "version": 2,
        "pitch_limit_deg": criterion.pitch_limit_deg,
        "roll_limit_deg": criterion.roll_limit_deg,
        "tumble_pitch_deg": criterion.tumble_pitch_deg,
        "tumble_roll_deg": criterion.tumble_roll_deg,
        "inclusive": criterion.inclusive,
        "require_no_tumble": criterion.require_no_tumble,
        "require_post_touchdown_stable": criterion.require_post_touchdown_stable,
    }


__all__ = [
    "LandingSafetyCriterion",
    "PAPER_SAFETY_V2",
    "touchdown_attitude_ok",
    "paper_safe_v2",
    "safety_schema",
]
