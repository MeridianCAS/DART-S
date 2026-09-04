"""System-level safety evidence for deploying suspension preconditioning."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SystemOutcomeDecision:
    available: bool
    deploy_preset: bool
    action: str
    reason: str
    query_distance: float | None
    evidence: dict | None


def query_system_outcome_table(
    angle_deg: float,
    speed_mps: float,
    *,
    table_path: Path | str,
    geometry_family: str,
    action_kind: str | None = None,
    schedule_id: str | None = None,
    max_query_distance: float | None = None,
) -> SystemOutcomeDecision:
    """Use only independent, system-level safety calibration evidence."""
    path = Path(table_path)
    if not path.exists():
        return SystemOutcomeDecision(
            False, False, "abstain", "system_outcome_table_missing", None, None)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        entries = [
            entry for entry in doc["entries"]
            if str(entry.get("geometry_family")) == str(geometry_family)
            and (
                action_kind is None
                or str(entry.get("preset_action")) == str(action_kind)
            )
            and (
                schedule_id is None
                or str(entry.get("schedule_id", "static_full"))
                == str(schedule_id)
            )
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return SystemOutcomeDecision(
            False, False, "abstain", "system_outcome_table_unreadable", None, None)
    if doc.get("endpoint") != "post_touchdown_safety_v2":
        return SystemOutcomeDecision(
            False, False, "abstain", "system_outcome_endpoint_mismatch", None, None)
    if doc.get("data_role") != "calibration_for_future_independent_evaluation":
        return SystemOutcomeDecision(
            False, False, "abstain", "system_outcome_role_not_calibration", None, None)
    if not entries:
        reason = (
            f"system_outcome_condition_unsupported("
            f"{geometry_family},action={action_kind},schedule={schedule_id})"
            if action_kind is not None or schedule_id is not None
            else f"system_outcome_geometry_unsupported({geometry_family})"
        )
        return SystemOutcomeDecision(
            False, False, "abstain",
            reason,
            None, None)

    def distance(entry):
        return math.hypot(
            (float(entry["angle_deg"]) - angle_deg) / 5.0,
            (float(entry["speed_mps"]) - speed_mps) / 1.0,
        )

    best = min(entries, key=distance)
    d_best = distance(best)
    radius = float(
        doc.get("max_query_distance", 0.25)
        if max_query_distance is None else max_query_distance)
    if d_best > radius:
        return SystemOutcomeDecision(
            True, False, "abstain",
            f"system_outcome_query_too_far({d_best:.2f}>{radius:.2f})",
            d_best, best)
    eligible = bool(best.get("deploy_eligible"))
    return SystemOutcomeDecision(
        True,
        eligible,
        str(best.get("preset_action", "static_preset")) if eligible else "dart_only",
        str(best.get("decision_reason", "insufficient_system_safety_evidence")),
        d_best,
        best,
    )


__all__ = ["SystemOutcomeDecision", "query_system_outcome_table"]
