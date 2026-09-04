#!/usr/bin/env python3
"""Freeze the 600-run DART-S arXiv confirmatory BeamNG design."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from control.darts.reachability_screen import (  # noqa: E402
    assess_operational_candidate,
    select_operational_action,
)
from control.darts.reachability import directional_rate_budgets  # noqa: E402
from control.darts.susp_precondition import (  # noqa: E402
    SelectorUncertainty,
    recommend_arm,
)


OUT_ROOT = Path("data/derived/confirmatory")
GATE_TABLE = Path("calibration/gate_table.json")
OUTCOME_TABLE = Path("calibration/system_outcome_table.json")
FROZEN_SOURCES = (
    "control/darts/reachability_screen.py",
    "control/darts/reachability.py",
    "control/darts/susp_precondition.py",
    "control/darts/cluster_stats.py",
    "control/darts/wheel_budget.py",
    "control/darts/landing_safety.py",
    "scripts/darts_bench.py",
    "scripts/confirmatory_design.py",
    "scripts/confirmatory_analysis.py",
    "scripts/experiments/queue_confirmatory.sh",
    "calibration/gate_table.json",
    "calibration/system_outcome_table.json",
)


BLOCKS = {
    "boundary_a35_v15": {
        "angle_deg": 35.0,
        "speed_mps": 15.0,
        "arms": ["dart_guarded", "arxiv_joint"],
        "rolls": 3,
        "kind": "boundary",
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.0,
        "expected_joint_arm": "nose_up",
    },
    "boundary_a40_v13": {
        "angle_deg": 40.0,
        "speed_mps": 13.0,
        "arms": ["dart_guarded", "arxiv_joint"],
        "rolls": 3,
        "kind": "boundary",
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.0,
        "expected_joint_arm": "nose_up",
    },
    "factorial_a35_v16": {
        "angle_deg": 35.0,
        "speed_mps": 16.0,
        "arms": [
            "neutral",
            "nose_up",
            "dart_guarded",
            "arxiv_outcome_only",
            "arxiv_budget_only",
            "arxiv_joint",
        ],
        "rolls": 3,
        "kind": "factorial",
        "target_offset_mps": -0.6,
        "angle_uncertainty_deg": 0.0,
        "expected_joint_arm": "nose_up",
    },
    "timing_a35_v11p5": {
        "angle_deg": 35.0,
        "speed_mps": 11.5,
        "arms": ["neutral", "nose_up", "phase_015", "phase_035"],
        "rolls": 3,
        "kind": "timing",
        "phase_sweep_leads": [0.15, 0.35],
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.0,
    },
    "washboard_a35_v13": {
        "angle_deg": 35.0,
        "speed_mps": 13.0,
        "arms": ["dart_guarded", "arxiv_joint", "dart_s_forced"],
        "rolls": 3,
        "kind": "washboard",
        "runup_bumps": 6,
        "bump_height_m": 0.08,
        "bump_spacing_m": 1.6,
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.0,
        "expected_joint_arm": "neutral",
    },
    "heldout_p1_a37_v15p25": {
        "angle_deg": 37.0,
        "speed_mps": 15.25,
        "arms": ["dart_guarded", "arxiv_joint"],
        "rolls": 3,
        "kind": "capture_positive",
        "expected_table_arm": "nose_up",
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.5,
        "expected_joint_arm": "nose_up",
    },
    "heldout_p2_a39_v13": {
        "angle_deg": 39.0,
        "speed_mps": 13.0,
        "arms": ["dart_guarded", "arxiv_joint"],
        "rolls": 3,
        "kind": "capture_positive",
        "expected_table_arm": "nose_up",
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.5,
        "expected_joint_arm": "nose_up",
    },
    "capture_negative_a36_v13p5": {
        "angle_deg": 36.0,
        "speed_mps": 13.5,
        "arms": ["arxiv_joint", "dart_s_forced"],
        "rolls": 3,
        "kind": "capture_negative",
        "expected_table_arm": "neutral",
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.5,
        "expected_joint_arm": "neutral",
    },
    "capture_boundary_a40_v12p65": {
        "angle_deg": 40.0,
        "speed_mps": 12.65,
        "arms": ["arxiv_joint", "dart_s_forced"],
        "rolls": 3,
        "kind": "capture_boundary",
        "expected_table_arm": "neutral",
        "target_offset_mps": 0.0,
        "angle_uncertainty_deg": 0.5,
        "expected_joint_arm": "neutral",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
    ).strip()


def design_fingerprint(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("design_sha256", None)
    encoded = json.dumps(
        clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_design(path: Path, *, check_tree: bool = False) -> list[str]:
    """Verify a frozen design document.

    The fingerprint and preflight checks are portable. The commit and the
    source-hash ledger can only match inside the working tree the design was
    frozen in, so ``check_tree`` is opt-in; the ledger shipped with this
    package records the paths of that tree.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    deviations = []
    if payload.get("design_sha256") != design_fingerprint(payload):
        deviations.append("design fingerprint mismatch")
    if payload.get("deviations") is not None:
        deviations.append("design contains preflight deviations")
    if not check_tree:
        return deviations
    if payload.get("expected_commit") != head_commit():
        deviations.append("HEAD differs from expected_commit")
    for source, expected in (payload.get("source_hashes") or {}).items():
        source_path = REPO / source
        if not source_path.exists():
            deviations.append(f"frozen source missing: {source}")
        elif sha256(source_path) != expected:
            deviations.append(f"frozen source hash mismatch: {source}")
    if len(payload.get("source_hashes") or {}) != len(FROZEN_SOURCES):
        deviations.append("frozen source ledger incomplete")
    return deviations


def candidate_preflight(block: dict) -> dict:
    angle = float(block["angle_deg"])
    speed = float(block["speed_mps"]) + float(
        block.get("target_offset_mps", 0.0))
    result = recommend_arm(
        angle,
        speed,
        require_calibration_map=True,
        calibration_support_max_distance=1.5,
        system_outcome_table_path=OUTCOME_TABLE,
        require_system_safety_evidence=True,
        system_safety_max_distance=0.75,
        geometry_family="tabletop_fillet",
        uncertainty=SelectorUncertainty(
            angle_half_width_deg=float(
                block.get("angle_uncertainty_deg", 0.0)),
            speed_half_width_mps=0.2,
            require_safety_v2_evidence=True,
        ),
    )
    entry = result.get("entry") or {}
    calmap = result.get("calibration_map") or {}
    if not entry or not calmap.get("in_domain"):
        return {
            "available": False,
            "table_arm": result.get("arm"),
            "reason": result.get("reason"),
        }
    theta0 = math.radians(float(entry["theta0_neutral"]))
    omega0 = math.radians(float(entry["omega_y0_neutral"]))
    flight_time = float(entry["T_flight_neutral"])
    wheel_value = entry.get("wheel_w0_neutral")
    wheel_source = "gate_table_measured_wheel"
    if wheel_value is None:
        wheel_value = float(entry.get("v0_neutral") or speed) / 0.36
        wheel_source = "A5_v_over_r"
    wheel_natural = float(wheel_value)
    dtheta = math.radians(float(calmap["dtheta0_deg"]))
    domega = math.radians(float(calmap["domega_y0_dps"]))
    dwheel = float(calmap["dwheel_radps"])

    def assessment(theta: float, omega: float, wheel: float):
        B_up, B_down = directional_rate_budgets(
            1.2, 125.7, -125.7, max(0.0, wheel), 2043.0)
        return assess_operational_candidate(
            theta0=theta,
            omega0=omega,
            theta_target=0.0,
            flight_time_s=flight_time,
            pitch_accel_radps2=math.radians(135.0),
            B_up=B_up,
            B_down=B_down,
            theta_half_width_rad=math.radians(35.0),
            omega_half_width_radps=math.radians(25.0),
        )

    natural = assessment(theta0, omega0, wheel_natural)
    preset = assessment(
        theta0 + dtheta,
        omega0 + domega,
        wheel_natural + dwheel,
    )
    counterfactuals = {
        strategy: select_operational_action(
            table_arm=str(result["arm"]),
            strategy=strategy,
            natural=natural,
            preset=preset,
        ).as_dict()
        for strategy in (
            "outcome_only",
            "state_only",
            "budget_only",
            "joint",
        )
    }
    return {
        "available": True,
        "table_arm": result["arm"],
        "reason": result["reason"],
        "calmap_support_distance": calmap["support_distance"],
        "outcome_query_distance": (
            (result.get("system_outcome_evidence") or {}).get(
                "query_distance")),
        "uncertainty_audit": result.get("uncertainty_audit"),
        "natural": natural.as_dict(),
        "preset": preset.as_dict(),
        "strategy_counterfactuals": counterfactuals,
        "mean_wheel_input": {
            "neutral_radps": wheel_natural,
            "preset_radps": wheel_natural + dwheel,
            "mode": "nominal_mean_wheel",
            "source": wheel_source,
        },
    }


def build_sessions() -> list[dict]:
    block_names = list(BLOCKS)
    sessions = []
    for session_index in range(8):
        rotated = (
            block_names[session_index:]
            + block_names[:session_index]
        )
        for order_index, block_name in enumerate(rotated):
            block = BLOCKS[block_name]
            session_id = (
                f"axv-{block_name.replace('_', '-')}-"
                f"s{session_index + 1:02d}"
            )
            sessions.append({
                "session_id": session_id,
                "session_index": session_index + 1,
                "within_round_order": order_index + 1,
                "block_id": block_name,
                "angle_deg": block["angle_deg"],
                "speed_mps": block["speed_mps"],
                "arms": block["arms"],
                "rolls": block["rolls"],
                "expected_runs": len(block["arms"]) * block["rolls"],
            })
    return sessions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_ROOT / "design.json",
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-tree",
        action="store_true",
        help="also check the commit and source-hash ledger, which only match "
             "in the working tree the design was frozen in",
    )
    args = parser.parse_args()
    if args.verify or args.verify_tree:
        deviations = verify_frozen_design(args.out, check_tree=args.verify_tree)
        print(f"[darts-design] verify deviations={deviations or None}")
        return 0 if not deviations else 2

    preflight = {
        name: candidate_preflight(block)
        for name, block in BLOCKS.items()
        if "arxiv_joint" in block["arms"]
    }
    expected = {
        name: block.get("expected_table_arm")
        for name, block in BLOCKS.items()
        if block.get("expected_table_arm") is not None
    }
    deviations = [
        f"{name}: table_arm={preflight[name].get('table_arm')!r}, "
        f"expected={arm!r}"
        for name, arm in expected.items()
        if preflight[name].get("table_arm") != arm
    ]
    deviations.extend(
        f"{name}: joint_arm="
        f"{preflight[name].get('strategy_counterfactuals', {}).get('joint', {}).get('selected_arm')!r}, "
        f"expected={block['expected_joint_arm']!r}"
        for name, block in BLOCKS.items()
        if block.get("expected_joint_arm") is not None
        and preflight[name].get("strategy_counterfactuals", {}).get(
            "joint", {}).get("selected_arm") != block["expected_joint_arm"]
    )
    sessions = build_sessions()
    total_runs = sum(session["expected_runs"] for session in sessions)
    if total_runs != 600:
        deviations.append(f"total_runs={total_runs}, expected=600")

    missing_sources = [
        source for source in FROZEN_SOURCES
        if not (REPO / source).exists()
    ]
    if missing_sources:
        deviations.append(f"missing frozen sources: {missing_sources}")
    source_hashes = {
        source: sha256(REPO / source)
        for source in FROZEN_SOURCES
        if (REPO / source).exists()
    }
    payload = {
        "schema_version": "darts-arxiv-confirmatory-design/1",
        "canonical_id": "DARTS_ARXIV_CONFIRMATORY_20260813",
        "expected_commit": head_commit(),
        "source_hashes": source_hashes,
        "threshold_semantics_radps": {
            "certificate_envelope": 125.7,
            "operational_guard": 200.0,
            "drivetrain_hard_limit": 221.2,
        },
        "selector_policy": "legacy",
        "operational_selector_schema": (
            "darts-operational-reachability-selector/1"),
        "sessions": sessions,
        "blocks": BLOCKS,
        "preflight": preflight,
        "expected_runs": total_runs,
        "attempt_limit": 650,
        "walltime_limit_hours": 20,
        "deviations": deviations or None,
    }
    payload["design_sha256"] = design_fingerprint(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[darts-design] runs={total_runs} sessions={len(sessions)} "
        f"deviations={deviations or None}")
    return 0 if not deviations else 1


if __name__ == "__main__":
    raise SystemExit(main())
