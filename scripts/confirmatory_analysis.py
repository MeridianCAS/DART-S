#!/usr/bin/env python3
"""Analyze the all-new 600-run DART-S arXiv confirmatory matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from control.darts.cluster_stats import (  # noqa: E402
    MultiArmSessionRecord,
    audit_completeness,
    audit_multi_arm_design,
    audit_session_design,
    audit_speed_match,
    holm_adjust,
    intraclass_correlation,
    leave_one_session_out,
    paired_contrast,
    session_level_exact_signflip,
    two_level_cluster_bootstrap,
)
from control.darts.landing_safety import paper_safe_v2  # noqa: E402
from control.darts.provenance import (  # noqa: E402
    aggregate_provenance,
    assert_time_integrity,
)


VIZ = REPO / "data/cohorts"
EXPECTED_STRATEGIES = (
    "outcome_only",
    "state_only",
    "budget_only",
    "joint",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design(path: Path) -> dict:
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("schema_version") != "darts-arxiv-confirmatory-design/1":
        raise ValueError("unexpected confirmatory design schema")
    if design.get("expected_runs") != 600 or design.get("deviations") is not None:
        raise ValueError("confirmatory design is not frozen cleanly")
    return design


def raw_path_for_session(session: dict) -> Path:
    tag = session["session_id"].replace("-", "_")
    return VIZ / f"darts_bench_{tag}.json"


def roll_index(run: dict, fallback: int) -> int:
    protocol = run.get("protocol") or {}
    return int(protocol.get("roll_index", run.get("roll_index", fallback)))


def run_speed(run: dict) -> float:
    takeoff = run.get("takeoff_state") or {}
    value = takeoff.get("v0_mps")
    if value is None:
        value = run.get("control_target_v_mps", run.get("v_entry"))
    return float(value)


def hard_limit_exceeded(run: dict) -> bool:
    return bool(
        (run.get("wheel_trajectory_audit") or {}).get(
            "hard_limit_exceeded"))


def load_session(
    *,
    design: dict,
    session: dict,
    path: Path,
) -> tuple[dict, list[MultiArmSessionRecord], list[str], list[dict]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert_time_integrity(doc, path=str(path), require_time_metrics=True)
    block = design["blocks"][session["block_id"]]
    params = doc.get("params") or {}
    deviations = []
    expected_params = {
        "protocol_tag": "darts_arxiv_confirmatory",
        "session_id": session["session_id"],
        "selector_policy": "legacy",
        "rolls": session["rolls"],
        "arms": session["arms"],
        "wheel_storage_hard_limit_radps": 221.2,
        "approach_wheel_spin_guard_radps": 200.0,
        "gate_wheel_omega_max_radps": 125.7,
        "gate_wheel_omega_min_radps": -125.7,
    }
    for key, expected in expected_params.items():
        if params.get(key) != expected:
            deviations.append(
                f"{session['session_id']}: {key}={params.get(key)!r}, "
                f"expected={expected!r}")
    records = []
    run_rows = []
    for speed_key, arm_rows in (doc.get("data") or {}).items():
        for arm in session["arms"]:
            runs = arm_rows.get(arm, [])
            if len(runs) != int(session["rolls"]):
                deviations.append(
                    f"{session['session_id']}/{arm}: n={len(runs)}")
            for fallback, run in enumerate(runs):
                protocol = run.get("protocol") or {}
                roll = roll_index(run, fallback)
                if protocol.get("session_id") != session["session_id"]:
                    deviations.append(
                        f"{session['session_id']}/{arm}/r{roll}: "
                        "protocol session mismatch")
                if hard_limit_exceeded(run):
                    deviations.append(
                        f"{session['session_id']}/{arm}/r{roll}: hard limit")
                if bool(run.get("took_off")):
                    wheels = (run.get("takeoff_state") or {}).get(
                        "wheels0_radps")
                    if (
                        not isinstance(wheels, dict)
                        or set(wheels) != {"FL", "FR", "RL", "RR"}
                    ):
                        deviations.append(
                            f"{session['session_id']}/{arm}/r{roll}: "
                            "signed per-wheel takeoff trace missing")
                window_complete = bool(
                    (run.get("post_touchdown") or {}).get(
                        "complete_window"))
                took_off = bool(run.get("took_off"))
                touched_down = run.get("land_pitch_deg") is not None
                records.append(MultiArmSessionRecord(
                    physical_session_id=session["session_id"],
                    roll_index=roll,
                    arm=arm,
                    value=float(bool(paper_safe_v2(run))),
                    speed_mps=run_speed(run),
                    took_off=took_off,
                    touched_down=touched_down,
                    window_complete=window_complete,
                ))
                gate = run.get("gate") or {}
                ftheta = gate.get("f_theta") or {}
                counterfactuals = ftheta.get(
                    "strategy_counterfactuals") or {}
                if arm.startswith("arxiv_"):
                    strategy = arm.removeprefix("arxiv_")
                    operational = ftheta.get("operational_selector") or {}
                    if operational.get("strategy") != strategy:
                        deviations.append(
                            f"{session['session_id']}/{arm}/r{roll}: "
                            "operational strategy mismatch")
                    if set(counterfactuals) != set(EXPECTED_STRATEGIES):
                        deviations.append(
                            f"{session['session_id']}/{arm}/r{roll}: "
                            "counterfactual strategy ledger incomplete")
                wheels = (run.get("takeoff_state") or {}).get(
                    "wheels0_radps") or {}
                realized_wheels = (
                    [abs(float(value)) for value in wheels.values()]
                    if isinstance(wheels, dict) else [])
                predicted = (
                    ftheta.get("preset_nose_up") or {}).get(
                    "wheel_w0_radps")
                run_rows.append({
                    "session_id": session["session_id"],
                    "block_id": session["block_id"],
                    "kind": block["kind"],
                    "speed_key": speed_key,
                    "roll_index": roll,
                    "arm": arm,
                    "safe": bool(paper_safe_v2(run)),
                    "took_off": took_off,
                    "touched_down": touched_down,
                    "window_complete": window_complete,
                    "hard_limit_exceeded": hard_limit_exceeded(run),
                    "gate_decision": gate.get("decision"),
                    "strategy_counterfactuals": counterfactuals,
                    "predicted_mean_wheel_radps": predicted,
                    "realized_mean_abs_wheel_radps": (
                        statistics.fmean(realized_wheels)
                        if realized_wheels else None),
                    "realized_max_abs_wheel_radps": (
                        max(realized_wheels) if realized_wheels else None),
                    "realized_selected_reachable": (
                        ftheta.get("realized_selected_arm") or {}).get(
                        "reachable"),
                })
    return doc, records, deviations + branch_deviations(
        session=session, block=block, rows=run_rows), run_rows


def branch_deviations(*, session: dict, block: dict, rows: list[dict]) -> list[str]:
    expected = block.get("expected_joint_arm")
    if expected is None:
        return []
    selected = [
        row["gate_decision"]
        for row in rows if row["arm"] == "arxiv_joint"
    ]
    return [
        f"{session['session_id']}/arxiv_joint: "
        f"branch={sorted(set(selected))}, expected={expected}"
    ] if any(value != expected for value in selected) else []


def check_one_session(
    *,
    design: dict,
    session_id: str,
    raw: Path,
) -> dict:
    matches = [
        session for session in design["sessions"]
        if session["session_id"] == session_id
    ]
    if len(matches) != 1:
        raise ValueError(f"session not uniquely frozen: {session_id}")
    doc, records, deviations, rows = load_session(
        design=design, session=matches[0], path=raw)
    audit = audit_multi_arm_design(
        records,
        expected_arms=matches[0]["arms"],
        expected_sessions=1,
        expected_rolls_per_session=matches[0]["rolls"],
    )
    if not audit.passed:
        deviations.append(f"{session_id}: multi-arm design mismatch")
    return {
        "session_id": session_id,
        "passed": not deviations,
        "deviations": deviations or None,
        "rows": len(rows),
        "source_sha256": sha256(raw),
        "time_integrity_ok": (
            (doc.get("_provenance") or {}).get(
                "time_integrity_ok") is True),
    }


def paired_analysis(
    records: list[MultiArmSessionRecord],
    *,
    baseline: str,
    candidate: str,
    expected_sessions: int = 8,
    expected_rolls: int = 3,
    speed_mode: str = "standard",
) -> dict:
    pairs = paired_contrast(
        records, baseline_arm=baseline, candidate_arm=candidate)
    design = audit_session_design(
        pairs,
        expected_sessions=expected_sessions,
        expected_rolls_per_session=expected_rolls,
    )
    bootstrap = two_level_cluster_bootstrap(
        pairs, n_bootstrap=100000, seed=20260813)
    signflip = session_level_exact_signflip(pairs)
    speed = audit_speed_match(pairs, mode=speed_mode)
    completeness = audit_completeness(pairs)
    icc = intraclass_correlation(pairs)
    loso = leave_one_session_out(pairs)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "design": asdict(design),
        "cluster_bootstrap": asdict(bootstrap),
        "session_signflip": asdict(signflip),
        "speed_match": asdict(speed),
        "completeness": asdict(completeness),
        "icc": asdict(icc),
        "leave_one_session_out": asdict(loso),
        "passed_unadjusted": bool(
            design.passed
            and bootstrap.ci_lo > 0.0
            and signflip.p_two_sided < 0.05
            and not loso.direction_flip
            and speed.passed
            and completeness.passed
        ),
    }


def interaction_records(
    records: list[MultiArmSessionRecord],
    *,
    arms: tuple[str, str, str, str],
) -> list:
    neutral, preset, dart, system = arms
    indexed = {
        (record.physical_session_id, record.roll_index, record.arm):
        record
        for record in records
    }
    output = []
    from control.darts.cluster_stats import PairedSessionRecord
    for session_id, roll in sorted({
        (record.physical_session_id, record.roll_index)
        for record in records
    }):
        cells = [
            indexed.get((session_id, roll, arm))
            for arm in arms
        ]
        if any(cell is None for cell in cells):
            continue
        contrast = (
            cells[3].value - cells[2].value
            - cells[1].value + cells[0].value)
        output.append(PairedSessionRecord(
            physical_session_id=session_id,
            roll_index=roll,
            baseline_value=0.0,
            candidate_value=contrast,
            baseline_speed_mps=cells[2].speed_mps,
            candidate_speed_mps=cells[3].speed_mps,
            baseline_took_off=cells[2].took_off,
            candidate_took_off=cells[3].took_off,
            baseline_touched_down=cells[2].touched_down,
            candidate_touched_down=cells[3].touched_down,
            baseline_window_complete=cells[2].window_complete,
            candidate_window_complete=cells[3].window_complete,
        ))
    return output


def interaction_analysis(records: list[MultiArmSessionRecord]) -> dict:
    pairs = interaction_records(
        records,
        arms=("neutral", "nose_up", "dart_guarded", "arxiv_joint"),
    )
    bootstrap = two_level_cluster_bootstrap(
        pairs, n_bootstrap=100000, seed=20260814)
    signflip = session_level_exact_signflip(pairs)
    loso = leave_one_session_out(pairs)
    return {
        "arms": ["neutral", "nose_up", "dart_guarded", "arxiv_joint"],
        "cluster_bootstrap": asdict(bootstrap),
        "session_signflip": asdict(signflip),
        "leave_one_session_out": asdict(loso),
        "passed_unadjusted": bool(
            bootstrap.ci_lo > 0.0
            and signflip.p_two_sided < 0.05
            and not loso.direction_flip),
    }


def mean_wheel_audit(rows: list[dict]) -> dict:
    errors = [
        row["realized_mean_abs_wheel_radps"]
        - row["predicted_mean_wheel_radps"]
        for row in rows
        if row["predicted_mean_wheel_radps"] is not None
        and row["realized_mean_abs_wheel_radps"] is not None
        and row["arm"].startswith("arxiv_")
    ]
    max_wheels = [
        row["realized_max_abs_wheel_radps"]
        for row in rows
        if row["realized_max_abs_wheel_radps"] is not None
        and row["arm"].startswith("arxiv_")
    ]
    return {
        "n": len(errors),
        "mean_error_radps": statistics.fmean(errors) if errors else None,
        "median_abs_error_radps": (
            statistics.median(abs(value) for value in errors)
            if errors else None),
        "max_abs_error_radps": (
            max(abs(value) for value in errors) if errors else None),
        "realized_per_wheel_above_certificate_envelope": sum(
            value > 125.7 for value in max_wheels),
        "realized_per_wheel_above_operational_guard": sum(
            value > 200.0 for value in max_wheels),
        "realized_per_wheel_above_hard_limit": sum(
            value > 221.2 for value in max_wheels),
    }


def counterfactual_ledger(rows: list[dict]) -> dict:
    ledger = {
        strategy: {"nose_up": 0, "neutral": 0, "avoid_preset": 0}
        for strategy in EXPECTED_STRATEGIES
    }
    for row in rows:
        if row["arm"] != "arxiv_joint":
            continue
        for strategy, payload in row["strategy_counterfactuals"].items():
            arm = payload["selected_arm"]
            ledger[strategy].setdefault(arm, 0)
            ledger[strategy][arm] += 1
    return ledger


def capture_branch_metrics(rows: list[dict]) -> dict:
    positive_blocks = {
        "heldout_p1_a37_v15p25",
        "heldout_p2_a39_v13",
    }
    negative_block = "capture_negative_a36_v13p5"
    boundary_block = "capture_boundary_a40_v12p65"
    joint_rows = [
        row for row in rows if row["arm"] == "arxiv_joint"]
    positive = [
        row for row in joint_rows if row["block_id"] in positive_blocks]
    negative = [
        row for row in joint_rows if row["block_id"] == negative_block]
    boundary = [
        row for row in joint_rows if row["block_id"] == boundary_block]
    return {
        "positive": {
            "n": len(positive),
            "selected_preset": sum(
                row["gate_decision"] == "nose_up" for row in positive),
            "realized_selected_reachable": sum(
                row["realized_selected_reachable"] is True
                for row in positive),
        },
        "negative": {
            "n": len(negative),
            "false_preset": sum(
                row["gate_decision"] == "nose_up" for row in negative),
        },
        "boundary": {
            "n": len(boundary),
            "abstained": sum(
                row["gate_decision"] != "nose_up" for row in boundary),
        },
    }


def full_analysis(design: dict, output_dir: Path) -> tuple[dict, int]:
    all_records: dict[str, list[MultiArmSessionRecord]] = {
        block_id: [] for block_id in design["blocks"]
    }
    all_rows = []
    files = []
    deviations = []
    for session in design["sessions"]:
        path = raw_path_for_session(session)
        if not path.exists():
            deviations.append(f"missing raw: {path.name}")
            continue
        doc, records, session_deviations, rows = load_session(
            design=design, session=session, path=path)
        files.append(path)
        all_records[session["block_id"]].extend(records)
        all_rows.extend(rows)
        deviations.extend(session_deviations)

    analyses = {
        "boundary_a35_v15": paired_analysis(
            all_records["boundary_a35_v15"],
            baseline="dart_guarded", candidate="arxiv_joint"),
        "boundary_a40_v13": paired_analysis(
            all_records["boundary_a40_v13"],
            baseline="dart_guarded", candidate="arxiv_joint"),
        "factorial_system": paired_analysis(
            all_records["factorial_a35_v16"],
            baseline="dart_guarded", candidate="arxiv_joint"),
        "factorial_interaction": interaction_analysis(
            all_records["factorial_a35_v16"]),
        "timing_vs_static": paired_analysis(
            all_records["timing_a35_v11p5"],
            baseline="nose_up", candidate="phase_035",
            speed_mode="timed"),
        "timing_vs_harm": paired_analysis(
            all_records["timing_a35_v11p5"],
            baseline="phase_015", candidate="phase_035",
            speed_mode="timed"),
        "washboard_selector_vs_forced": paired_analysis(
            all_records["washboard_a35_v13"],
            baseline="dart_s_forced", candidate="arxiv_joint"),
        "washboard_selector_vs_dart": paired_analysis(
            all_records["washboard_a35_v13"],
            baseline="dart_guarded", candidate="arxiv_joint"),
        "heldout_p1": paired_analysis(
            all_records["heldout_p1_a37_v15p25"],
            baseline="dart_guarded", candidate="arxiv_joint"),
        "heldout_p2": paired_analysis(
            all_records["heldout_p2_a39_v13"],
            baseline="dart_guarded", candidate="arxiv_joint"),
    }
    families = {
        "boundary": [
            "boundary_a35_v15",
            "factorial_system",
            "boundary_a40_v13",
        ],
        "timing": ["timing_vs_static", "timing_vs_harm"],
        "heldout": ["heldout_p1", "heldout_p2"],
        "washboard": ["washboard_selector_vs_forced"],
    }
    adjusted = {}
    for family, names in families.items():
        p_values = {
            name: analyses[name]["session_signflip"]["p_two_sided"]
            for name in names
        }
        family_adjusted = holm_adjust(p_values)
        adjusted[family] = family_adjusted
        for name, value in family_adjusted.items():
            analyses[name]["holm_p"] = value
            analyses[name]["passed"] = bool(
                analyses[name]["passed_unadjusted"] and value < 0.05)
    factorial_holm = holm_adjust({
        "factorial_system": analyses[
            "factorial_system"]["session_signflip"]["p_two_sided"],
        "factorial_interaction": analyses[
            "factorial_interaction"]["session_signflip"]["p_two_sided"],
    })
    analyses["factorial_system"]["factorial_holm_p"] = factorial_holm[
        "factorial_system"]
    analyses["factorial_interaction"]["factorial_holm_p"] = factorial_holm[
        "factorial_interaction"]
    analyses["factorial_interaction"]["passed"] = bool(
        analyses["factorial_interaction"]["passed_unadjusted"]
        and factorial_holm["factorial_interaction"] < 0.05)
    analyses["factorial_system"]["passed"] = bool(
        analyses["factorial_system"].get("passed", False)
        and factorial_holm["factorial_system"] < 0.05)

    capture_metrics = capture_branch_metrics(all_rows)
    branch_fidelity = bool(
        capture_metrics["positive"]["n"] == 48
        and capture_metrics["positive"]["selected_preset"] == 48
        and capture_metrics["negative"]["n"] == 24
        and capture_metrics["negative"]["false_preset"] == 0
        and capture_metrics["boundary"]["n"] == 24
        and capture_metrics["boundary"]["abstained"] == 24
    )
    claim_decisions = {
        "three_boundary_effect": all(
            analyses[name].get("passed", False)
            for name in families["boundary"]),
        "factorial_interaction": analyses[
            "factorial_interaction"]["passed"],
        "timing": all(
            analyses[name].get("passed", False)
            for name in families["timing"]),
        "washboard_abstention": analyses[
            "washboard_selector_vs_forced"].get("passed", False),
        "heldout_deploy": all(
            analyses[name].get("passed", False)
            for name in families["heldout"]),
        "branch_fidelity": branch_fidelity,
    }
    branch_counts = {
        block: {
            "joint_nose_up": sum(
                row["gate_decision"] == "nose_up"
                for row in all_rows
                if row["block_id"] == block
                and row["arm"] == "arxiv_joint"),
            "joint_total": sum(
                row["arm"] == "arxiv_joint"
                for row in all_rows if row["block_id"] == block),
        }
        for block in design["blocks"]
    }
    overall = bool(
        not deviations
        and all(claim_decisions.values())
        and not any(row["hard_limit_exceeded"] for row in all_rows)
    )
    payload = {
        "schema_version": "darts-arxiv-confirmatory-verdict/1",
        "canonical_id": "DARTS_ARXIV_CONFIRMATORY_20260813",
        "design_sha256": design["design_sha256"],
        "expected_runs": design["expected_runs"],
        "observed_runs": len(all_rows),
        "analyses": analyses,
        "holm_families": adjusted,
        "factorial_holm": factorial_holm,
        "branch_counts": branch_counts,
        "capture_branch_metrics": capture_metrics,
        "strategy_counterfactuals": counterfactual_ledger(all_rows),
        "mean_wheel_vs_realized_per_wheel": mean_wheel_audit(all_rows),
        "claim_decisions": claim_decisions,
        "decision": "PASS" if overall else "PARTIAL_OR_FAIL",
        "protocol_deviations": deviations or None,
        "_provenance": aggregate_provenance(
            files,
            note="all-new DART-S arXiv session-level confirmatory matrix"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[darts-analysis] decision={payload['decision']} "
        f"runs={len(all_rows)}/{design['expected_runs']} "
        f"claims={claim_decisions}")
    return payload, 0 if overall else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-session")
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    design = load_design(args.design)
    if args.check_session:
        if args.raw is None:
            raise ValueError("--check-session requires --raw")
        audit = check_one_session(
            design=design,
            session_id=args.check_session,
            raw=args.raw,
        )
        print(json.dumps(audit, sort_keys=True))
        return 0 if audit["passed"] else 3
    if args.output_dir is None:
        raise ValueError("full analysis requires --output-dir")
    _, return_code = full_analysis(design, args.output_dir)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
