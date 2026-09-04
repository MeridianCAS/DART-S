#!/usr/bin/env python3
"""DART-S bench: ramp-face suspension preconditioning, paired against baselines.

Question: does presetting active-suspension corner heights during the run-up
(front-up/rear-down or the mirror) shift the actual takeoff state (theta0,
omega_y0) enough to change the landing attitude? Actuator authority itself is
measured separately by ``scripts/actuator_probe.py`` (static pitch +/-3.5 deg).
Platform: darts_4motor.pc (four-corner hydro channels dartsSuspFL/FR/RL/RR).
One ramp angle per session (queue scripts split the angles) crossed with the
entry speeds given by --speeds. Paired arms:

  neutral    (0,0,0,0)      baseline
  nose_up    (+1,+1,-1,-1)  front-up/rear-down takeoff preset
  nose_down  (-1,-1,+1,+1)  mirror preset (symmetry control)
  torque_up  suspension neutral, rear-biased run-up torque (competing baseline)

The preset is commanded at the start of the run-up, reaches its travel well
before the lip, and is held to takeoff. N rolls per arm, medians reported.

Landing is latched at first contact after apex (nc>=1); the pz<0.75 branch is
only a fallback, so the +/-8 cm ride-height difference between presets cannot
bias the landing threshold. Impact az_peak is sampled over the same window.

Deterministic stepping (SR-17): set_deterministic plus control.pause() before
the loop, never resumed inside it; dt_eff is calibrated ballistically.

Usage: python scripts/darts_bench.py --angle 20 --speeds 10 13 16 --rolls 5
Menu-wedge guard: DARTS_BNG_PRE_LAUNCH_KILL=1
Output: data/cohorts/darts_bench_{tag}.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import scripts._ramp_feather as rf
import scripts._natural_jump as nj
import scripts._ramp_bench as bench
import scripts._tp_perramp as tp
from control.darts.provenance import make_provenance
from control.darts.approach_speed_governor import (
    ApproachSpeedGovernorConfig,
    symmetric_speed_command,
)
from control.darts.cut_speed_compensation import (
    pre_cut_speed_target,
    speed_retention_ready,
)
from control.darts.airborne_attitude import PitchAttitudeController
from control.darts.dart_pitch_air import DARTPitchRateConfig, dart_pitch_rate_command
from control.darts.landing_safety import PAPER_SAFETY_V2, paper_safe_v2, safety_schema
from control.darts.reachability_screen import (
    assess_operational_candidate,
    select_operational_action,
)
from control.darts.reachability import TakeoffReachableSet, directional_rate_budgets
from control.darts.stats_util import median_or_none
from control.darts.wheel_budget import (
    BEAMNG_SBR4_SPIN_CONVENTION_ID,
    DRIVETRAIN_HARD_LIMIT_RADPS,
    OPERATIONAL_SPIN_GUARD_RADPS,
    SUSTAINED_EFFECTIVE_ENVELOPE_RADPS,
    WheelAuthorityLimits,
    wheel_authority_audit,
)
from control.darts.susp_precondition import (
    GATE_TABLE_PATH,
    PresetAwareTakeoffSet,
    SelectorUncertainty,
    SYSTEM_OUTCOME_TABLE_PATH,
    preset_effect,
    recommend_arm,
)
from control.darts.susp_release_phase import (
    ReleasePulseConfig,
    pitch_mode_length,
    scaled_preset,
)
from control.darts.strict_selector_runtime import strict_v2_runtime_select

# Named jump geometry is loaded from sim/scenarios; point the library there
# before it is imported.
import os, sys
for _i, _arg in enumerate(sys.argv):
    if _arg == "--jump-scenario" and _i + 1 < len(sys.argv):
        os.environ.setdefault("DARTS_JUMP_DIR", "sim/scenarios")
        break
    if _arg.startswith("--jump-scenario="):
        os.environ.setdefault("DARTS_JUMP_DIR", "sim/scenarios")
        break
import control.darts.jump_library as jlib
from control.darts.runup_ground import (
    RUNUP_GROUND_CHOICES,
    apply_ramp_material,
    build_runup_pad_segments,
    place_ramp_with_ground,
    runup_ground_audit,
)
from scripts._jump_geometry import (
    build_ramp_pts,
    valley_landing_segments,
    gap_landing_segments,
    gap_ramp_landing_segments,
    ballistic_landing_segments,
    landing_slope_segments,
    bank_landing_segments,
    _post_root_ground,
    _camber_pivot_y,
    _spawn_z_for_lane,
    _set_wheel_factors,
    susp_stroke_sample,
    wheel_w,
    wheel_w_all,
)


def _ramp_pts_for(args, base_x):
    """Same ramp construction as main(): convex-lip kicker or tabletop fillet."""
    if getattr(args, "ramp_mode", "ramp") == "kicker":
        pts, _ = build_ramp_pts(args, base_x, args.angle)
        return pts
    fl_len = tp.fillet_len_for(args.angle, args.fillet_rise_budget, args.fillet_len_max)
    return bench.ramp_polyline(base_x, args.angle, args.rise, fl_len,
                               args.n_fillet, args.n_main, args.sink)

ART = REPO / "data/cohorts"
# No authorized catalog ships, so the strict branch fails closed; paper cohorts
# run under --selector-policy legacy.
STRICT_V2_CATALOG_PATH = REPO / "calibration/strict_v2_catalog.json"
DT = 0.01

ARMS = {
    "neutral":    {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0},
    "nose_up":    {"susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0},
    "nose_down":  {"susp": (-1.0, -1.0, 1.0, 1.0), "bias": 0.0},
    # Competing baseline: suspension neutral, rear-biased run-up torque
    # (b=-1 gives front 0 / rear 2, the nose-up direction). The bias is inert
    # once airborne (throttle=0), and landing braking ignores throttleFactor.
    "torque_up":  {"susp": (0.0, 0.0, 0.0, 0.0), "bias": -1.0},
    # Landing-end presets, takeoff end neutral: soft damping after apex, or soft
    # damping plus a four-corner lift that widens the available stroke.
    "land_soft":       {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "landing": "soft"},
    "land_soft_lift":  {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "landing": "soft_lift"},
    # Ablation: both ends preset
    "both_ends":  {"susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0, "landing": "soft_lift"},
    # 2x2 channel interaction: airborne pitch inner loop crossed with the preset
    "air_only":   {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "air": True},
    "air_susp":   {"susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0, "air": True},
    # Full stack: DART differential rate-tracking air law.
    "dart_only":   {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "air_law": "dart_diff"},
    "dart_cut_only": {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0,
                      "air_law": "dart_diff", "lip_cut_m": 1.0},
    "dart_s_full": {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
                    "f_theta": True, "air_law": "dart_diff"},
    "dart_s_no_cut": {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
                      "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0},
    # Formal default after matched-speed E9 + rough pass-through confirmation.
    # Keep the explicit no-cut arm for historical artifact compatibility.
    "dart_s_system": {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
                      "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0},
    # S3 guard-mechanism confirmatory aliases.  Guard settings live in the arm
    # spec so guarded/unguarded laws can run inside one randomized block.
    "dart_guarded": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "air_law": "dart_diff",
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
    },
    "dart_unguarded": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "air_law": "dart_diff",
        "dart_wheel_speed_cap_radps": 0.0,
        "approach_wheel_spin_guard_radps": 0.0,
    },
    "dart_s_guarded": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
    },
    "dart_s_state_only": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "certificate_mode": "state_only",
        "capture_required": True,
    },
    "dart_s_budget_only": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "certificate_mode": "budget_only",
        "capture_required": False,
    },
    "dart_s_joint": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "certificate_mode": "joint",
        "capture_required": True,
    },
    "dart_s_joint_capture_off": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "certificate_mode": "joint",
        "capture_required": False,
    },
    # arXiv confirmatory selector ablations. These use the shared operational
    # interval-reachability screen on the legacy empirical outcome stack and
    # are intentionally independent of the deployment-disabled strict catalog.
    "arxiv_outcome_only": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "operational_strategy": "outcome_only",
    },
    "arxiv_state_only": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "operational_strategy": "state_only",
    },
    "arxiv_budget_only": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "operational_strategy": "budget_only",
    },
    "arxiv_joint": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "operational_strategy": "joint",
    },
    "dart_s_neural": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
        "certificate_mode": "joint",
        "capture_required": False,
        "authorization_mode": "progressive_evidence",
        "minimum_evidence_grade": "S2",
    },
    "best_classical_dart": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0,
        "air_law": "dart_diff",
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
    },
    "best_classical_static": {
        "susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0,
        "air_law": "dart_diff",
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
    },
    "best_classical_timed": {
        "susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0,
        "air_law": "dart_diff", "phase_pulse_lead_sec": 0.35,
        "dart_wheel_speed_cap_radps": 200.0,
        "approach_wheel_spin_guard_radps": 200.0,
    },
    "forced_preferred_static": {
        "susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0,
        "air_law": "dart_diff",
    },
    "forced_preferred_timed": {
        "susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0,
        "air_law": "dart_diff", "phase_pulse_lead_sec": 0.35,
    },
    "dart_s_unguarded": {
        "susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
        "f_theta": True, "air_law": "dart_diff", "lip_cut_m": 0.0,
        "dart_wheel_speed_cap_radps": 0.0,
        "approach_wheel_spin_guard_radps": 0.0,
    },
    # Forced-preset probe: nose_up plus the DART air law, bypassing the gate table
    # so that grid gaps (35 deg / 16 m/s falls back to neutral as
    # nearest_grid_too_far) do not hide the preset's upper bound. It measures how
    # much the preset could do, so read it apart from the selector arms.
    "dart_s_forced": {"susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0,
                      "air_law": "dart_diff"},
    # Forced-timing probe: same idea as the forced static preset, but the action is
    # the formal timing schedule (lead fixed at 0.35 s, duration and release from
    # the --phase-pulse-* defaults 0.12 s / 0.50). Used to fill the 11.5 and
    # 12 m/s integration gap and to test the 0.25-unit query radius.
    "dart_timed_forced": {"susp": (1.0, 1.0, -1.0, -1.0), "bias": 0.0,
                          "air_law": "dart_diff", "phase_pulse_lead_sec": 0.35},
    # Full DART-S system arm: when the gate commands prepare (nose_up) it also cuts
    # throttle 1 m before the lip, per arm, so the other arms in the same session
    # stay clean. Neutral and avoid branches do not cut: the cut is part of prepare.
    "dart_s_cut": {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True,
                   "f_theta": True, "air_law": "dart_diff",
                   "lip_cut_on_prepare": 1.0},
    # Closed-loop prepare-then-go: the gate table is queried once at the decision
    # point and returns prepare (nose_up), abort (avoid_preset), or pass-through.
    "gated":      {"susp": (0.0, 0.0, 0.0, 0.0), "bias": 0.0, "gate": True},
}
DART_S_SYSTEM_ARM = "dart_s_system"


def _imu_drain(imu):
    """Drain the AdvancedIMU buffer to list[dict] (single / dict-bulk / list forms)."""
    try:
        data = imu.poll()
    except Exception:
        return []
    if data is None:
        return []
    if isinstance(data, dict):
        if "time" in data:  # single reading
            return [data]
        try:
            keys = sorted(data.keys(), key=lambda x: int(x))
        except (TypeError, ValueError):
            keys = list(data.keys())
        return [data[k] for k in keys if isinstance(data[k], dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def set_susp(veh, fl, fr, rl, rr):
    nj._vlua(veh, f"electrics.values.dartsSuspFL={fl:.3f} electrics.values.dartsSuspFR={fr:.3f} "
                  f"electrics.values.dartsSuspRL={rl:.3f} electrics.values.dartsSuspRR={rr:.3f}")


def set_split(veh, b):
    """Axle torque split: frontFactor=1+b, rearFactor=1-b, clamped at 0; b=0 is even."""
    ff = max(0.0, 1.0 + b); rr = max(0.0, 1.0 - b)
    nj._vlua(veh, f"electrics.values.throttleFactorFL={ff} electrics.values.throttleFactorFR={ff} "
                  f"electrics.values.throttleFactorRL={rr} electrics.values.throttleFactorRR={rr}")


def set_damper_mode(veh, mode):
    """Switch the front and rear adaptive dampers; returns the ack list for logging."""
    out = []
    for name in ("adaptiveFrontDamper", "adaptiveRearDamper"):
        r = nj._vlua(veh, f"local c=controller.getController('{name}') "
                          f"if c then c.setParameters({{damperMode='{mode}'}}) return 'ok' else return 'nil' end")
        out.append(str(r).strip() if r is not None else "err")
    return out


def timing_config_for_gate(recommendation: dict, final_arm: str) -> ReleasePulseConfig | None:
    """Build a pulse only when the final selector action remains timed nose-up."""
    if final_arm != "nose_up" or recommendation.get("action_type") != "timed_preset":
        return None
    timing = recommendation.get("release_timing")
    if not isinstance(timing, dict):
        return None
    return ReleasePulseConfig(
        lead_sec=float(timing["lead_sec"]),
        duration_sec=float(timing["duration_sec"]),
        release_fraction=float(timing["release_fraction"]),
        dt_sec=DT,
    )


def build_cut_speed_arm_specs(biases) -> tuple[dict, dict]:
    """Build the 2x2 cut/suspension calibration variants for each speed bias."""
    specs = {}
    mapping = {}
    for raw in biases:
        bias = float(raw)
        if bias < 0 or not math.isfinite(bias):
            raise ValueError("cut speed biases must be finite and >= 0")
        key = f"b{int(round(bias * 100)):03d}"
        if key in mapping:
            raise ValueError(f"duplicate cut speed bias key: {key}")
        cut_only = f"dart_cut_{key}"
        system = f"dart_s_cut_{key}"
        specs[cut_only] = {**ARMS["dart_cut_only"], "lip_cut_speed_bias_mps": bias}
        specs[system] = {**ARMS["dart_s_cut"], "lip_cut_speed_bias_mps": bias}
        mapping[key] = {"bias_mps": bias, "cut_only_arm": cut_only, "system_arm": system}
    return specs, mapping


def build_nocut_speed_arm_specs(offsets) -> tuple[dict, dict]:
    """Build no-cut DART-S variants with action-specific control targets."""
    specs = {}
    mapping = {}
    for raw in offsets:
        offset = float(raw)
        if not math.isfinite(offset):
            raise ValueError("no-cut speed offsets must be finite")
        key = ("m" if offset < 0 else "p") + f"{int(round(abs(offset) * 100)):03d}"
        if key in mapping:
            raise ValueError(f"duplicate no-cut speed offset key: {key}")
        arm = f"dart_s_no_cut_{key}"
        specs[arm] = {**ARMS["dart_s_no_cut"], "entry_speed_offset_mps": offset}
        mapping[key] = {"offset_mps": offset, "arm": arm}
    return specs, mapping


def _bump_segments(args):
    """Washboard run-up: N transverse planks, the last one bump_end metres before the

    Same schema as the flat ramp segments (box, identity rotation). Plank thickness
    is the effective height plus a 2 cm sink so the top face sits at bump_height.
    """
    h = float(args.bump_height); L = float(args.bump_len); T = h + 0.02
    segs = []
    for j in range(int(args.runup_bumps)):
        cx = float(args.base_x) - float(args.bump_end) - j * float(args.bump_spacing)
        segs.append({"name": f"bump{j}", "pos": (cx, 0.0, T / 2.0 - 0.02),
                     "size": (args.width, L, T), "rot": (0.0, 0.0, 0.0, 1.0)})
    return segs


def one_run(bng, veh, *, arm, spec, base_x, run_up, v_entry, R_flight, args, imu=None):
    """One roll: teleport, preset, run-up under P control, latch takeoff state, land."""
    preset_scale = float(getattr(args, "preset_scale", 1.0))
    control_v_entry = max(
        0.1,
        float(v_entry)
        + float(getattr(args, "approach_speed_target_offset_mps", 0.0) or 0.0)
        + float(spec.get("entry_speed_offset_mps", 0.0) or 0.0),
    )
    speed_governor_mode = str(getattr(args, "approach_speed_governor", "throttle_only"))
    speed_governor_cfg = ApproachSpeedGovernorConfig(
        deadband_mps=float(getattr(args, "approach_speed_deadband_mps", 0.05)),
        brake_gain=float(getattr(args, "approach_speed_brake_gain", 2.0)),
        max_brake=float(getattr(args, "approach_speed_max_brake", 1.0)),
    )
    governor_throttle_steps = 0
    governor_brake_steps = 0
    base_fac = tuple(preset_scale * float(x) for x in spec["susp"])
    fac = base_fac
    bias = float(spec.get("bias", 0.0))
    phase_cfg = None
    if spec.get("phase_pulse_lead_sec") is not None:
        phase_cfg = ReleasePulseConfig(
            lead_sec=float(spec["phase_pulse_lead_sec"]),
            duration_sec=float(args.phase_pulse_duration_sec),
            release_fraction=float(args.phase_pulse_release_frac),
            dt_sec=DT,
        )
    phase_trigger_i = None
    phase_restore_i = None
    phase_trigger_distance_m = None
    phase_trigger_speed_mps = None
    phase_trigger_t_to_lip_sec = None
    phase_state_now = "disabled" if phase_cfg is None else "armed"
    stroke_rows = [] if int(getattr(args, "susp_stroke_trace", 0) or 0) else None
    # Record the vertical-velocity time series over the flight and landing window so
    # that filtered impact metrics (20/50 ms moving-average az) can be recomputed
    # offline: a single-step az peak overstates the load.
    impact_rows = [] if int(getattr(args, "impact_trace", 0) or 0) else None
    # A 1 kHz AdvancedIMU channel is attached when requested, for offline SAE J211
    # filtering; a 100 Hz stream only supports CFC 10 under the J211 sampling rule.
    imu_rows = [] if imu is not None else None
    if imu is not None:
        _imu_drain(imu)  # discard whatever the previous roll left in the buffer
    landing_arm = spec.get("landing")  # None / "soft" / "soft_lift"(E4)
    landing_fired = False
    # Airborne pitch inner loop: throttle pitches the body up, brake pitches it down
    air_ctrl = PitchAttitudeController() if spec.get("air") else None
    air_law = spec.get("air_law")
    effective_air_guard_radps = float(
        spec.get(
            "dart_wheel_speed_cap_radps",
            getattr(
                args,
                "dart_wheel_speed_cap_radps",
                OPERATIONAL_SPIN_GUARD_RADPS,
            ),
        )
    )
    effective_approach_guard_radps = float(
        spec.get(
            "approach_wheel_spin_guard_radps",
            getattr(args, "approach_wheel_spin_guard_radps", 0.0),
        )
        or 0.0
    )
    dart_cfg = DARTPitchRateConfig(
        pitch_rate_kp=float(getattr(args, "dart_pitch_rate_kp", 2.0)),
        rate_max_dps=float(getattr(args, "dart_rate_max_dps", 25.0)),
        drive_gain=float(getattr(args, "dart_drive_gain", 0.4)),
        engage_air_steps=int(getattr(args, "dart_engage_air_steps", 6)),
        wheel_speed_cap_radps=effective_air_guard_radps,
    )
    # Closed-loop gate state; the decision point x_dec is set once peak_x is known
    gate_mode = bool(spec.get("gate"))
    gate_decision = None; gate_abort = False; stopped_before_lip = None; stop_margin = None
    gate_rec = None; gate_f_theta = None; gate_merge = None; gate_action_type = None
    # Lip throttle cut (m). The session-level --lip-throttle-cut is the default;
    # spec["lip_cut_m"] overrides unconditionally, and spec["lip_cut_on_prepare"]
    # only fires when the gate commands prepare, so the cut stays on the system arm.
    cut_m = float(getattr(args, "lip_throttle_cut", 0.0) or 0.0)
    if spec.get("lip_cut_m") is not None:
        cut_m = float(spec["lip_cut_m"])
    cut_requested_m = cut_m
    cut_speed_bias_mps = float(
        spec.get("lip_cut_speed_bias_mps",
                 getattr(args, "lip_cut_speed_bias_mps", 0.0)) or 0.0)
    cut_speed_policy = str(getattr(args, "cut_speed_retention_policy", "audit"))
    cut_speed_tol_mps = float(getattr(args, "cut_speed_retention_tol_mps", 0.15))
    cut_speed_audit = None
    dart_engaged_steps = 0; dart_factor_absmax = 0.0
    dart_guard_limited_steps = 0
    dart_air_command_steps = 0
    approach_spin_guard_steps = 0
    approach_max_wheel_abs_radps = 0.0
    wheel_trace_samples = 0
    wheel_trace_air_samples = 0
    wheel_trace_post_samples = 0
    wheel_abs_peak_radps = 0.0
    wheel_abs_peak_air_radps = 0.0
    wheel_abs_peak_post_radps = 0.0
    wheel_hard_exceed_steps = 0
    wheel_guard_exceed_steps = 0
    wheel_abs_peak_by_name: dict[str, float] = {}
    per_step_state_trace = (
        [] if bool(getattr(args, "per_step_state_trace", 0)) else None)
    pts = _ramp_pts_for(args, base_x)
    peak_x = pts[-1][0]
    peak_z = pts[-1][1]
    # Decision point: far enough back that (a) the 0.5 s preset travel completes with
    # margin (lead is tunable; at >=1.5 s it finishes before the fillet) and (b) the
    # avoid branch still has braking distance (a_brake ~ 6 m/s^2 plus 2 m). Every arm
    # records decision_state at the same x_dec, which is what pairs a measured
    # decision-point state with the takeoff state it leads to.
    lead = float(getattr(args, "gate_lead_sec", 0.8))
    x_dec = peak_x - max(v_entry * lead, v_entry * v_entry / (2.0 * 6.0) + 2.0)
    dec_state = None
    syaw = math.radians(270.0); rot = (0.0, 0.0, math.sin(syaw / 2), math.cos(syaw / 2))
    spawn_y = 0.0
    spawn_z = _spawn_z_for_lane(spawn_y, args) if abs(float(getattr(args, "runup_camber_deg", 0.0))) > 1e-6 else 0.3
    try: veh.teleport((base_x - run_up, spawn_y, spawn_z), rot, reset=True)
    except Exception: pass
    try: veh.ai.set_mode("disabled")
    except Exception: pass
    rf._step(bng, 3)
    tp.set_side_cam(bng, base_x, peak_x, R_flight, run_up)
    bench.force_gameplay(bng)          # clear the menu state (includes a resume)
    try: bng.control.pause()           # SR-17: back to stepped, never resumed below
    except Exception: pass
    # settle on the brakes
    for _ in range(12):
        try: veh.control(throttle=0.0, brake=1.0, steering=0.0); rf._step(bng, 1)
        except Exception: pass
    # Command the preset (hydro rate 2/s reaches full travel in 0.5 s; 100 ticks to
    # settle) and reset or command the torque bias, which can survive a teleport.
    r0, p0, _ = nj._rpy(veh, nj._poll(veh))
    set_susp(veh, *fac)
    set_split(veh, bias)
    set_damper_mode(veh, "regular")  # a previous arm may have left the soft mode
    for _ in range(100):
        rf._step(bng, 1)
    r1, p1, _ = nj._rpy(veh, nj._poll(veh))
    ground_pitch_before = round(math.degrees(p0), 3)
    ground_pitch_after = round(math.degrees(p1), 3)
    ground_dpitch = round(math.degrees(p1 - p0), 2)
    # kick the EV out of standstill under paused stepping, then run up
    try: veh.set_velocity(float(min(control_v_entry, 8.0)), 0.3)
    except Exception: pass
    rf._step(bng, 2)

    took_off = False; t_takeoff = t_land = None; air_streak = 0
    apex_z = -9.0; past_apex = False; pd_prev = None; roll_prev = None
    takeoff_state = None; land_pitch = None; prelip_pitch = None
    t_wall_prev = None; dt_samples = []; stall = 0
    vz_last = None; i_last_air = None
    vz_prev = None; az_peak = 0.0; land_roll = None
    tumbled = False; post_land_flip = False
    post_max_abs_pitch = 0.0; post_max_abs_roll = 0.0
    post_contact_samples = 0
    post_steps = max(60, int(round(float(getattr(args, "post_land_sec", 2.0)) / DT)))
    for i in range(args.max_steps):
        _t = time.time()
        if took_off and t_land is None and t_wall_prev is not None:
            dt_samples.append(min(0.5, max(0.005, _t - t_wall_prev)))
        t_wall_prev = _t
        st = nj._poll(veh)
        pos = st.get("pos") or (0, 0, 0); vel = st.get("vel") or (0, 0, 0)
        px, pz = float(pos[0]), float(pos[2])
        gspd = math.hypot(float(vel[0]), float(vel[1]))
        roll, pitch, yaw = nj._rpy(veh, st); nc = nj._contact(veh)
        pitch_d = math.degrees(pitch)
        roll_d = math.degrees(roll)
        pdot = (pitch_d - pd_prev) / DT if pd_prev is not None else 0.0
        rdot = (roll_d - roll_prev) / DT if roll_prev is not None else 0.0
        pd_prev = pitch_d
        roll_prev = roll_d
        air_streak = air_streak + 1 if (nc == 0) else 0
        _wheels_step = wheel_w_all(veh)
        _wheel_mean_abs = None
        if _wheels_step:
            _wheel_abs_values = {
                str(name): abs(float(value))
                for name, value in _wheels_step.items()
            }
            _wheel_mean_abs = sum(_wheel_abs_values.values()) / len(_wheel_abs_values)
            _wheel_step_peak = max(_wheel_abs_values.values())
            wheel_trace_samples += 1
            wheel_abs_peak_radps = max(wheel_abs_peak_radps, _wheel_step_peak)
            for name, value in _wheel_abs_values.items():
                wheel_abs_peak_by_name[name] = max(
                    wheel_abs_peak_by_name.get(name, 0.0), value)
            if took_off and t_land is None:
                wheel_trace_air_samples += 1
                wheel_abs_peak_air_radps = max(
                    wheel_abs_peak_air_radps, _wheel_step_peak)
            if t_land is not None:
                wheel_trace_post_samples += 1
                wheel_abs_peak_post_radps = max(
                    wheel_abs_peak_post_radps, _wheel_step_peak)
            if _wheel_step_peak > float(args.wheel_storage_hard_limit_radps):
                wheel_hard_exceed_steps += 1
            if (
                effective_air_guard_radps > 0.0
                and _wheel_step_peak > effective_air_guard_radps
            ):
                wheel_guard_exceed_steps += 1
        if took_off and (abs(roll_d) > PAPER_SAFETY_V2.tumble_roll_deg
                         or abs(pitch_d) > PAPER_SAFETY_V2.tumble_pitch_deg):
            tumbled = True
            if t_land is not None:
                post_land_flip = True
        if t_land is not None:
            post_contact_samples += 1
            post_max_abs_pitch = max(post_max_abs_pitch, abs(pitch_d))
            post_max_abs_roll = max(post_max_abs_roll, abs(roll_d))
        # Snapshot the measured state at the decision point on every arm, which is
        # the corpus the online membership predictor and its audit read.
        if dec_state is None and not took_off and px >= x_dec:
            _w_dec_all = _wheels_step
            dec_state = {"x_to_peak_m": round(px - peak_x, 2), "v_mps": round(gspd, 2),
                         "vx_mps": round(gspd, 2),
                         "vz_mps": round(float(vel[2]), 3),
                         "pitch_deg": round(pitch_d, 2), "pitch_rate_dps": round(pdot, 1),
                         "roll_deg": round(roll_d, 2), "roll_rate_dps": round(rdot, 1),
                         "pz_m": round(pz, 3), "z_m": round(pz, 3), "i": i,
                         "state_id": (
                             f"{getattr(args, 'session_id', '')}:"
                             f"{spec.get('_roll_index', 'na')}:"
                             f"{arm}:{i}"),
                         "wheel_spin_convention_id": BEAMNG_SBR4_SPIN_CONVENTION_ID,
                         "wheel_w_radps": (
                             round(_wheel_mean_abs, 2)
                             if _wheel_mean_abs is not None else wheel_w(veh)),
                         "wheels_radps": _w_dec_all or None}
        # One gate query at the decision point: table, timing action, curvature veto.
        if gate_mode and gate_decision is None and not took_off and px >= x_dec:
            _r_lip = (float(getattr(args, "lip_radius", 0.0) or 0.0)
                      if getattr(args, "ramp_mode", "ramp") == "kicker" else None)
            _roughness_context = {
                "runup_bumps": int(getattr(args, "runup_bumps", 0) or 0),
                "bump_height_m": float(getattr(args, "bump_height", 0.0) or 0.0),
                "bump_spacing_m": float(getattr(args, "bump_spacing", 0.0) or 0.0),
                "runup_ground_type": str(
                    getattr(args, "runup_ground_type", "ASPHALT")),
            }
            _selector_uncertainty = None
            if (
                float(getattr(args, "selector_angle_uncertainty_deg", 0.0)) > 0.0
                or float(getattr(args, "selector_speed_uncertainty_mps", 0.0)) > 0.0
                or bool(getattr(args, "selector_require_safety_v2_evidence", 0))
            ):
                _selector_uncertainty = SelectorUncertainty(
                    angle_half_width_deg=float(
                        getattr(args, "selector_angle_uncertainty_deg", 0.0)),
                    speed_half_width_mps=float(
                        getattr(args, "selector_speed_uncertainty_mps", 0.0)),
                    require_safety_v2_evidence=bool(
                        getattr(args, "selector_require_safety_v2_evidence", 0)),
                )
            _selector_policy = str(getattr(
                args, "selector_policy", "strict_v2"))
            if _selector_policy == "strict_v2":
                rec = strict_v2_runtime_select(
                    catalog_path=str(getattr(
                        args,
                        "selector_strict_catalog",
                        STRICT_V2_CATALOG_PATH,
                    )),
                    decision_state=dec_state or {},
                    query_angle_deg=float(args.angle),
                    query_speed_mps=float(gspd),
                    geometry_profile=str(getattr(
                        args,
                        "selector_geometry_family",
                        "tabletop_fillet",
                    )),
                    roughness_context=_roughness_context,
                    lip_radius_m=(
                        _r_lip if _r_lip and _r_lip > 0 else None),
                    landing_slope_deg=float(getattr(
                        args, "landing_slope_deg", 0.0) or 0.0),
                    landing_slope_uncertainty_deg=float(getattr(
                        args,
                        "selector_landing_slope_uncertainty_deg",
                        1.0,
                    )),
                    physical_session_id=str(getattr(
                        args, "session_id", "") or ""),
                    can_stop_before_lip=True,
                    certificate_mode=str(spec.get(
                        "certificate_mode",
                        getattr(args, "selector_certificate_mode", "joint"),
                    )),
                    capture_required=bool(spec.get(
                        "capture_required",
                        getattr(args, "selector_capture_required", 1),
                    )),
                    authorization_mode=str(spec.get(
                        "authorization_mode",
                        getattr(
                            args,
                            "selector_authorization_mode",
                            "certificate_first",
                        ),
                    )),
                    minimum_evidence_grade=str(spec.get(
                        "minimum_evidence_grade",
                        getattr(args, "selector_min_evidence_grade", "S2"),
                    )),
                    terrain_features={
                        **{
                            str(key): float(value)
                            for key, value in _roughness_context.items()
                            if isinstance(value, (int, float))
                        },
                        "rise_m": float(args.rise),
                        "fillet_len_m": float(tp.fillet_len_for(
                            args.angle,
                            args.fillet_rise_budget,
                            args.fillet_len_max,
                        )),
                        "roughness_amplitude_m": (
                            float(args.bump_height)
                            if int(args.runup_bumps) > 0 else 0.0
                        ),
                        "roughness_spacing_m": (
                            float(args.bump_spacing)
                            if int(args.runup_bumps) > 0 else 0.0
                        ),
                        "material_code": float({
                            "ASPHALT": 0,
                            "GRAVEL": 1,
                            "DIRT": 2,
                        }[str(args.runup_ground_type).upper()]),
                    },
                    phase="final",
                )
            else:
                rec = recommend_arm(
                    float(args.angle),
                    float(gspd),
                    table_path=str(getattr(args, "gate_table", GATE_TABLE_PATH)),
                    lip_radius=_r_lip if _r_lip and _r_lip > 0 else None,
                    roughness_context=_roughness_context,
                    uncertainty=_selector_uncertainty,
                    geometry_family=str(
                        getattr(args, "selector_geometry_family", "tabletop_fillet")),
                    require_calibration_map=bool(
                        getattr(args, "selector_require_calibration_map", 1)),
                    calibration_support_max_distance=float(getattr(
                        args, "selector_calibration_support_max_distance", 1.5)),
                    system_outcome_table_path=str(
                        getattr(
                            args,
                            "selector_system_outcome_table",
                            SYSTEM_OUTCOME_TABLE_PATH,
                        )),
                    require_system_safety_evidence=bool(
                        getattr(args, "selector_require_system_safety_evidence", 0)),
                    system_safety_max_distance=float(getattr(
                        args, "selector_system_safety_max_distance", 0.75)),
                )
            rec["query_angle_deg"] = float(args.angle)
            rec["query_speed_mps"] = float(gspd)
            rec["nominal_condition_speed_mps"] = float(v_entry)
            gate_rec = {k: rec.get(k) for k in
                        ("arm", "action_type", "reason", "release_timing", "timed_effect",
                         "curvature_margin_mps2", "curvature_veto", "roughness_veto",
                         "query_angle_deg", "query_speed_mps",
                         "nominal_condition_speed_mps",
                         "query_distance", "deployment_status", "abstained",
                         "uncertainty_audit", "calibration_map",
                         "system_outcome_evidence", "table_audit",
                         "strict_selector")}
            table_arm = rec["arm"]
            gate_decision = table_arm
            gate_action_type = rec.get("action_type")
            if _selector_policy == "strict_v2":
                gate_f_theta = rec.get("strict_selector")
                gate_merge = {
                    "policy": "strict_v2_certificate_first",
                    "table_arm": None,
                    "final_arm": gate_decision,
                    "reason": rec.get("reason"),
                }
            if _selector_policy == "legacy" and spec.get("f_theta"):
                entry = rec.get("entry") or {}
                needed = ("theta0_neutral", "omega_y0_neutral", "T_flight_neutral")
                if all(entry.get(k) is not None for k in needed):
                    _th0 = math.radians(float(entry["theta0_neutral"]))
                    _om0 = math.radians(float(entry["omega_y0_neutral"]))
                    _T = float(entry["T_flight_neutral"])
                    _v0 = float(entry.get("v0_neutral") or v_entry)
                    _eff = preset_effect("nose_up", float(args.angle))
                    _calmap = rec.get("calibration_map") or {}
                    if (
                        _calmap.get("in_domain")
                        and _calmap.get("dtheta0_deg") is not None
                        and _calmap.get("domega_y0_dps") is not None
                    ):
                        from control.darts.susp_precondition import SuspPresetEffect
                        _eff = SuspPresetEffect.from_deg(
                            float(_calmap["dtheta0_deg"]),
                            float(_calmap["domega_y0_dps"]),
                        )
                    elif entry.get("d_theta0") is not None and entry.get("d_omega_y0") is not None:
                        from control.darts.susp_precondition import SuspPresetEffect
                        _eff = SuspPresetEffect.from_deg(
                            float(entry["d_theta0"]), float(entry["d_omega_y0"]))

                    def _cert(arm_name, theta, omega, wheel_w0):
                        _b_up, _b_dn = directional_rate_budgets(
                            float(args.gate_wheel_iw),
                            float(args.gate_wheel_omega_max_radps),
                            float(args.gate_wheel_omega_min_radps),
                            max(0.0, float(wheel_w0)),
                            float(args.gate_j_y))
                        _assessment = assess_operational_candidate(
                            theta0=theta,
                            omega0=omega,
                            theta_target=math.radians(
                                float(args.gate_theta_l_deg)),
                            flight_time_s=_T,
                            pitch_accel_radps2=math.radians(
                                float(args.gate_a_pitch_dps2)),
                            B_up=_b_up,
                            B_down=_b_dn,
                            omega_half_width_radps=math.radians(
                                float(args.gate_omega_bar_dps)),
                            theta_half_width_rad=math.radians(
                                float(args.gate_theta_bar_deg)),
                        )
                        _rs = TakeoffReachableSet(
                            theta_L=math.radians(float(args.gate_theta_l_deg)),
                            T=_T, a=math.radians(float(args.gate_a_pitch_dps2)),
                            B_up=_b_up, B_down=_b_dn,
                            omega_bar=math.radians(float(args.gate_omega_bar_dps)),
                            theta_bar=math.radians(float(args.gate_theta_bar_deg)))
                        _pr = _rs.query(theta, omega)
                        return {
                            "arm": arm_name,
                            "reachable": _pr.reachable,
                            "budget_ok": _pr.budget_ok,
                            "velocity_nullable": _pr.velocity_nullable,
                            "displacement_ok": _pr.displacement_ok,
                            "margin_deg": round(math.degrees(_pr.margin), 2),
                            "B_up_dps": round(math.degrees(_b_up), 2),
                            "B_down_dps": round(math.degrees(_b_dn), 2),
                            "wheel_w0_radps": round(float(wheel_w0), 2),
                            "operational_assessment": _assessment.as_dict(),
                        }, _assessment

                    _w_neutral = float(entry.get("wheel_w0_neutral")
                                       or (_v0 / float(args.wheel_r)))
                    _w_preset = _w_neutral + float(
                        _calmap.get("dwheel_radps")
                        if _calmap.get("in_domain")
                        and _calmap.get("dwheel_radps") is not None
                        else (entry.get("d_wheel_w0") or 0.0))
                    _natural, _natural_assessment = _cert(
                        "neutral", _th0, _om0, _w_neutral)
                    _preset, _preset_assessment = _cert(
                        "nose_up", _th0 + _eff.dtheta0,
                        _om0 + _eff.domega_y0, _w_preset)
                    gate_f_theta = {
                        "tier": "ob+directional_budget",
                        "terminal_set": {
                            "theta_center_deg": float(args.gate_theta_l_deg),
                            "theta_half_width_deg": float(args.gate_theta_bar_deg),
                            "omega_half_width_dps": float(args.gate_omega_bar_dps),
                            "scope": "airborne_pitch_only",
                        },
                        "input_source": (
                            "gate_table_measured_wheel"
                            if entry.get("wheel_w0_neutral") is not None else "gate_table_A5"),
                        "T_est_s": _T,
                        "natural": _natural,
                        "preset_nose_up": _preset,
                    }
                    _policy = str(getattr(args, "gate_f_theta_policy", "audit"))
                    _operational_strategy = spec.get("operational_strategy")
                    _operational_decision = None
                    gate_f_theta["strategy_counterfactuals"] = {
                        strategy: select_operational_action(
                            table_arm=table_arm,
                            strategy=strategy,
                            natural=_natural_assessment,
                            preset=_preset_assessment,
                        ).as_dict()
                        for strategy in (
                            "outcome_only",
                            "state_only",
                            "budget_only",
                            "joint",
                        )
                    }
                    if _operational_strategy is not None:
                        _operational_decision = select_operational_action(
                            table_arm=table_arm,
                            strategy=str(_operational_strategy),
                            natural=_natural_assessment,
                            preset=_preset_assessment,
                        )
                        gate_decision = _operational_decision.selected_arm
                        _policy = f"operational_{_operational_strategy}"
                        gate_f_theta["operational_selector"] = (
                            _operational_decision.as_dict())
                    elif _policy == "veto" and table_arm == "nose_up" and not _preset["reachable"]:
                        gate_decision = "neutral" if _natural["reachable"] else "avoid_preset"
                    elif _policy == "require_flip" and table_arm == "nose_up":
                        gate_decision = (
                            "nose_up" if (not _natural["reachable"] and _preset["reachable"])
                            else ("neutral" if _natural["reachable"] else "avoid_preset"))
                    gate_merge = {
                        "policy": _policy,
                        "table_arm": table_arm,
                        "final_arm": gate_decision,
                        "reason": (
                            _operational_decision.reason
                            if _operational_decision is not None
                            else (
                                "table_and_cert_merged"
                                if gate_decision != table_arm
                                else "table_decision_preserved")),
                    }
                else:
                    gate_f_theta = {
                        "error": "gate_table_missing_takeoff_state",
                        "missing": [k for k in needed if entry.get(k) is None],
                    }
                    gate_merge = {
                        "policy": str(getattr(args, "gate_f_theta_policy", "audit")),
                        "table_arm": table_arm,
                        "final_arm": gate_decision,
                        "reason": "f_theta_unavailable",
                    }
            gate_action_type = {
                "avoid_preset": "abort",
                "neutral": (
                    "airborne_only_fallback"
                    if rec.get("abstained") else "pass_through"),
                "nose_up": (
                    "timed_preset"
                    if rec.get("release_timing") and gate_decision == "nose_up"
                    else "static_preset"),
                "nose_down": "static_preset",
            }.get(gate_decision, "pass_through")
            if gate_decision == "nose_up":
                base_fac = tuple(preset_scale * float(x) for x in ARMS["nose_up"]["susp"])
                fac = base_fac
                set_susp(veh, *fac)          # 0.5 s travel, the lead already covers it
                timed_cfg = timing_config_for_gate(rec, gate_decision)
                if timed_cfg is not None:
                    phase_cfg = timed_cfg
                    phase_state_now = "armed"
                if spec.get("lip_cut_on_prepare"):
                    cut_m = float(spec["lip_cut_on_prepare"])   # cut rides with prepare
                    cut_requested_m = cut_m
            elif gate_decision == "avoid_preset":
                gate_abort = True            # refuse the jump: brake before the lip

        # Phase-sweep intervention: briefly move the full preset toward neutral,
        # then restore the identical final command before contact release.  Lead
        # is defined by measured longitudinal time-to-lip, not by loop count.
        if phase_cfg is not None and not took_off:
            _d_lip = peak_x - px
            _ttl = phase_cfg.time_to_lip(_d_lip, gspd)
            if phase_trigger_i is None and phase_cfg.should_trigger(_d_lip, gspd):
                phase_trigger_i = i
                phase_trigger_distance_m = round(_d_lip, 3)
                phase_trigger_speed_mps = round(gspd, 3)
                phase_trigger_t_to_lip_sec = round(_ttl, 4)
                phase_state_now = phase_cfg.state(i, phase_trigger_i)
                fac = scaled_preset(base_fac, phase_cfg.command_scale(i, phase_trigger_i))
                set_susp(veh, *fac)
            elif phase_trigger_i is not None:
                phase_state_now = phase_cfg.state(i, phase_trigger_i)
                if phase_state_now == "restored" and phase_restore_i is None:
                    phase_restore_i = i
                    fac = base_fac
                    set_susp(veh, *fac)
        if not took_off and prelip_pitch is None and px >= peak_x - args.lip_zone:
            prelip_pitch = round(pitch_d, 2)
        if not took_off and air_streak >= 3 and px >= peak_x - 3.0 and pz >= (peak_z - 0.7):
            took_off = True; t_takeoff = i; apex_z = pz
            _w0_all = _wheels_step
            takeoff_state = {"theta0_deg": round(pitch_d, 1), "omega_y0_dps": round(pdot, 0),
                             "v0_mps": round(gspd, 2), "vx0_mps": round(gspd, 2),
                             "vz0_mps": round(float(vel[2]), 2), "z0_m": round(pz, 3),
                             "wheel_w0_radps": (
                                 round(_wheel_mean_abs, 2)
                                 if _wheel_mean_abs is not None else wheel_w(veh)),
                             "wheels0_radps": _w0_all or None,
                             "wheel_spin_convention_id": BEAMNG_SBR4_SPIN_CONVENTION_ID}
        if (stroke_rows is not None and t_land is None and px >= peak_x - float(args.phase_trace_prelip_m)
                and (not took_off or (t_takeoff is not None and i - t_takeoff <= 30))):
            _sr = susp_stroke_sample(veh)
            if _sr:
                _q_pitch = pitch_mode_length(_sr)
                _sr.update({
                    "i": i,
                    "px": round(px, 3),
                    "pz": round(pz, 3),
                    "d_to_lip_m": round(peak_x - px, 3),
                    "t_to_lip_est_sec": (
                        round(max(0.0, peak_x - px) / gspd, 4) if gspd > 0.1 else None),
                    "pitch_deg": round(pitch_d, 3),
                    "pdot_dps": round(pdot, 2),
                    "gspd_mps": round(gspd, 3),
                    "nc": nc,
                    "took_off": bool(took_off),
                    "pitch_mode_length_m": (
                        round(_q_pitch, 6) if _q_pitch is not None else None),
                    "phase_pulse_state": phase_state_now,
                    "phase_command_scale": (
                        phase_cfg.command_scale(i, phase_trigger_i)
                        if phase_cfg is not None else 1.0),
                })
                stroke_rows.append(_sr)
        if took_off and t_land is None:
            vz_last = float(vel[2]); i_last_air = i
            if pz > apex_z: apex_z = pz
            if not past_apex and pz < apex_z - 0.5: past_apex = True
            # Landing-end preset fires at apex, 0.4-0.9 s before touchdown: the damper
            if past_apex and landing_arm and not landing_fired:
                landing_fired = True
                set_damper_mode(veh, "soft")
                if landing_arm == "soft_lift":
                    fac = (1.0, 1.0, 1.0, 1.0)
                    set_susp(veh, *fac)
            # Landing latch: first contact after apex (nc>=1); pz<0.75 is the fallback
            if past_apex and ((nc is not None and nc >= 1) or pz < 0.75):
                t_land = i; land_pitch = round(pitch_d, 1); land_roll = round(roll_d, 1)
                post_max_abs_pitch = abs(pitch_d)
                post_max_abs_roll = abs(roll_d)
        # Impact and settling window: az_peak = max |dvz|/dt (dt is 0.01 when stepped)
        if t_land is not None and i <= t_land + post_steps and vz_prev is not None:
            az_peak = max(az_peak, abs(float(vel[2]) - vz_prev) / DT)
        vz_prev = float(vel[2])
        if (impact_rows is not None and took_off
                and (t_land is None or i <= t_land + post_steps)):
            impact_rows.append({
                "i": i, "vz": round(float(vel[2]), 4), "pz": round(pz, 3),
                "nc": nc, "pitch_deg": round(pitch_d, 2),
            })
        if imu is not None:
            # drain every step to bound the buffer; only keep the flight-to-post window
            _in_win = (took_off and (t_land is None or i <= t_land + post_steps))
            for _rd in _imu_drain(imu):
                if not _in_win:
                    continue
                _acc = _rd.get("accRaw")
                try:
                    imu_rows.append([round(float(_rd.get("time", -1.0)), 5),
                                     round(float(_acc[0]), 3),
                                     round(float(_acc[1]), 3),
                                     round(float(_acc[2]), 3)])
                except (TypeError, ValueError, IndexError, KeyError):
                    pass
        if per_step_state_trace is not None:
            _phase = (
                "post_touchdown"
                if t_land is not None
                else ("airborne" if took_off else "ground")
            )
            _wheel_values = {
                str(name): round(float(value), 3)
                for name, value in (_wheels_step or {}).items()
            }
            _step_wheel_peak = (
                max(abs(value) for value in _wheel_values.values())
                if _wheel_values else None)
            per_step_state_trace.append({
                "i": i,
                "sim_time_s": round(i * DT, 3),
                "phase": _phase,
                "px_m": round(px, 3),
                "pz_m": round(pz, 3),
                "speed_mps": round(gspd, 3),
                "pitch_deg": round(pitch_d, 3),
                "pitch_rate_dps": round(pdot, 3),
                "roll_deg": round(roll_d, 3),
                "contact_count": nc,
                "wheels_radps": _wheel_values,
                "max_abs_wheel_radps": (
                    round(_step_wheel_peak, 3)
                    if _step_wheel_peak is not None else None),
                "hard_limit_exceeded": bool(
                    _step_wheel_peak is not None
                    and _step_wheel_peak
                    > float(args.wheel_storage_hard_limit_radps)),
            })
        # Control: P to v_entry during run-up, coast after takeoff, brake after landing
        if gate_abort and not took_off:
            thr, brk, pb = 0.0, 1.0, 0.0
            if gspd < 0.3:
                stopped_before_lip = bool(px < peak_x - 0.5)
                stop_margin = round(peak_x - px, 2)
                break
        elif not took_off:
            # Throttle cut: coast from cut metres before the lip to unload the flywheel.
            # gravity-only energy compensation is augmented by a calibrated
            # per-cell bias.  Optional fallback disables the cut when the
            # measured cut-entry speed cannot meet the retention target.
            v_ctrl = (
                pre_cut_speed_target(
                    control_v_entry, cut_m, args.angle,
                    calibrated_bias_mps=cut_speed_bias_mps)
                if cut_m > 0 else control_v_entry)
            in_cut_zone = cut_m > 0 and px >= peak_x - cut_m
            if in_cut_zone and cut_speed_audit is None:
                ready = speed_retention_ready(gspd, v_ctrl, cut_speed_tol_mps)
                fallback = cut_speed_policy == "fallback_no_cut" and not ready
                cut_speed_audit = {
                    "policy": cut_speed_policy,
                    "requested_cut_m": cut_requested_m,
                    "calibrated_bias_mps": cut_speed_bias_mps,
                    "target_at_cut_mps": round(v_ctrl, 3),
                    "measured_at_cut_mps": round(gspd, 3),
                    "margin_mps": round(gspd - v_ctrl, 3),
                    "tolerance_mps": cut_speed_tol_mps,
                    "ready": ready,
                    "fallback_no_cut": fallback,
                }
                if fallback:
                    cut_m = 0.0
                    in_cut_zone = False
                    v_ctrl = control_v_entry
            if in_cut_zone:
                thr, brk, pb = 0.0, 0.0, 0.0
            else:
                if speed_governor_mode == "symmetric":
                    thr, brk = symmetric_speed_command(
                        gspd, v_ctrl, speed_governor_cfg)
                else:
                    thr, brk = (1.0 if gspd < v_ctrl else 0.0), 0.0
                pb = 0.0
                governor_throttle_steps += int(thr > 0)
                governor_brake_steps += int(brk > 0)
            _approach_guard = effective_approach_guard_radps
            if _approach_guard > 0.0:
                _approach_wheels = _wheels_step
                if _approach_wheels:
                    _approach_max = max(
                        abs(float(value)) for value in _approach_wheels.values())
                    approach_max_wheel_abs_radps = max(
                        approach_max_wheel_abs_radps, _approach_max)
                    if _approach_max > _approach_guard:
                        thr = 0.0
                        approach_spin_guard_steps += 1
            stall = stall + 1 if gspd < 0.5 else 0
            if stall > 150:
                break  # fail fast when the paused-mode launch did not take
        elif t_land is None:
            if air_law == "dart_diff":
                _target_deg = float(getattr(args, "landing_slope_deg", 0.0) or 0.0)
                _wheel_guard_values = _wheels_step
                _wheel_guard_speed = (
                    max(abs(float(value)) for value in _wheel_guard_values.values())
                    if _wheel_guard_values else wheel_w(veh))
                dart_air_command_steps += 1
                _cmd = dart_pitch_rate_command(
                    pitch_d, pdot, _target_deg,
                    air_streak=air_streak, contact_count=int(nc or 0),
                    wheel_speed_abs_radps=_wheel_guard_speed, config=dart_cfg)
                _set_wheel_factors(
                    veh, _cmd.wheel_factor, _cmd.wheel_factor,
                    _cmd.wheel_factor, _cmd.wheel_factor)
                thr, brk, pb = _cmd.throttle, _cmd.brake, 0.0
                if _cmd.engaged:
                    dart_engaged_steps += 1
                    dart_factor_absmax = max(dart_factor_absmax, abs(_cmd.wheel_factor))
                if _cmd.guard_limited:
                    dart_guard_limited_steps += 1
            elif air_ctrl is not None:
                pitch_ref = math.radians(float(args.landing_slope_deg)) if float(getattr(args, "landing_slope_deg", 0.0)) != 0.0 else 0.0
                thr, brk = air_ctrl.compute(pitch, math.radians(pdot), pitch_ref)
                pb = 0.0
            else:
                thr, brk, pb = 0.0, 0.0, 0.0
        else:
            if air_law == "dart_diff":
                _set_wheel_factors(veh, 1.0, 1.0, 1.0, 1.0)
            thr, brk, pb = 0.0, 1.0, 1.0
        if i % 10 == 0:
            set_susp(veh, *fac)  # re-command the preset in case a reset dropped it
            if bias != 0.0 and not took_off:
                set_split(veh, bias)
        try: veh.control(throttle=thr, brake=brk, steering=0.0, parkingbrake=pb)
        except Exception: pass
        rf._step(bng, 1)
        if t_land is not None and i > t_land + post_steps:
            break
    # Time base: each loop step makes two Lua round trips, so wall dt (~0.05) is not
    # the sim step. Calibrate ballistically: dvz = -g*N*dt over the flight segment.
    dt_wall = round(sum(dt_samples) / len(dt_samples), 4) if dt_samples else None
    dt_phys = None
    if takeoff_state is not None and vz_last is not None and i_last_air is not None and t_takeoff is not None:
        n = i_last_air - t_takeoff
        if n >= 20:
            dt_phys = round((takeoff_state["vz0_mps"] - vz_last) / (9.81 * n), 4)
    T_flight = round((t_land - t_takeoff) * DT, 2) if (t_takeoff is not None and t_land is not None) else None
    post_window_complete = bool(
        t_land is not None and post_contact_samples >= post_steps + 1)
    post_stable = bool(
        post_window_complete
        and not tumbled
        and post_max_abs_pitch <= PAPER_SAFETY_V2.pitch_limit_deg
        and post_max_abs_roll <= PAPER_SAFETY_V2.roll_limit_deg)
    out = {"arm": arm, "v_entry": v_entry,
           "control_target_v_mps": control_v_entry,
           "took_off": took_off,
           "ground_pitch_before_deg": ground_pitch_before,
           "ground_pitch_after_deg": ground_pitch_after,
           "ground_dpitch_deg": ground_dpitch,
           "prelip_pitch_deg": prelip_pitch, "decision_state": dec_state,
           "takeoff_state": takeoff_state,
           "T_flight": T_flight, "airtime_steps": T_flight, "land_pitch_deg": land_pitch,
           "land_roll_deg": land_roll, "az_peak_mps2": round(az_peak, 1),
           "tumbled": tumbled, "post_land_flip": post_land_flip,
           "post_touchdown": {
               "window_steps": post_steps,
               "window_s": round(post_steps * DT, 2),
               "required_samples": post_steps + 1,
               "samples": post_contact_samples,
               "complete_window": post_window_complete,
               "max_abs_pitch_deg": round(post_max_abs_pitch, 2),
               "max_abs_roll_deg": round(post_max_abs_roll, 2),
               "stable": post_stable,
           },
           "air_law": {
               "name": air_law or ("legacy_pitch_pd" if air_ctrl is not None else "passive"),
               "engaged_steps": dart_engaged_steps,
               "command_opportunities": dart_air_command_steps,
               "guard_limited_steps": dart_guard_limited_steps,
               "guard_limited_fraction": (
                   round(dart_guard_limited_steps / dart_air_command_steps, 4)
                   if dart_air_command_steps > 0 else None),
               "operational_spin_guard_radps": dart_cfg.wheel_speed_cap_radps,
               "wheel_factor_absmax": round(dart_factor_absmax, 4),
           },
           "approach_wheel_guard": {
               "limit_radps": effective_approach_guard_radps,
               "limited_steps": approach_spin_guard_steps,
               "observed_max_abs_radps": round(
                   approach_max_wheel_abs_radps, 2),
           },
           "wheel_trajectory_audit": {
               "samples": wheel_trace_samples,
               "air_samples": wheel_trace_air_samples,
               "post_touchdown_samples": wheel_trace_post_samples,
               "max_abs_radps": round(wheel_abs_peak_radps, 2),
               "max_abs_air_radps": round(wheel_abs_peak_air_radps, 2),
               "max_abs_post_touchdown_radps": round(
                   wheel_abs_peak_post_radps, 2),
               "per_wheel_max_abs_radps": {
                   name: round(value, 2)
                   for name, value in sorted(wheel_abs_peak_by_name.items())
               },
               "operational_guard_radps": effective_air_guard_radps,
               "operational_guard_exceed_steps": wheel_guard_exceed_steps,
               "drivetrain_hard_limit_radps": float(
                   args.wheel_storage_hard_limit_radps),
               "hard_limit_exceed_steps": wheel_hard_exceed_steps,
               "hard_limit_exceeded": wheel_hard_exceed_steps > 0,
           },
           "dt_eff": dt_phys, "dt_wall_mean": dt_wall}
    if per_step_state_trace is not None:
        out["per_step_state_trace"] = per_step_state_trace
    if stroke_rows is not None:
        out["stroke_trace"] = stroke_rows
        out["stroke_trace_schema_version"] = 1
    if impact_rows is not None:
        out["impact_trace"] = impact_rows
        out["impact_trace_schema_version"] = 1
    if imu_rows is not None:
        out["impact_imu_trace"] = {
            "hz": int(getattr(args, "impact_imu_hz", 0) or 0),
            "schema_version": 1,
            "frame": "imu-local [t, ax, ay, az(up)], gravity included",
            "rows": imu_rows,
        }
    out["approach_speed_governor"] = {
        "mode": speed_governor_mode,
        "nominal_request_mps": float(v_entry),
        "control_target_mps": control_v_entry,
        "deadband_mps": speed_governor_cfg.deadband_mps,
        "brake_gain": speed_governor_cfg.brake_gain,
        "max_brake": speed_governor_cfg.max_brake,
        "throttle_steps": governor_throttle_steps,
        "brake_steps": governor_brake_steps,
    }
    if phase_cfg is not None:
        out["release_phase_pulse"] = {
            "lead_sec": phase_cfg.lead_sec,
            "duration_sec": phase_cfg.duration_sec,
            "duration_steps": phase_cfg.duration_steps,
            "release_fraction": phase_cfg.release_fraction,
            "trigger_step": phase_trigger_i,
            "restore_step": phase_restore_i,
            "takeoff_step": t_takeoff,
            "trigger_distance_to_lip_m": phase_trigger_distance_m,
            "trigger_speed_mps": phase_trigger_speed_mps,
            "trigger_t_to_lip_est_sec": phase_trigger_t_to_lip_sec,
            "restore_to_takeoff_sec": (
                round((t_takeoff - phase_restore_i) * DT, 3)
                if t_takeoff is not None and phase_restore_i is not None else None),
            "command_restored_before_takeoff": bool(
                t_takeoff is not None and phase_restore_i is not None and phase_restore_i < t_takeoff),
            "final_command": list(base_fac),
        }
    out["safe"] = {
        "criterion_version": 2,
        "touchdown_attitude_ok": bool(
            took_off and land_pitch is not None and land_roll is not None
            and abs(land_pitch) <= PAPER_SAFETY_V2.pitch_limit_deg
            and abs(land_roll) <= PAPER_SAFETY_V2.roll_limit_deg),
        "no_tumble": not tumbled,
        "post_touchdown_stable": post_stable,
    }
    out["safe"]["overall"] = bool(paper_safe_v2(out))
    _wheel_values = []
    if takeoff_state:
        _per_wheel = takeoff_state.get("wheels0_radps")
        if isinstance(_per_wheel, dict):
            _wheel_values = list(_per_wheel.values())
        elif takeoff_state.get("wheel_w0_radps") is not None:
            _wheel_values = [takeoff_state["wheel_w0_radps"]]
    out["wheel_authority_audit"] = wheel_authority_audit(
        _wheel_values,
        WheelAuthorityLimits(
            sustained_effective_envelope_radps=float(
                args.gate_wheel_omega_max_radps),
            operational_spin_guard_radps=float(
                effective_air_guard_radps),
            drivetrain_hard_limit_radps=float(
                args.wheel_storage_hard_limit_radps),
        ),
    )
    out["lip_cut_m_requested"] = cut_requested_m if cut_requested_m > 0 else None
    out["lip_cut_m_effective"] = cut_m if cut_m > 0 else None
    out["cut_speed_retention_audit"] = cut_speed_audit
    if gate_mode:
        if gate_f_theta is not None and takeoff_state is not None and T_flight is not None:
            _wheels = takeoff_state.get("wheels0_radps") or {}
            _omega_now = ([abs(float(v)) for v in _wheels.values()]
                          if isinstance(_wheels, dict) and _wheels
                          else max(0.0, float(takeoff_state.get("wheel_w0_radps")
                                              or takeoff_state["v0_mps"] / float(args.wheel_r))))
            _b_up, _b_dn = directional_rate_budgets(
                float(args.gate_wheel_iw),
                float(args.gate_wheel_omega_max_radps),
                float(args.gate_wheel_omega_min_radps),
                _omega_now, float(args.gate_j_y))
            _real = TakeoffReachableSet(
                theta_L=math.radians(float(args.gate_theta_l_deg)),
                T=float(T_flight), a=math.radians(float(args.gate_a_pitch_dps2)),
                B_up=_b_up, B_down=_b_dn,
                omega_bar=math.radians(float(args.gate_omega_bar_dps)),
                theta_bar=math.radians(float(args.gate_theta_bar_deg))).query(
                    math.radians(float(takeoff_state["theta0_deg"])),
                    math.radians(float(takeoff_state["omega_y0_dps"])))
            gate_f_theta["realized_selected_arm"] = {
                "reachable": _real.reachable,
                "budget_ok": _real.budget_ok,
                "velocity_nullable": _real.velocity_nullable,
                "displacement_ok": _real.displacement_ok,
                "margin_deg": round(math.degrees(_real.margin), 2),
                "B_up_dps": round(math.degrees(_b_up), 2),
                "B_down_dps": round(math.degrees(_b_dn), 2),
                "wheel_input_source": (
                    "measured_per_wheel" if isinstance(_omega_now, list)
                    else "measured_mean_or_A5"),
            }
        out["gate"] = {"decision": gate_decision, "action_type": gate_action_type,
                       "release_timing": (
                           gate_rec.get("release_timing") if gate_rec else None),
                       "aborted": gate_abort,
                       "stopped_before_lip": stopped_before_lip, "stop_margin_m": stop_margin,
                       "recommend": gate_rec, "f_theta": gate_f_theta, "merge": gate_merge}
    return out


def med(vals):
    return median_or_none(vals)


def arm_summary(runs):
    ts = [r["takeoff_state"] for r in runs if r.get("takeoff_state")]
    safe_v2 = [r for r in runs if paper_safe_v2(r)]
    return {
        "n_takeoff": len(ts), "n": len(runs),
        "n_safe_v2": len(safe_v2),
        "safe_rate_v2": round(len(safe_v2) / len(runs), 3) if runs else None,
        "ground_dpitch_med": med([r["ground_dpitch_deg"] for r in runs]),
        "prelip_pitch_med": med([r["prelip_pitch_deg"] for r in runs]),
        "theta0_med": med([t["theta0_deg"] for t in ts]),
        "omega_y0_med": med([t["omega_y0_dps"] for t in ts]),
        "v0_med": med([t["v0_mps"] for t in ts]),
        "wheel_w0_med": med([t.get("wheel_w0_radps") for t in ts]),
        "T_flight_med": med([r["T_flight"] for r in runs]),
        "land_pitch_med": med([r["land_pitch_deg"] for r in runs]),
        "az_peak_med": med([r["az_peak_mps2"] for r in runs]),
    }


def audit_neural_calmap_run(
    run: dict,
    *,
    expected_session_id: str,
    collection_mode: str,
) -> dict:
    """Fail-closed schema audit for prospective Neural CalMap evidence."""
    reasons = []
    protocol = run.get("protocol") or {}
    if not expected_session_id:
        reasons.append("session_id_missing")
    if protocol.get("session_id") != expected_session_id:
        reasons.append("params_protocol_session_mismatch")
    if protocol.get("roll_index") is None:
        reasons.append("roll_index_missing")
    decision = run.get("decision_state")
    if not isinstance(decision, dict):
        reasons.append("decision_state_missing")
    else:
        for field in (
            "pitch_deg", "pitch_rate_dps", "roll_deg", "roll_rate_dps",
            "vx_mps", "vz_mps", "z_m", "state_id",
            "wheel_spin_convention_id",
        ):
            if decision.get(field) is None:
                reasons.append(f"decision_{field}_missing")
        wheels = decision.get("wheels_radps")
        if not isinstance(wheels, dict) or set(wheels) != {"FL", "FR", "RL", "RR"}:
            reasons.append("decision_signed_wheels4_missing")
    takeoff = run.get("takeoff_state")
    model_eligible = isinstance(takeoff, dict) and run.get("T_flight") is not None
    if model_eligible:
        for field in (
            "theta0_deg", "omega_y0_dps", "vx0_mps", "vz0_mps", "z0_m",
            "wheel_spin_convention_id",
        ):
            if takeoff.get(field) is None:
                reasons.append(f"takeoff_{field}_missing")
        wheels = takeoff.get("wheels0_radps")
        if not isinstance(wheels, dict) or set(wheels) != {"FL", "FR", "RL", "RR"}:
            reasons.append("takeoff_signed_wheels4_missing")
    elif collection_mode == "model":
        reasons.append("model_takeoff_or_flight_time_missing")
    passed = not reasons
    return {
        "schema_version": "darts-neural-calmap-run-audit/1",
        "collection_mode": collection_mode,
        "passed": passed,
        "model_eligible": bool(model_eligible and passed),
        "reasons": reasons,
        "session_id": expected_session_id,
        "roll_index": protocol.get("roll_index"),
    }


def sign_test_p(diffs):
    """Two-sided paired sign test (exact binomial) over per-roll paired differences."""
    d = [x for x in diffs if x is not None and x != 0]
    n = len(d)
    if n == 0:
        return None
    k = sum(1 for x in d if x > 0)
    from math import comb
    kk = min(k, n - k)
    p = sum(comb(n, j) for j in range(0, kk + 1)) * 2 / (2 ** n)
    return round(min(1.0, p), 4)


def _phase_protocol_fingerprint(args, arm_specs) -> str:
    """Fingerprint the experiment design so checkpoints fail closed on drift."""
    source_sha256 = {
        "scripts/darts_bench.py": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "control/darts/susp_release_phase.py": hashlib.sha256(
            (REPO / "control/darts/susp_release_phase.py").read_bytes()).hexdigest(),
        "control/darts/calibration_map.py": hashlib.sha256(
            (REPO / "control/darts/calibration_map.py").read_bytes()).hexdigest(),
        "control/darts/susp_precondition.py": hashlib.sha256(
            (REPO / "control/darts/susp_precondition.py").read_bytes()).hexdigest(),
        "control/darts/system_outcome_selector.py": hashlib.sha256(
            (REPO / "control/darts/system_outcome_selector.py").read_bytes()).hexdigest(),
        "control/darts/wheel_budget.py": hashlib.sha256(
            (REPO / "control/darts/wheel_budget.py").read_bytes()).hexdigest(),
        "control/darts/dart_pitch_air.py": hashlib.sha256(
            (REPO / "control/darts/dart_pitch_air.py").read_bytes()).hexdigest(),
    }
    if any(bool(arm_specs[k].get("gate")) for k in args.arms):
        strict_catalog = Path(getattr(
            args, "selector_strict_catalog", STRICT_V2_CATALOG_PATH))
        source_sha256["selector_strict_catalog"] = (
            hashlib.sha256(strict_catalog.read_bytes()).hexdigest()
            if strict_catalog.exists() else "MISSING")
        gate_path = Path(getattr(args, "gate_table", GATE_TABLE_PATH))
        source_sha256["selector_gate_table"] = (
            hashlib.sha256(gate_path.read_bytes()).hexdigest()
            if gate_path.exists() else "MISSING")
        outcome_path = Path(getattr(
            args, "selector_system_outcome_table", SYSTEM_OUTCOME_TABLE_PATH))
        source_sha256["selector_system_outcome_table"] = (
            hashlib.sha256(outcome_path.read_bytes()).hexdigest()
            if outcome_path.exists() else "MISSING")
    payload = {
        "angle": args.angle,
        "speeds": list(args.speeds),
        "rolls": args.rolls,
        "arms": list(args.arms),
        "arm_specs": {k: arm_specs[k] for k in args.arms},
        "pc": args.pc,
        "rise": args.rise,
        "fillet_len_max": args.fillet_len_max,
        "fillet_rise_budget": args.fillet_rise_budget,
        "run_up": args.run_up,
        "base_x": args.base_x,
        "max_steps": args.max_steps,
        "post_land_sec": args.post_land_sec,
        "lip_zone": args.lip_zone,
        "v_jitter": args.v_jitter,
        "lip_throttle_cut": args.lip_throttle_cut,
        "lip_cut_speed_bias_mps": args.lip_cut_speed_bias_mps,
        "cut_speed_retention_policy": args.cut_speed_retention_policy,
        "cut_speed_retention_tol_mps": args.cut_speed_retention_tol_mps,
        "cut_speed_bias_sweep": list(args.cut_speed_bias_sweep),
        "nocut_speed_offset_sweep": list(args.nocut_speed_offset_sweep),
        "dart_s_no_cut_speed_offset_mps": args.dart_s_no_cut_speed_offset_mps,
        "approach_speed_governor": args.approach_speed_governor,
        "approach_speed_target_offset_mps": args.approach_speed_target_offset_mps,
        "approach_speed_deadband_mps": args.approach_speed_deadband_mps,
        "approach_speed_brake_gain": args.approach_speed_brake_gain,
        "approach_speed_max_brake": args.approach_speed_max_brake,
        "runup_ground_type": args.runup_ground_type,
        "approach_wheel_spin_guard_radps": args.approach_wheel_spin_guard_radps,
        "dart_wheel_speed_cap_radps": args.dart_wheel_speed_cap_radps,
        "wheel_storage_hard_limit_radps": args.wheel_storage_hard_limit_radps,
        "selector_angle_uncertainty_deg": args.selector_angle_uncertainty_deg,
        "selector_speed_uncertainty_mps": args.selector_speed_uncertainty_mps,
        "selector_require_safety_v2_evidence": (
            args.selector_require_safety_v2_evidence),
        "selector_geometry_family": args.selector_geometry_family,
        "selector_require_calibration_map": args.selector_require_calibration_map,
        "selector_calibration_support_max_distance": (
            args.selector_calibration_support_max_distance),
        "selector_require_system_safety_evidence": (
            args.selector_require_system_safety_evidence),
        "selector_system_safety_max_distance": (
            args.selector_system_safety_max_distance),
        "selector_policy": args.selector_policy,
        "selector_strict_catalog": args.selector_strict_catalog,
        "selector_certificate_mode": args.selector_certificate_mode,
        "selector_capture_required": args.selector_capture_required,
        "selector_authorization_mode": args.selector_authorization_mode,
        "selector_min_evidence_grade": args.selector_min_evidence_grade,
        "selector_landing_slope_uncertainty_deg": (
            args.selector_landing_slope_uncertainty_deg),
        "gate_table": args.gate_table,
        "selector_system_outcome_table": args.selector_system_outcome_table,
        "gate_f_theta_policy": args.gate_f_theta_policy,
        "gate_theta_l_deg": args.gate_theta_l_deg,
        "gate_theta_bar_deg": args.gate_theta_bar_deg,
        "gate_omega_bar_dps": args.gate_omega_bar_dps,
        "jump_scenario": args.jump_scenario,
        "ramp_mode": getattr(args, "ramp_mode", "ramp"),
        "lip_radius": getattr(args, "lip_radius", None),
        "lip_sweep_deg": getattr(args, "lip_sweep_deg", None),
        "randomize_arm_order": args.randomize_arm_order,
        "arm_order_seed": args.arm_order_seed,
        "phase_sweep_leads": list(args.phase_sweep_leads),
        "phase_pulse_duration_sec": args.phase_pulse_duration_sec,
        "phase_pulse_release_frac": args.phase_pulse_release_frac,
        "susp_stroke_trace": args.susp_stroke_trace,
        "phase_trace_prelip_m": args.phase_trace_prelip_m,
        "impact_trace": getattr(args, "impact_trace", 0),
        "per_step_state_trace": getattr(args, "per_step_state_trace", 0),
        "session_id": getattr(args, "session_id", ""),
        "launch_id": getattr(args, "launch_id", ""),
        "neural_calmap_collection": getattr(
            args, "neural_calmap_collection", "off"),
        "source_sha256": source_sha256,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pc", default="vehicles/sbr/darts_4motor.pc")
    p.add_argument("--angle", type=float, default=20)
    p.add_argument("--speeds", type=float, nargs="*", default=[13.0], help="entry-speed sweep in m/s (several speeds per session)")
    p.add_argument("--v-jitter", type=float, default=0.0,
                   help="entry-speed jitter: each roll draws from U(v-J, v+J), shared by every arm in that roll so the pairing holds; "
                        "a positive value makes the aggregator pool separately (params.v_jitter)")
    p.add_argument("--lip-throttle-cut", type=float, default=0.0,
                   help="throttle cut: coast from this many metres before the lip, unloading the flywheel; "
                        "a positive value makes the aggregator pool separately")
    p.add_argument("--lip-cut-speed-bias-mps", type=float, default=0.0,
                   help="per-cell calibration bias on the pre-cut target speed (m/s), on top of the gravity-energy term")
    p.add_argument("--cut-speed-bias-sweep", type=float, nargs="*", default=[],
                   help="speed-deconfounding scout: generate cut-only and system cut-bias variants")
    p.add_argument("--nocut-speed-offset-sweep", type=float, nargs="*", default=[],
                   help="no-cut matched-speed scout: generate action-specific target offsets")
    p.add_argument("--dart-s-no-cut-speed-offset-mps", type=float, default=0.0,
                   help="per-cell control-target offset for the formal no-cut DART-S arm")
    p.add_argument("--approach-speed-governor",
                   choices=("throttle_only", "symmetric"), default="throttle_only")
    p.add_argument("--approach-speed-target-offset-mps", type=float, default=0.0,
                   help="lip speed-target offset shared by both arms")
    p.add_argument("--approach-speed-deadband-mps", type=float, default=0.05)
    p.add_argument("--approach-speed-brake-gain", type=float, default=2.0)
    p.add_argument("--approach-speed-max-brake", type=float, default=1.0)
    p.add_argument("--cut-speed-retention-policy",
                   choices=("audit", "fallback_no_cut"), default="audit",
                   help="when the pre-cut speed target is missed: audit only, or fail closed to no-cut")
    p.add_argument("--cut-speed-retention-tol-mps", type=float, default=0.15)
    p.add_argument("--runup-bumps", type=int, default=0,
                   help="washboard run-up: place N transverse planks before the ramp base to "
                        "excite entry-attitude spread; positive values pool as _bp{height_cm}")
    p.add_argument("--bump-height", type=float, default=0.05,
                   help="effective plank height in m (top face above ground), default 0.05")
    p.add_argument("--bump-spacing", type=float, default=1.3,
                   help="plank centre spacing in m (~half the wheelbase, excites pitch), default 1.3")
    p.add_argument("--bump-end", type=float, default=3.0,
                   help="distance from the last plank to the ramp base in m (smaller leaves more residual oscillation), default 3.0")
    p.add_argument("--bump-len", type=float, default=0.35,
                   help="plank length along the direction of travel in m, default 0.35")
    p.add_argument(
        "--runup-ground-type",
        choices=RUNUP_GROUND_CHOICES,
        default="ASPHALT",
        help="unified run-up and takeoff-ramp contact material",
    )
    p.add_argument("--protocol-tag", default="",
                   help="protocol pool suffix (e.g. j10 for jitter +/-1.0): appended to the platform key so "
                        "same-named protocols at different amplitudes pool separately")
    p.add_argument("--session-id", default="",
                   help="independent simulator-session identifier written to every run")
    p.add_argument("--launch-id", default="",
                   help="immutable experiment launch identifier written to every run")
    p.add_argument(
        "--neural-calmap-collection",
        choices=("off", "model", "system"),
        default="off",
        help="enable fail-closed prospective Neural CalMap run schema audit",
    )
    p.add_argument("--gate-lead-sec", type=float, default=2.0,
                   help="gated decision lead in seconds: x_dec = peak - max(v*lead, braking distance). "
                        "The default 2.0 completes the preset before the fillet (measured: lead 0.8 lands "
                        "inside the fillet at safe 0.70, lead 2.0 recovers 0.93)")
    p.add_argument("--gate-table", default=str(GATE_TABLE_PATH),
                   help="empirical preset-action table used by the online selector")
    p.add_argument(
        "--selector-policy",
        choices=("strict_v2", "legacy"),
        default="strict_v2",
        help=(
            "strict_v2=certificate-first fail-closed authorization; "
            "legacy=historical table-first reproduction only"
        ),
    )
    p.add_argument(
        "--selector-strict-catalog",
        default=str(STRICT_V2_CATALOG_PATH),
        help="immutable strict_v2 candidate/model/hash catalog",
    )
    p.add_argument(
        "--selector-certificate-mode",
        choices=("state_only", "budget_only", "joint"),
        default="joint",
        help="preregistered strict_v2 certificate ablation mode",
    )
    p.add_argument(
        "--selector-capture-required",
        type=int,
        choices=(0, 1),
        default=1,
        help="1=consume full capture membership; 0=preregistered gate ablation",
    )
    p.add_argument(
        "--selector-authorization-mode",
        choices=("certificate_first", "progressive_evidence"),
        default="certificate_first",
        help="authorization identity; reachability is audit-only in progressive mode",
    )
    p.add_argument(
        "--selector-min-evidence-grade",
        choices=("W0", "W1", "W2", "S1", "S2"),
        default="S2",
        help="minimum local evidence grade for progressive deployment",
    )
    p.add_argument("--gate-f-theta-policy", choices=("audit", "veto", "require_flip"),
                   default="audit", help="how the F_theta screen and the gate table are combined")
    p.add_argument("--selector-angle-uncertainty-deg", type=float, default=0.0,
                   help="bounded ramp-angle uncertainty; conflicting corner actions abstain")
    p.add_argument("--selector-speed-uncertainty-mps", type=float, default=0.0,
                   help="bounded approach-speed uncertainty; conflicting corner actions abstain")
    p.add_argument("--selector-require-safety-v2-evidence", type=int, choices=(0, 1),
                   default=0, help="deploy preset only from post-touchdown-v2 evidence")
    p.add_argument("--selector-geometry-family", default="tabletop_fillet",
                   help="calibration-map geometry family; unknown families abstain")
    p.add_argument("--selector-require-calibration-map", type=int, choices=(0, 1),
                   default=1, help="fail closed when takeoff-shift CalMap lacks support")
    p.add_argument("--selector-calibration-support-max-distance", type=float,
                   default=1.5,
                   help="maximum normalized support distance for static CalMap queries")
    p.add_argument("--selector-system-outcome-table",
                   default=str(SYSTEM_OUTCOME_TABLE_PATH),
                   help="independent system-level safety calibration table")
    p.add_argument("--selector-require-system-safety-evidence", type=int,
                   choices=(0, 1), default=0,
                   help="deploy preset only with independent DART-S-vs-DART safety evidence")
    p.add_argument("--selector-system-safety-max-distance", type=float,
                   default=0.75,
                   help="maximum normalized distance to independent system evidence")
    p.add_argument(
        "--selector-landing-slope-uncertainty-deg",
        type=float,
        default=1.0,
        help="bounded landing-plane angle error consumed by strict_v2",
    )
    p.add_argument("--gate-theta-l-deg", type=float, default=0.0)
    p.add_argument("--gate-theta-bar-deg", type=float, default=0.0,
                   help="airborne pitch terminal-set half width; 0 keeps exact-pair certificate")
    p.add_argument("--gate-a-pitch-dps2", type=float, default=135.0)
    p.add_argument("--gate-omega-bar-dps", type=float, default=5.0)
    p.add_argument("--gate-wheel-iw", type=float, default=1.2)
    p.add_argument("--gate-wheel-omega-max-radps", type=float,
                   default=SUSTAINED_EFFECTIVE_ENVELOPE_RADPS,
                   help="conservative sustained-effective certificate envelope")
    p.add_argument("--gate-wheel-omega-min-radps", type=float,
                   default=-SUSTAINED_EFFECTIVE_ENVELOPE_RADPS)
    p.add_argument("--gate-j-y", type=float, default=2043.0)
    p.add_argument("--wheel-r", type=float, default=0.36)
    p.add_argument("--wheel-storage-hard-limit-radps", type=float,
                   default=DRIVETRAIN_HARD_LIMIT_RADPS,
                   help="physical drivetrain storage limit; audit only")
    p.add_argument("--approach-wheel-spin-guard-radps", type=float,
                   default=OPERATIONAL_SPIN_GUARD_RADPS,
                   help="shared ground-phase wheel-slip guard; <=0 disables")
    p.add_argument("--dart-pitch-rate-kp", type=float, default=2.0)
    p.add_argument("--dart-rate-max-dps", type=float, default=25.0)
    p.add_argument("--dart-drive-gain", type=float, default=0.4)
    p.add_argument("--dart-engage-air-steps", type=int, default=6)
    p.add_argument("--dart-wheel-speed-cap-radps", type=float,
                   default=OPERATIONAL_SPIN_GUARD_RADPS,
                   help="runtime spin-up guard; <=0 disables for historical reproduction")
    p.add_argument("--post-land-sec", type=float, default=2.0,
                   help="post-touchdown stability observation window, at least 0.6 s")
    p.add_argument("--preset-scale", type=float, default=1.0,
                   help="preset amplitude scale; 0 is neutral, 1 is the nominal full preset")
    p.add_argument("--phase-sweep-leads", type=float, nargs="*", default=[],
                   help="DART-S release-phase scout: lip-relative partial-release pulse leads (s). "
                        "When set, arms become neutral/static nose-up plus one variant per lead.")
    p.add_argument("--phase-pulse-duration-sec", type=float, default=0.12,
                   help="partial-release command duration for each phase variant")
    p.add_argument("--phase-pulse-release-frac", type=float, default=0.50,
                   help="fraction of the full preset temporarily released toward neutral")
    p.add_argument("--susp-stroke-trace", type=int, choices=(0, 1), default=0,
                   help="record four-corner spring lengths around the lip")
    p.add_argument("--phase-trace-prelip-m", type=float, default=25.0,
                   help="start four-corner stroke trace this far before the lip")
    p.add_argument("--impact-trace", type=int, choices=(0, 1), default=0,
                   help="record per-step vz/pz/nc from takeoff through the post-landing "
                        "window for offline filtered-impact recomputation")
    p.add_argument("--per-step-state-trace", type=int, choices=(0, 1), default=0,
                   help="record full ground/air/post state and per-wheel trajectories")
    p.add_argument("--impact-imu-hz", type=int, default=0,
                   help="M2: attach an AdvancedIMU high-rate channel at this rate "
                        "(e.g. 1000) and record imu-local accel rows in the same "
                        "takeoff->post-landing window, for offline SAE J211 CFC-60 "
                        "filtering. 0 = off. Observational only (not in the protocol "
                        "fingerprint).")
    p.add_argument("--resume-checkpoint", type=int, choices=(0, 1), default=0,
                   help="atomically checkpoint every completed run and resume matching protocol")
    p.add_argument("--randomize-arm-order", type=int, choices=(0, 1), default=1,
                   help="deterministically shuffle the arm order each roll to remove order effects")
    p.add_argument("--arm-order-seed", type=int, default=20260731)
    p.add_argument("--rolls", type=int, default=5)
    p.add_argument("--arms", nargs="*", default=["neutral", "nose_up", "nose_down", "torque_up"])
    p.add_argument("--lip-zone", type=float, default=12.0)
    p.add_argument("--rise", type=float, default=3.0)
    p.add_argument("--fillet-len-max", type=float, default=15.0)
    p.add_argument("--fillet-rise-budget", type=float, default=3.0)
    p.add_argument("--width", type=float, default=36.0)
    p.add_argument("--thick", type=float, default=0.8)
    p.add_argument("--n-fillet", type=int, default=14)
    p.add_argument("--n-main", type=int, default=3)
    p.add_argument("--overlap", type=float, default=1.10)
    p.add_argument("--sink", type=float, default=0.05)
    p.add_argument("--base-x", type=float, default=50.0)
    p.add_argument("--run-up", type=float, default=45.0)
    p.add_argument("--max-steps", type=int, default=1400)
    p.add_argument("--tag", default="ab1")
    p.add_argument("--jump-scenario", default=None,
                   help="load a jump-scenario library entry (e.g. darts_a20r7_valley12) for its geometry")
    args = p.parse_args(argv)
    if not 0.0 <= float(args.preset_scale) <= 1.0:
        raise SystemExit("--preset-scale must be in [0, 1]")
    if float(args.lip_cut_speed_bias_mps) < 0:
        raise SystemExit("--lip-cut-speed-bias-mps must be >= 0")
    if float(args.selector_calibration_support_max_distance) <= 0:
        raise SystemExit("--selector-calibration-support-max-distance must be > 0")
    if float(args.selector_system_safety_max_distance) <= 0:
        raise SystemExit("--selector-system-safety-max-distance must be > 0")
    if args.neural_calmap_collection != "off":
        if not args.session_id or not args.launch_id:
            raise SystemExit(
                "Neural CalMap collection requires --session-id and --launch-id")
        if (
            args.neural_calmap_collection == "model"
            and int(args.per_step_state_trace) != 1
        ):
            raise SystemExit(
                "Neural CalMap model collection requires --per-step-state-trace 1")
    arm_specs = dict(ARMS)
    if abs(float(args.dart_s_no_cut_speed_offset_mps)) > 1e-12:
        arm_specs["dart_s_no_cut"] = {
            **ARMS["dart_s_no_cut"],
            "entry_speed_offset_mps": float(args.dart_s_no_cut_speed_offset_mps),
        }
    phase_variant_map = {}
    cut_speed_variant_map = {}
    nocut_speed_variant_map = {}
    n_sweeps = sum(bool(x) for x in (
        args.phase_sweep_leads, args.cut_speed_bias_sweep,
        args.nocut_speed_offset_sweep))
    if n_sweeps > 1:
        raise SystemExit("phase/cut-bias/no-cut-offset sweeps cannot be combined")
    if args.cut_speed_bias_sweep:
        try:
            variants, cut_speed_variant_map = build_cut_speed_arm_specs(
                args.cut_speed_bias_sweep)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        arm_specs.update(variants)
        args.arms = [
            "dart_only", "dart_s_no_cut",
            *[name for item in cut_speed_variant_map.values()
              for name in (item["cut_only_arm"], item["system_arm"])],
        ]
        args.resume_checkpoint = 1
        print(f"[darts-cut-speed] variants={cut_speed_variant_map}; arms={args.arms}",
              flush=True)
    if args.nocut_speed_offset_sweep:
        try:
            variants, nocut_speed_variant_map = build_nocut_speed_arm_specs(
                args.nocut_speed_offset_sweep)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        arm_specs.update(variants)
        args.arms = ["dart_only", *[x["arm"] for x in nocut_speed_variant_map.values()]]
        args.resume_checkpoint = 1
        print(f"[darts-nocut-speed] variants={nocut_speed_variant_map}; arms={args.arms}",
              flush=True)
    if args.phase_sweep_leads:
        leads = [float(v) for v in args.phase_sweep_leads]
        if len(set(leads)) != len(leads):
            raise SystemExit("--phase-sweep-leads must be unique")
        if any(v < 0.05 or v > 2.5 for v in leads):
            raise SystemExit("--phase-sweep-leads must lie in [0.05, 2.5] s")
        # Validate shared pulse parameters before launching BeamNG.
        for lead in leads:
            ReleasePulseConfig(
                lead_sec=lead,
                duration_sec=float(args.phase_pulse_duration_sec),
                release_fraction=float(args.phase_pulse_release_frac),
                dt_sec=DT,
            )
            name = f"phase_{int(round(lead * 100)):03d}"
            if name in arm_specs:
                raise SystemExit(f"phase variant name collision: {name}")
            arm_specs[name] = {
                **ARMS["nose_up"],
                "phase_pulse_lead_sec": lead,
            }
            phase_variant_map[name] = lead
        args.arms = ["neutral", "nose_up", *phase_variant_map]
        args.susp_stroke_trace = 1
        args.resume_checkpoint = 1
        print(f"[darts-phase] variants={phase_variant_map}; arms={args.arms}; "
              f"pulse={args.phase_pulse_release_frac:.2f}×/{args.phase_pulse_duration_sec:.2f}s",
              flush=True)
    args.phase_variant_map = phase_variant_map
    args.cut_speed_variant_map = cut_speed_variant_map
    args.nocut_speed_variant_map = nocut_speed_variant_map

    if args.jump_scenario:
        _applied = jlib.apply_to_args(args, args.jump_scenario)
        if hasattr(args, "angles") and isinstance(args.angles, (list, tuple)) and len(args.angles) > 0:
            args.angle = args.angles[0]
        print(f"[darts] loaded jump scenario {args.jump_scenario!r}; applied keys: {_applied}", flush=True)

    bad = [a for a in args.arms if a not in arm_specs]
    if bad:
        raise SystemExit(f"unknown arm: {bad}, choose from {list(arm_specs)}")
    if (
        args.neural_calmap_collection == "model"
        and set(args.arms)
        != {"dart_only", "dart_s_forced", "dart_timed_forced"}
    ):
        raise SystemExit(
            "Neural CalMap model collection requires exactly "
            "dart_only/dart_s_forced/dart_timed_forced")
    # Steep ramps and high entry speeds need a longer run-up to reach target speed
    if args.angle >= 30 and args.run_up < 60.0:
        args.run_up = 60.0
        print(f"[darts] angle={args.angle} deg >=30 -> run_up extended to {args.run_up} m", flush=True)
    if max(args.speeds) >= 16.0 and args.run_up < 60.0:
        args.run_up = 60.0
        print(f"[darts] max speed >=16 -> run_up extended to {args.run_up} m", flush=True)

    _, _, R = bench.ramp_speeds(args.rise)
    aidx = bench.ANGLES.index(args.angle) if args.angle in bench.ANGLES else 5
    R_flight = R[aidx]
    from beamngpy import Scenario, Vehicle  # type: ignore
    from beamngpy.sensors import Electrics, State  # type: ignore
    from data_pipeline.beamng_session import (
        BeamNGSessionConfig, BeamNGSession, wait_scenario_ready, ensure_freerun)
    cfg = BeamNGSessionConfig.from_env(
        home=(BeamNGSessionConfig.from_env().home or r"C:\BeamNG.tech"),
        linux_binary=None, open_timeout_sec=240)
    syaw = math.radians(270.0); spawn_rot = (0.0, 0.0, math.sin(syaw / 2), math.cos(syaw / 2))
    ART.mkdir(parents=True, exist_ok=True)
    outp = ART / f"darts_bench_{args.tag}.json"
    checkpoint_path = ART / f"darts_bench_{args.tag}.checkpoint.json"
    protocol_fingerprint = _phase_protocol_fingerprint(args, arm_specs)
    data = {}
    resumed_from_checkpoint = False
    checkpoint_loaded_run_count = 0
    if int(args.resume_checkpoint) and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("protocol_fingerprint") != protocol_fingerprint:
            raise SystemExit(
                f"checkpoint protocol mismatch: {checkpoint_path}; remove or use a new --tag")
        data = checkpoint.get("data") or {}
        resumed_from_checkpoint = True
        checkpoint_loaded_run_count = sum(
            len(rs) for vd in data.values() for rs in vd.values())
        print(f"[darts-ckpt] resumed {sum(len(rs) for vd in data.values() for rs in vd.values())} "
              f"runs from {checkpoint_path}", flush=True)

    def write_checkpoint() -> None:
        if not int(args.resume_checkpoint):
            return
        doc = {
            "protocol_fingerprint": protocol_fingerprint,
            "params": vars(args),
            "data": data,
            "updated_unix": time.time(),
        }
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(checkpoint_path)

    with BeamNGSession(cfg) as bng:
        try: bng.settings.set_deterministic(100)
        except Exception: pass
        sc = Scenario("smallgrid", "darts_bench")
        veh = Vehicle("ego", model="sbr", part_config=args.pc)
        try:
            veh.attach_sensor("state", State())
            veh.attach_sensor("electrics", Electrics())
        except Exception: pass
        sc.add_vehicle(veh, pos=(args.base_x - args.run_up, 0.0, 0.3), rot_quat=spawn_rot)
        sc.make(bng)
        def qlua(c): return bng.queue_lua_command(c, response=True)
        bng.scenario.load(sc); bng.scenario.start()
        wait_scenario_ready(bng, expected_vid="ego", timeout_sec=20.0); ensure_freerun(bng)
        for _ in range(4): bench.force_gameplay(bng); rf._step(bng, 4)
        if not bench.ensure_gameplay_live(bng, veh):
            print("[darts] ABORT(exit 6): BeamNG wedged in the menu. Kill everything and rerun.", flush=True)
            return 6
        # SR-18 spawn liveness gate: only continue while the vehicle Lua channel answers
        alive = nj._vlua(veh, "return 'alive'")
        if str(alive).strip().strip("'\"") != "alive":
            print(f"[darts] ABORT(exit 7): vehicle Lua channel is dead (alive={alive!r}). "
                  "taskkill every BeamNG process and rerun.", flush=True)
            return 7

        # High-rate IMU channel for offline CFC filtering. Observation only, so it is
        # not part of the protocol fingerprint; attached once per session and drained
        # by one_run on every roll.
        imu = None
        if int(getattr(args, "impact_imu_hz", 0) or 0) > 0:
            from beamngpy.sensors import AdvancedIMU
        # Mount low, near the CoM: the default (0,0,1.7) top node reads a ~80 Hz local
        # soft-body mode at +/-1500 m/s^2 that swamps the whole-vehicle deceleration
        # signal.
            imu = AdvancedIMU(
                "m2imu", bng, veh,
                physics_update_time=1.0 / float(args.impact_imu_hz),
                pos=(0.0, 0.0, 0.5),
                is_send_immediately=False,
                is_using_gravity=True,
                is_visualised=False,
                is_snapping_desired=True,
                is_allow_wheel_nodes=False,
            )
            print(f"[darts] M2: AdvancedIMU attached @ {args.impact_imu_hz} Hz "
                  f"(gravity included, imu-local frame)", flush=True)

        # build the ramp once
        pts = _ramp_pts_for(args, args.base_x)
        segs = bench.ramp_segments(pts, 0, args.width, args.thick, args.overlap)
        peak_x, peak_z = pts[-1]
        if int(args.runup_bumps) > 0:
            bsegs = _bump_segments(args)
            segs = segs + bsegs
            print(f"[darts] washboard: {len(bsegs)} planks h={args.bump_height} m "
                  f"x∈[{bsegs[-1]['pos'][0]:.1f},{bsegs[0]['pos'][0]:.1f}] "
                  f"(base={args.base_x})", flush=True)

        # landing area from the scenario library
        lsegs = []
        R_flight_scene = None
        if args.jump_scenario:
            a = math.radians(args.angle)
            v0 = max(args.speeds) if args.speeds else float(args.v_entry)
            # approach mode: size landing surface for the actual entry speed, not the
            # template ballistic_v0 (which is for air-impulse mode).
            if getattr(args, "launch_mode", "approach") != "approach" and float(args.ballistic_v0) > 0:
                v0 = float(args.ballistic_v0)
            gamma = (math.degrees(a) - float(args.lip_sweep_deg)) if getattr(args, "ramp_mode", "ramp") == "kicker" else math.degrees(a)
            beta = abs(float(args.landing_slope_deg))
            _cross_phi = float(getattr(args, "landing_cross_slope_deg", 0.0))
            _post_rg = _post_root_ground(runup_camber_deg=float(args.runup_camber_deg),
                                          cross_slope_deg=_cross_phi)
            if args.landing_slope_mode == "valley":
                _lip_exit = math.degrees(math.atan2(pts[-1][1] - pts[-2][1],
                                                    pts[-1][0] - pts[-2][0])) if len(pts) >= 2 else 0.0
                lsegs, lpts, _meta = valley_landing_segments(
                    peak_x, peak_z, v0, gamma, beta,
                    clearance=float(args.ballistic_clearance),
                    floor_depth=float(args.valley_floor_depth),
                    length_req=float(args.landing_slope_len),
                    width=args.width, thick=args.thick, ri=0, visual=True,
                    feather_len=float(args.valley_feather_len),
                    floor_run=float(args.valley_floor_run),
                    lip_exit_slope_deg=_lip_exit,
                    crest_blend_len=float(args.valley_crest_len),
                    post_root_ground=_post_rg)
                if not _meta.get("infeasible"):
                    R_flight_scene = float(_meta.get("dx_land", 0) or 0)
            elif args.landing_slope_mode == "gap":
                lsegs, lpts, _meta = gap_landing_segments(
                    peak_x, peak_z, v0, gamma,
                    clearance_catch=float(args.ballistic_clearance),
                    gap_max=float(args.landing_gap_max),
                    face_max_deg=float(args.dune_face_deg),
                    length_req=float(args.landing_slope_len),
                    width=args.width, thick=args.thick, ri=0, visual=True,
                    post_root_ground=_post_rg)
            elif args.landing_slope_mode == "gap-ramp":
                lsegs, lpts, _meta = gap_ramp_landing_segments(
                    peak_x, peak_z, v0, gamma, beta,
                    clearance=float(args.ballistic_clearance),
                    gap_max=float(args.landing_gap_max),
                    face_max_deg=float(args.dune_face_deg),
                    front_bank_deg=float(args.landing_front_bank_deg),
                    run_req=float(args.landing_gap_run),
                    width=args.width, thick=args.thick, ri=0, visual=True,
                    post_root_ground=_post_rg)
            elif args.landing_slope_mode == "ballistic":
                lsegs, lpts, _meta = ballistic_landing_segments(
                    peak_x, peak_z, v0, gamma,
                    clearance=float(args.ballistic_clearance),
                    length=float(args.landing_slope_len),
                    width=args.width, thick=args.thick, ri=0, visual=True,
                    post_root_ground=_post_rg)
            else:
                # fixed / auto-to-flight
                lsegs = landing_slope_segments(
                    peak_x, peak_z, -beta,
                    length=float(args.landing_slope_len),
                    width=args.width, thick=args.thick, ri=0, visual=True,
                    post_root_ground=_post_rg)

            if abs(_cross_phi) > 1e-6 and lsegs:
                lsegs = bank_landing_segments(lsegs, _cross_phi, pivot_z=0.0)
            if abs(float(args.runup_camber_deg)) > 1e-6:
                _pivot_y = _camber_pivot_y(args)
                segs = bank_landing_segments(segs, float(args.runup_camber_deg),
                                             pivot_y=_pivot_y, pivot_z=0.0)
                if lsegs:
                    lsegs = bank_landing_segments(lsegs, float(args.runup_camber_deg),
                                                  pivot_y=_pivot_y, pivot_z=0.0)

        ground_type = getattr(args, "runup_ground_type", "ASPHALT")
        takeoff_segments = apply_ramp_material(list(segs), ground_type)
        runup_pad = build_runup_pad_segments(
            base_x=args.base_x,
            run_up=args.run_up,
            width=args.width,
            thick=args.thick,
            ground_type=ground_type,
        )
        place_ramp_with_ground(
            bng,
            runup_pad + takeoff_segments + lsegs,
            qlua,
            ground_type,
        )
        rf._step(bng, 3)
        if R_flight_scene and R_flight_scene > 1.0:
            R_flight = R_flight_scene

        # Interleave speed x roll x arm so drift over the session cancels in the pairs
        import random as _rnd
        for v in args.speeds:
            vkey = f"v{v:g}"
            vdata = data.setdefault(vkey, {})
            for k in range(args.rolls):
                # Entry jitter: one draw per roll, shared by every arm, seeded and reproducible
                if args.v_jitter > 0:
                    rng = _rnd.Random(int(args.angle) * 10007 + int(v * 100) * 101 + k)
                    v_run = v + rng.uniform(-args.v_jitter, args.v_jitter)
                else:
                    v_run = v
                arm_order = list(args.arms)
                if int(args.randomize_arm_order):
                    order_rng = _rnd.Random(
                        int(args.arm_order_seed) + int(args.angle * 1000)
                        + int(v * 100) * 101 + k * 1009)
                    order_rng.shuffle(arm_order)
                for order_idx, arm in enumerate(arm_order):
                    prior = [
                        r for r in vdata.get(arm, [])
                        if (r.get("protocol") or {}).get("roll_index") == k]
                    if prior:
                        print(f"[darts-ckpt] SKIP {vkey} roll={k} arm={arm}", flush=True)
                        continue
                    run_spec = dict(arm_specs[arm])
                    run_spec["_roll_index"] = k
                    r = one_run(bng, veh, arm=arm, spec=run_spec, base_x=args.base_x,
                                run_up=args.run_up, v_entry=v_run, R_flight=R_flight, args=args,
                                imu=imu)
                    r["protocol"] = {
                        "tag": args.tag,
                        "session_id": args.session_id,
                        "launch_id": args.launch_id,
                        "runner_resumed_from_checkpoint": resumed_from_checkpoint,
                        "checkpoint_loaded_run_count": checkpoint_loaded_run_count,
                        "roll_index": k,
                        "arm_order_index": order_idx,
                        "arm_order": arm_order,
                        "command_v_mps": v,
                        "actual_command_v_mps": v_run,
                    }
                    if args.neural_calmap_collection != "off":
                        r["neural_calmap_data_audit"] = audit_neural_calmap_run(
                            r,
                            expected_session_id=args.session_id,
                            collection_mode=args.neural_calmap_collection,
                        )
                        if not r["neural_calmap_data_audit"]["passed"]:
                            raise RuntimeError(
                                "Neural CalMap prospective data gate failed: "
                                + json.dumps(
                                    r["neural_calmap_data_audit"],
                                    sort_keys=True,
                                )
                            )
                    vdata.setdefault(arm, []).append(r)
                    write_checkpoint()
                    ts = r.get("takeoff_state") or {}
                    gtxt = ""
                    if r.get("gate"):
                        g = r["gate"]
                        gtxt = (f" GATE={g['decision']}"
                                + (f" abort_stop={g['stopped_before_lip']} margin={g['stop_margin_m']}m"
                                   if g["aborted"] else ""))
                    print(f"[darts] {vkey} roll={k} arm={arm} took_off={r['took_off']} "
                          f"ground_dpitch={r['ground_dpitch_deg']}° prelip={r['prelip_pitch_deg']}° "
                          f"theta0={ts.get('theta0_deg')}° omega_y0={ts.get('omega_y0_dps')}°/s v0={ts.get('v0_mps')} "
                          f"T={r['T_flight']}s land={r['land_pitch_deg']}° az={r['az_peak_mps2']} "
                          f"dt_eff={r['dt_eff']}{gtxt}", flush=True)

        summary = {}; paired = {}
        for vkey, vdata in data.items():
            summary[vkey] = {arm: arm_summary(runs) for arm, runs in vdata.items()}
            print(f"[darts] === medians {vkey} angle={args.angle} deg ===", flush=True)
            for arm, s in summary[vkey].items():
                print(f"  {arm}: takeoff {s['n_takeoff']}/{s['n']} safe_v2={s['n_safe_v2']}/{s['n']} "
                      f"ground_dpitch={s['ground_dpitch_med']}° "
                      f"prelip={s['prelip_pitch_med']}° theta0={s['theta0_med']}° omega_y0={s['omega_y0_med']}°/s "
                      f"v0={s['v0_med']} wheel_w0={s['wheel_w0_med']} T={s['T_flight_med']}s land={s['land_pitch_med']}° "
                      f"az={s['az_peak_med']}", flush=True)
            base = summary[vkey].get("neutral")
            if not base or not base["n_takeoff"]:
                continue
            paired[vkey] = {}
            base_runs = vdata.get("neutral", [])
            for arm, s in summary[vkey].items():
                if arm == "neutral" or not s["n_takeoff"]:
                    continue
                arm_runs = vdata.get(arm, [])
                # per-roll paired differences (matched by roll index) feeding the sign test
                land_diffs = [
                    (a["land_pitch_deg"] - b["land_pitch_deg"])
                    for a, b in zip(arm_runs, base_runs)
                    if a.get("land_pitch_deg") is not None and b.get("land_pitch_deg") is not None]
                paired[vkey][arm] = {
                    "d_theta0": (round(s["theta0_med"] - base["theta0_med"], 2)
                                 if None not in (s["theta0_med"], base["theta0_med"]) else None),
                    "d_omega_y0": (round(s["omega_y0_med"] - base["omega_y0_med"], 1)
                                   if None not in (s["omega_y0_med"], base["omega_y0_med"]) else None),
                    "d_land_pitch": (round(s["land_pitch_med"] - base["land_pitch_med"], 2)
                                     if None not in (s["land_pitch_med"], base["land_pitch_med"]) else None),
                    "d_T_flight": (round(s["T_flight_med"] - base["T_flight_med"], 2)
                                   if None not in (s["T_flight_med"], base["T_flight_med"]) else None),
                    "d_az_peak": (round(s["az_peak_med"] - base["az_peak_med"], 1)
                                  if None not in (s["az_peak_med"], base["az_peak_med"]) else None),
                    "sign_p_land": sign_test_p(land_diffs),
                }
                print(f"[darts] === paired diff {vkey} {arm} - neutral === "
                      f"{json.dumps(paired[vkey][arm])}", flush=True)

        dt_med = med(r["dt_eff"] for vd in data.values() for runs in vd.values()
                     for r in runs if r.get("dt_eff"))
        neural_audits = [
            r.get("neural_calmap_data_audit")
            for vd in data.values()
            for runs in vd.values()
            for r in runs
            if r.get("neural_calmap_data_audit") is not None
        ]
        out = {"params": vars(args), "data": data, "summary": summary, "paired_vs_neutral": paired,
               "runup_ground_audit": runup_ground_audit(args.runup_ground_type),
               "takeoff_state_schema_version": 3,
               "release_phase_schema_version": 1 if args.phase_sweep_leads else None,
               "protocol_fingerprint": protocol_fingerprint,
               "phase_source_sha256": {
                   "scripts/darts_bench.py": hashlib.sha256(
                       Path(__file__).read_bytes()).hexdigest(),
                   "control/darts/susp_release_phase.py": hashlib.sha256(
                       (REPO / "control/darts/susp_release_phase.py").read_bytes()).hexdigest(),
               },
               "safe_criterion": safety_schema(),
               "median_fn": "statistics.median",
               "neural_calmap_collection_audit": {
                   "mode": args.neural_calmap_collection,
                   "n": len(neural_audits),
                   "n_passed": sum(bool(audit["passed"]) for audit in neural_audits),
                   "n_model_eligible": sum(
                       bool(audit["model_eligible"]) for audit in neural_audits),
                   "all_passed": all(
                       bool(audit["passed"]) for audit in neural_audits),
               },
               "_provenance": make_provenance(
                   "deterministic_stepped", sps=100, dt_eff=dt_med,
                   note="DART-S takeoff A/B; SR-17 pause-stepped; dt_eff from ballistic calibration; "
                        "T_flight=steps x 0.01 s; landing latched at first contact")}
        outp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print(f"[darts] saved {outp}", flush=True)
        if imu is not None:
            try:
                imu.remove()
                print("[darts] M2: AdvancedIMU removed", flush=True)
            except Exception as exc:  # a reused session drops sensors on scenario reload
                print(f"[darts] M2: IMU remove failed ({exc}); scenario reload will clear it", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
