"""Preset-aware takeoff gating: suspension preconditioning meets reachability.

This module adds a suspension-preset bias term on top of the reachability
certificate in :mod:`control.darts.reachability`. Presetting the four corner
heights before the lip shifts the realised takeoff state from its natural
value ``(theta0, omega_y0)`` to ``(theta0 + dtheta0, omega_y0 + domega_y0)``,
which is equivalent to translating the feasible takeoff set. A landing-end
preset (damper mode or travel) instead widens the terminal tolerance.

The bias magnitudes come from two sources, in order of precedence:

* Closed-loop calibration. Paired ``nose_up - neutral`` differences across
  ramp angles are fitted and stored in
  ``calibration/preset_effect_fit.json``; :func:`preset_effect` reads that
  file when it exists.
* Otherwise the measured anchors in :data:`DEFAULT_ANCHORS_NOSE_UP`. As an
  analytic lower bound the static geometry term is
  ``dtheta0 ~ arctan((dz_f - dz_r) / L)``, and the actuator probe measures
  +/-3.5 deg of static pitch authority over the +/-15% corner travel.

Pure functions, no simulator or tensor dependencies.
"""
from __future__ import annotations

import json
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from control.darts.calibration_map import ActionCalibrationMap
from control.darts.reachability import PitchReachability, TakeoffReachableSet
from control.darts.system_outcome_selector import query_system_outcome_table
from control.darts.susp_release_phase import ReleasePulseConfig

_REPO = Path(__file__).resolve().parents[2]
FIT_PATH = _REPO / "calibration/preset_effect_fit.json"

# Measured closed-loop anchors, {angle_deg: (dtheta0_deg, domega_y0_dps)}.
# The fitted file overrides these whenever it is present.
DEFAULT_ANCHORS_NOSE_UP = {
    20.0: (3.1, 6.0),
    35.0: (2.8, 7.0),
}


@dataclass(frozen=True)
class SuspPresetEffect:
    """Takeoff-state bias and terminal-tolerance relief from a preset."""

    dtheta0: float = 0.0          # takeoff pitch bias (rad)
    domega_y0: float = 0.0        # takeoff pitch-rate bias (rad/s)
    dtheta_land_tol: float = 0.0  # terminal pitch tolerance added by a landing preset (rad)

    @classmethod
    def from_deg(cls, dtheta0_deg: float, domega_y0_dps: float,
                 dtheta_land_tol_deg: float = 0.0) -> "SuspPresetEffect":
        return cls(math.radians(dtheta0_deg), math.radians(domega_y0_dps),
                   math.radians(dtheta_land_tol_deg))


def _interp_anchor(anchors: dict[float, tuple[float, float]], angle_deg: float) -> tuple[float, float]:
    """Interpolate linearly between anchors; clamp rather than extrapolate."""
    ks = sorted(anchors)
    if angle_deg <= ks[0]:
        return anchors[ks[0]]
    if angle_deg >= ks[-1]:
        return anchors[ks[-1]]
    for lo, hi in zip(ks, ks[1:]):
        if lo <= angle_deg <= hi:
            f = (angle_deg - lo) / (hi - lo)
            a, b = anchors[lo], anchors[hi]
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
    return anchors[ks[-1]]


def preset_effect(arm: str, angle_deg: float, fit_path: Path | str | None = None) -> SuspPresetEffect:
    """Return the preset bias for ``arm`` at a given ramp angle.

    ``arm`` is one of ``neutral``, ``nose_up`` or ``nose_down``. The fitted
    file takes precedence over :data:`DEFAULT_ANCHORS_NOSE_UP`. Unless the
    fit carries its own ``nose_down`` anchors, that arm is taken as the sign
    flip of ``nose_up``, which the paired A/B data supports.
    """
    if arm == "neutral":
        return SuspPresetEffect()
    p = Path(fit_path) if fit_path else FIT_PATH
    anchors = None
    if p.exists():
        try:
            fit = json.loads(p.read_text(encoding="utf-8"))
            arm_fit = fit.get("arms", {}).get(arm) or fit.get("arms", {}).get("nose_up")
            if arm_fit:
                anchors = {float(k): tuple(v) for k, v in arm_fit["anchors"].items()}
        except Exception:
            anchors = None
    sign = 1.0
    if anchors is None:
        anchors = DEFAULT_ANCHORS_NOSE_UP
        if arm == "nose_down":
            sign = -1.0
    elif arm == "nose_down" and "nose_down" not in json.loads(p.read_text(encoding="utf-8")).get("arms", {}):
        sign = -1.0
    dth, dom = _interp_anchor(anchors, float(angle_deg))
    return SuspPresetEffect.from_deg(sign * dth, sign * dom)


@dataclass(frozen=True)
class PresetAwareTakeoffSet:
    """Reachability gate evaluated at the preset takeoff state.

    Equivalently, the feasible set is translated by ``(-dtheta0, -domega_y0)``
    at the takeoff end and the terminal tolerance widened at the landing end.
    :meth:`contains_natural` takes the natural, un-preset prediction and adds
    the bias internally, so the question the gate answers changes from "is
    this takeoff state acceptable" to "will it be acceptable once preset".
    """

    base: TakeoffReachableSet
    effect: SuspPresetEffect

    def query_natural(
        self,
        theta0: float,
        omega_y0: float,
        *,
        B_up: float | None = None,
        B_down: float | None = None,
    ) -> PitchReachability:
        """Query the shifted state with optional arm-specific budgets.

        ``B_up/B_down`` must come from the candidate arm's takeoff wheel speeds;
        omitting them preserves the base-set budget for legacy callers.
        """
        if (B_up is None) != (B_down is None):
            raise ValueError("B_up and B_down must be provided together")
        eff_theta = theta0 + self.effect.dtheta0
        eff_omega = omega_y0 + self.effect.domega_y0
        # Terminal tolerance relief. The certificate treats theta_L as a
        # point target, so widening the tolerance by tol is equivalent to
        # moving theta_L towards theta0 by min(tol, |e|).
        theta_L = self.base.theta_L
        tol = self.effect.dtheta_land_tol
        if tol > 0.0:
            e = eff_theta - theta_L
            theta_L = theta_L + math.copysign(min(tol, abs(e)), e)
        shifted = TakeoffReachableSet(
            theta_L=theta_L, T=self.base.T, a=self.base.a,
            H_max=self.base.H_max, J_y=self.base.J_y,
            B_up=self.base.B_up if B_up is None else B_up,
            B_down=self.base.B_down if B_down is None else B_down,
            omega_bar=self.base.omega_bar,
            theta_bar=self.base.theta_bar)
        return shifted.query(eff_theta, eff_omega)

    def contains_natural(self, theta0: float, omega_y0: float) -> bool:
        return self.query_natural(theta0, omega_y0).reachable


def go_nogo_with_preset(
    theta0: float, omega_y0: float, base: TakeoffReachableSet,
    arm: str = "nose_up", angle_deg: float = 20.0,
    effect: Optional[SuspPresetEffect] = None,
) -> dict:
    """Compare the gate with and without the preset.

    ``flipped`` marks the states the natural gate refuses but the preset gate
    admits, which is the net gain of preparing the vehicle over only being
    able to say no.
    """
    eff = effect or preset_effect(arm, angle_deg)
    natural = base.query(theta0, omega_y0)
    preset = PresetAwareTakeoffSet(base=base, effect=eff).query_natural(theta0, omega_y0)
    return {
        "no_preset_go": natural.reachable,
        "preset_go": preset.reachable,
        "flipped": preset.reachable and not natural.reachable,
        "effect_deg": {"dtheta0": round(math.degrees(eff.dtheta0), 2),
                       "domega_y0": round(math.degrees(eff.domega_y0), 1)},
        "natural_margin": natural.margin,
        "preset_margin": preset.margin,
    }


GATE_TABLE_PATH = _REPO / "calibration/gate_table.json"
SYSTEM_OUTCOME_TABLE_PATH = _REPO / "calibration/system_outcome_table.json"
VALID_GATE_ARMS = frozenset({"neutral", "nose_up", "nose_down", "avoid_preset"})
TIMING_MODE = "partial_release_return"


def canonical_release_timing(entry: dict) -> tuple[dict | None, str | None]:
    """Validate and canonicalize an optional timing-conditioned gate action.

    The deployable variable is the nominal pulse schedule, not measured
    rebound phase.  Invalid timing metadata fails closed in ``recommend_arm``.
    """
    raw = entry.get("release_timing")
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "release_timing_not_object"
    if raw.get("mode") != TIMING_MODE:
        return None, f"unsupported_timing_mode({raw.get('mode')!r})"
    if raw.get("final_arm", "nose_up") != "nose_up":
        return None, "timing_final_arm_not_nose_up"
    try:
        cfg = ReleasePulseConfig(
            lead_sec=float(raw["lead_sec"]),
            duration_sec=float(raw["duration_sec"]),
            release_fraction=float(raw["release_fraction"]),
        )
        max_dist = float(raw.get("max_query_distance", 0.25))
        if not math.isfinite(max_dist) or max_dist < 0:
            raise ValueError("max_query_distance")
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"invalid_release_timing({exc})"
    return {
        "mode": TIMING_MODE,
        "lead_sec": cfg.lead_sec,
        "duration_sec": cfg.duration_sec,
        "release_fraction": cfg.release_fraction,
        "final_arm": "nose_up",
        "restore_before_takeoff_required": bool(
            raw.get("restore_before_takeoff_required", True)),
        "max_query_distance": max_dist,
        "source": raw.get("source"),
        "protocol_fingerprint": raw.get("protocol_fingerprint"),
    }, None

# Curvature-saturation veto: as n(v) = g cos(alpha) - v^2 / R_lip approaches
# zero the front axle unloads early and the preset becomes adverse. The
# threshold n(v) <= n_min is calibrated on the R = 35 m ladder and does not
# transfer across lip radii.
G_MPS2 = 9.81
CURVATURE_N_MIN = 2.0
ROUGH_VETO_MIN_BUMPS = 6
ROUGH_VETO_MIN_HEIGHT_M = 0.08
ROUGH_VETO_MIN_SPACING_M = 1.5


@dataclass(frozen=True)
class SelectorUncertainty:
    """Bounded online-input uncertainty used for fail-closed deployment."""

    angle_half_width_deg: float = 0.0
    speed_half_width_mps: float = 0.0
    require_safety_v2_evidence: bool = False

    def __post_init__(self):
        if self.angle_half_width_deg < 0.0:
            raise ValueError("angle_half_width_deg must be >= 0")
        if self.speed_half_width_mps < 0.0:
            raise ValueError("speed_half_width_mps must be >= 0")


def lip_curvature_margin(v_entry: float, lip_radius: float, angle_deg: float,
                         g: float = G_MPS2) -> float:
    """Residual normal acceleration on a convex lip, in m/s^2.

    ``n(v) = g cos(alpha) - v^2 / R_lip``; at or below zero the vehicle is
    fully unloaded along the arc.
    """
    return g * math.cos(math.radians(angle_deg)) - v_entry * v_entry / float(lip_radius)


def rough_runup_veto_active(context: dict | None) -> bool:
    """Calibrated S4-g5 guard for the measured rough-runup reverse domain."""
    if not isinstance(context, dict):
        return False
    try:
        return (
            int(context.get("runup_bumps", 0) or 0) >= ROUGH_VETO_MIN_BUMPS
            and float(context.get("bump_height_m", 0.0) or 0.0)
            >= ROUGH_VETO_MIN_HEIGHT_M
            and float(context.get("bump_spacing_m", 0.0) or 0.0)
            >= ROUGH_VETO_MIN_SPACING_M
        )
    except (TypeError, ValueError):
        return False


def recommend_arm(angle_deg: float, v_entry: float,
                  table_path: Path | str | None = None,
                  lip_radius: float | None = None,
                  n_min: float = CURVATURE_N_MIN,
                  roughness_context: dict | None = None,
                  uncertainty: SelectorUncertainty | None = None,
                  geometry_family: str = "tabletop_fillet",
                  require_calibration_map: bool = False,
                  calibration_support_max_distance: float = 1.5,
                  system_outcome_table_path: Path | str | None = None,
                  require_system_safety_evidence: bool = False,
                  system_safety_max_distance: float | None = None) -> dict:
    """Recommend an arm from the calibrated action map plus the safeguards.

    The measured preset gain is non-monotone in ramp angle and entry speed,
    so the analytic translation alone is not enough: the gate first looks up
    an arm in the calibrated table, which may return an explicit
    ``avoid_preset``, and only then evaluates go/no-go through
    :class:`PresetAwareTakeoffSet`. The lookup takes the nearest grid point
    under a Euclidean metric in which 5 m/s of speed is one angle cell; with
    no table, or a nearest cell further than 1.5 cells away, the
    recommendation falls back to ``neutral``.

    Passing ``lip_radius`` enables the curvature safeguard: where
    ``n(v) <= n_min`` a ``nose_up`` recommendation is demoted to ``neutral``,
    leaving the jump/no-jump call to the reachability gate on the natural
    state. See the module comment for the two competing channels and the
    domain over which the threshold was calibrated. A table
    ``avoid_preset`` is more conservative still and is never overridden.
    """
    calibration_support_max_distance = float(calibration_support_max_distance)
    if calibration_support_max_distance <= 0.0:
        raise ValueError("calibration_support_max_distance must be positive")
    if system_safety_max_distance is not None:
        system_safety_max_distance = float(system_safety_max_distance)
        if system_safety_max_distance <= 0.0:
            raise ValueError("system_safety_max_distance must be positive")
    p = Path(table_path) if table_path else GATE_TABLE_PATH
    n_v = (lip_curvature_margin(v_entry, lip_radius, angle_deg)
           if lip_radius and lip_radius > 0.0 else None)

    def _finish(rec: dict) -> dict:
        arm = rec.get("arm")
        if arm not in VALID_GATE_ARMS:
            rec.update(
                arm="neutral",
                reason=f"unknown_gate_arm({arm!r}); was: {rec.get('reason')}",
                release_timing=None,
                timed_effect=None,
            )
        if rec["arm"] != "nose_up":
            rec["release_timing"] = None
            rec["timed_effect"] = None
        rec["roughness_veto"] = False
        if rec["arm"] == "nose_up" and rough_runup_veto_active(roughness_context):
            rec.update(
                arm="neutral",
                roughness_veto=True,
                reason=(
                    "rough_runup_reverse_domain"
                    f"(bumps={roughness_context.get('runup_bumps')},"
                    f"h={roughness_context.get('bump_height_m')},"
                    f"spacing={roughness_context.get('bump_spacing_m')}); "
                    f"was: {rec.get('reason')}"
                ),
                release_timing=None,
                timed_effect=None,
            )
        if n_v is not None:
            rec["curvature_margin_mps2"] = round(n_v, 2)
            rec["curvature_veto"] = False
            if rec["arm"] == "nose_up" and n_v <= n_min:
                rec.update(arm="neutral", curvature_veto=True,
                           reason=f"curvature_saturation(n={n_v:.2f}<= {n_min:g}); was: {rec['reason']}",
                           release_timing=None, timed_effect=None)
        rec["deployment_status"] = rec.get("deployment_status") or {
            "avoid_preset": "abort",
            "neutral": "airborne_only",
            "nose_up": "preset",
            "nose_down": "preset",
        }[rec["arm"]]
        rec["action_type"] = {
            "avoid_preset": "abort",
            "neutral": (
                "airborne_only_fallback"
                if rec.get("abstained") else "pass_through"),
            "nose_up": (
                "timed_preset" if rec.get("release_timing") else "static_preset"),
            "nose_down": "static_preset",
        }[rec["arm"]]
        return rec

    if not p.exists():
        return _finish({
            "arm": "neutral",
            "reason": "no_gate_table",
            "entry": None,
            "abstained": True,
            "deployment_status": "abstain_to_airborne_only",
        })
    try:
        raw_bytes = p.read_bytes()
        table_doc = json.loads(raw_bytes.decode("utf-8"))
        tab = table_doc["entries"]
    except Exception:
        return _finish({"arm": "neutral", "reason": "gate_table_unreadable", "entry": None})
    if not tab:
        return _finish({
            "arm": "neutral",
            "reason": "gate_table_empty",
            "entry": None,
            "abstained": True,
            "deployment_status": "abstain_to_airborne_only",
        })
    def dist(e):
        return math.hypot((e["angle"] - angle_deg) / 5.0, (e["v"] - v_entry) / 1.0)
    best = min(tab, key=dist)
    d_best = dist(best)
    if d_best > 1.5:
        return _finish({
            "arm": "neutral",
            "reason": f"nearest_grid_too_far({d_best:.2f})",
            "entry": best,
            "query_distance": d_best,
            "release_timing": None,
            "timed_effect": None,
            "abstained": True,
            "deployment_status": "abstain_to_airborne_only",
        })
    timing, timing_error = canonical_release_timing(best)
    if timing_error:
        return _finish({
            "arm": "neutral",
            "reason": f"{timing_error}; was: {best.get('reason')}",
            "entry": best,
            "release_timing": None,
            "timed_effect": None,
        })
    arm = best.get("recommended_arm")
    reason = best.get("reason")
    if timing is not None and d_best > timing["max_query_distance"]:
        arm = best.get("static_recommended_arm", "neutral")
        reason = (
            f"timing_query_too_far({d_best:.2f}>{timing['max_query_distance']:.2f}); "
            f"static_fallback: {best.get('static_reason', reason)}")
        timing = None
    action_kind = "timed_preset" if timing is not None else "static_preset"
    calibration_map_model = ActionCalibrationMap.from_gate_entries(tab)
    calibration_estimate = calibration_map_model.query(
        angle_deg,
        v_entry,
        action_kind=action_kind,
        geometry_family=geometry_family,
        max_support_distance=(
            min(
                timing["max_query_distance"],
                calibration_support_max_distance,
            )
            if timing is not None
            else calibration_support_max_distance),
    )
    if arm == "nose_up" and require_calibration_map and not calibration_estimate.in_domain:
        return _finish({
            "arm": "neutral",
            "reason": f"calibration_map_unavailable({calibration_estimate.reason}); was: {reason}",
            "entry": best,
            "query_distance": d_best,
            "release_timing": None,
            "timed_effect": None,
            "abstained": True,
            "deployment_status": "abstain_to_airborne_only",
            "calibration_map": calibration_estimate.as_dict(),
        })
    system_outcome_evidence = None
    if arm == "nose_up" and require_system_safety_evidence:
        system_decision = query_system_outcome_table(
            angle_deg,
            v_entry,
            table_path=(
                system_outcome_table_path or SYSTEM_OUTCOME_TABLE_PATH),
            geometry_family=geometry_family,
            max_query_distance=system_safety_max_distance,
        )
        system_outcome_evidence = {
            "available": system_decision.available,
            "deploy_preset": system_decision.deploy_preset,
            "action": system_decision.action,
            "reason": system_decision.reason,
            "query_distance": system_decision.query_distance,
            "evidence": system_decision.evidence,
        }
        action_matches = system_decision.action == action_kind
        if not system_decision.deploy_preset or not action_matches:
            why = (
                system_decision.reason
                if not system_decision.deploy_preset
                else f"system_action_mismatch({system_decision.action}!={action_kind})"
            )
            return _finish({
                "arm": "neutral",
                "reason": f"{why}; was: {reason}",
                "entry": best,
                "query_distance": d_best,
                "release_timing": None,
                "timed_effect": None,
                "abstained": True,
                "deployment_status": "abstain_to_airborne_only",
                "calibration_map": calibration_estimate.as_dict(),
                "system_outcome_evidence": system_outcome_evidence,
            })
    uncertainty_audit = None
    if uncertainty is not None:
        def decision_at(query_angle, query_speed):
            def query_dist(entry):
                return math.hypot(
                    (entry["angle"] - query_angle) / 5.0,
                    (entry["v"] - query_speed) / 1.0,
                )
            entry = min(tab, key=query_dist)
            distance = query_dist(entry)
            if distance > 1.5:
                return "abstain"
            alt_timing, alt_error = canonical_release_timing(entry)
            if alt_error:
                return "abstain"
            alt_arm = entry.get("recommended_arm")
            if alt_timing is not None and distance > alt_timing["max_query_distance"]:
                alt_arm = entry.get("static_recommended_arm", "neutral")
                alt_timing = None
            if alt_arm == "nose_up" and rough_runup_veto_active(roughness_context):
                return "neutral"
            if (
                alt_arm == "nose_up"
                and lip_radius is not None
                and lip_radius > 0.0
                and lip_curvature_margin(
                    query_speed, lip_radius, query_angle) <= n_min
            ):
                return "neutral"
            alt_action_kind = (
                "timed_preset" if alt_timing is not None else "static_preset")
            alt_calmap = calibration_map_model.query(
                query_angle,
                query_speed,
                action_kind=alt_action_kind,
                geometry_family=geometry_family,
                max_support_distance=(
                    min(
                        alt_timing["max_query_distance"],
                        calibration_support_max_distance,
                    )
                    if alt_timing is not None
                    else calibration_support_max_distance),
            )
            if (
                alt_arm == "nose_up"
                and require_calibration_map
                and not alt_calmap.in_domain
            ):
                return "abstain"
            if alt_arm == "nose_up" and require_system_safety_evidence:
                alt_system = query_system_outcome_table(
                    query_angle,
                    query_speed,
                    table_path=(
                        system_outcome_table_path or SYSTEM_OUTCOME_TABLE_PATH),
                    geometry_family=geometry_family,
                    max_query_distance=system_safety_max_distance,
                )
                if (
                    not alt_system.deploy_preset
                    or alt_system.action != alt_action_kind
                ):
                    return "abstain"
            if (
                uncertainty.require_safety_v2_evidence
                and alt_arm == "nose_up"
                and entry.get("safe_criterion_basis") != "safety_v2"
            ):
                return "abstain"
            return alt_arm

        da = float(uncertainty.angle_half_width_deg)
        dv = float(uncertainty.speed_half_width_mps)
        corners = {
            (angle_deg + sa * da, v_entry + sv * dv)
            for sa in (-1.0, 1.0)
            for sv in (-1.0, 1.0)
        }
        decisions = sorted({decision_at(a, v) for a, v in corners})
        uncertainty_audit = {
            "angle_half_width_deg": da,
            "speed_half_width_mps": dv,
            "corner_decisions": decisions,
            "robust": decisions == [arm],
            "require_safety_v2_evidence": (
                uncertainty.require_safety_v2_evidence),
        }
        evidence_ok = (
            not uncertainty.require_safety_v2_evidence
            or arm != "nose_up"
            or best.get("safe_criterion_basis") == "safety_v2"
        )
        if decisions != [arm] or not evidence_ok:
            why = (
                "safety_v2_evidence_required"
                if not evidence_ok
                else f"uncertainty_decision_conflict({','.join(decisions)})"
            )
            return _finish({
                "arm": "neutral",
                "reason": f"{why}; was: {reason}",
                "entry": best,
                "query_distance": d_best,
                "release_timing": None,
                "timed_effect": None,
                "abstained": True,
                "deployment_status": "abstain_to_airborne_only",
                "uncertainty_audit": uncertainty_audit,
                "calibration_map": calibration_estimate.as_dict(),
                "system_outcome_evidence": system_outcome_evidence,
            })
    return _finish({
        "arm": arm,
        "reason": reason,
        "entry": best,
        "query_distance": d_best,
        "release_timing": timing,
        "timed_effect": best.get("timed_effect") if timing is not None else None,
        "uncertainty_audit": uncertainty_audit,
        "calibration_map": calibration_estimate.as_dict(),
        "system_outcome_evidence": system_outcome_evidence,
        "table_audit": {
            "schema_version": table_doc.get("schema_version"),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "timing_overlay": table_doc.get("timing_overlay"),
        },
    })


__all__ = [
    "SuspPresetEffect",
    "PresetAwareTakeoffSet",
    "preset_effect",
    "go_nogo_with_preset",
    "recommend_arm",
    "canonical_release_timing",
    "lip_curvature_margin",
    "rough_runup_veto_active",
    "SelectorUncertainty",
    "CURVATURE_N_MIN",
    "FIT_PATH",
    "GATE_TABLE_PATH",
    "SYSTEM_OUTCOME_TABLE_PATH",
    "DEFAULT_ANCHORS_NOSE_UP",
    "TIMING_MODE",
    "ROUGH_VETO_MIN_BUMPS",
    "ROUGH_VETO_MIN_HEIGHT_M",
    "ROUGH_VETO_MIN_SPACING_M",
]
