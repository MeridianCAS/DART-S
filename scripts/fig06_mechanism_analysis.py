#!/usr/bin/env python3
"""Fig. 6 mechanism analysis over the queue_fig06_mechanism.sh cohorts.

Pairs dart_s_forced (nose-up preset plus DART) against dart_only (DART alone)
by roll index within each cohort and reports:

* ladders: per-speed median takeoff pitch-rate difference with its sign-test
  p-value, plus the pitch, wheel-speed and |landing error| differences, for
  the R=35 m reference lip and the R=22 m held-out lip; v* = sqrt(g R cos a);
* arc cells: the same contrast on the 2x2 radius-by-arc-length grid at n(v)~0;
* stroke windows: front-to-rear lift-off window from the four-corner spring
  lengths, per arm, for every cohort that recorded a stroke trace.

Output: data/derived/fig06_mechanism.json
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from control.darts.provenance import aggregate_provenance  # noqa: E402

COHORTS = REPO / "data/cohorts"
OUT = REPO / "data/derived/fig06_mechanism.json"
G = 9.81
DT = 0.01
BASE_ARM, PRESET_ARM = "dart_only", "dart_s_forced"
FRONT = ("aspring_FL", "aspring_FR")
REAR = ("aspring_RL1", "aspring_RR1")

# (label, lip radius, sweep deg) -> ordered (speed, tag) rungs.
LADDERS = {
    "R35": (35.0, 8.0, [(13.0, "fig06_r35_v13"), (15.0, "fig06_r35_v15"),
                        (16.5, "fig06_r35_v16p5"), (18.0, "fig06_r35_v18"),
                        (19.5, "fig06_r35_v19p5")]),
    "R22": (22.0, 8.0, [(12.0, "fig06_r22_v12"), (13.0, "fig06_r22_v13"),
                        (14.25, "fig06_r22_v14p25"), (15.5, "fig06_r22_v15p5")]),
}
# 2x2 radius x arc length at n(v) ~ 0: (radius, sweep deg, speed, tag).
ARC_CELLS = [
    (35.0, 5.0, 18.0, "fig06_r35s5_v18"),
    (35.0, 8.0, 18.0, "fig06_r35_v18"),
    (22.0, 8.0, 14.25, "fig06_r22_v14p25"),
    (22.0, 12.73, 14.25, "fig06_r22s12p73_v14p25"),
]


def med(values, nd=2):
    v = [x for x in values if x is not None]
    return round(st.median(v), nd) if v else None


def sign_p(diffs):
    """Two-sided exact sign test over paired differences; zeros are dropped."""
    d = [x for x in diffs if x is not None and x != 0]
    n = len(d)
    if n == 0:
        return None
    k = min(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    return round(min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n), 5)


def load_cohort(tag):
    p = COHORTS / f"darts_bench_{tag}.json"
    return (json.loads(p.read_text(encoding="utf-8")), p) if p.exists() else (None, p)


def runs_by_roll(doc, arm):
    """{roll_index: record} for runs of ``arm`` that took off."""
    out = {}
    for vdata in (doc.get("data") or {}).values():
        for r in vdata.get(arm, []):
            if r.get("took_off"):
                out[(r.get("protocol") or {}).get("roll_index")] = r
    return out


def pair(doc):
    """Roll-matched (base, preset) record pairs."""
    a, b = runs_by_roll(doc, BASE_ARM), runs_by_roll(doc, PRESET_ARM)
    return [(a[k], b[k]) for k in sorted(set(a) & set(b), key=lambda x: -1 if x is None else x)]


def ts(r, key):
    return (r.get("takeoff_state") or {}).get(key)


def land_err(r, slope_deg):
    lp = r.get("land_pitch_deg")
    return None if lp is None else abs(float(lp) - slope_deg)


def contrast(doc, radius, alpha_deg, speed):
    pairs = pair(doc)
    slope = float((doc.get("params") or {}).get("landing_slope_deg") or 0.0)
    d_om = [ts(b, "omega_y0_dps") - ts(a, "omega_y0_dps") for a, b in pairs
            if None not in (ts(a, "omega_y0_dps"), ts(b, "omega_y0_dps"))]
    d_th = [ts(b, "theta0_deg") - ts(a, "theta0_deg") for a, b in pairs
            if None not in (ts(a, "theta0_deg"), ts(b, "theta0_deg"))]
    d_w = [ts(b, "wheel_w0_radps") - ts(a, "wheel_w0_radps") for a, b in pairs
           if None not in (ts(a, "wheel_w0_radps"), ts(b, "wheel_w0_radps"))]
    d_err = [land_err(b, slope) - land_err(a, slope) for a, b in pairs
             if None not in (land_err(a, slope), land_err(b, slope))]
    return {
        "v": speed,
        "n_normal_mps2": round(G * math.cos(math.radians(alpha_deg)) - speed ** 2 / radius, 2),
        "n_pairs": len(d_om),
        "d_omega_med": med(d_om), "p_sign_omega": sign_p(d_om),
        "d_theta0_med": med(d_th), "p_sign_theta0": sign_p(d_th),
        "d_wheel_w0_med": med(d_w),
        "d_abs_land_err_med": med(d_err), "p_sign_land_err": sign_p(d_err),
        "omega_neutral_med": med([ts(a, "omega_y0_dps") for a, _ in pairs]),
        "v0_neutral_med": med([ts(a, "v0_mps") for a, _ in pairs]),
    }


def plateau_step(rows, keys, tol_frac=0.0015):
    """First step at which the shorter of ``keys`` reaches its droop plateau
    (within ``tol_frac`` of the window maximum) and never drops back."""
    series = [(r["i"], min(r[k] for k in keys)) for r in rows
              if all(r.get(k) is not None for k in keys)]
    if len(series) < 8:
        return None
    vmax = max(v for _, v in series)
    tol = vmax * tol_frac
    for idx, (i, v) in enumerate(series):
        if v >= vmax - tol and all(v2 >= vmax - 2 * tol for _, v2 in series[idx:]):
            return i
    return None


def liftoff_window(rows):
    """(front step, rear step, rear-minus-front window in s) or None."""
    fp, rp = plateau_step(rows, FRONT), plateau_step(rows, REAR)
    if fp is None or rp is None:
        return None
    return fp, rp, round((rp - fp) * DT, 3)


def stroke_windows(doc):
    arms = {}
    for arm in (BASE_ARM, PRESET_ARM):
        per = []
        for r in runs_by_roll(doc, arm).values():
            w = liftoff_window(r.get("stroke_trace") or [])
            if w is not None:
                per.append({"front_i": w[0], "rear_i": w[1], "window_s": w[2],
                            "omega_y0_dps": ts(r, "omega_y0_dps")})
        arms[arm] = {"n_traced": len(per),
                     "window_med_s": med([e["window_s"] for e in per], 3),
                     "omega_med_dps": med([e["omega_y0_dps"] for e in per], 1),
                     "per_jump": per}
    a, b = arms[BASE_ARM], arms[PRESET_ARM]
    out = {"arms": arms, "d_window_s": None, "implied_dps_per_s_of_window": None}
    if a["window_med_s"] is not None and b["window_med_s"] is not None:
        dw = round(b["window_med_s"] - a["window_med_s"], 3)
        out["d_window_s"] = dw
        if abs(dw) > 1e-6 and None not in (a["omega_med_dps"], b["omega_med_dps"]):
            out["implied_dps_per_s_of_window"] = round(
                (b["omega_med_dps"] - a["omega_med_dps"]) / dw, 1)
    return out


def main() -> int:
    sources, ladders, arcs, strokes = [], {}, [], {}
    alpha = None
    for label, (radius, sweep, rungs) in LADDERS.items():
        rows = []
        for speed, tag in rungs:
            doc, p = load_cohort(tag)
            if doc is None:
                print(f"[fig06] missing {p.name}")
                continue
            sources.append(p)
            alpha = float(doc["params"]["angle"])
            rows.append({"tag": tag, **contrast(doc, radius, alpha, speed)})
        if rows:
            ladders[label] = {
                "lip_radius_m": radius, "lip_sweep_deg": sweep,
                "arc_length_m": round(radius * math.radians(sweep), 2),
                "v_star_mps": round(math.sqrt(G * radius * math.cos(math.radians(alpha))), 2),
                "rungs": rows}
    for radius, sweep, speed, tag in ARC_CELLS:
        doc, p = load_cohort(tag)
        if doc is None:
            continue
        if p not in sources:
            sources.append(p)
        arcs.append({"tag": tag, "lip_radius_m": radius, "lip_sweep_deg": sweep,
                     "arc_length_m": round(radius * math.radians(sweep), 2),
                     **contrast(doc, radius, float(doc["params"]["angle"]), speed)})
    for p in sources:
        doc = json.loads(p.read_text(encoding="utf-8"))
        if int((doc.get("params") or {}).get("susp_stroke_trace") or 0):
            strokes[p.stem.replace("darts_bench_", "")] = stroke_windows(doc)
    if not sources:
        print("[fig06] no fig06_* cohorts under data/cohorts; run queue_fig06_mechanism.sh first")
        return 2

    for label, lad in ladders.items():
        print(f"[fig06] {label}: v*={lad['v_star_mps']} m/s  arc={lad['arc_length_m']} m")
        for r in lad["rungs"]:
            print(f"  v={r['v']:>5}  n(v)={r['n_normal_mps2']:>6}  pairs={r['n_pairs']:>3}  "
                  f"d_omega={r['d_omega_med']}  p={r['p_sign_omega']}  d|err|={r['d_abs_land_err_med']}")
    for c in arcs:
        print(f"[fig06] arc R={c['lip_radius_m']} sweep={c['lip_sweep_deg']} ({c['arc_length_m']} m) "
              f"v={c['v']}: d_omega={c['d_omega_med']} p={c['p_sign_omega']} pairs={c['n_pairs']}")
    for tag, s in strokes.items():
        a, b = s["arms"][BASE_ARM], s["arms"][PRESET_ARM]
        print(f"[fig06] stroke {tag}: window {a['window_med_s']} -> {b['window_med_s']} s "
              f"(n={a['n_traced']}/{b['n_traced']}), implied {s['implied_dps_per_s_of_window']} deg/s per s")

    out = {"contrast": f"{PRESET_ARM} minus {BASE_ARM}, paired by roll",
           "alpha_deg": alpha, "ladders": ladders, "arc_cells": arcs, "stroke_windows": strokes,
           "_provenance": aggregate_provenance(
               sources, note="Fig. 6 mechanism aggregate over queue_fig06_mechanism.sh cohorts")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[fig06] saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
