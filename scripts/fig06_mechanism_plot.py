#!/usr/bin/env python3
"""Fig. 6 from data/derived/fig06_mechanism.json and the 18 m/s stroke cohort.

(a) Paired takeoff pitch-rate difference against entry speed for the R=35 m
    and R=22 m lips; filled markers are sign-test p<0.05; dashed lines mark v*.
(b) Front and rear spring extension around front lift-off for one
    representative jump per arm (window closest to the arm median), with the
    lift-off windows shaded.

Output: data/derived/fig06_mechanism.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.fig06_mechanism_analysis import (  # noqa: E402
    BASE_ARM, DT, FRONT, OUT, PRESET_ARM, REAR, liftoff_window, load_cohort, runs_by_roll)

STROKE_TAG = "fig06_r35_v18"
C_R35, C_R22, C_NEU = "#D55E00", "#0072B2", "#000000"


def panel_ladder(ax, doc):
    for label, color, marker in (("R35", C_R35, "o"), ("R22", C_R22, "s")):
        lad = (doc.get("ladders") or {}).get(label)
        if not lad:
            continue
        rungs = [r for r in lad["rungs"] if r.get("d_omega_med") is not None]
        ax.plot([r["v"] for r in rungs], [r["d_omega_med"] for r in rungs], "-",
                color=color, lw=1.4, label=f"R = {lad['lip_radius_m']:.0f} m")
        for r in rungs:
            sig = r.get("p_sign_omega") is not None and r["p_sign_omega"] < 0.05
            ax.plot(r["v"], r["d_omega_med"], marker, ms=5, color=color,
                    mfc=color if sig else "none", mew=1.1)
        ax.axvline(lad["v_star_mps"], color=color, lw=0.9, ls="--")
        ax.annotate(f"$v^*$={lad['v_star_mps']:.1f}", (lad["v_star_mps"], 1.0),
                    xycoords=("data", "axes fraction"), xytext=(3, -3),
                    textcoords="offset points", color=color, fontsize=7, va="top")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("entry speed (m/s)")
    ax.set_ylabel(r"$\Delta\omega_{\theta 0}$ (deg/s)")
    ax.legend(loc="lower left", frameon=False, fontsize=7)


def representative(doc, arm):
    """Stroke series of the jump whose lift-off window is closest to the arm median."""
    cands = []
    for r in runs_by_roll(doc, arm).values():
        rows = r.get("stroke_trace") or []
        w = liftoff_window(rows)
        if w is not None:
            cands.append((rows, *w))
    if not cands:
        return None
    windows = sorted(c[3] for c in cands)
    target = windows[len(windows) // 2]
    rows, fp, _, window = min(cands, key=lambda c: abs(c[3] - target))
    rows = [r for r in rows if -0.35 <= (r["i"] - fp) * DT <= 0.55
            and all(r.get(k) is not None for k in FRONT + REAR)]
    t = [(r["i"] - fp) * DT for r in rows]
    front = [min(r[k] for k in FRONT) * 1000 for r in rows]
    rear = [min(r[k] for k in REAR) * 1000 for r in rows]
    return {"t": t, "front": [x - front[0] for x in front],
            "rear": [x - rear[0] for x in rear], "window": window}


def panel_stroke(ax, cohort):
    base = representative(cohort, BASE_ARM) if cohort else None
    preset = representative(cohort, PRESET_ARM) if cohort else None
    if not base or not preset:
        ax.text(0.5, 0.5, "no stroke trace", transform=ax.transAxes, ha="center")
        return
    ax.axvspan(0, preset["window"], color=C_R35, alpha=0.08)
    ax.axvspan(0, base["window"], color="0.45", alpha=0.16)
    for data, color, name in ((base, C_NEU, "DART"), (preset, C_R35, "preset + DART")):
        ax.plot(data["t"], data["front"], "-", color=color, lw=1.25, label=f"{name} front")
        ax.plot(data["t"], data["rear"], "--", color=color, lw=1.25, label=f"{name} rear")
    ax.set_xlim(-0.35, 0.55)
    ax.axvline(0, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("time from front lift-off (s)")
    ax.set_ylabel("spring extension (mm)")
    ax.text(0.98, 0.98, f"lift-off windows {base['window']:.2f}/{preset['window']:.2f} s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color="0.25")
    ax.legend(loc="lower right", frameon=False, ncols=2, fontsize=7)


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fig06] matplotlib is required for the figure: pip install matplotlib")
        return 2
    if not OUT.exists():
        print(f"[fig06] {OUT} missing; run scripts/fig06_mechanism_analysis.py first")
        return 2
    doc = json.loads(OUT.read_text(encoding="utf-8"))
    cohort, _ = load_cohort(STROKE_TAG)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 2.45))
    panel_ladder(ax_a, doc)
    panel_stroke(ax_b, cohort)
    for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.tick_params(labelsize=7)
        ax.set_title(tag, loc="left", fontweight="bold", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=200)
    print(f"[fig06] saved {OUT.with_suffix('.png')} and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
