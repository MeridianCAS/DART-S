import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import scripts.fig06_mechanism_analysis as m  # noqa: E402


def _trace(front_step, rear_step, n=60):
    """Spring lengths that reach the droop plateau at the given steps."""
    rows = []
    for i in range(n):
        f = 0.40 + 0.02 * min(1.0, (i / front_step) ** 4)
        r = 0.40 + 0.02 * min(1.0, (i / rear_step) ** 4)
        rows.append({"i": i, "aspring_FL": f, "aspring_FR": f + 0.0005,
                     "aspring_RL1": r, "aspring_RR1": r + 0.0005})
    return rows


def _run(roll, omega, theta, land, trace=None):
    rec = {"took_off": True, "protocol": {"roll_index": roll},
           "takeoff_state": {"theta0_deg": theta, "omega_y0_dps": omega,
                             "v0_mps": 18.0, "wheel_w0_radps": 80.0},
           "land_pitch_deg": land}
    if trace is not None:
        rec["stroke_trace"] = trace
    return rec


def _cohort(path, tag, n=10, d_omega=-8.0):
    base = [_run(k, -30.0 + k, 17.0, -8.0, _trace(20, 24)) for k in range(n)]
    preset = [_run(k, -30.0 + k + d_omega, 19.5, -4.0, _trace(20, 56)) for k in range(n)]
    doc = {"params": {"angle": 20.0, "landing_slope_deg": -12.0, "susp_stroke_trace": 1},
           "data": {"v18": {m.BASE_ARM: base, m.PRESET_ARM: preset}},
           "_provenance": {"time_integrity_ok": True, "dt_mode": "deterministic_stepped",
                           "sps": 100, "dt_eff": 0.01}}
    (path / f"darts_bench_{tag}.json").write_text(json.dumps(doc), encoding="utf-8")
    return doc


def test_sign_test_matches_exact_binomial():
    assert m.sign_p([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) == round(2 / 2 ** 10, 5)
    assert m.sign_p([1, -1, 1, -1]) == 1.0
    assert m.sign_p([0, 0]) is None


def test_contrast_pairs_by_roll_and_uses_landing_slope(tmp_path):
    doc = _cohort(tmp_path, "x")
    c = m.contrast(doc, 35.0, 20.0, 18.0)
    assert c["n_pairs"] == 10
    assert c["d_omega_med"] == -8.0 and c["p_sign_omega"] == round(2 / 2 ** 10, 5)
    assert c["d_theta0_med"] == 2.5
    # |(-4) - (-12)| - |(-8) - (-12)| = 8 - 4
    assert c["d_abs_land_err_med"] == 4.0
    assert c["n_normal_mps2"] == round(9.81 * math.cos(math.radians(20)) - 18 ** 2 / 35, 2)


def test_liftoff_window_from_plateaus():
    fp, rp, w = m.liftoff_window(_trace(20, 56))
    assert (fp, rp, w) == (20, 56, 0.36)
    assert m.liftoff_window(_trace(20, 56)[:5]) is None


def test_main_writes_ladders_arcs_and_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "COHORTS", tmp_path)
    monkeypatch.setattr(m, "OUT", tmp_path / "fig06_mechanism.json")
    for _, tag in m.LADDERS["R35"][2]:
        _cohort(tmp_path, tag)
    assert m.main() == 0
    out = json.loads((tmp_path / "fig06_mechanism.json").read_text(encoding="utf-8"))
    lad = out["ladders"]["R35"]
    assert lad["v_star_mps"] == round(math.sqrt(9.81 * 35 * math.cos(math.radians(20))), 2)
    assert len(lad["rungs"]) == 5 and "R22" not in out["ladders"]
    assert [c["tag"] for c in out["arc_cells"]] == ["fig06_r35_v18"]
    win = out["stroke_windows"]["fig06_r35_v18"]
    assert win["d_window_s"] == 0.32
    assert win["implied_dps_per_s_of_window"] == -25.0
    assert out["_provenance"]["time_integrity_ok"] is True
