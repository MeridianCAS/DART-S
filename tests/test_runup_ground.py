"""Unit tests for run-up ground type (no BeamNG)."""
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from control.darts import runup_ground as rg  # noqa: E402


def test_normalize_and_material_map():
    assert rg.normalize_ground_type("gravel") == "GRAVEL"
    assert rg.resolve_runup_material("ASPHALT") == "track_editor_C_border"
    assert rg.resolve_runup_material("GRAVEL") == "darts_runup_gravel"
    assert rg.resolve_runup_material("DIRT") == "darts_runup_dirt"


def test_asphalt_unified_runup_pad():
    pad = rg.build_runup_pad_segments(base_x=50.0, run_up=50.0, width=36.0, thick=0.8)
    assert len(pad) == 1
    assert pad[0]["material"] == "track_editor_C_border"
    assert pad[0]["size"][2] == 0.8
    top_z = pad[0]["pos"][2] + pad[0]["size"][2] * 0.5
    assert abs(top_z) < 0.05
    cx = pad[0]["pos"][0]
    half_len = pad[0]["size"][1] * 0.5
    assert abs((cx - half_len) - (-11.0)) < 0.01  # x0 = spawn(0) - margin_back(11)


def test_gravel_runup_pad_geometry():
    pad = rg.build_runup_pad_segments(
        base_x=50.0, run_up=50.0, width=36.0, ground_type="GRAVEL", thick=0.8)
    assert len(pad) == 1
    assert pad[0]["name"] == "darts_runup_pad"
    assert pad[0]["material"] == "darts_runup_gravel"
    cx = pad[0]["pos"][0]
    assert 0.0 < cx < 50.0
    half_len = pad[0]["size"][1] * 0.5
    assert abs((cx - half_len) - (-11.0)) < 0.01


def test_apply_ramp_material():
    segs = [{"name": "r0s0", "material": "old"}]
    rg.apply_ramp_material(segs, "DIRT")
    assert segs[0]["material"] == "darts_runup_dirt"


def test_audit_fields():
    a = rg.runup_ground_audit("GRAVEL")
    assert a["runup_ground_type"] == "GRAVEL"
    assert a["runup_pad"] is True
    assert a["unified_runup_surface"] is True
    assert a["nominal_static_friction"] == 0.69
    assert rg.runup_ground_audit("ASPHALT")["runup_pad"] is True


def test_cli_flag_registered():
    src = (REPO / "scripts" / "darts_bench.py").read_text(encoding="utf-8")
    assert "--runup-ground-type" in src


if __name__ == "__main__":
    test_normalize_and_material_map()
    test_asphalt_unified_runup_pad()
    test_gravel_runup_pad_geometry()
    test_apply_ramp_material()
    test_audit_fields()
    test_cli_flag_registered()
    print("OK: test_darts_runup_ground")
