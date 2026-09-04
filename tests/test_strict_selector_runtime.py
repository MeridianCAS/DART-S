from __future__ import annotations

import json
from pathlib import Path

from control.darts.strict_selector_runtime import (
    load_strict_runtime_bundle,
    strict_v2_runtime_select,
)


REPO = Path(__file__).resolve().parent.parent
CATALOG = REPO / "calibration/strict_v2_catalog.json"


def test_runtime_aborts_without_an_authorized_catalog():
    """No strict catalog ships: the landing-capture and CalMap held-out gates
    both failed, so the strict branch must refuse to act rather than fall
    back on the unaudited prediction. Paper runs use the legacy policy."""
    assert not CATALOG.exists()
    result = strict_v2_runtime_select(
        catalog_path=CATALOG,
        decision_state={},
        query_angle_deg=35.0,
        query_speed_mps=15.0,
        geometry_profile="tabletop_fillet",
        roughness_context={},
        lip_radius_m=None,
        landing_slope_deg=0.0,
        landing_slope_uncertainty_deg=0.0,
        physical_session_id="test",
        can_stop_before_lip=True,
    )
    assert result["arm"] == "avoid_preset"
    assert result["deployment_status"] == "abort"
    assert result["strict_selector"]["authorized"] is False


def test_runtime_rejects_catalog_fingerprint_tamper(tmp_path):
    payload = {
        "schema_version": "darts-strict-selector-v2/1",
        "method_fingerprint": "0" * 64,
        "deployment_eligible": True,
        "blockers": [],
    }
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bundle = load_strict_runtime_bundle(str(path.resolve()))
    assert "strict_catalog_fingerprint_mismatch" in bundle.blockers
