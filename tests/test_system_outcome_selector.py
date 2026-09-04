import json

from control.darts.system_outcome_selector import query_system_outcome_table


def _table(tmp_path):
    path = tmp_path / "system.json"
    path.write_text(json.dumps({
        "endpoint": "post_touchdown_safety_v2",
        "data_role": "calibration_for_future_independent_evaluation",
        "max_query_distance": 0.25,
        "entries": [{
            "angle_deg": 35.0,
            "speed_mps": 15.0,
            "geometry_family": "tabletop_fillet",
            "preset_action": "static_preset",
            "deploy_eligible": True,
            "decision_reason": "independent_system_safety_gain",
        }],
    }), encoding="utf-8")
    return path


def test_system_outcome_selector_deploys_only_inside_frozen_support(tmp_path):
    path = _table(tmp_path)

    exact = query_system_outcome_table(
        35.0, 15.0, table_path=path, geometry_family="tabletop_fillet")
    far = query_system_outcome_table(
        35.0, 15.5, table_path=path, geometry_family="tabletop_fillet")

    assert exact.deploy_preset is True
    assert exact.action == "static_preset"
    assert far.deploy_preset is False
    assert far.action == "abstain"


def test_system_outcome_selector_rejects_noncalibration_role(tmp_path):
    path = _table(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["data_role"] = "evaluation"
    path.write_text(json.dumps(doc), encoding="utf-8")

    result = query_system_outcome_table(
        35.0, 15.0, table_path=path, geometry_family="tabletop_fillet")

    assert result.available is False
    assert result.reason == "system_outcome_role_not_calibration"


def test_system_outcome_selector_rejects_wrong_geometry_family(tmp_path):
    result = query_system_outcome_table(
        35.0,
        15.0,
        table_path=_table(tmp_path),
        geometry_family="convex_kicker_r35",
    )

    assert result.deploy_preset is False
    assert result.reason == "system_outcome_geometry_unsupported(convex_kicker_r35)"


def test_system_outcome_selector_never_substitutes_static_for_timed(tmp_path):
    result = query_system_outcome_table(
        35.0,
        15.0,
        table_path=_table(tmp_path),
        geometry_family="tabletop_fillet",
        action_kind="timed_preset",
        schedule_id="release_l0p35_d0p12_f0p5",
    )
    assert result.deploy_preset is False
    assert result.reason.startswith("system_outcome_condition_unsupported")
