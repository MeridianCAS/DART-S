"""Run-up / takeoff-ramp ground model selection (ASPHALT / GRAVEL / DIRT).

Procedural ramp segments use BeamNG ``Material.groundType`` for tire contact physics.
All ground types lay a unified flat run-up slab (same material + thickness as the ramp)
over smallgrid so spawn→lip has one visual surface and one contact material.
"""
from __future__ import annotations

from typing import Any

RUNUP_GROUND_CHOICES = ("ASPHALT", "GRAVEL", "DIRT")

# Nominal coeffs from gameengine.zip art/groundmodels.json (BeamNG 0.38.x).
NOMINAL_FRICTION: dict[str, dict[str, float]] = {
    "ASPHALT": {"static": 0.98, "sliding": 0.70, "roughness": 0.0},
    "GRAVEL": {"static": 0.69, "sliding": 0.74, "roughness": 0.44},
    "DIRT": {"static": 0.70, "sliding": 0.73, "roughness": 0.42},
}

_MATERIAL_BY_GROUND: dict[str, str] = {
    "ASPHALT": "track_editor_C_border",
    "GRAVEL": "darts_runup_gravel",
    "DIRT": "darts_runup_dirt",
}

# (groundType, diffuseColor stage0, diffuseColor stage1) for custom materials.
_CUSTOM_SPECS: dict[str, tuple[str, str, str]] = {
    "GRAVEL": (
        "GRAVEL",
        "0.55 0.50 0.46 1",
        "0.42 0.38 0.34 1",
    ),
    "DIRT": (
        "DIRT",
        "0.52 0.40 0.30 1",
        "0.40 0.30 0.22 1",
    ),
}

# Register custom run-up materials via Lua. Plain diffuseColor is used instead
# of texture maps (session-staged dds files are unreliable); groundType drives
# the contact/friction model, which is what the surface axis varies.
LUA_ENSURE_CUSTOM = r"""
local function _darts_ensure_runup_mat(name, gtype, dc0)
  local m = scenetree.findObject(name)
  if m then return end
  m = createObject("Material")
  m.mapTo = name
  m:setField("diffuseColor", 0, dc0)
  m:setField("specularPower", 0, "8")
  m:setField("specular", 0, "0.1 0.1 0.1 1")
  m:setField("materialTag0", 0, "beamng")
  m:setField("materialTag1", 0, "RoadAndPath")
  m:setField("groundType", 0, gtype)
  m:registerObject(name)
end
_darts_ensure_runup_mat("darts_runup_gravel", "GRAVEL", "0.62 0.58 0.52 1")
_darts_ensure_runup_mat("darts_runup_dirt", "DIRT", "0.45 0.32 0.20 1")
return "ok"
"""

def normalize_ground_type(value: str | None) -> str:
    gt = (value or "ASPHALT").strip().upper()
    if gt not in RUNUP_GROUND_CHOICES:
        raise ValueError(
            f"runup_ground_type must be one of {RUNUP_GROUND_CHOICES}, got {value!r}"
        )
    return gt

def resolve_runup_material(ground_type: str) -> str:
    return _MATERIAL_BY_GROUND[normalize_ground_type(ground_type)]

def friction_nominal(ground_type: str) -> dict[str, float]:
    return dict(NOMINAL_FRICTION[normalize_ground_type(ground_type)])

def apply_ramp_material(segs: list[dict[str, Any]], ground_type: str) -> list[dict[str, Any]]:
    """Set material on takeoff ramp segments only (not landing mesh)."""
    mat = resolve_runup_material(ground_type)
    for seg in segs:
        seg["material"] = mat
    return segs

def build_runup_pad_segments(
    *,
    base_x: float,
    run_up: float,
    width: float,
    thick: float = 0.8,
    ground_type: str = "ASPHALT",
    margin_back: float = 11.0,
    margin_front: float = 1.5,
) -> list[dict[str, Any]]:
    """Flat run-up slab: spawn→ramp-base, same material/thickness as takeoff ramp.

    Covers smallgrid so tires never mix grid contact with procedural ramp mesh.
    Top surface at z≈0 aligns with ramp segment tops (flat prepend no longer needed).
    margin_back: flat slab extends this far behind spawn (x = base_x - run_up).
    """
    gt = normalize_ground_type(ground_type)
    x0 = float(base_x) - float(run_up) - float(margin_back)
    x1 = float(base_x) + float(margin_front)
    length = max(1.0, x1 - x0)
    cx = 0.5 * (x0 + x1)
    slab_thick = max(0.55, float(thick))
    top_z = 0.0
    cz = top_z - slab_thick * 0.5 + 0.03
    return [{
        "name": "darts_runup_pad",
        "pos": (cx, 0.0, cz),
        "size": (float(width), length, slab_thick),
        "rot": (0.0, 0.0, 0.0, 1.0),
        "material": resolve_runup_material(gt),
    }]

def runup_ground_audit(ground_type: str) -> dict[str, Any]:
    gt = normalize_ground_type(ground_type)
    fr = friction_nominal(gt)
    return {
        "runup_ground_type": gt,
        "material": resolve_runup_material(gt),
        "nominal_static_friction": fr["static"],
        "nominal_sliding_friction": fr["sliding"],
        "nominal_roughness": fr["roughness"],
        "runup_pad": True,
        "unified_runup_surface": True,
    }

def ensure_runup_materials(queue_lua, ground_type: str) -> None:
    """Register darts_runup_* materials once per session (no-op for ASPHALT)."""
    if normalize_ground_type(ground_type) == "ASPHALT":
        return
    try:
        queue_lua(LUA_ENSURE_CUSTOM)
    except Exception as exc:  # noqa: BLE001
        print(f"[runup-ground] WARN ensure materials: {exc!r}", flush=True)

def place_ramp_with_ground(bng, segs, queue_lua, ground_type: str = "ASPHALT"):
    """Ensure ground materials then delegate to _ramp_feather.place_ramp."""
    import scripts._ramp_feather as rf  # noqa: WPS433

    ensure_runup_materials(queue_lua, ground_type)
    if normalize_ground_type(ground_type) != "ASPHALT":
        rf._step(bng, 2)   # let the material registration settle before mesh placement
    return rf.place_ramp(bng, segs, queue_lua)
