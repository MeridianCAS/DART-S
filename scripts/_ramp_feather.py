"""BeamNG ProceduralMesh placement used by the DART-S experiment runner.

Places the ramp segments built in ``scripts/_ramp_bench.py`` and steps the
sim between Lua round trips.
"""
from __future__ import annotations

import json
import time

LUA_ENUM = (
    "local out={} "
    "local ok,names=pcall(function() return scenetree.findClassObjects('ProceduralMesh') end) "
    "if ok and names then for _,n in ipairs(names) do "
    "local o=scenetree.findObject(n) "
    "if o then local bb=o:getWorldBox() "
    "out[n]={bb.minExtents.x,bb.minExtents.y,bb.minExtents.z,"
    "bb.maxExtents.x,bb.maxExtents.y,bb.maxExtents.z} end end "
    "else out['_enum_error']=tostring(names) end "
    "return jsonEncode(out)"
)
LUA_DELETE_ALL = (
    "local deleted={} "
    "local ok,names=pcall(function() return scenetree.findClassObjects('ProceduralMesh') end) "
    "if ok and names then for _,n in ipairs(names) do "
    "local o=scenetree.findObject(n) "
    "if o then pcall(function() o:delete() end) deleted[#deleted+1]=n end end end "
    "return jsonEncode(deleted)"
)


def _step(bng, n):
    try:
        bng.control.step(n)
        return True
    except Exception:
        try:
            bng.step(n)
            return True
        except Exception:
            time.sleep(n / 100.0)
            return False


def place_ramp(bng, segs, queue_lua):
    """Delete existing ProceduralMesh objects, place ``segs``, and enumerate.

    Returns ``(deleted, placed, enum)``.
    """
    from beamngpy import ProceduralCube  # type: ignore
    try:
        deleted = json.loads(queue_lua(LUA_DELETE_ALL))
    except Exception as exc:  # noqa: BLE001
        deleted = {"error": repr(exc)}
    _step(bng, 3)
    placed = 0
    for s in segs:
        try:
            ProceduralCube(pos=s["pos"], size=s["size"], name=s["name"],
                           rot_quat=s["rot"],
                           material=s.get("material", "track_editor_C_border")).place(bng)
            placed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"place {s['name']} FAILED: {exc!r}", flush=True)
    _step(bng, 3)
    try:
        enum = json.loads(queue_lua(LUA_ENUM))
    except Exception as exc:  # noqa: BLE001
        enum = {"error": repr(exc)}
    return deleted, placed, enum
