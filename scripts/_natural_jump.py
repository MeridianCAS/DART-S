"""Vehicle poll and Lua helpers used by the DART-S experiment runner.

Not a standalone bench: ``scripts/darts_bench.py`` imports ``_poll``,
``_vlua``, ``_contact``, ``_gear`` and ``_rpy``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# wheel ground-contact count (contactMaterialID1 >= 0); runs in the vehicle Lua VM
LUA_CONTACT = (
    "local c=0 if wheels~=nil and wheels.wheels~=nil then "
    "for i,wd in pairs(wheels.wheels) do "
    "if wd.contactMaterialID1~=nil and wd.contactMaterialID1>=0 then c=c+1 end end end "
    "return tostring(c)"
)
# roll/pitch/yaw in rad via the vehicle Lua VM; better than dir-asin (no roll there)
LUA_RPY = (
    "local ok,r,p,y=pcall(function() return obj:getRollPitchYaw() end) "
    "if ok and r~=nil then return string.format('%.5f,%.5f,%.5f',r,p,y) else return 'NA' end"
)

def _poll(vehicle):
    try:
        vehicle.poll_sensors()
    except Exception:
        try:
            vehicle.sensors.poll()
        except Exception:
            pass
    try:
        raw = vehicle.sensors["state"].data
        if raw:
            return dict(raw)
    except Exception:
        pass
    try:
        st = vehicle.state
        data = getattr(st, "data", st)
        return dict(data) if data else {}
    except Exception:
        return {}

def _sensor(vehicle, name):
    s = getattr(vehicle, "sensors", None)
    try:
        return s[name].data if s is not None else None
    except Exception:
        return None

def _vlua(vehicle, cmd):
    fn = getattr(vehicle, "queue_lua_command", None)
    if not callable(fn):
        return None
    try:
        return fn(cmd, response=True)
    except Exception:
        return None

def _contact(vehicle):
    r = _vlua(vehicle, LUA_CONTACT)
    try:
        return int(str(r).strip())
    except Exception:
        return None

def _gear(vehicle):
    """Return the current gear from electrics ('gear' or 'gearIndex'), '?' if unknown."""
    el = _sensor(vehicle, "electrics")
    if isinstance(el, dict):
        g = el.get("gear")
        if g is not None:
            return str(g)
        gi = el.get("gearIndex")
        if gi is not None:
            return str(gi)
    return "?"

def _rpy(vehicle, state):
    """Return (roll, pitch, yaw) in rad via getRollPitchYaw; fall back to
    dir-asin pitch with roll=0."""
    r = _vlua(vehicle, LUA_RPY)
    if r is not None and str(r).strip() != "NA":
        try:
            roll, pitch, yaw = (float(x) for x in str(r).strip().split(","))
            return roll, pitch, yaw
        except Exception:
            pass
    d = state.get("dir") or (1.0, 0.0, 0.0)
    return 0.0, math.asin(max(-1.0, min(1.0, float(d[2])))), 0.0
