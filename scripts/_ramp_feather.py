"""BeamNG ProceduralMesh helpers used by the DART-S experiment runner.

Builds the ramp as a fan of thin rotated cubes, places them, steps the sim,
and keeps the BeamNG window in the foreground. The named ramp and kicker
polylines live in ``scripts/_ramp_bench.py``.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PREFIX = "darts_ramp_seg"


def foreground():
    ps = ('Add-Type @"\nusing System;using System.Runtime.InteropServices;\n'
          'public class W{[DllImport("user32.dll")]public static extern bool SetForegroundWindow(IntPtr h);'
          '[DllImport("user32.dll")]public static extern bool ShowWindow(IntPtr h,int n);}\n"@\n'
          'Get-Process|?{$_.MainWindowTitle -like "*BeamNG*"}|'
          '%{[W]::ShowWindow($_.MainWindowHandle,9);[W]::SetForegroundWindow($_.MainWindowHandle)}')
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=15,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

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


def build_polyline(args):
    """Return the ramp centreline as ``[(x, z)]`` plus the fillet radius.

    The profile starts flat, curves through the fillet arc, and continues as
    a straight face at the requested angle. The whole polyline is then sunk
    by ``args.sink`` so the leading edge sits inside the ground and only the
    lip stands proud.
    """
    th = math.radians(args.angle)
    R = args.fillet_len / math.sin(th)   # arc radius, horizontal span = fillet_len
    pts = []
    for k in range(args.n_fillet + 1):
        phi = th * k / args.n_fillet
        x = args.x_near + R * math.sin(phi)
        z = R * (1.0 - math.cos(phi))
        pts.append((x, z))
    x_f, z_f = pts[-1]
    for j in range(1, args.n_main + 1):
        x = x_f + args.main_run * j / args.n_main
        z = z_f + (x - x_f) * math.tan(th)
        pts.append((x, z))
    pts = [(x, z - args.sink) for (x, z) in pts]
    return pts, R


def build_segments(pts, args):
    """Turn the polyline into thin cubes whose top faces lie on it.

    Each segment is sized ``(width_Y, slope_len_X, thick_Z)`` and rotated to
    the local slope, with its centre offset half a thickness below the top
    face along the surface normal.
    """
    T = args.thick
    segs = []
    for i in range(len(pts) - 1):
        x0, z0 = pts[i]
        x1, z1 = pts[i + 1]
        theta = math.atan2(z1 - z0, x1 - x0)
        slope_len = math.hypot(x1 - x0, z1 - z0) * args.overlap
        mx, mz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
        cx = mx + (T / 2.0) * math.sin(theta)
        cz = mz - (T / 2.0) * math.cos(theta)
        rot = (0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0))
        segs.append({"name": f"{PREFIX}{i}", "pos": (cx, 0.0, cz),
                     "size": (args.width, slope_len, T), "rot": rot,
                     "angle_deg": round(math.degrees(theta), 2)})
    return segs


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


def scene_alive(queue_lua, expected_n):
    """Liveness check on the placed geometry rather than on ``gamestate``.

    A scenario loaded through beamngpy can report ``menu`` while the vehicle
    and terrain are in fact live, so gamestate is not a usable signal. What a
    real scene loss does show is that every ProceduralMesh is gone, leaving
    the enumeration empty. Returns ``(alive, n_meshes)``.
    """
    try:
        enum = json.loads(queue_lua(LUA_ENUM))
    except Exception:  # noqa: BLE001
        return False, -1
    if not isinstance(enum, dict):
        return False, -1
    n = len([k for k in enum if not k.startswith("_")])
    return n >= expected_n, n
