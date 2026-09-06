"""Ramp and landing geometry shared by the DART-S bench.

Procedural construction of the tabletop and convex-lip ramp faces, the landing
profiles (slope, ballistic hill, gap, valley), cross-slope banking, and the
per-wheel readback helpers. Identical to the geometry layer of the DART
reproduction package, so both papers build the same surfaces.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import scripts._natural_jump as nj
import scripts._ramp_bench as bench
import scripts._tp_perramp as tp

# Beam lengths of the four active-suspension springs; cids are resolved once and
# cached in the vehicle Lua VM because the beam table scan is not free per step.
LUA_SUSP_STROKE = (
    "if _G.__darts_susp_cids==nil then local t={} for i,b in pairs(v.data.beams) do "
    "local n=b.name if n=='aspring_FR' or n=='aspring_FL' or n=='aspring_RR1' or n=='aspring_RL1' "
    "then t[n]=b.cid end end _G.__darts_susp_cids=t end "
    "local o={} for n,c in pairs(_G.__darts_susp_cids) do o[n]=obj:getBeamLength(c) end "
    "return jsonEncode(o)")


def susp_stroke_sample(veh):
    """Per-corner active-spring length (m); {} when the channel is unreadable."""
    try:
        return json.loads(str(nj._vlua(veh, LUA_SUSP_STROKE)))
    except Exception:
        return {}

ANGLES = [10, 15, 20, 25, 30, 35, 40, 45]

LUA_WMEAN = ("local s=0 local n=0 if wheels~=nil and wheels.wheels~=nil then "
             "for _,wd in pairs(wheels.wheels) do s=s+math.abs(wd.angularVelocity) n=n+1 end end "
             "if n>0 then return tostring(s/n) else return '0' end")

LUA_WHEELS_ALL = ("local out={} if wheels~=nil and wheels.wheels~=nil then "
                  "for _,wd in pairs(wheels.wheels) do "
                  "if wd.name~=nil then out[tostring(wd.name)]=wd.angularVelocity end end end "
                  "return jsonEncode(out)")

def wheel_w(veh):
    """Mean wheel angular velocity at touchdown (rad/s): DART pitch drive can spin wheels to ~200; hard slide at contact risks blowout; land-prep should reduce this sharply."""
    try: return round(float(nj._vlua(veh, LUA_WMEAN)), 0)
    except Exception: return None

def wheel_w_all(veh):
    """Per-wheel angular velocity dict {wheel_name: angularVelocity} (4WIDS proxy). Returns {} if unreadable."""
    try:
        raw = nj._vlua(veh, LUA_WHEELS_ALL)
        if not raw:
            return {}
        obj = json.loads(str(raw))
        return {k: round(float(v), 2) for k, v in obj.items()} if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _slope_marker(name, x, z, theta, *, y=0.0, width=1.0, along=1.0, thick=0.18,
                  clearance=0.05, material="track_editor_C_border"):
    """Place a visual bump marker above the slope surface. size=(worldY width, along-slope length, thickness)."""
    normal_x = -math.sin(theta)
    normal_z = math.cos(theta)
    cx = x + (clearance + thick / 2.0) * normal_x
    cz = z + (clearance + thick / 2.0) * normal_z
    return {"name": name, "pos": (cx, y, cz), "size": (width, along, thick),
            "rot": (0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0)),
            "material": material}

def _vertical_marker(name, x, z, *, y, size=(1.0, 1.0, 3.0), material="track_editor_C_border",
                     root_ground=False):
    """Exaggerated vertical post/wall marker: no slope rotation so distant camera still sees landing bounds.

    root_ground=True: post rooted at SmallGrid z=0, top above slope (z+size[2]). **Flat/no-bank only**;
    after bank_landing_segments camber/cross-slope, root_ground base no longer sits on slope -> use root_ground=False.
    root_ground=False: base on slope z, height size[2], then rigid bank rotation -> flush with slope.
    """
    if root_ground:
        h = max(0.5, z + size[2])
        return {"name": name, "pos": (x, y, h / 2.0), "size": (size[0], size[1], h),
                "rot": (0.0, 0.0, 0.0, 1.0), "material": material}
    return {"name": name, "pos": (x, y, z + size[2] / 2.0), "size": size,
            "rot": (0.0, 0.0, 0.0, 1.0), "material": material}

def _post_root_ground(*, runup_camber_deg: float = 0.0, cross_slope_deg: float = 0.0) -> bool:
    """With camber / landing cross-slope, posts must sit on slope; forbid root_ground at z=0."""
    return abs(float(runup_camber_deg)) < 1e-6 and abs(float(cross_slope_deg)) < 1e-6

def _append_ramp_rail_visuals(segs, pts, *, prefix, ri, width, post_root_ground=True):
    """Add outer slope-hugging rails + gate posts along polyline slope (landing_slope family; takeoff/landing shared)."""
    if len(pts) < 2:
        return segs
    rail_w = 0.45
    edge_y = width / 2.0 - 1.0
    for i in range(len(pts) - 1):
        x0, z0 = pts[i]; x1, z1 = pts[i + 1]
        theta = math.atan2(z1 - z0, x1 - x0)
        mx, mz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
        seg_len = math.hypot(x1 - x0, z1 - z0)
        for side, y in (("L", edge_y), ("R", -edge_y)):
            segs.append(_slope_marker(
                f"{prefix}{ri}_rail_{side}_{i}", mx, mz, theta, y=y,
                width=rail_w, along=seg_len, thick=0.35, clearance=0.22))
    step = max(1, len(pts) // 8)
    for j in range(0, len(pts), step):
        x, z = pts[j]
        for side, y in (("L", edge_y), ("R", -edge_y)):
            segs.append(_vertical_marker(
                f"{prefix}{ri}_post_{side}_{j}", x, z, y=y,
                size=(0.7, 0.7, 1.4), root_ground=post_root_ground))
    return segs

def _quat_mul(a, b):
    """Hamilton product a⊗b, quaternion (x,y,z,w). Apply b then a (left-multiply = further world-frame rotation)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )

def bank_landing_segments(lsegs, phi_deg, *, pivot_y=0.0, pivot_z=0.0):
    """Rigid bank phi_deg about ground line {y=pivot_y, z=pivot_z} along forward (landing / run-up+takeoff shared).

    A plane rotated about a line stays a plane, so the whole rig (rails included) stays coherent, only laterally tilted.
    - pivot_y=0 (landing cross-slope): bank about centerline; centerline fixed.
    - pivot_y=∓W/2 (run-up camber): bank about low edge -> low edge at z=0 (natural feather), rest lifted,
      both wheels on incline -> true lateral roll (=γ). Center pivot lets low half sink into flat ground and lose contact, no roll.
    phi_deg>0: viewed along +X, dips down-right (right side low); <0 opposite. Returns new segs list (does not mutate input).
    """
    if abs(float(phi_deg)) < 1e-6:
        return lsegs
    phi = math.radians(float(phi_deg))
    c, s = math.cos(phi), math.sin(phi)
    q_bank = (math.sin(phi / 2.0), 0.0, 0.0, math.cos(phi / 2.0))
    out = []
    for seg in lsegs:
        x, y, z = seg["pos"]
        yr = y - pivot_y
        zr = z - pivot_z
        new_y = pivot_y + yr * c - zr * s
        new_z = pivot_z + yr * s + zr * c
        ns = dict(seg)
        ns["pos"] = (x, new_y, new_z)
        ns["rot"] = _quat_mul(q_bank, tuple(seg.get("rot", (0.0, 0.0, 0.0, 1.0))))
        out.append(ns)
    return out

def _camber_pivot_y(args) -> float:
    """Run-up camber pivot y = ∓W/2 about low edge (γ>0 low side at y<0 -> pivot=-W/2).

    Under camber use apron width (default 40 m) not narrow ramp width (12 m) so low edge truly grounds and center lifts enough."""
    gamma = _runup_camber_deg(args)
    half_w = _camber_apron_width(args) / 2.0
    return -math.copysign(half_w, gamma)

def _runup_camber_deg(args) -> float:
    return float(getattr(args, "runup_camber_deg", 0.0) or 0.0)

def _camber_apron_width(args) -> float:
    """Camber wide apron width (decoupled from narrow ramp width); pivot_y uses this."""
    if abs(_runup_camber_deg(args)) < 1e-6:
        return float(getattr(args, "width", 36.0))
    return float(getattr(args, "runup_apron_width", 40.0) or 40.0)

def _spawn_z_for_lane(y: float, args, *, base_z: float = 0.3) -> float:
    """Banked run-up (rotate about low-edge pivot_y): road height at lane y ≈ (y - pivot_y)·sin(γ) + clearance.
    Low edge (y=pivot_y) z=0 grounded, center raised (W/2)·sin|γ| -> vehicle on lifted strip, both wheels contact."""
    camber = _runup_camber_deg(args)
    if abs(camber) < 1e-6:
        return base_z
    pivot_y = _camber_pivot_y(args)
    return base_z + (float(y) - pivot_y) * math.sin(math.radians(camber))

def landing_slope_segments(peak_x, peak_z, beta_deg, *, length, width, thick, ri, visual=True,
                           post_root_ground=True):
    """Matched downslope landing (A): **lip-continuous** downslope back-face, target landing pitch=beta_deg.

    beta_deg<0 means downslope along +X. Real jumps do not place a downslope patch at a flat landing point,
    but continue a downslope back-face from lip; vehicle lands downstream on that slope ballistically.
    This function starts at (peak_x, peak_z), extends length m by beta_deg, entire slope above grid
    visible and continuous with lip. Score |pitch-beta_deg| not |pitch|.
    """
    if abs(beta_deg) < 1e-6 or length <= 0:
        return []
    theta = math.radians(beta_deg)
    if beta_deg < 0:
        max_visible = max(0.5, (peak_z - 0.05) / abs(math.tan(theta)))
        length = min(float(length), max_visible)
    start_x = peak_x
    end_x = peak_x + float(length)
    z0 = peak_z
    z1 = z0 + (end_x - start_x) * math.tan(theta)
    pts = [(start_x, z0), (end_x, z1)]
    segs = bench.ramp_segments(pts, 100 + ri, width, thick, 1.0)
    for s in segs:
        s["name"] = s["name"].replace(f"r{100 + ri}", f"landing{ri}")
        s["material"] = "track_editor_C_border"
    if not visual:
        return segs

    mid_x = 0.5 * (start_x + end_x)
    mid_z = 0.5 * (z0 + z1)
    slope_len = math.hypot(end_x - start_x, z1 - z0)
    rail_w = 0.45
    for side, y in (("L", width / 2.0 - 1.0), ("R", -width / 2.0 + 1.0)):
        segs.append(_slope_marker(
            f"landing{ri}_rail_{side}", mid_x, mid_z, theta, y=y,
            width=rail_w, along=slope_len, thick=0.35, clearance=0.22))
        for end_name, x, z in (("start", start_x, z0), ("end", end_x, z1)):
            segs.append(_vertical_marker(
                f"landing{ri}_post_{side}_{end_name}", x, z, y=y,
                size=(0.8, 0.8, 1.6), root_ground=post_root_ground))
    return segs

def ballistic_curve(peak_x, peak_z, v0, gamma_deg, clearance, *, g=9.81):
    """Vehicle flight parabola (shifted down clearance) as landing z(dx). Returns (k, slope0, dx_ground, fn_z, fn_tan).

    Parabola (lip origin, dx=x-peak_x): z = peak_z + tanγ·dx - k·dx², k = g/(2·(v0·cosγ)²).
    Landing surface = curve shifted down clearance (~CG-to-wheel vertical), tangential wheel contact.
    dx_ground = where landing surface hits z=0 (SmallGrid); beyond that slope goes underground.
    """
    gx = math.radians(gamma_deg)
    vx = v0 * math.cos(gx)
    vz = v0 * math.sin(gx)
    if vx < 0.1:
        return None
    k = g / (2.0 * vx * vx)
    slope0 = vz / vx
    a, b, c = -k, slope0, peak_z - clearance
    disc = b * b - 4 * a * c
    dx_ground = (-b - math.sqrt(disc)) / (2 * a) if disc > 0 else 0.0

    def fn_z(dx):
        return peak_z - clearance + slope0 * dx - k * dx * dx

    def fn_slope(dx):
        return slope0 - 2.0 * k * dx

    def fn_tan(dx):
        return math.atan(fn_slope(dx))

    return {"k": k, "slope0": slope0, "dx_ground": dx_ground, "vx": vx, "vz": vz,
            "fn_z": fn_z, "fn_slope": fn_slope, "fn_tan": fn_tan}

def landing_hill_profile(peak_z, v0, gamma_deg, clearance, *, face_max_deg=33.0,
                         length=60.0, g=9.81, ds=0.6):
    """Natural dune landing profile z(dx), dx from lip. Three C1-smooth segments:

      segment 1 ballistic face: follows flight parabola (shifted clearance), convex brink + steepening slide,
                  until tangent reaches -face_max_deg (dune angle of repose cap, default 33°);
      segment 2 concave toe runout: arc smooths tangent from -face_max to 0°, tangent to flat at z=0,
                  removes hard "knife into ground" kink.

    Landing surface shifted clearance (z0=peak_z-clearance), start below kicker top by clearance;
    Seam removed by lowering kicker top clearance on main side; this function keeps natural convex downhill shape.
    Returns (pts[(dx,z)], meta). pts monotonic in dx; last point (dx_toe, 0) horizontal tangent.
    """
    bc = ballistic_curve(0.0, peak_z, v0, gamma_deg, clearance, g=g)
    if bc is None:
        return [], {}
    k, slope0 = bc["k"], bc["slope0"]
    face_max = math.radians(abs(face_max_deg))
    s_face = -math.tan(face_max)
    dx_a = (slope0 - s_face) / (2.0 * k)
    z_a = bc["fn_z"](dx_a)
    if z_a <= clearance * 0.5 or dx_a <= 0.0:
        dx_g = bc["dx_ground"]
        n = max(8, int(dx_g / ds))
        pts = [(dx_g * i / n, max(0.0, bc["fn_z"](dx_g * i / n))) for i in range(n + 1)]
        meta = {"mode": "parabola_only", "dx_a": round(dx_a, 2), "z_a": round(z_a, 2),
                "dx_toe": round(dx_g, 2), "face_max_deg": round(face_max_deg, 1)}
        return pts, meta
    R = z_a / (1.0 - math.cos(face_max))
    dx_toe = dx_a + R * math.sin(face_max)
    pts = []
    n1 = max(4, int(dx_a / ds))
    for i in range(n1 + 1):
        dx = dx_a * i / n1
        pts.append((dx, bc["fn_z"](dx)))
    n2 = max(4, int((dx_toe - dx_a) / ds))
    for i in range(1, n2 + 1):
        alpha = face_max * (1.0 - i / n2)
        x = dx_toe - R * math.sin(alpha)
        z = R * (1.0 - math.cos(alpha))
        pts.append((x, max(0.0, z)))
    if dx_toe > length:
        pts = [(dx, z) for (dx, z) in pts if dx <= length]
    meta = {"mode": "parabola+toe", "dx_a": round(dx_a, 2), "z_a": round(z_a, 2),
            "R_toe": round(R, 2), "dx_toe": round(dx_toe, 2),
            "face_max_deg": round(face_max_deg, 1),
            "z_start": round(bc["fn_z"](0.0), 3)}
    return pts, meta

def ballistic_landing_segments(peak_x, peak_z, v0, gamma_deg, *, clearance, length,
                               width, thick, ri, visual=True, g=9.81, face_max_deg=33.0,
                               post_root_ground=True):
    """Natural dune ballistic landing: upper ballistic face (tangential soft landing + convex brink), lower concave runout to flat.

    Replaces unnatural straight downslope (cannot match parabola) and "parabola stab into ground" (hard kink).
    Returns (segs, pts_xz, meta).
    """
    prof, meta = landing_hill_profile(peak_z, v0, gamma_deg, clearance,
                                      face_max_deg=face_max_deg, length=length, g=g)
    if not prof or len(prof) < 2:
        return [], [], {}
    pts = [(peak_x + dx, z) for (dx, z) in prof]
    segs = bench.ramp_segments(pts, 200 + ri, width, thick, 1.0)
    for s in segs:
        s["name"] = s["name"].replace(f"r{200 + ri}", f"balland{ri}")
        s["material"] = "track_editor_C_border"
    meta = dict(meta)
    meta.update({"vx": round(bc_vx(v0, gamma_deg), 2), "n_seg": len(segs)})
    if not visual:
        return segs, pts, meta
    for side, y in (("L", width / 2.0 - 1.0), ("R", -width / 2.0 + 1.0)):
        for j in range(0, len(pts), max(1, len(pts) // 6)):
            x, z = pts[j]
            segs.append(_vertical_marker(f"balland{ri}_post_{side}_{j}", x, z, y=y,
                                         size=(0.7, 0.7, 1.4), root_ground=post_root_ground))
    return segs, pts, meta

def gap_landing_profile(peak_z, v0, gamma_deg, *, clearance_catch=0.4, gap_max=2.5,
                        shape_p=1.5, face_max_deg=33.0, feather_len=6.5,
                        length=80.0, g=9.81, ds=0.5):
    """True jump landing profile (gap mode): from lip, smooth **unimodal-slope** landing hill (natural hill / ski-jump style),
    Vehicle flies freely ~1.5 s above then lands at toe. Profile z(dx), dx from lip.

    Slope σ(t) = σ_max·W(t) (t=dx/L, unimodal: ends 0, mid peak) -> z(dx)=peak_z-∫tan(σ)ds:
      - unimodal slope, no reversal -> monotonic descent, no slope inflection -> **no ridge**;
      - W->0 at ends -> smooth lip join, natural feather to ground, no hard kink or separate runout;
      - L = ballistic landing (T=0), toe at vehicle landing; σ_max solves drop = peak_z;
      - shape_p shifts slope peak: =1 symmetric (smoothest), >1 peak right (steeper landing end), <1 peak left (steeper apex).
    Note: no tangential soft landing (gap vs tangential conflict on single surface; ω_y0 overspin dominates landing),
        traded for ridge-free natural hill shape + true gap. Returns (pts[(dx,z)], meta).
    """
    gx = math.radians(gamma_deg)
    vx = v0 * math.cos(gx); vz = v0 * math.sin(gx)
    if vx < 0.1:
        return [], {}
    k = g / (2.0 * vx * vx); slope0 = vz / vx

    def T(dx):
        return peak_z + slope0 * dx - k * dx * dx

    L = (slope0 + math.sqrt(slope0 * slope0 + 4.0 * k * peak_z)) / (2.0 * k)
    if L <= 0:
        return [], {}

    def W(t):
        tt = min(1.0, max(0.0, t)) ** shape_p
        return math.sin(math.pi * tt)

    nn = max(60, int(L / ds))
    def drop(sm_deg):
        sm = math.radians(sm_deg); s = 0.0
        for i in range(nn):
            s += math.tan(sm * W((i + 0.5) / nn)) * (L / nn)
        return s
    lo, hi = 1.0, 80.0
    for _ in range(44):
        mid = 0.5 * (lo + hi)
        if drop(mid) < peak_z:
            lo = mid
        else:
            hi = mid
    sigma_max = 0.5 * (lo + hi)
    sm_rad = math.radians(sigma_max)
    pts = [(0.0, peak_z)]; z = peak_z; max_face = 0.0
    for i in range(1, nn + 1):
        sloc = sm_rad * W((i - 0.5) / nn)
        max_face = max(max_face, sloc)
        z -= math.tan(sloc) * (L / nn)
        pts.append((L * i / nn, max(0.0, z)))
    if pts[-1][1] > 1e-3:
        pts.append((pts[-1][0] + 1.0, 0.0))
    if length > pts[-1][0] + 1e-6:
        pts.append((float(length), 0.0))
    pts = [(dx, zz) for (dx, zz) in pts if dx <= length]
    gap_samples = [T(dx) - zz for (dx, zz) in pts if dx < L - 0.5]
    land_slope_deg = -round(math.degrees(sm_rad * W(0.98)), 1)
    meta = {"mode": "gap_valley", "dx_land": round(L, 2),
            "z_land": 0.0, "land_slope_deg": land_slope_deg,
            "gap_max_actual": round(max(gap_samples), 2) if gap_samples else 0.0,
            "gap_min_flight": round(min(gap_samples), 2) if gap_samples else 0.0,
            "sigma_max_deg": round(sigma_max, 1), "max_face_deg": round(math.degrees(max_face), 1),
            "shape_p": shape_p, "clearance_catch": clearance_catch,
            "dx_toe": round(pts[-1][0], 2), "z_start": round(peak_z, 2)}
    return pts, meta

def gap_landing_segments(peak_x, peak_z, v0, gamma_deg, *, clearance_catch, gap_max,
                         shape_p, feather_len, length, width, thick, ri, visual=True, g=9.81,
                         face_max_deg=33.0, post_root_ground=True):
    """Gap-mode landing (steep valley + tangential catch + lower feather). Returns (segs, pts_xz, meta)."""
    prof, meta = gap_landing_profile(peak_z, v0, gamma_deg, clearance_catch=clearance_catch,
                                     gap_max=gap_max, shape_p=shape_p, feather_len=feather_len,
                                     face_max_deg=face_max_deg, length=length, g=g)
    if not prof or len(prof) < 2:
        return [], [], {}
    pts = [(peak_x + dx, z) for (dx, z) in prof]
    segs = bench.ramp_segments(pts, 300 + ri, width, thick, 1.0)
    for s in segs:
        s["name"] = s["name"].replace(f"r{300 + ri}", f"gapland{ri}")
        s["material"] = "track_editor_C_border"
    meta = dict(meta); meta.update({"vx": round(bc_vx(v0, gamma_deg), 2), "n_seg": len(segs)})
    if not visual:
        return segs, pts, meta
    for side, y in (("L", width / 2.0 - 1.0), ("R", -width / 2.0 + 1.0)):
        for j in range(0, len(pts), max(1, len(pts) // 6)):
            x, z = pts[j]
            segs.append(_vertical_marker(f"gapland{ri}_post_{side}_{j}", x, z, y=y,
                                         size=(0.7, 0.7, 1.4), root_ground=post_root_ground))
    return segs, pts, meta

def gap_ramp_landing_segments(peak_x, peak_z, v0, gamma_deg, beta_deg, *, clearance,
                              gap_run, length, width, thick, ri, visual=True, g=9.81,
                              post_root_ground=True):
    """Gap + tilted catch slope (downslope generalization): vehicle crosses **air gap (true free flight gap_run m)** from lip,
    lands on **straight downslope catch top** at angle β (nose-down); DART must match pitch to β = tests airborne pitch authority on steeper landing.

    vs existing modes (confirmed): gap lands on ~0° feathered valley floor (target_pitch≈0, no slope-steepness test);
    ballistic follows trajectory without gap. This mode fills gap: true air gap + tilted landing surface.

    Geometry (gap_run decoupled; tangential placement covers trajectory, discarded): catch **top** at ballistic dx=gap_run
    (z_top=trajectory height-clearance, wheels touch at top), slope extends down at β. **β independent of landing point ->
    sweep β at fixed airtime (=gap_run/vx), decouple steepness from airtime**; gap_run is separate length/airtime axis.
    lip->ramp top is air gap (no mesh, true free flight; short fall into gap = fail = real envelope).
    Feasibility: gap_run past apex (descent leg) and z_top>0, else infeasible. Returns (segs, pts, meta)."""
    cur = ballistic_curve(peak_x, peak_z, v0, gamma_deg, 0.0, g=g)
    if cur is None:
        return [], [], {"infeasible": True, "reason": "vx~0"}
    slope0 = cur["slope0"]; k = cur["k"]; fn_z = cur["fn_z"]; fn_slope = cur["fn_slope"]
    apex_dx = slope0 / (2.0 * k)
    if gap_run <= apex_dx + 0.5:
        return [], [], {"infeasible": True, "reason": "gap_run_before_apex",
                        "gap_run": round(gap_run, 2), "apex_dx": round(apex_dx, 2)}
    z_top = fn_z(gap_run) - clearance
    if z_top <= 0.05:
        return [], [], {"infeasible": True, "reason": "ramp_top_below_ground",
                        "gap_run": round(gap_run, 2), "z_top": round(z_top, 2)}
    x_top = peak_x + gap_run
    segs = landing_slope_segments(x_top, z_top, -abs(beta_deg), length=length,
                                  width=width, thick=thick, ri=ri, visual=visual,
                                  post_root_ground=post_root_ground)
    pts = [(peak_x, peak_z), (x_top, z_top)]
    meta = {"mode": "gap_ramp", "beta_deg": -abs(beta_deg), "land_slope_deg": -abs(beta_deg),
            "dx_land": round(gap_run, 2), "x_land": round(x_top, 2), "z_land": round(z_top, 2),
            "airtime_s": round(gap_run / cur["vx"], 3), "gap_run": round(gap_run, 2),
            "traj_slope_at_land_deg": round(math.degrees(math.atan(fn_slope(gap_run))), 1),
            "v0_design": round(v0, 2), "gamma_deg": gamma_deg, "vx": round(cur["vx"], 2),
            "n_seg": len(segs)}
    return segs, pts, meta

def _valley_crest_pts(peak_z, lip_exit_slope_deg, beta_deg, dx_c, n=None):
    """Valley lip rounding bulge (gap-style smooth join): quadratic Bezier on [0,dx_c] blends lip exit slope s0
    smooth transition to -β, **bulges above -β straight line**, end (dx_c, peak_z-tanβ·dx_c) tangent-merge to -β.

    Key: -β line still through lip -> landing/feather/floor/auto-rise use original through-lip closed form; crest is lip rounding only,
    landing geometry unchanged (vehicle lands on constant -β line beyond dx_c, flies over crest). Returns crest points [(dx,z)] (t>0, not dx=0)."""
    s0 = math.radians(max(0.0, float(lip_exit_slope_deg)))
    br = math.radians(abs(float(beta_deg))); tb = math.tan(br)
    if dx_c <= 0.1:
        return []
    if n is None:
        n = max(24, int(dx_c / 0.15))
    P0 = (0.0, peak_z)
    P3 = (dx_c, peak_z - tb * dx_c)
    h = dx_c
    B1 = (P0[0] + (h / 3.0) * math.cos(s0), P0[1] + (h / 3.0) * math.sin(s0))
    B2 = (P3[0] - (h / 3.0) * math.cos(br), P3[1] - (h / 3.0) * (-math.sin(br)))
    pts = []
    for i in range(1, n + 1):
        t = i / n; u = 1.0 - t
        x = u**3 * P0[0] + 3 * u * u * t * B1[0] + 3 * u * t * t * B2[0] + t**3 * P3[0]
        z = u**3 * P0[1] + 3 * u * u * t * B1[1] + 3 * u * t * t * B2[1] + t**3 * P3[1]
        pts.append((x, z))
    return pts

def valley_landing_segments(peak_x, peak_z, v0, gamma_deg, beta_deg, *, clearance,
                            floor_depth, length_req, width, thick, ri, visual=True, g=9.81,
                            feather_len=6.0, floor_run=14.0, lip_exit_slope_deg=0.0,
                            crest_blend_len=6.0, post_root_ground=True):
    """Continuous downslope landing (valley: baseline slope + feathered floor + convex lip join): from lip through
    **convex crest** (lip exit->-β, gap-style end slope fade, removes lip ridge kink) -> **constant -β straight slope**
    (vehicle lands ballistically; DART matches pitch to β = tests landing-slope airborne pitch authority) -> **concave toe feather** (tangent -β->0°)
    -> **flat valley floor** (anchored at smallgrid z=0, no underground drop-off).

    Four C1 segments: 1) convex crest (s0->-β) 2) straight -β (covers landing + margin) 3) concave toe (R=feather_len/sinβ, -β->0°)
    4) flat floor (floor_run). Crest shape independent of peak_z -> landing still closed form. floor_depth kept for compat (no dig currently).
    Returns (segs, pts, meta)."""
    beta = abs(float(beta_deg))
    br = math.radians(beta)
    tb = math.tan(br)
    if tb < 1e-4:
        return [], [], {"infeasible": True, "reason": "beta~0"}
    cur = ballistic_curve(peak_x, peak_z, v0, gamma_deg, clearance, g=g)
    if cur is None:
        return [], [], {"infeasible": True, "reason": "vx~0"}
    k = cur["k"]; slope0 = cur["slope0"]
    disc = (slope0 + tb) ** 2 - 4.0 * k * clearance
    if disc <= 0:
        return [], [], {"infeasible": True, "reason": "no_landing_crossing"}
    dx_land = ((slope0 + tb) + math.sqrt(disc)) / (2.0 * k)
    R = max(2.0, float(feather_len)) / max(math.sin(br), 1e-3)
    dz_fillet = R * (1.0 - math.cos(br))
    dx_straight = max(float(length_req or 0.0), dx_land + 6.0)
    z_straight = peak_z - dx_straight * tb
    floor_z = z_straight - dz_fillet
    if floor_z < 0.0:
        floor_z = 0.0
        z_straight = dz_fillet
        dx_straight = max(0.5, (peak_z - z_straight) / tb)
    dx_toe = dx_straight + R * math.sin(br)
    dx_c = min(float(crest_blend_len), max(0.0, dx_straight - 1.0))
    crest_pts = _valley_crest_pts(peak_z, lip_exit_slope_deg, beta, dx_c)
    ds = 0.8
    prof = [(0.0, peak_z)] + list(crest_pts)
    n1 = max(2, int((dx_straight - dx_c) / ds))
    for i in range(1, n1 + 1):
        dx = dx_c + (dx_straight - dx_c) * i / n1
        prof.append((dx, peak_z - dx * tb))
    n2 = max(3, int((dx_toe - dx_straight) / ds))
    for i in range(1, n2 + 1):
        alpha = br * (1.0 - i / n2)
        prof.append((dx_toe - R * math.sin(alpha), floor_z + R * (1.0 - math.cos(alpha))))
    prof.append((dx_toe + float(floor_run), floor_z))
    pts = [(peak_x + dx, z) for (dx, z) in prof]
    segs = bench.ramp_segments(pts, 100 + ri, width, thick, 1.0)
    for s in segs:
        s["name"] = s["name"].replace(f"r{100 + ri}", f"valley{ri}")
        s["material"] = "track_editor_C_border"
    total_len = dx_toe + float(floor_run)
    meta = {"mode": "valley", "beta_deg": -beta, "land_slope_deg": -beta,
            "dx_land": round(dx_land, 2), "x_land": round(peak_x + dx_land, 2),
            "z_land": round(peak_z - dx_land * tb, 2), "airtime_s": round(dx_land / cur["vx"], 3),
            "floor_z": round(floor_z, 2), "R_toe": round(R, 2), "dx_toe": round(dx_toe, 2),
            "dx_crest": round(dx_c, 2), "lip_exit_slope_deg": round(float(lip_exit_slope_deg), 1),
            "feather_len": round(R * math.sin(br), 2), "floor_run": round(float(floor_run), 2),
            "length": round(total_len, 2), "v0_design": round(v0, 2),
            "gamma_deg": gamma_deg, "vx": round(cur["vx"], 2), "n_seg": len(segs)}
    if dx_straight < dx_land + 0.05:
        meta["warn"] = "straight_short_land_on_fillet"
    if visual and total_len > 1.0:
        segs = _append_ramp_rail_visuals(segs, pts, prefix="valley", ri=ri, width=width,
                                         post_root_ground=post_root_ground)
    return segs, pts, meta

def bc_vx(v0, gamma_deg):
    return v0 * math.cos(math.radians(gamma_deg))

def build_ramp_pts(args, base_x, angle):
    """Dispatch ramp polyline by --ramp-mode. tabletop=original convex crest with abrupt lip; kicker=straight face + controlled lip arc (B1).

    kicker lip radius: use --lip-radius>0 directly; if ==0 auto-compute from --lip-omega-target-dps
    R_lip=v_peak/|omega_target| (v_peak from ramp_speeds takeoff speed for this angle)."""
    mode = getattr(args, "ramp_mode", "tabletop")
    if mode == "kicker":
        R_lip = float(args.lip_radius)
        if R_lip <= 0.0:
            vp, _vb, _R = bench.ramp_speeds(args.rise)
            v_peak = vp[ANGLES.index(angle)] if angle in ANGLES else vp[0]
            omega = max(1e-3, abs(math.radians(args.lip_omega_target_dps)))
            R_lip = v_peak / omega
        efl = float(args.entry_fillet_len)
        r_min = float(getattr(args, "entry_fillet_radius", 0.0) or 0.0)
        n_entry = int(args.n_entry)
        if r_min > 0.0:
            efl_adapt = r_min * math.sin(math.radians(angle))
            if efl_adapt > efl:
                n_entry = max(n_entry, int(math.ceil(n_entry * efl_adapt / max(efl, 1e-6))))
                efl = efl_adapt
        return bench.kicker_polyline(
            base_x, angle, args.rise, efl, R_lip,
            args.lip_sweep_deg, n_entry, args.n_straight, args.n_lip, args.sink), R_lip
    fl = tp.fillet_len_for(angle, args.fillet_rise_budget, args.fillet_len_max)
    return bench.ramp_polyline(base_x, angle, args.rise, fl, args.n_fillet,
                               args.n_main, args.sink), None

def _set_wheel_factors(veh, fl, fr, rl, rr):
    """Per-motor throttleFactor (differential actuator; probe confirmed independent+invertible).
    factor>0=forward drive (wheels forward -> nose-up reaction); factor<0=reverse (wheels back -> nose-down).
    Global throttle supplies torque; sign from factor (_dart_4motor_diff_probe)."""
    try:
        nj._vlua(veh, f"electrics.values.throttleFactorFL={float(fl):.3f} electrics.values.throttleFactorFR={float(fr):.3f} "
                      f"electrics.values.throttleFactorRL={float(rl):.3f} electrics.values.throttleFactorRR={float(rr):.3f}")
    except Exception:
        pass
