"""Ramp geometry and BeamNG gameplay-live helpers for ``scripts/darts_bench.py``.

Library only (no standalone bench): takeoff-speed tables, tabletop/kicker
polylines, mesh segments, and the menu-overlay / gameplay-live probes the
experiment runner calls between jumps.
"""
from __future__ import annotations
import math, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import scripts._ramp_feather as rf      # noqa: E402  step / scene infra
import scripts._natural_jump as nj      # noqa: E402  poll / Lua helpers

ANGLES = [10, 15, 20, 25, 30, 35, 40, 45]
VX_CONST = 8.33   # 8.33 m/s = 30 km/h approach speed; v_peak = VX / cos(theta)
G = 9.81

def ramp_speeds(rise):
    """Return per-angle (v_peak, v_base, R_flight): v_peak = VX/cos(theta),
    v_base = sqrt(v_peak^2 + 2*g*h) * 1.10, R = flight range from height h."""
    vp, vb, R = [], [], []
    for th in ANGLES:
        t = math.radians(th)
        v_peak = VX_CONST / math.cos(t)
        v_base = math.sqrt(v_peak * v_peak + 2 * G * rise) * 1.10
        vz = v_peak * math.sin(t)
        tf = (vz + math.sqrt(vz * vz + 2 * G * rise)) / G
        vp.append(v_peak); vb.append(v_base); R.append(VX_CONST * tf)
    return vp, vb, R

def force_gameplay(bng):
    """Clear the BeamNG menu overlay and force the scenario into gameplay.
    Fires several setGameState/guihook lua variants, resumes the sim, then
    reads back the gamestate. Returns (gameplay_view, menu_cleared)."""
    cleared = False
    # Try multiple guihook/setGameState variants; any one may clear the overlay
    for _lua in (
        "core_gamestate.setGameState('scenario','scenario','freeroam')",
        "guihooks.trigger('ChangeState', {state='play', params={}})",
        "guihooks.trigger('MenuHide')",
        "core_gamestate.setGameState('scenario','play','')",
        "guihooks.trigger('ChangeState', {state='play'}) guihooks.trigger('MenuHide')",
        "if core_gamestate and core_gamestate.requestExitOptionsMenu then core_gamestate.requestExitOptionsMenu() end",
    ):
        try:
            bng.queue_lua_command(_lua); cleared = True
        except Exception:
            pass
    for attr in ("control.resume", "resume"):
        try:
            obj = bng
            for a in attr.split("."):
                obj = getattr(obj, a)
            obj(); break
        except Exception:
            continue
    gs = None
    try:
        gs = bng.control.get_gamestate()
    except Exception:
        pass
    gameplay = bool(gs) and ("menu" not in str(gs).lower())
    return gameplay, cleared

def bring_beamng_foreground(*, hard=False):
    """Bring the BeamNG window to the foreground (Windows only).

    Works around SetForegroundWindow restrictions: 1) ALT keydown/up nudge;
    2) AttachThreadInput to the foreground/target threads; 3) hard=True adds
    a minimize/restore cycle. Returns True if any window was handled."""
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        u32 = ctypes.windll.user32
        k32 = ctypes.windll.kernel32
        hwnds = _find_beamng_hwnds(u32)         # excludes the CONSOLE window
        ok = False
        for h in hwnds:
            # 1) ALT key nudge so Windows permits SetForegroundWindow
            u32.keybd_event(0x12, 0, 0, 0)      # ALT down
            u32.keybd_event(0x12, 0, 2, 0)      # ALT up
            if hard:                            # 3) minimize/restore cycle
                u32.ShowWindow(h, 6)            # SW_MINIMIZE
                u32.ShowWindow(h, 9)            # SW_RESTORE
            else:
                u32.ShowWindow(h, 9)            # SW_RESTORE
            # 2) AttachThreadInput so SetForegroundWindow succeeds
            fg = u32.GetForegroundWindow()
            cur_tid = k32.GetCurrentThreadId()
            fg_tid = u32.GetWindowThreadProcessId(fg, None)
            tgt_tid = u32.GetWindowThreadProcessId(h, None)
            u32.AttachThreadInput(cur_tid, fg_tid, True)
            u32.AttachThreadInput(cur_tid, tgt_tid, True)
            u32.BringWindowToTop(h)
            u32.SetForegroundWindow(h)
            u32.SetActiveWindow(h)
            u32.SetFocus(h)
            u32.AttachThreadInput(cur_tid, fg_tid, False)
            u32.AttachThreadInput(cur_tid, tgt_tid, False)
            ok = True
        return ok
    except Exception:
        return False

def _find_beamng_hwnds(u32):
    """Return visible BeamNG top-level HWNDs sorted by client area (largest
    first), excluding the [CONSOLE] window."""
    import ctypes
    from ctypes import wintypes
    cands = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _lp):
        n = u32.GetWindowTextLengthW(hwnd)
        if n:
            buf = ctypes.create_unicode_buffer(n + 1)
            u32.GetWindowTextW(hwnd, buf, n + 1)
            t = buf.value
            if "BeamNG" in t and "CONSOLE" not in t.upper() and u32.IsWindowVisible(hwnd):
                r = wintypes.RECT()
                u32.GetClientRect(hwnd, ctypes.byref(r))
                area = (r.right - r.left) * (r.bottom - r.top)
                cands.append((area, hwnd))
        return True

    u32.EnumWindows(_cb, 0)
    cands.sort(reverse=True)                      # largest client area first
    return [h for _a, h in cands]

def _postmsg_click(u32, h, cx, cy):
    """Post a synthetic left click at client coords (cx, cy) via PostMessage.
    Needs no focus, z-order change, or real cursor movement."""
    lp = ((int(cy) & 0xFFFF) << 16) | (int(cx) & 0xFFFF)
    u32.PostMessageW(h, 0x0200, 0, lp)            # WM_MOUSEMOVE
    u32.PostMessageW(h, 0x0201, 0x0001, lp)       # WM_LBUTTONDOWN (MK_LBUTTON)
    time.sleep(0.04)
    u32.PostMessageW(h, 0x0202, 0, lp)            # WM_LBUTTONUP

def postmsg_click_beamng_px(points_px, *, log=print):
    """Sweep PostMessage clicks over pixel points in the BeamNG client area,
    e.g. to hit the menu play button. points_px=[(cx,cy),...]; out-of-range
    points are skipped. Returns the number of clicks sent."""
    if not sys.platform.startswith("win"):
        return 0
    try:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        hwnds = _find_beamng_hwnds(u32)
        if not hwnds:
            return 0
        h = hwnds[0]
        rect = wintypes.RECT(); u32.GetClientRect(h, ctypes.byref(rect))
        cw, ch = rect.right - rect.left, rect.bottom - rect.top
        n = 0
        for (cx, cy) in points_px:
            if 0 <= cx < cw and 0 <= cy < ch:
                _postmsg_click(u32, h, cx, cy); n += 1
        log(f"[bng-click] postmsg swept {n} px-points on clientWH=({cw}x{ch})", flush=True)
        return n
    except Exception as e:
        log(f"[bng-click] px err={e}", flush=True)
        return 0

def postmsg_click_beamng_frac(points_frac, *, log=print):
    """Like postmsg_click_beamng_px but with fractional client coords, robust
    to window size changes. points_frac=[(fx,fy),...]."""
    if not sys.platform.startswith("win"):
        return 0
    try:
        import ctypes
        from ctypes import wintypes
        u32 = ctypes.windll.user32
        hwnds = _find_beamng_hwnds(u32)
        if not hwnds:
            return 0
        h = hwnds[0]
        rect = wintypes.RECT(); u32.GetClientRect(h, ctypes.byref(rect))
        cw, ch = rect.right - rect.left, rect.bottom - rect.top
        n = 0
        for (fx, fy) in points_frac:
            _postmsg_click(u32, h, int(cw * fx), int(ch * fy)); n += 1
        log(f"[bng-click] postmsg swept {n} frac-points on clientWH=({cw}x{ch})", flush=True)
        return n
    except Exception as e:
        log(f"[bng-click] frac err={e}", flush=True)
        return 0

def ensure_gameplay_live(bng, veh, *, tries=12, step_each=12, move_eps=0.5, log=print):
    """Verify the sim is really in live gameplay, not stuck behind a menu.

    scenario.start() may leave the gamestate at menu/None while readiness
    probes look fine; the symptom is a car that never moves. Each try:
    foreground + force_gameplay, then a short full-throttle pulse; ground
    speed > move_eps counts as live. Returns False after `tries` failures
    (caller should abort). Re-run after any teleport-heavy sequence.
    """
    # gamestate readback alone is unreliable; use "does the car move under
    # throttle" as ground truth: 0.4s pulse, gspd > 0 = live, ~0 = menu/frozen
    for k in range(tries):
        bring_beamng_foreground(hard=(k >= 1))   # hard restore from try 2 on
        # The spawn-time overlay can ignore lua force_gameplay; from try 2 on
        # also sweep PostMessage clicks near the menu play button (env gated).
        if k >= 1 and os.environ.get("DART_BNG_AUTOCLICK", "1") == "1":
            # Sweep several fractional points around the play button: window
            # size and UI scale vary, and PostMessage avoids stealing focus
            # from other windows (unlike SetCursorPos + mouse_event).
            postmsg_click_beamng_frac(
                [(0.0127, 0.027), (0.018, 0.035), (0.010, 0.022), (0.024, 0.045), (0.0127, 0.055)],
                log=log)
        force_gameplay(bng)
        rf._step(bng, step_each)
        # dart_4motor EV: re-assert per-wheel throttleFactor=1 (may get reset)
        nj._vlua(veh, "electrics.values.throttleFactorFL=1 electrics.values.throttleFactorFR=1 "
                      "electrics.values.throttleFactorRL=1 electrics.values.throttleFactorRR=1")
        for _ in range(40):                  # 0.4s full-throttle pulse
            try: veh.control(throttle=1.0, brake=0.0, steering=0.0)
            except Exception: pass
            rf._step(bng, 1)
        st1 = nj._poll(veh); vel = st1.get("vel") or (0, 0, 0)
        gspd = math.hypot(float(vel[0]), float(vel[1]))
        try: veh.control(throttle=0.0, brake=1.0, steering=0.0)
        except Exception: pass
        rf._step(bng, 3)
        if gspd > move_eps:                  # car moved -> gameplay is live
            log(f"[bng] gameplay LIVE (try {k+1}: pulse_gspd={gspd:.2f}m/s, foreground+force_gameplay)")
            return True
        log(f"[bng] gameplay NOT live (try {k+1}/{tries}: pulse_gspd={gspd:.2f}m/s), "
            f"foreground+force_gameplay retry", flush=True)
    log(f"[bng] WARN gameplay NOT live after {tries} tries = /, abort", flush=True)
    return False

def ramp_polyline(base_x, angle, rise, fillet_len, n_fillet, n_main, sink):
    th = math.radians(angle)
    R = fillet_len / math.sin(th)
    pts = []
    for k in range(n_fillet + 1):
        phi = th * k / n_fillet
        pts.append((base_x + R * math.sin(phi), R * (1.0 - math.cos(phi))))
    xf, zf = pts[-1]
    main_h = (rise - zf) / math.tan(th)        # main run so the ramp tops at z=rise
    # skip the main run when the fillet already reaches rise (avoid a zero-size ProceduralCube)
    if main_h > 0.1:
        for j in range(1, n_main + 1):
            x = xf + main_h * j / n_main
            pts.append((x, zf + (x - xf) * math.tan(th)))
    return [(x, z - sink) for (x, z) in pts]

def kicker_polyline(base_x, alpha_deg, rise, entry_fillet_len, lip_radius,
                    lip_sweep_deg, n_entry, n_straight, n_lip, sink):
    """Build a kicker ramp polyline (B1): flat -> concave entry fillet
    (0 -> alpha) -> straight at alpha -> convex lip arc of radius R_lip.

    Unlike a tabletop, the lip arc shapes the takeoff: the exit angle is
    roughly alpha - lip_sweep and the initial pitch rate scales with
    -v_peak / R_lip. Returns [(x, z)] ending at the lip (z = rise); see
    scripts/_dart_jump_geometry_design.py for the derivation.
    """
    a = math.radians(alpha_deg)
    sweep = math.radians(lip_sweep_deg)                 # >0: flatten exit; <0: kick-up
    kick_up = sweep < 0
    asw = abs(sweep)
    exit_ang = (a + asw) if kick_up else (a - asw)      # lip exit angle
    pts = []
    # 1) concave entry fillet from 0 to alpha (same as the tabletop entry)
    R_in = entry_fillet_len / math.sin(a)
    for k in range(n_entry + 1):
        phi = a * k / n_entry
        pts.append((base_x + R_in * math.sin(phi), R_in * (1.0 - math.cos(phi))))
    xe, ze = pts[-1]
    # 2) lip arc height below z=rise: R*(cos(exit)-cos(alpha)), sign by direction
    dz_lip = lip_radius * (math.cos(exit_ang) - math.cos(a)) if not kick_up \
        else lip_radius * (math.cos(a) - math.cos(exit_ang))
    z_arc0 = rise - dz_lip                               # z where the lip arc starts
    # 3) straight section at alpha joining the fillet end to the lip arc start
    dz_straight = z_arc0 - ze
    if dz_straight > 0.05:
        run = dz_straight / math.tan(a)
        for j in range(1, n_straight + 1):
            x = xe + run * j / n_straight
            pts.append((x, ze + (x - xe) * math.tan(a)))
    x_s, z_s = pts[-1]
    # 4) lip arc of radius R_lip sweeping alpha -> exit_ang, ending at z=rise
    #    convex (flatten): psi decreases from alpha toward alpha - sweep
    #    concave (kick-up): psi increases from alpha toward alpha + kick
    for k in range(1, n_lip + 1):
        if kick_up:
            psi = a + asw * k / n_lip
            x = x_s + lip_radius * (math.sin(psi) - math.sin(a))
            z = z_s + lip_radius * (math.cos(a) - math.cos(psi))
        else:
            psi = a - asw * k / n_lip
            x = x_s + lip_radius * (math.sin(a) - math.sin(psi))
            z = z_s + lip_radius * (math.cos(psi) - math.cos(a))
        pts.append((x, z))
    return [(x, z - sink) for (x, z) in pts]

def ramp_segments(pts, ri, width, thick, overlap):
    T = thick
    segs = []
    for i in range(len(pts) - 1):
        x0, z0 = pts[i]; x1, z1 = pts[i + 1]
        theta = math.atan2(z1 - z0, x1 - x0)
        slope_len = math.hypot(x1 - x0, z1 - z0) * overlap
        mx, mz = (x0 + x1) / 2.0, (z0 + z1) / 2.0
        cx = mx + (T / 2.0) * math.sin(theta)
        cz = mz - (T / 2.0) * math.cos(theta)
        rot = (0.0, math.sin(theta / 2.0), 0.0, math.cos(theta / 2.0))
        segs.append({"name": f"r{ri}s{i}", "pos": (cx, 0.0, cz),
                     "size": (width, slope_len, T), "rot": rot})
    return segs
