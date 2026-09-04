#!/usr/bin/env python3
"""Actuator-layer probe for the active-suspension platform.

Measures three things on the standing vehicle, which is what the reported
actuator authority in the paper comes from:

  1. Height and attitude authority. The four corner hydros
     (dartsSuspFL/FR/RL/RR, each in [-1, +1]) are driven together for ride
     height, front-up/rear-down for the pitch channel the takeoff preset
     uses, and left-up/right-down for roll.
  2. Per-corner independence. Lifting FL alone must move pitch and roll
     together, with the correct signs.
  3. Semi-active damper modes. Switching adaptiveFront/RearDamper between
     soft and hard must be acknowledged, and a 0.8 m drop test must leave a
     larger post-landing oscillation in soft than in hard.

Usage: python scripts/actuator_probe.py [--pc vehicles/sbr/darts_4motor.pc]
Output: data/derived/actuator_probe.json
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LUA_POSE = ("local r,p,y=obj:getRollPitchYaw() local pos=obj:getPosition() "
            "return string.format('%.5f,%.5f,%.5f,%.4f',r,p,y,pos.z)")


def _vlua(veh, c):
    try:
        return veh.queue_lua_command(c, response=True)
    except Exception:
        return None


def pose(veh):
    r = _vlua(veh, LUA_POSE)
    try:
        a, b, c, z = (float(x) for x in str(r).strip().split(","))
        return a, b, c, z
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def set_susp(veh, fl, fr, rl, rr):
    _vlua(veh, f"electrics.values.dartsSuspFL={fl:.3f} electrics.values.dartsSuspFR={fr:.3f} "
               f"electrics.values.dartsSuspRL={rl:.3f} electrics.values.dartsSuspRR={rr:.3f}")


def set_damper_mode(veh, mode):
    out = []
    for name in ("adaptiveFrontDamper", "adaptiveRearDamper"):
        r = _vlua(veh, f"local c=controller.getController('{name}') "
                       f"if c then c.setParameters({{damperMode='{mode}'}}) return 'ok' else return 'nil' end")
        out.append(str(r).strip() if r is not None else "err")
    return out


def steps(bng, n):
    for _ in range(n):
        try:
            bng.control.step(1)
        except Exception:
            pass


def ddeg(a, b):
    d = math.degrees(b - a)
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return round(d, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc", type=str, default="vehicles/sbr/darts_4motor.pc")
    args = ap.parse_args()

    from beamngpy import Scenario, Vehicle  # type: ignore
    from data_pipeline.beamng_session import (
        BeamNGSessionConfig, BeamNGSession, wait_scenario_ready, ensure_freerun)
    cfg = BeamNGSessionConfig.from_env(
        home=(BeamNGSessionConfig.from_env().home or r"C:\BeamNG.tech"),
        linux_binary=None, open_timeout_sec=240)
    spawn_pos = (0.0, 0.0, 0.5)
    spawn_rot = (0.0, 0.0, 0.0, 1.0)
    result = {"pc": args.pc}
    with BeamNGSession(cfg) as bng:
        sc = Scenario("smallgrid", "darts_actuator_probe")
        veh = Vehicle("ego", model="sbr", part_config=args.pc)
        sc.add_vehicle(veh, pos=spawn_pos, rot_quat=spawn_rot)
        sc.make(bng); bng.scenario.load(sc); bng.scenario.start()
        wait_scenario_ready(bng, expected_vid="ego"); ensure_freerun(bng)
        try:
            bng.settings.set_deterministic(100)
        except Exception:
            pass

        # Spawn liveness gate: the vehicle Lua channel has to answer, or every
        # reading would be a silent zero.
        alive = _vlua(veh, "return 'alive'")
        if str(alive).strip().strip("'\"") != "alive":
            raise SystemExit(
                f"[probe] FATAL spawn/Lua channel is dead (alive probe={alive!r}). "
                "Either this attached to a BeamNG instance started before the mod was "
                "deployed, or the jbeam failed to parse and no vehicle was created. "
                "Kill every BeamNG.tech.x64.exe and rerun; if it persists, check "
                "beamng.log for a jbeam error.")

        # ---- 1) standing baseline ----
        steps(bng, 200)
        r0, p0, y0, z0 = pose(veh)
        print(f"[probe] baseline roll={math.degrees(r0):.2f}° pitch={math.degrees(p0):.2f}° z={z0:.4f}", flush=True)

        # ---- 2) height and attitude arms; the hydro rate of 2/s reaches full
        #         travel well inside 250 ticks ----
        arms = {
            "lift_all":  (1, 1, 1, 1),
            "drop_all":  (-1, -1, -1, -1),
            "nose_up":   (1, 1, -1, -1),   # front up, rear down: the takeoff preset
            "nose_down": (-1, -1, 1, 1),
            "roll_left": (1, -1, 1, -1),   # left up, right down
            "corner_FL": (1, 0, 0, 0),     # per-corner independence
        }
        res = {}
        for arm, (fl, fr, rl, rr) in arms.items():
            set_susp(veh, fl, fr, rl, rr)
            steps(bng, 250)
            r1, p1, _, z1 = pose(veh)
            res[arm] = {"droll_deg": ddeg(r0, r1), "dpitch_deg": ddeg(p0, p1),
                        "dz_m": round(z1 - z0, 4)}
            print(f"[probe] arm={arm} fac=({fl},{fr},{rl},{rr}) "
                  f"dz={res[arm]['dz_m']}m dpitch={res[arm]['dpitch_deg']}° droll={res[arm]['droll_deg']}°",
                  flush=True)
            set_susp(veh, 0, 0, 0, 0)
            steps(bng, 200)
        result["height_arms"] = res

        # ---- 3) damper drop test: 0.8 m drop, ride-height swing over the 300
        #         ticks after landing ----
        drop = {}
        for mode in ("soft", "hard"):
            acks = set_damper_mode(veh, mode)
            try:
                veh.teleport((spawn_pos[0], spawn_pos[1], spawn_pos[2] + 0.8), spawn_rot, reset=True)
            except Exception:
                pass
            steps(bng, 5)
            acks2 = set_damper_mode(veh, mode)  # a teleport reset can clear it
            zs = []
            for _ in range(300):
                steps(bng, 1)
                zs.append(pose(veh)[3])
            # Peak-to-peak over the settled tail is the oscillation amplitude.
            tail = zs[100:]
            drop[mode] = {"ack": acks, "ack_after_reset": acks2,
                          "z_pp_mm": round((max(tail) - min(tail)) * 1000, 1),
                          "z_min": round(min(zs), 4)}
            print(f"[probe] drop mode={mode} ack={acks2} z_pp={drop[mode]['z_pp_mm']}mm", flush=True)
        result["damper_drop"] = drop

        # ---- 4) verdict ----
        lift = res["lift_all"]["dz_m"]; dropz = res["drop_all"]["dz_m"]
        pitch_auth = abs(res["nose_up"]["dpitch_deg"]) + abs(res["nose_down"]["dpitch_deg"])
        roll_auth = abs(res["roll_left"]["droll_deg"])
        fl_ok = (res["corner_FL"]["dpitch_deg"] * res["nose_up"]["dpitch_deg"] > 0 and
                 res["corner_FL"]["droll_deg"] * res["roll_left"]["droll_deg"] > 0)
        damper_ok = ("ok" in drop["soft"]["ack_after_reset"] and
                     drop["soft"]["z_pp_mm"] > drop["hard"]["z_pp_mm"])
        result["verdict"] = {
            "height_authority_m": {"lift": lift, "drop": dropz},
            "pitch_authority_deg_sum": round(pitch_auth, 2),
            "roll_authority_deg": round(roll_auth, 2),
            "height_active": lift > 0.01 and dropz < -0.01,
            "per_corner_independent": fl_ok,
            "damper_modes_effective": damper_ok,
        }
        print(f"[probe] === VERDICT === {json.dumps(result['verdict'])}", flush=True)

    out = REPO / "data/derived/actuator_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
