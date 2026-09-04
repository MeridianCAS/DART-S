"""Jump-scenario JSON library. CLI flags override loaded geometry."""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _resolve_library_dir() -> Path:
    d = os.environ.get("DARTS_JUMP_DIR")
    if not d:
        return REPO / "sim" / "scenarios"
    p = Path(d)
    return p if p.is_absolute() else (REPO / p)


LIBRARY_DIR = _resolve_library_dir()


def load_scenario(name: str) -> dict:
    path = LIBRARY_DIR / f"{name}.json"
    if not path.exists():
        avail = ", ".join(s["name"] for s in list_scenarios()) or "(none)"
        raise FileNotFoundError(f"scenario {name!r} not found; available: {avail}")
    return json.loads(path.read_text(encoding="utf-8")).get("geometry", {})


def apply_to_args(args, name: str, *, skip=None):
    skip = set(skip or [])
    geo = load_scenario(name)
    applied = []
    for k, v in geo.items():
        if k in skip:
            continue
        args.__dict__[k] = v
        applied.append(k)
    return applied


def list_scenarios() -> list[dict]:
    if not LIBRARY_DIR.exists():
        return []
    out = []
    for p in sorted(LIBRARY_DIR.glob("*.json")):
        if p.name == "_index.json":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append({"name": d.get("name", p.stem), "description": d.get("description", ""),
                        "mode": d.get("geometry", {}).get("landing_slope_mode"),
                        "created": d.get("created", "")})
        except Exception:
            continue
    return out


if __name__ == "__main__":
    rows = list_scenarios()
    if not rows:
        print("no scenarios found")
    else:
        print(f"{len(rows)} scenarios @ {LIBRARY_DIR}:")
        for s in rows:
            print(f"  - {s['name']:32s} [{s['mode']}] {s['description']}")
