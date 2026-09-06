"""Self-describing ``_provenance`` stamps plus a fail-closed time-base gate.

Every cohort JSON this package writes carries a ``_provenance`` block whose
``time_integrity_ok`` flag is computed at write time, and every analysis
script checks its inputs with :func:`assert_time_integrity`. Data with no
stamp, a stamp that fails, or a path listed in the contaminated registry
raises instead of being silently treated as clean.

The verdict is outcome-based rather than flag-based:

* ``deterministic_stepped`` with ``dt_eff`` close to ``1 / sps`` passes, so
  time-dependent quantities are trustworthy.
* ``freerun_wallclock`` fails, because airtime, flight time and any per-step
  rate drift with computational load. Time-independent quantities such as
  landing angles remain valid.
* A missing stamp or a registry hit is treated as contaminated.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

SCHEMA_VERSION = "darts-jump-report/2"   # v2 semantics: airtime is sim flight time
AIRTIME_FIX_COMMIT = "5809bb3"           # deterministic-stepping fix
REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "calibration" / "contaminated_registry.json"


class ContaminatedDataError(RuntimeError):
    """Raised when data has an unverifiable or failing time base."""


def git_commit(short: bool = True):
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        return subprocess.check_output(args, cwd=str(REPO), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def make_provenance(dt_mode: str, sps=None, dt_eff=None, note: str | None = None) -> dict:
    """Build a ``_provenance`` block for one cohort file.

    ``dt_mode`` is ``deterministic_stepped`` or ``freerun_wallclock``.
    """
    ok = False
    if dt_mode == "deterministic_stepped":
        if sps and dt_eff is not None:
            tol = 0.2 * (1.0 / sps)
            if abs(float(dt_eff) - 1.0 / sps) <= tol:
                ok, reason = True, f"deterministic, dt_eff={dt_eff} matches 1/{sps}"
            else:
                reason = (
                    f"deterministic but dt_eff={dt_eff} deviates from "
                    f"1/{sps}={1.0/sps:.4f} (tol={tol:.4f}): stepping did not take"
                )
        else:
            reason = "deterministic mode but sps/dt_eff missing, time base unverifiable"
    elif dt_mode == "freerun_wallclock":
        reason = ("freerun: airtime, flight time and per-step rates drift with load; "
                  "landing pitch/roll/yaw and other time-independent metrics stay valid")
    else:
        reason = f"unknown dt_mode={dt_mode!r}"
    prov = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": git_commit(),
        "airtime_fix_commit": AIRTIME_FIX_COMMIT,
        "dt_mode": dt_mode,
        "sps": sps,
        "dt_eff": dt_eff,
        "time_integrity_ok": ok,
        "time_integrity_reason": reason,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if note:
        prov["note"] = note
    return prov


def aggregate_provenance(
    source_paths,
    note: str | None = None,
    require_time_metrics: bool = True,
) -> dict:
    """Chain the time-base gate through a derived artifact.

    By default every source file must carry ``time_integrity_ok=true``. An
    analysis that only consumes branch decisions, geometry or landing angles
    may pass ``require_time_metrics=False``, which still requires a stamp
    and a clean registry check but waives ``dt_eff`` integrity.
    """
    import json as _json
    from pathlib import Path as _Path
    srcs, ok_all = [], True
    for p in source_paths:
        p = _Path(p)
        accepted = False
        source_time_ok = False
        try:
            report = _json.loads(p.read_text(encoding="utf-8")) or {}
            prov = report.get("_provenance") or {}
            source_time_ok = prov.get("time_integrity_ok") is True
            assert_time_integrity(
                report, path=str(p), require_time_metrics=require_time_metrics)
            accepted = True
        except Exception:
            pass
        ok_all = ok_all and accepted
        srcs.append({
            "file": p.name,
            "time_integrity_ok": source_time_ok,
            "time_metrics_required": bool(require_time_metrics),
            "accepted": accepted,
        })
    prov = {
        "schema_version": "darts-aggregate/1",
        "code_commit": git_commit(),
        "time_integrity_ok": ok_all,
        "time_integrity_reason": (
            "every source passed the time-base check" if ok_all and require_time_metrics
            else "no time metrics consumed; every source is stamped and off the registry"
            if ok_all
            else "at least one source failed the required provenance gate"),
        "sources": srcs,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if note:
        prov["note"] = note
    return prov


def _load_registry() -> dict:
    try:
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    except Exception:
        return {"contaminated": []}


def registry_hit(path):
    if not path:
        return None
    p = str(Path(path).resolve()).replace("\\", "/")
    for entry in _load_registry().get("contaminated", []):
        for g in entry.get("globs", []):
            gg = g.replace("\\", "/")
            if Path(p).match(gg) or (gg.strip("*") and gg.strip("*") in p):
                return entry
    return None


def assert_time_integrity(report, path=None, require_time_metrics: bool = True):
    """Fail closed on a report dict or path whose time base is not verified.

    With ``require_time_metrics=True`` the stamp must pass, which is what
    conclusions about airtime, flight time or rates need. Analyses that only
    use time-independent quantities may pass ``False``; the stamp must still
    exist and the path must still be off the contaminated registry.
    """
    if isinstance(report, (str, Path)):
        path = path or str(report)
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    hit = registry_hit(path)
    if hit:
        raise ContaminatedDataError(
            f"[CONTAMINATED] {path}: {hit.get('reason')} "
            f"(registry hit; fixed in commit {hit.get('fix_commit')})")
    prov = report.get("_provenance") if isinstance(report, dict) else None
    if prov is None:
        raise ContaminatedDataError(
            f"[NO-PROVENANCE] {path or '<dict>'}: no _provenance block, so the "
            f"time base cannot be confirmed (fail-closed; see {REGISTRY.name})")
    if require_time_metrics and not prov.get("time_integrity_ok", False):
        raise ContaminatedDataError(
            f"[TIME-INTEGRITY-FAIL] {path or '<dict>'}: {prov.get('time_integrity_reason')}")
    return prov
