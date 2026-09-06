"""BeamNG session helpers for DART jump experiments.

Provides :class:`BeamNGSession` for connect / scenario-ready wait / teardown,
plus small utilities (``ensure_freerun``, ``wait_scenario_ready``, process kill).
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, "")
    v = v.strip() if isinstance(v, str) else v
    return v if v else default

def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")

# Image names ``taskkill`` is allowed to terminate on Windows. Purposefully
# wide because BeamNG.tech and BeamNG.drive both ship multiple shipping
# binaries plus a launcher.
DEFAULT_KILL_TARGETS: tuple[str, ...] = (
    "BeamNG.tech.x64.exe",
    "BeamNG.drive.x64.exe",
    "BeamNG.tech.exe",
    "BeamNG.drive.exe",
    "BeamNG.x64.exe",
    "BeamNG.exe",
    "BeamNGLauncher.exe",
)

# Linux process names ``pkill -9 -f`` matches against. The Linux binary names
# don't have an .exe suffix; DART's recipe also matches via -f against the
# full command line so a ``-rport 25252`` argument suffix on a re-launch will
# still terminate stale instances.
DEFAULT_LINUX_KILL_PATTERNS: tuple[str, ...] = (
    "BeamNG.tech.x64",
    "BeamNG.drive.x64",
)

# Some BeamNG.tech builds need ``-tcom`` or the BeamNGpy listener never starts,
# even when ``-rport`` is already on the command line.
DEFAULT_LINUX_LAUNCH_ARGS: tuple[str, ...] = (
    "-headless",
    "-nosteam",
    "-tcom",
    "-console",
)

@dataclass
class BeamNGSessionConfig:
    """Settings for a single BeamNG session lifecycle."""

    host: str = "127.0.0.1"
    port: int = 25252
    home: str | None = None
    user_path: str | None = None
    auto_launch: bool = True
    open_timeout_sec: int = 60
    open_retries: int = 3
    open_retry_sleep_sec: float = 3.0
    pre_launch_kill: bool = False  # destructive on a shared box; default off
    ready_timeout_sec: float = 60.0
    ready_poll_sec: float = 0.5
    ready_min_settle_sec: float = 2.5
    ready_fast_probe: bool = True
    ready_fast_probe_min_wait_sec: float = 2.0
    quit_after: bool = True
    quit_grace_sec: float = 60.0
    verbose: bool = True
    log_prefix: str = "[bng]"
    kill_targets: tuple[str, ...] = field(default_factory=lambda: DEFAULT_KILL_TARGETS)
    linux_kill_patterns: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_LINUX_KILL_PATTERNS
    )
    # "blanket" kills every BeamNG image; "port" only the process on this rport
    # (required if two cohorts share one box).
    kill_scope: str = "blanket"
    linux_binary: str | None = None  # if set, pre-launch; do not use BeamNGpy's launcher
    linux_launch_args: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_LINUX_LAUNCH_ARGS
    )
    linux_launch_log: str | None = None
    linux_launch_wait_sec: float = 90.0
    linux_launch_poll_sec: float = 1.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "BeamNGSessionConfig":
        """Build a config from DARTS_BNG_* / BEAMNG_* environment variables.

        Both prefixes are honoured. ``DARTS_BNG_*`` overrides DART-S-specific
        knobs without colliding with generic ``BEAMNG_*`` exports.
        """
        cfg = cls(
            host=_env("DARTS_BNG_HOST") or _env("BEAMNG_HOST") or "127.0.0.1",
            port=_env_int("DARTS_BNG_PORT", _env_int("BEAMNG_PORT", 25252)),
            home=_env("DARTS_BNG_HOME") or _env("BEAMNG_HOME"),
            user_path=_env("DARTS_BNG_USER_PATH") or _env("BEAMNG_USER_PATH"),
            auto_launch=_env_bool("DARTS_BNG_AUTO_LAUNCH", True),
            open_timeout_sec=_env_int("DARTS_BNG_OPEN_TIMEOUT_SEC", 60),
            open_retries=_env_int("DARTS_BNG_OPEN_RETRIES", 3),
            open_retry_sleep_sec=_env_float("DARTS_BNG_OPEN_RETRY_SLEEP_SEC", 3.0),
            pre_launch_kill=_env_bool("DARTS_BNG_PRE_LAUNCH_KILL", False),
            kill_scope=(_env("DARTS_BNG_KILL_SCOPE") or "blanket"),
            ready_timeout_sec=_env_float("DARTS_BNG_READY_TIMEOUT_SEC", 60.0),
            ready_poll_sec=_env_float("DARTS_BNG_READY_POLL_SEC", 0.5),
            ready_min_settle_sec=_env_float("DARTS_BNG_READY_MIN_SETTLE_SEC", 2.5),
            ready_fast_probe=_env_bool("DARTS_BNG_READY_FAST_PROBE", True),
            ready_fast_probe_min_wait_sec=_env_float(
                "DARTS_BNG_READY_FAST_PROBE_MIN_WAIT_SEC", 2.0
            ),
            quit_after=_env_bool("DARTS_BNG_QUIT_AFTER", True),
            quit_grace_sec=_env_float("DARTS_BNG_QUIT_GRACE_SEC", 60.0),
            verbose=_env_bool("DARTS_BNG_VERBOSE", True),
            linux_binary=_env("DARTS_BNG_LINUX_BINARY"),
            linux_launch_log=_env("DARTS_BNG_LINUX_LOG"),
            linux_launch_wait_sec=_env_float(
                "DARTS_BNG_LINUX_LAUNCH_WAIT_SEC", 90.0
            ),
            linux_launch_poll_sec=_env_float(
                "DARTS_BNG_LINUX_LAUNCH_POLL_SEC", 1.0
            ),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        # On Linux, derive linux_binary from home if not explicitly set.
        if (
            sys.platform.startswith("linux")
            and cfg.linux_binary is None
            and cfg.home
        ):
            cand = os.path.join(cfg.home, "BinLinux", "BeamNG.tech.x64")
            if os.path.exists(cand):
                cfg.linux_binary = cand
        return cfg

def tcp_port_open(host: str, port: int, timeout_sec: float = 2.0) -> bool:
    """Cheap pre-flight: is *something* listening on ``host:port``?"""
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False

def kill_beamng_processes(
    targets: tuple[str, ...] = DEFAULT_KILL_TARGETS,
    *,
    grace_sec: float = 60.0,
    runner: Any | None = None,
    log: Any = print,
    log_prefix: str = "[bng]",
    linux_patterns: tuple[str, ...] = DEFAULT_LINUX_KILL_PATTERNS,
    platform_override: str | None = None,
) -> int:
    """Terminate every BeamNG-related image.

    On Windows uses ``taskkill /F /T /IM <exe>`` over ``targets``.
    On Linux uses ``pkill -9 -f <pattern>`` over ``linux_patterns``.
    Returns the number of (target / pattern) entries the kill helper
    claims to have terminated. The ``runner`` argument is injected for
    testability and defaults to ``subprocess.run``. The
    ``platform_override`` argument is for tests that want to exercise
    one branch on the other OS.
    """
    runner = runner or subprocess.run
    plat = platform_override or sys.platform
    if plat == "win32":
        n = 0
        for exe in targets:
            try:
                res = runner(
                    ["taskkill", "/F", "/T", "/IM", exe],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=grace_sec,
                    check=False,
                )
                stdout = (getattr(res, "stdout", b"") or b"")
                stderr = (getattr(res, "stderr", b"") or b"")
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="ignore")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="ignore")
                text = (stdout + stderr).upper()
                # taskkill returns non-zero when no process matched; treat as ok.
                if "SUCCESS" in text or "" in (stdout + stderr):
                    n += 1
                    log(f"{log_prefix} kill: terminated {exe}")
            except Exception as exc:                                # noqa: BLE001
                log(f"{log_prefix} kill: WARN failed for {exe}: {exc!r}")
        if n == 0:
            log(f"{log_prefix} kill: no BeamNG process matched")
        return n

    if plat.startswith("linux"):
        n = 0
        for pat in linux_patterns:
            try:
                res = runner(
                    ["pkill", "-9", "-f", pat],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=grace_sec,
                    check=False,
                )
                # pkill exit codes: 0 == matched, 1 == no match, 2 == syntax error.
                rc = getattr(res, "returncode", 1)
                if rc == 0:
                    n += 1
                    log(f"{log_prefix} kill: pkill terminated -f {pat!r}")
            except Exception as exc:                                # noqa: BLE001
                log(f"{log_prefix} kill: WARN pkill -f {pat!r} failed: {exc!r}")
        if n == 0:
            log(f"{log_prefix} kill: no BeamNG process matched")
        return n

    log(f"{log_prefix} kill: platform {plat} not supported, skipped")
    return 0

def build_port_kill_commands(
    port: int, *, pid: int | None = None
) -> list[list[str]]:
    """Pure helper: the argv list for a port-scoped (non-blanket) kill.

    Returns the commands to terminate *only* the BeamNG instance bound to
    ``port``. When a launch PID is known we kill it directly first (precise),
    and always append ``fuser -k <port>/tcp`` as a backstop in case the PID
    has re-forked or the caller never recorded it. With ``pid`` None the
    builder degrades to the port backstop alone. Kept side-effect-free so the
    targeting logic can be unit-tested without spawning anything.
    """
    cmds: list[list[str]] = []
    if pid is not None and int(pid) > 0:
        cmds.append(["kill", "-9", str(int(pid))])
    cmds.append(["fuser", "-k", f"{int(port)}/tcp"])
    return cmds

def kill_beamng_by_port(
    port: int,
    *,
    pid: int | None = None,
    grace_sec: float = 60.0,
    runner: Any | None = None,
    log: Any = print,
    log_prefix: str = "[bng]",
    platform_override: str | None = None,
) -> int:
    """Port-scoped kill: terminate only the BeamNG bound to ``port``.

    The Linux counterpart to the blanket :func:`kill_beamng_processes`, used
    when ``kill_scope="port"`` so parallel cohorts on one box never kill each
    other. Executes :func:`build_port_kill_commands` (PID then ``fuser``).
    Returns the number of commands that exited 0. Non-Linux platforms are a
    no-op; ``runner``/``platform_override`` are injected for tests.
    """
    runner = runner or subprocess.run
    plat = platform_override or sys.platform
    if not plat.startswith("linux"):
        log(
            f"{log_prefix} kill(port): platform {plat} not supported, skipped"
        )
        return 0

    n = 0
    for cmd in build_port_kill_commands(port, pid=pid):
        try:
            res = runner(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=grace_sec,
                check=False,
            )
            rc = getattr(res, "returncode", 1)
            if rc == 0:
                n += 1
                log(f"{log_prefix} kill(port): {' '.join(cmd)} -> rc=0")
        except Exception as exc:                                    # noqa: BLE001
            log(f"{log_prefix} kill(port): WARN {' '.join(cmd)} failed: {exc!r}")
    if n == 0:
        log(f"{log_prefix} kill(port): nothing matched on port {port}")
    return n

def launch_beamng_linux(
    binary: str,
    *,
    port: int = 25252,
    user_path: str | None = None,
    extra_args: tuple[str, ...] = DEFAULT_LINUX_LAUNCH_ARGS,
    log_path: str | None = None,
    cwd: str | None = None,
    runner: Any | None = None,
    log: Any = print,
    log_prefix: str = "[bng]",
) -> int:
    """Spawn the Linux native BeamNG.tech binary headless in the background.

    The returned int is the spawned process PID. The default args are
    ``-headless -nosteam -console`` plus ``-rport <port>`` (added here so the
    config can vary the port without rebuilding the args tuple). stdout/stderr
    is redirected to ``log_path`` (defaults to
    ``$HOME/DART_beamng_<port>.log``). The function does **not** wait for
    the listener — callers should use :func:`tcp_port_open` polling.

    ``user_path`` isolates the BeamNG **user folder** per instance so that two
    BeamNG processes on the same box never share cache / mods / temp state.
    When set, ``-userpath <dir>`` is appended to the launch args. This is the
    documented BeamNG.tech/BeamNG.drive CLI flag (mirrors BeamNGpy's
    ``user=`` kwarg; see BeamNGpy README + BeamNG startup.ini ``UserPath``).
    **When ``user_path`` is None the arg is omitted entirely**, keeping the
    single-instance argv byte-for-byte identical to the legacy behaviour.
    NOTE: BeamNG.tech quietly crashes on userpaths containing spaces, so the
    slot launcher uses space-free directories.

    ``runner`` is injected for testability — defaults to ``subprocess.Popen``.
    """
    if not os.path.exists(binary):
        raise FileNotFoundError(f"BeamNG Linux binary not found: {binary}")
    args = [binary, *extra_args, "-rport", str(port)]
    if user_path is not None:
        args += ["-userpath", user_path]
    if cwd is None:
        cwd = os.path.dirname(os.path.dirname(binary))               # one above BinLinux/
    if log_path is None:
        home = os.environ.get("HOME") or os.path.expanduser("~")
        log_path = os.path.join(home, f"DART_beamng_{port}.log")
    log(f"{log_prefix} launching Linux native: {' '.join(args)}")
    log(f"{log_prefix} stdout/stderr -> {log_path}")
    log_fp = open(log_path, "ab")
    Popen = runner or subprocess.Popen
    # start_new_session=True so signals to the parent don't propagate.
    proc = Popen(
        args,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        start_new_session=True,
    )
    pid = getattr(proc, "pid", -1)
    log(f"{log_prefix} BeamNG Linux pid={pid}")
    return int(pid)

def ensure_freerun(bng: Any, *, log: Any = print, log_prefix: str = "[bng]") -> None:
    """Resume the simulation if a previous run left it paused.

    Tries ``bng.control.resume()`` first (BeamNGpy >= 1.26) and falls back
    to ``bng.resume()`` for older versions.
    """
    for attr_path in ("control.resume", "resume"):
        obj = bng
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            obj()
            log(f"{log_prefix} ensure_freerun via bng.{attr_path}()")
            return
        except Exception:
            continue
    log(f"{log_prefix} WARN: could not resume — simulation may stay paused")

def warn_if_pre_running(bng: Any, *, log: Any = print, log_prefix: str = "[bng]") -> bool:
    """If the scenario is stuck on the start screen, emit a single hint."""
    try:
        gs = bng.control.get_gamestate()
    except Exception:
        return False
    if not isinstance(gs, dict):
        return False
    if gs.get("state") == "scenario" and gs.get("scenario_state") == "pre-running":
        log(
            f"{log_prefix} WARN scenario_state=pre-running (start screen). "
            "If the BeamNG window looks frozen, click Start in the game UI."
        )
        return True
    return False

def wait_scenario_ready(
    bng: Any,
    *,
    expected_vid: str | None = None,
    timeout_sec: float = 60.0,
    poll_sec: float = 0.5,
    min_settle_sec: float = 2.5,
    fast_probe: bool = True,
    fast_probe_min_wait_sec: float = 2.0,
    log: Any = print,
    log_prefix: str = "[bng]",
    _now: Any = time.time,
    _sleep: Any = time.sleep,
) -> bool:
    """Block until the scenario is ready for vehicle RPCs.

    Returns ``True`` when either the gamestate transitioned to
    ``scenario/running`` *or* the fast-probe (vehicle present) succeeded.
    Returns ``False`` on timeout, but in either case applies the
    ``min_settle_sec`` baseline before returning.
    """
    t0 = _now()
    last_combined: str | None = None
    reached_running = False
    probe_passed = False
    while _now() - t0 < timeout_sec:
        try:
            gs = bng.control.get_gamestate()
        except Exception as exc:                                    # noqa: BLE001
            log(
                f"{log_prefix} wait_scenario_ready: get_gamestate failed at "
                f"t+{_now()-t0:.1f}s: {exc!r}"
            )
            break
        if isinstance(gs, dict):
            combined = f"{gs.get('state')}/{gs.get('scenario_state')}"
            if combined != last_combined:
                log(f"{log_prefix} gamestate={combined} (t+{_now()-t0:.1f}s)")
                last_combined = combined
            if gs.get("state") == "scenario" and gs.get("scenario_state") == "running":
                reached_running = True
                break

        if fast_probe and (_now() - t0) >= max(0.0, fast_probe_min_wait_sec):
            try:
                vehs = bng.vehicles.get_current() or {}
            except Exception as exc:                                # noqa: BLE001
                log(
                    f"{log_prefix} fast_probe ignored exception "
                    f"t+{_now()-t0:.1f}s: {exc!r}"
                )
                vehs = {}
            have = set(vehs.keys()) if isinstance(vehs, dict) else set()
            ok = (str(expected_vid) in have) if expected_vid is not None else bool(have)
            if ok:
                probe_passed = True
                log(
                    f"{log_prefix} fast_probe OK at t+{_now()-t0:.1f}s "
                    f"(vehicles={sorted(have)}, expected={expected_vid!r})"
                )
                break
        _sleep(poll_sec)

    ready = reached_running or probe_passed
    if not ready:
        log(
            f"{log_prefix} wait_scenario_ready: no running signal after "
            f"{_now()-t0:.1f}s (last={last_combined}); proceeding with "
            f"{min_settle_sec}s settle"
        )
    _sleep(max(0.0, float(min_settle_sec)))
    return ready

class BeamNGSession(contextlib.AbstractContextManager):
    """Open / yield / disconnect / kill — the four pillars of robust BeamNG."""

    def __init__(
        self,
        cfg: BeamNGSessionConfig | None = None,
        *,
        beamngpy_factory: Any | None = None,
        log: Any = print,
    ):
        self.cfg = cfg or BeamNGSessionConfig.from_env()
        self._log = log
        self._bng: Any | None = None
        self._beamngpy_factory = beamngpy_factory
        self._launched_pid: int | None = None   # PID from Linux pre-launch,
                                                 # used for port-scoped teardown.

    @property
    def bng(self) -> Any:
        if self._bng is None:
            raise RuntimeError("BeamNGSession not open; use as a context manager")
        return self._bng

    def open(self) -> Any:
        if self._bng is not None:
            return self._bng

        cfg = self.cfg
        if cfg.pre_launch_kill:
            if cfg.kill_scope == "port":
                self._info(
                    f"pre_launch_kill enabled (scope=port), terminating only "
                    f"BeamNG bound to port {cfg.port}"
                )
                kill_beamng_by_port(
                    cfg.port,
                    grace_sec=cfg.quit_grace_sec,
                    log=self._log,
                    log_prefix=cfg.log_prefix,
                )
            else:
                self._info("pre_launch_kill enabled, terminating any dangling BeamNG")
                kill_beamng_processes(
                    cfg.kill_targets,
                    grace_sec=cfg.quit_grace_sec,
                    log=self._log,
                    log_prefix=cfg.log_prefix,
                )

        BNGCls = self._resolve_beamngpy_factory()

        last_exc: Exception | None = None
        for attempt in range(1, max(1, cfg.open_retries) + 1):
            port_ok = tcp_port_open(cfg.host, cfg.port, timeout_sec=2.0)
            need_launch = cfg.auto_launch and not port_ok

            # Linux native pre-launch: we always pre-spawn the binary
            # ourselves and then connect with launch=False. BeamNGpy 1.35's
            # internal launcher is Windows-tuned and doesn't pass the
            # ``-headless`` flag the Linux build needs.
            launch = need_launch
            if (
                need_launch
                and sys.platform.startswith("linux")
                and cfg.linux_binary
            ):
                self._linux_pre_launch(cfg)
                launch = False
                port_ok = tcp_port_open(cfg.host, cfg.port, timeout_sec=2.0)

            self._info(
                f"connect attempt {attempt}/{cfg.open_retries} "
                f"host={cfg.host} port={cfg.port} port_ok={port_ok} "
                f"launch={launch} timeout={cfg.open_timeout_sec}s"
            )

            bng_kwargs: dict[str, Any] = {}
            if cfg.home is not None:
                bng_kwargs["home"] = cfg.home
            if cfg.user_path is not None:
                bng_kwargs["user"] = cfg.user_path
            bng = BNGCls(cfg.host, cfg.port, **bng_kwargs)

            outcome: dict[str, Any] = {"ok": False, "err": None}

            def _do_open() -> None:
                try:
                    bng.open(launch=launch)
                    outcome["ok"] = True
                except Exception as exc:                            # noqa: BLE001
                    outcome["err"] = exc

            t0 = time.time()
            th = threading.Thread(target=_do_open, daemon=True)
            th.start()
            th.join(timeout=max(1, cfg.open_timeout_sec))

            try:
                if th.is_alive():
                    raise TimeoutError(
                        f"BeamNG open timed out after {cfg.open_timeout_sec}s "
                        f"(launch={launch})"
                    )
                if not bool(outcome["ok"]):
                    err = outcome["err"]
                    if isinstance(err, Exception):
                        raise err
                    raise RuntimeError("BeamNG open failed with unknown error")
                self._info(f"connected in {time.time() - t0:.1f}s")
                self._bng = bng
                with contextlib.suppress(Exception):
                    setattr(bng, "_DART_session", self)
                return bng
            except Exception as exc:                                # noqa: BLE001
                last_exc = exc
                self._info(f"WARN connect attempt {attempt} failed: {exc!r}")
                with contextlib.suppress(Exception):
                    bng.disconnect()
                if attempt < cfg.open_retries:
                    time.sleep(max(0.5, cfg.open_retry_sleep_sec))

        raise RuntimeError(
            f"BeamNG open failed after {cfg.open_retries} attempts: {last_exc}"
        )

    def close(self, *, kill: bool | None = None) -> None:
        cfg = self.cfg
        if self._bng is not None:
            with contextlib.suppress(Exception):
                self._bng.disconnect()
            self._info("disconnected")
        self._bng = None

        do_kill = cfg.quit_after if kill is None else kill
        if do_kill:
            if cfg.kill_scope == "port":
                kill_beamng_by_port(
                    cfg.port,
                    pid=self._launched_pid,
                    grace_sec=cfg.quit_grace_sec,
                    log=self._log,
                    log_prefix=cfg.log_prefix,
                )
            else:
                kill_beamng_processes(
                    cfg.kill_targets,
                    grace_sec=cfg.quit_grace_sec,
                    log=self._log,
                    log_prefix=cfg.log_prefix,
                    linux_patterns=cfg.linux_kill_patterns,
                )
        self._launched_pid = None

    def hard_refresh(self, *, sleep_sec: float = 5.0, kill: bool = True) -> Any:
        """Disconnect, kill BeamNG process tree, reconnect (cohort mid-run recovery)."""
        self._info(f"hard_refresh: close(kill={kill}) then reopen")
        self.close(kill=kill)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        return self.open()

    def _linux_pre_launch(self, cfg: "BeamNGSessionConfig") -> None:
        """Spawn the Linux BeamNG binary and wait for the listener."""
        binary = cfg.linux_binary or ""
        if not binary or not os.path.exists(binary):
            self._info(
                f"WARN linux_binary missing or absent ({binary!r}); "
                "falling back to BeamNGpy launcher"
            )
            return
        self._launched_pid = launch_beamng_linux(
            binary=binary,
            port=cfg.port,
            user_path=cfg.user_path,
            extra_args=cfg.linux_launch_args,
            log_path=cfg.linux_launch_log,
            log=self._log,
            log_prefix=cfg.log_prefix,
        )
        deadline = time.time() + max(5.0, cfg.linux_launch_wait_sec)
        last_log = 0.0
        while time.time() < deadline:
            if tcp_port_open(cfg.host, cfg.port, timeout_sec=1.0):
                self._info(
                    f"Linux BeamNG TCP up after {time.time() - (deadline - cfg.linux_launch_wait_sec):.1f}s"
                )
                return
            now = time.time()
            if now - last_log >= 5.0:
                last_log = now
                self._info(
                    f"waiting for Linux BeamNG listener on {cfg.host}:{cfg.port}"
                )
            time.sleep(max(0.1, cfg.linux_launch_poll_sec))
        self._info(
            f"WARN Linux BeamNG listener not up within "
            f"{cfg.linux_launch_wait_sec:.0f}s; will let BeamNGpy try"
        )

    def __enter__(self) -> Any:
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:                  # noqa: D401
        try:
            self.close()
        except Exception as err:                                    # noqa: BLE001
            self._info(f"WARN close raised: {err!r}")

    def _resolve_beamngpy_factory(self) -> Any:
        if self._beamngpy_factory is not None:
            return self._beamngpy_factory
        try:
            from beamngpy import BeamNGpy        # type: ignore
        except ImportError as exc:               # pragma: no cover
            raise RuntimeError(
                "beamngpy is not installed. `pip install -e .[beamng]` or "
                "set --no-eval to use the runner skeleton offline."
            ) from exc
        return BeamNGpy

    def _info(self, msg: str) -> None:
        if self.cfg.verbose:
            self._log(f"{self.cfg.log_prefix} {msg}")


@contextlib.contextmanager
def beamng_session(
    cfg: BeamNGSessionConfig | None = None,
    **overrides: Any,
) -> Iterator[Any]:
    """Functional context manager wrapper around :class:`BeamNGSession`."""
    if cfg is None:
        cfg = BeamNGSessionConfig.from_env(**overrides)
    elif overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    s = BeamNGSession(cfg)
    bng = s.open()
    try:
        yield bng
    finally:
        s.close()

__all__ = [
    "BeamNGSession",
    "BeamNGSessionConfig",
    "DEFAULT_KILL_TARGETS",
    "DEFAULT_LINUX_KILL_PATTERNS",
    "DEFAULT_LINUX_LAUNCH_ARGS",
    "beamng_session",
    "build_port_kill_commands",
    "ensure_freerun",
    "kill_beamng_by_port",
    "kill_beamng_processes",
    "launch_beamng_linux",
    "tcp_port_open",
    "wait_scenario_ready",
    "warn_if_pre_running",
]
