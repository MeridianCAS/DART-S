#!/usr/bin/env bash
# Minimal cohort runner for paper reproduction (sequential, resumable via DONE markers).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-python}"
cd "$REPO"
OUT="${OUT:-data/queue_runs}"
LOG="$OUT/logs"
COHORTS="data/cohorts"
mkdir -p "$LOG" "$OUT" "$COHORTS"

export DARTS_JUMP_DIR="${DARTS_JUMP_DIR:-sim/scenarios}"
export DARTS_BNG_PRE_LAUNCH_KILL="${DARTS_BNG_PRE_LAUNCH_KILL:-1}"
export PYTHONUNBUFFERED=1

# A cohort that stops writing to its log for this long is treated as wedged.
STALL_SEC="${STALL_SEC:-1500}"
POLL_SEC="${POLL_SEC:-30}"

log() { echo "=== [$(date -Iseconds)] $*"; }

# One simulator port per machine, so one queue at a time.
LOCK_DIR="$OUT/_QUEUE_LOCK.d"
acquire_lock() {
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    log "ABORT another queue holds $LOCK_DIR"
    exit 3
  fi
  printf 'pid=%s\nstarted=%s\n' "$$" "$(date -Iseconds)" >"$LOCK_DIR/owner"
  trap 'rm -rf "$LOCK_DIR"' EXIT
}

kill_sim() {
  taskkill //IM BeamNG.tech.x64.exe //F //T >/dev/null 2>&1 \
    || pkill -f BeamNG.tech >/dev/null 2>&1 || true
  sleep 3
}

# run_cohort <tag> [darts_bench.py flags...]
run_cohort() {
  local tag="$1"
  shift
  local done_m="$OUT/${tag}.DONE"
  local logf="$LOG/${tag}.log"
  local raw="$COHORTS/darts_bench_${tag}.json"
  if [[ -f "$done_m" && -f "$raw" ]]; then
    log "SKIP $tag (already done)"
    return 0
  fi
  log "START $tag"
  kill_sim
  "$PY" scripts/darts_bench.py "$@" --resume-checkpoint 1 --tag "$tag" >"$logf" 2>&1 &
  local runner=$!
  local mt
  while kill -0 "$runner" 2>/dev/null; do
    sleep "$POLL_SEC"
    mt=$(stat -c %Y "$logf" 2>/dev/null || echo 0)
    if (( $(date +%s) - mt > STALL_SEC )); then
      log "WEDGE $tag (no log output for ${STALL_SEC}s)"
      kill -9 "$runner" 2>/dev/null || true
      kill_sim
      return 1
    fi
  done
  wait "$runner" 2>/dev/null
  local rc=$?
  if (( rc != 0 )) || [[ ! -f "$raw" ]]; then
    log "FAIL $tag rc=$rc (see $logf)"
    return 1
  fi
  touch "$done_m"
  log "OK $tag"
}
