#!/usr/bin/env bash
# Headline confirmation (Fig. 5 plus the timing, washboard, and held-out blocks):
# 72 independent simulator sessions and 600 formal runs, driven session by session
# from the frozen design in calibration/confirmatory_design.json.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

DESIGN="${DESIGN:-calibration/confirmatory_design.json}"
RESULTS="${RESULTS:-data/derived/confirmatory}"
mkdir -p "$RESULTS"

acquire_lock
log "QUEUE START: confirmatory"

"$PY" scripts/confirmatory_design.py --out "$DESIGN" --verify \
  || { log "ABORT design fingerprint gate"; exit 2; }

LAUNCH_ID="darts-confirmatory-$(date +%Y%m%d)"

mapfile -t SESSION_ROWS < <("$PY" - "$DESIGN" <<'PY'
import json
import sys

design = json.load(open(sys.argv[1], encoding="utf-8"))
for session in design["sessions"]:
    block = design["blocks"][session["block_id"]]
    leads = block.get("phase_sweep_leads") or []
    print("\t".join([
        session["session_id"],
        str(session["angle_deg"]),
        str(session["speed_mps"]),
        str(session["rolls"]),
        ",".join(session["arms"]),
        str(block.get("target_offset_mps", 0.0)),
        str(block.get("runup_bumps", 0)),
        str(block.get("bump_height_m", 0.05)),
        str(block.get("bump_spacing_m", 1.3)),
        ",".join(str(v) for v in leads) or "-",
        str(block.get("angle_uncertainty_deg", 0.0)),
    ]))
PY
)

# The confirmatory protocol pins every selector and guard threshold, so the
# blocks differ only in geometry, speed, arms, run-up roughness, and the
# declared angle uncertainty.
run_session() {
  local sid="$1" angle="$2" speed="$3" rolls="$4" arms_csv="$5" offset="$6"
  local bumps="$7" bump_h="$8" bump_spacing="$9" phase_csv="${10}" angle_unc="${11}"
  local tag="${sid//-/_}"
  local arms=() phase_args=()
  IFS=',' read -r -a arms <<<"$arms_csv"
  if [[ "$phase_csv" != "-" ]]; then
    local leads=()
    IFS=',' read -r -a leads <<<"$phase_csv"
    phase_args=(--phase-sweep-leads "${leads[@]}")
  fi
  run_cohort "$tag" \
    --angle "$angle" --speeds "$speed" --rolls "$rolls" --arms "${arms[@]}" \
    --run-up 75 --post-land-sec 2.0 \
    --runup-bumps "$bumps" --bump-height "$bump_h" --bump-spacing "$bump_spacing" \
    --impact-trace 1 --per-step-state-trace 1 \
    --randomize-arm-order 1 --arm-order-seed "$((2026081300 + ${#sid} * 17 + 10#${sid: -2}))" \
    --approach-speed-governor symmetric \
    --approach-speed-target-offset-mps "$offset" \
    --approach-wheel-spin-guard-radps 200 --dart-wheel-speed-cap-radps 200 \
    --wheel-storage-hard-limit-radps 221.2 \
    --selector-policy legacy --gate-f-theta-policy audit \
    --selector-angle-uncertainty-deg "$angle_unc" \
    --selector-speed-uncertainty-mps 0.2 \
    --selector-require-safety-v2-evidence 1 \
    --selector-geometry-family tabletop_fillet \
    --selector-require-calibration-map 1 \
    --selector-calibration-support-max-distance 1.5 \
    --selector-require-system-safety-evidence 1 \
    --selector-system-safety-max-distance 0.75 \
    --gate-lead-sec 2.0 --gate-theta-bar-deg 35 --gate-omega-bar-dps 25 \
    --gate-wheel-omega-max-radps 125.7 --gate-wheel-omega-min-radps -125.7 \
    --protocol-tag darts_confirmatory --session-id "$sid" --launch-id "$LAUNCH_ID" \
    "${phase_args[@]}" || return 1
  # Per-session gate: a session that misses its expected runs or arms is not
  # allowed to enter the pooled analysis.
  "$PY" scripts/confirmatory_analysis.py --design "$DESIGN" \
    --check-session "$sid" --raw "$COHORTS/darts_bench_${tag}.json" \
    >>"$LOG/${tag}.log" 2>&1 || { log "SESSION GATE FAIL $sid"; return 1; }
}

log "sessions=${#SESSION_ROWS[@]}"
for row in "${SESSION_ROWS[@]}"; do
  IFS=$'\t' read -r sid angle speed rolls arms offset bumps bump_h bump_spacing \
    phase_csv angle_unc <<<"$row"
  run_session "$sid" "$angle" "$speed" "$rolls" "$arms" "$offset" \
    "$bumps" "$bump_h" "$bump_spacing" "$phase_csv" "$angle_unc" || exit 5
done
kill_sim

"$PY" scripts/confirmatory_analysis.py --design "$DESIGN" --output-dir "$RESULTS" \
  2>&1 | tee "$LOG/_confirmatory_analysis.log"
rc=${PIPESTATUS[0]}
(( rc == 0 )) || { log "ANALYSIS GATE FAIL rc=$rc"; exit 8; }
log "QUEUE DONE -> $RESULTS"
