#!/usr/bin/env bash
# Table I platform robustness: mass +/- 20 % and a forward mass-distribution
# shift, against the nominal +7.4 deg landing gain on the 20 deg geometry.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

acquire_lock
log "QUEUE START: tab01_robustness"

PAIR=(--angle 20 --speeds 13 --rolls 30 --arms neutral nose_up)

run_cohort "tab01_rob_m080" "${PAIR[@]}" \
  --pc vehicles/sbr/darts_4motor_m080.pc || true
run_cohort "tab01_rob_m120" "${PAIR[@]}" \
  --pc vehicles/sbr/darts_4motor_m120.pc || true
run_cohort "tab01_rob_cogfront" "${PAIR[@]}" \
  --pc vehicles/sbr/darts_4motor_cogfront.pc || true

# Entry-speed dispersion on the two reference geometries.
run_cohort "tab01_rob_jitter_a20" --angle 20 --speeds 13 --rolls 30 --v-jitter 0.5 || true
run_cohort "tab01_rob_jitter_a35" --angle 35 --speeds 13 --rolls 30 --v-jitter 0.5 || true

kill_sim
log "QUEUE DONE"
