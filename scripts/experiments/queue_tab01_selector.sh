#!/usr/bin/env bash
# Table I selector execution and timing. The three gated cohorts exercise the
# prepare, refuse, and pass-through branches of the calibrated decision table;
# the lead pair shows that the preset must finish before the fillet loads.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

acquire_lock
log "QUEUE START: tab01_selector"

# The gated arm reproduces the calibrated table-first decision path, so it runs
# under the legacy selector policy rather than the fail-closed strict policy.
COMMON=(--rolls 30 --arms gated --post-land-sec 2.0
        --arm-order-seed 20260805
        --selector-policy legacy --gate-f-theta-policy audit
        --protocol-tag darts_selector)

# prepare (35 deg / 13 m/s -> nose_up), refuse (35 deg / 11.5 m/s -> abort),
# pass-through (20 deg / 15 m/s -> neutral).
run_cohort "tab01_sel_prepare" "${COMMON[@]}" \
  --angle 35 --speeds 13 --gate-lead-sec 2.0 || true
run_cohort "tab01_sel_refuse" "${COMMON[@]}" \
  --angle 35 --speeds 11.5 --gate-lead-sec 2.0 || true
run_cohort "tab01_sel_passthrough" "${COMMON[@]}" \
  --angle 20 --speeds 15 --gate-lead-sec 2.0 || true

# Decision lead: 0.8 s places the stroke inside the fillet, 2.0 s completes it
# beforehand.
run_cohort "tab01_sel_lead0p8" "${COMMON[@]}" \
  --angle 35 --speeds 13 --gate-lead-sec 0.8 || true

kill_sim
log "QUEUE DONE"
