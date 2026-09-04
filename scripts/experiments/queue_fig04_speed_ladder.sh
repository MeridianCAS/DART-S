#!/usr/bin/env bash
# Fig. 4: preset-minus-neutral response across the 35 deg speed ladder.
# Three sessions of 10 paired rolls per speed give the 30 pairs per speed and
# the session-level sign agreement; the speed order is permuted per session so
# that the reversal cannot be an artefact of within-session drift.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

acquire_lock
log "QUEUE START: fig04_speed_ladder"

LAUNCH_ID="darts-fig04-$(date +%Y%m%d)"

# run_session <index> <arm-order-seed> <speed...>
run_session() {
  local session="$1" seed="$2"
  shift 2
  run_cohort "fig04_ladder_s${session}" \
    --angle 35 --speeds "$@" --rolls 10 --arms neutral nose_up \
    --run-up 75 --post-land-sec 2.0 \
    --impact-trace 1 --per-step-state-trace 1 \
    --randomize-arm-order 1 --arm-order-seed "$seed" \
    --approach-speed-governor symmetric \
    --approach-speed-target-offset-mps 0.0 \
    --approach-wheel-spin-guard-radps 200 \
    --wheel-storage-hard-limit-radps 221.2 \
    --selector-geometry-family tabletop_fillet \
    --protocol-tag darts_speed_ladder \
    --session-id "ladder-session-${session}" --launch-id "$LAUNCH_ID"
}

run_session 1 20260810101 10 11 11.5 12 13 || exit 5
run_session 2 20260810102 13 12 11.5 11 10 || exit 5
run_session 3 20260810103 11.5 10 13 11 12 || exit 5

kill_sim
log "QUEUE DONE"
