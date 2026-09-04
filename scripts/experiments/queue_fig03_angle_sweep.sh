#!/usr/bin/env bash
# Fig. 3: action-minus-neutral landing pitch across the five tabletop geometries
# at 13 m/s, for the two suspension presets and the torque-bias baseline.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

acquire_lock
log "QUEUE START: fig03_angle_sweep"

# The default arm set is neutral / nose_up / nose_down / torque_up, which is
# exactly the Fig. 3 panel.
COMMON=(--speeds 13 --rolls 30 --arm-order-seed 20260731)

for angle in 15 20 25 30 35; do
  run_cohort "fig03_a${angle}_v13" --angle "$angle" "${COMMON[@]}" || true
done

# Speed trend on the 20 deg reference geometry; supporting context, not plotted.
run_cohort "fig03_a20_v10" --angle 20 --speeds 10 --rolls 30 --arm-order-seed 20260731 || true
run_cohort "fig03_a20_v16" --angle 20 --speeds 16 --rolls 30 --arm-order-seed 20260731 || true

kill_sim
log "QUEUE DONE"
