#!/usr/bin/env bash
# Table I ablations: stroke scaling, takeoff end versus both ends, landing-end
# variants, and the airborne-channel interaction.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

acquire_lock
log "QUEUE START: tab01_ablations"

PAIR=(--speeds 13 --rolls 30 --arms neutral nose_up)

# Stroke scaling: 33 / 50 / 67 mm per corner, i.e. 10 / 15 / 20 % of the
# available stroke. The nominal part config is the 15 % build.
run_cohort "tab01_stroke_r010" --angle 20 "${PAIR[@]}" \
  --pc vehicles/sbr/darts_4motor_r010.pc || true
run_cohort "tab01_stroke_r015" --angle 20 "${PAIR[@]}" || true
run_cohort "tab01_stroke_r020" --angle 20 "${PAIR[@]}" \
  --pc vehicles/sbr/darts_4motor_r020.pc || true

# Takeoff-end preset versus presetting both ends.
run_cohort "tab01_both_ends_a20" --angle 20 --speeds 13 --rolls 30 \
  --arms neutral nose_up both_ends || true
run_cohort "tab01_both_ends_a35" --angle 35 --speeds 13 --rolls 30 \
  --arms neutral nose_up both_ends || true

# Landing-end-only variants: the null control for the pre-liftoff mechanism.
run_cohort "tab01_landing_end_a20" --angle 20 --speeds 13 --rolls 30 \
  --arms neutral land_soft land_soft_lift || true
run_cohort "tab01_landing_end_a35" --angle 35 --speeds 13 --rolls 30 \
  --arms neutral land_soft land_soft_lift || true

# Ground preset crossed with the airborne pitch loop.
run_cohort "tab01_channels_a20" --angle 20 --speeds 13 --rolls 30 \
  --arms neutral nose_up air_only air_susp || true
run_cohort "tab01_channels_a35" --angle 35 --speeds 13 --rolls 30 \
  --arms neutral nose_up air_only air_susp || true

kill_sim
log "QUEUE DONE"
