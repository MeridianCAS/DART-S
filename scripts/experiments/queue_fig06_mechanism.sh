#!/usr/bin/env bash
# Fig. 6: convex-lip mechanism on the 20 deg kicker. DART alone (dart_only)
# against nose-up preset plus DART (dart_s_forced), paired by roll.
#   (a) R=35 m speed ladder and R=22 m held-out ladder: takeoff pitch-rate
#       difference against entry speed, with v* = sqrt(g R cos alpha).
#   (b) four-corner stroke traces at 18 m/s: the front-to-rear lift-off window.
#   2x2 radius-by-arc-length check at n(v) ~ 0: R=35 m sweep 5 deg (3.05 m arc)
#       and R=22 m sweep 12.73 deg (4.89 m arc) against the ladder cells.
# The 13 and 18 m/s cells on R=35 m carry 30 pairs; the others carry 10.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_queue_common.sh"

acquire_lock
log "QUEUE START: fig06_mechanism"

COMMON=(--jump-scenario darts_a20r7_valley12 --arms dart_only dart_s_forced
        --run-up 63 --v-jitter 0.25 --post-land-sec 2.0
        --protocol-tag darts_fig06_mechanism)

# run_cell <tag> <rolls> <speed> [flags...]
run_cell() {
  local tag="$1" rolls="$2" speed="$3"
  shift 3
  run_cohort "$tag" "${COMMON[@]}" --speeds "$speed" --rolls "$rolls" "$@"
}

# (a) R=35 m ladder; the 18 m/s cell also records the stroke traces for (b).
run_cell fig06_r35_v13    30 13   || exit 5
run_cell fig06_r35_v15    10 15   || exit 5
run_cell fig06_r35_v16p5  10 16.5 || exit 5
run_cell fig06_r35_v18    30 18   --susp-stroke-trace 1 || exit 5
run_cell fig06_r35_v19p5  10 19.5 || exit 5

# (a) R=22 m held-out ladder; the lip radius overrides the scenario value.
for v in 12 13 14.25 15.5; do
  run_cell "fig06_r22_v${v/./p}" 10 "$v" --lip-radius 22 --susp-stroke-trace 1 || exit 5
done

# 2x2 arc-length cells at n(v) ~ 0.
run_cell fig06_r35s5_v18        10 18    --lip-sweep-deg 5     --susp-stroke-trace 1 || exit 5
run_cell fig06_r22s12p73_v14p25 10 14.25 --lip-radius 22 --lip-sweep-deg 12.73 --susp-stroke-trace 1 || exit 5

kill_sim
"$PY" scripts/fig06_mechanism_analysis.py || exit 6
log "QUEUE DONE"
