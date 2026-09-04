# DART-S — Experiment Reproduction Package

Everything needed to **re-run the experiments** behind *DART-S:
Reachability-Audited Active-Suspension Preconditioning for Off-Road Vehicle
Jumps*. The package ships the BeamNG experiment runner, the selector and
reachability code it calls, the calibration tables the selector reads, the
frozen confirmatory design, cohort queue scripts, and vehicle configs.

**Authors.** Yu Hu<sup>1,2,\*</sup>, Fangzhou Zhao<sup>1</sup>, Liang
Chen<sup>1</sup>, Chen Min<sup>1</sup>, Wei Li<sup>1,2</sup>, Mingyuan
Sang<sup>1,2</sup>, Jiajia Ma<sup>3</sup>, Shican Chen<sup>3</sup>, Di
Pang<sup>3</sup>, Baolei Chen<sup>3</sup>.

**Affiliations.**
<sup>1</sup>Research Center for Intelligent Computing Systems, Institute of
Computing Technology, CAS;
<sup>2</sup>School of Computer Science and Technology, University of Chinese
Academy of Sciences;
<sup>3</sup>Dong Feng Off-Road Vehicle Co., Ltd.
Corresponding author: Yu Hu (`huyu@ict.ac.cn`).

**No pre-computed cohort data is included.** You run the experiments locally;
each output file is named to map onto the paper's figures and table (e.g.
`data/cohorts/darts_bench_fig04_ladder_s1.json` → Fig. 4, first speed-ladder
session). The goal is **design-faithful reproduction** — same geometry,
calibration tables, selector policy, protocol, and statistics — not
byte-identical medians: BeamNG soft-body physics carries session-level
variation, so expect the paper's orderings, branch decisions, and significance
patterns to reproduce rather than exact numbers.

All experiments use deterministic-stepping full-scale simulation
(BeamNG.tech 0.38.3, 0.01 s step, 100 Hz control, custom 4WIDS electric
vehicle with series-hydraulic active suspension, curb mass ≈ 1383 kg).

## Layout

```
scripts/
  darts_bench.py               Main runner: takeoff A/B cohorts → data/cohorts/darts_bench_<TAG>.json
  confirmatory_design.py       Freeze/verify the 600-run confirmatory design (no simulator)
  confirmatory_analysis.py     Per-session gate and pooled cluster inference for the confirmatory runs
  actuator_probe.py            Standing-vehicle actuator authority probe (Table I, first row)
  reachcert_numeric_xval.py    LP cross-validation of the closed-form certificate (no simulator)
  experiments/*.sh             Cohort queue scripts (see table below)
control/darts/                 Selector, calibration map, reachability screen, erosion, wheel budget,
                               landing safety, release-phase action, provenance gate
calibration/                   Calibration tables the selector reads, plus the frozen design
data_pipeline/beamng_session.py  Simulator attach / ready / freerun / teardown helpers
tests/                         Unit tests for the certificate, selector, calibration map, and statistics
sim/scenarios/                 Geometry records loadable with --jump-scenario (reference and held-out)
vehicles/sbr/                  4WIDS part configs: nominal, stroke ratios, mass and CoG variants
data/cohorts/                  Generated cohort JSONs (empty until you run)
data/derived/                  Probe, analysis, and cross-validation outputs
data/queue_runs/               Queue logs and DONE markers (resumable reruns)
```

## Requirements

- Windows with **BeamNG.tech 0.38.3** (research license; set `DARTS_BNG_HOME`)
- Python 3.11+, `pip install -r requirements.txt`
- Vehicle configs from `vehicles/sbr/` copied into the BeamNG user folder
  (`.../BeamNG.tech/current/vehicles/sbr/`)

## Checks that need no simulator

```bash
python -m pytest tests -q                 # 173 tests: certificate, selector, calibration map, statistics
python scripts/confirmatory_design.py --out calibration/confirmatory_design.json --verify
python scripts/reachcert_numeric_xval.py  # → data/derived/cert_numeric_xval.json
```

`--verify` recomputes the design fingerprint and confirms the frozen design
carries no preflight deviations. `reachcert_numeric_xval.py` cross-validates
the exact-pair pitch certificate against direct-collocation linear programs
(`scipy` HiGHS): 64 parameter cases, a control witness for every case the
certificate accepts, a subset check against the earlier screen, and query
latency percentiles. Expected result is `"decision": "PASS"` with zero false
accepts, zero bound-tolerance violations, and zero witness failures.

## Calibration inputs

The selector is table-driven, so these files are part of the experiment
definition rather than incidental data.

| File                                    | Role                                                                                    |
| --------------------------------------- | --------------------------------------------------------------------------------------- |
| `calibration/gate_table.json`           | Empirical preset-action table over the calibrated angle/speed grid, with the neutral takeoff state per condition |
| `calibration/system_outcome_table.json` | Local system-level outcome evidence queried alongside the reachability screen            |
| `calibration/preset_effect_fit.json`    | Local calibration map: predicted pitch, pitch-rate, and wheel-speed shift per preset     |
| `calibration/confirmatory_design.json`  | Frozen 72-session, 600-run confirmatory design with its preflight record                 |
| `calibration/contaminated_registry.json`| Time-integrity registry read by the provenance gate; empty, since no data ships          |

The frozen design's `source_hashes` and `expected_commit` refer to the private
working tree it was frozen in, so those paths differ from this package's
layout. Portable verification is the fingerprint check above; `--verify-tree`
additionally checks the ledger and only passes in the original tree.

**The strict certificate-first selector ships disabled.** Its authorization
catalog is absent because the landing-capture and calibration-map held-out
gates both failed, so `--selector-policy strict_v2` (the default) refuses to
act. Paper cohorts that exercise the calibrated decision path therefore pass
`--selector-policy legacy`, as the queue scripts do.

## Simulator session helper

`data_pipeline/beamng_session.py` is **not** the experiment runner. It is the
session-lifecycle layer that `scripts/darts_bench.py` imports so one
BeamNG.tech process can stay attached across a long cohort: connect, wait until
the scenario is actually running, resume a paused sim (`ensure_freerun`), and
kill stale processes that still hold the research port.

This file is what remained after the authors burned a large amount of
wall-clock time on silent BeamNG failures: the scenario never leaving
`wait_scenario_running`, a vehicle that looks live but never moves, a freeze at
the instant of impact, an orphan process still bound to port 25252, and
overnight GPU throttle after the display sleeps. Do not replace it with a bare
`BeamNGpy` connect — those failure modes will return.

## Arm naming (paper-aligned)

Every cohort is a set of arms run in a deterministically randomized order
within each roll, so the paired differences share a roll.

| CLI / JSON arm                     | Paper role                                                              |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `neutral`                          | No preconditioning; the paired reference for every reported difference  |
| `nose_up` / `nose_down`            | Static ramp-face preset and its mirror (directionality control)         |
| `torque_up`                        | Rear-biased run-up torque baseline; suspension stays neutral            |
| `land_soft` / `land_soft_lift`     | Landing-end-only variants (null control for pre-liftoff shaping)        |
| `both_ends`                        | Preset at both the takeoff and landing ends                             |
| `air_only` / `air_susp`            | Airborne pitch loop alone, and crossed with the preset                  |
| `dart_only` / `dart_guarded`       | DART airborne law; `dart_guarded` adds the 200 rad/s command guard      |
| `dart_s_system`                    | Full DART-S stack (selector, preset, guarded DART law)                  |
| `dart_s_forced`                    | Preset forced on with the selector bypassed                             |
| `gated`                            | Calibrated table-first selector: prepare, refuse, or pass through       |
| `arxiv_outcome_only` / `arxiv_state_only` / `arxiv_budget_only` / `arxiv_joint` | The four selector-screen variants named by the frozen confirmatory design |
| `phase_015` / `phase_035`          | Timing actions generated by `--phase-sweep-leads 0.15 0.35`             |

## Queue scripts (recommended run order)

Each script is resumable: cohorts run sequentially and skip tags that already
have a `.DONE` marker under `data/queue_runs/`. Each script takes the queue
lock on entry, so run one at a time — the machine has a single simulator port.

| Script                       | Paper coverage                                                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `queue_confirmatory.sh`      | **Fig. 5** boundary and factorial blocks, plus the timing, washboard, held-out, and capture blocks: 72 sessions, 600 runs, headline result |
| `queue_fig03_angle_sweep.sh` | **Fig. 3** action-minus-neutral landing pitch across the five geometries at 13 m/s                                                    |
| `queue_fig04_speed_ladder.sh`| **Fig. 4** speed-dependent response on the 35° fillet (three sessions, permuted speed order)                                         |
| `queue_tab01_ablations.sh`   | **Table I** stroke scaling, takeoff end versus both ends, landing-end variants, airborne-channel interaction                          |
| `queue_tab01_robustness.sh`  | **Table I** mass ±20 %, forward CoG shift, entry-speed dispersion                                                                    |
| `queue_tab01_selector.sh`    | **Table I** selector execution branches and the 2.0 versus 0.8 s decision lead                                                       |

```bash
bash scripts/experiments/queue_confirmatory.sh       # ~20 h; gates each session as it lands
bash scripts/experiments/queue_fig03_angle_sweep.sh
bash scripts/experiments/queue_fig04_speed_ladder.sh
bash scripts/experiments/queue_tab01_ablations.sh
bash scripts/experiments/queue_tab01_robustness.sh
bash scripts/experiments/queue_tab01_selector.sh
python scripts/actuator_probe.py                     # Table I static authority row
```

The confirmatory queue is the only one with a hard gate: a session that misses
its expected runs or arms stops the queue instead of entering the pooled
analysis, and the pooled analysis itself must pass before the queue reports
success. The pooled verdict lands in `data/derived/confirmatory/verdict.json`.

Single-cohort example:

```bash
python scripts/darts_bench.py --angle 35 --speeds 13 --rolls 30 \
  --arms neutral nose_up --tag MY_TAG
```

Output: `data/cohorts/darts_bench_MY_TAG.json`, holding per-run records, the
per-arm medians, the paired differences against `neutral` with their sign-test
p-values, the safety criterion in force, and a `_provenance` block recording
that the run used deterministic stepping.

## Cohort file ↔ paper mapping

Every cohort file is `data/cohorts/darts_bench_<TAG>.json`.

| Paper item                                   | Queue script                  | Example tags                                                        |
| -------------------------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| Fig. 3 — geometry sweep at 13 m/s            | `queue_fig03_angle_sweep`     | `fig03_a15_v13` … `fig03_a35_v13`                                   |
| Fig. 4 — 35° speed ladder                    | `queue_fig04_speed_ladder`    | `fig04_ladder_s1`, `fig04_ladder_s2`, `fig04_ladder_s3`             |
| Fig. 5(a) — confirmed boundaries             | `queue_confirmatory`          | `axv_boundary_a40_v13_s01` … `_s08`, `axv_boundary_a35_v15_s01` …   |
| Fig. 5(b) — 35°/16 m/s factorial             | `queue_confirmatory`          | `axv_factorial_a35_v16_s01` … `_s08`                                |
| Timing action at 11.5 m/s                    | `queue_confirmatory`          | `axv_timing_a35_v11p5_s01` … `_s08`                                 |
| Washboard evaluation                         | `queue_confirmatory`          | `axv_washboard_a35_v13_s01` … `_s08`                                |
| Held-out interpolation conditions            | `queue_confirmatory`          | `axv_heldout_p1_a37_v15p25_s01` …, `axv_heldout_p2_a39_v13_s01` …   |
| Capture boundary and negative controls       | `queue_confirmatory`          | `axv_capture_boundary_a40_v12p65_s01` …, `axv_capture_negative_a36_v13p5_s01` … |
| Table I — static actuator authority          | `actuator_probe.py`           | `data/derived/actuator_probe.json`                                  |
| Table I — stroke scaling (33/50/67 mm)       | `queue_tab01_ablations`       | `tab01_stroke_r010`, `tab01_stroke_r015`, `tab01_stroke_r020`       |
| Table I — takeoff end versus both ends       | `queue_tab01_ablations`       | `tab01_both_ends_a20`, `tab01_both_ends_a35`                        |
| Table I — landing-end variants               | `queue_tab01_ablations`       | `tab01_landing_end_a20`, `tab01_landing_end_a35`                    |
| Channel interaction (sub-additive gains)     | `queue_tab01_ablations`       | `tab01_channels_a20`, `tab01_channels_a35`                          |
| Table I — platform robustness                | `queue_tab01_robustness`      | `tab01_rob_m080`, `tab01_rob_m120`, `tab01_rob_cogfront`            |
| Table I — selector execution                 | `queue_tab01_selector`        | `tab01_sel_prepare`, `tab01_sel_refuse`, `tab01_sel_passthrough`    |
| Table I — decision lead                      | `queue_tab01_selector`        | `tab01_sel_prepare` (2.0 s) versus `tab01_sel_lead0p8` (0.8 s)      |

## Scope

The mechanism measurements in Fig. 6 come from a different rig: the convex-lip
kicker bench of the prior DART work, instrumented with suspension presets and
per-axle stroke traces. That instrumented bench is not part of this package or
of the DART reproduction package, so Fig. 6 cannot be re-run from here. The
vehicle configs in `vehicles/sbr/` are the ones those cohorts used, and the two
entries in `sim/scenarios/` record the reference and held-out kicker
geometries.

## Reproducibility

Cohorts step the physics deterministically at a fixed 0.01 s and pair arms
within a roll. Decision-grade cells use 30 pairs per condition; the
confirmatory blocks use eight independent sessions of three paired rolls and
session-equal cluster inference, so a session is the unit of replication.
Apply the paper's paired conventions within cohorts: medians via
`statistics.median`, sign tests on paired differences, Holm adjustment across
the prespecified family, and Wilson intervals on criterion-success counts.
Reference numbers in the paper come from the authors' cohort archive; this
repository re-runs the same experimental designs.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

## Citation

If you use this reproduction package or build on DART-S, please cite:

```bibtex
@misc{hu2026darts,
  title  = {DART-S: Reachability-Audited Active-Suspension Preconditioning
            for Off-Road Vehicle Jumps},
  author = {Yu Hu and Fangzhou Zhao and Liang Chen and Chen Min and Wei Li and
            Mingyuan Sang and Jiajia Ma and Shican Chen and Di Pang and
            Baolei Chen},
  year   = {2026},
  note   = {arXiv preprint},
}
```

The prior work this package builds on is DART
([arXiv:2607.29011](https://arxiv.org/abs/2607.29011)).
