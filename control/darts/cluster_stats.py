"""Physical-session-aware statistics for DART-S confirmatory cohorts."""
from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from itertools import product
from math import comb
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PairedSessionRecord:
    physical_session_id: str
    roll_index: int
    baseline_value: float
    candidate_value: float
    baseline_speed_mps: float
    candidate_speed_mps: float
    baseline_took_off: bool = True
    candidate_took_off: bool = True
    baseline_touched_down: bool = True
    candidate_touched_down: bool = True
    baseline_window_complete: bool = True
    candidate_window_complete: bool = True

    def __post_init__(self):
        if not self.physical_session_id or self.roll_index < 0:
            raise ValueError("physical session ID and nonnegative roll index required")
        numeric = (
            self.baseline_value,
            self.candidate_value,
            self.baseline_speed_mps,
            self.candidate_speed_mps,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("paired-session numeric values must be finite")

    @property
    def difference(self) -> float:
        return self.candidate_value - self.baseline_value


@dataclass(frozen=True)
class MultiArmSessionRecord:
    physical_session_id: str
    roll_index: int
    arm: str
    value: float
    speed_mps: float
    took_off: bool = True
    touched_down: bool = True
    window_complete: bool = True

    def __post_init__(self):
        if (
            not self.physical_session_id
            or not self.arm
            or self.roll_index < 0
            or not math.isfinite(float(self.value))
            or not math.isfinite(float(self.speed_mps))
        ):
            raise ValueError("invalid multi-arm session record")


@dataclass(frozen=True)
class SessionDesignAudit:
    passed: bool
    n_sessions: int
    rolls_per_session: tuple[int, ...]
    duplicate_keys: tuple[tuple[str, int], ...]
    missing_keys: tuple[tuple[str, int], ...]
    reason: str


@dataclass(frozen=True)
class ClusterBootstrapResult:
    estimate: float
    ci_lo: float
    ci_hi: float
    n_bootstrap: int
    n_sessions: int
    statistic: str
    seed: int


@dataclass(frozen=True)
class SessionSignflipResult:
    estimate: float
    p_two_sided: float
    n_sessions: int
    session_effects: tuple[float, ...]


@dataclass(frozen=True)
class IntraclassCorrelationResult:
    estimable: bool
    classification: str
    n_sessions: int
    mean_rolls_per_session: float
    icc_1_1: float | None
    icc_1_k: float | None
    design_effect: float | None
    effective_n: float | None


@dataclass(frozen=True)
class LeaveOneSessionOutResult:
    estimate: float
    n_sessions: int
    effects_by_session: Mapping[str, float]
    estimates_without_session: Mapping[str, float]
    min_estimate: float
    max_estimate: float
    direction_flip: bool


@dataclass(frozen=True)
class SpeedMatchAudit:
    passed: bool
    mode: str
    n_pairs: int
    median_abs_gap_mps: float
    q95_abs_gap_mps: float
    median_limit_mps: float
    q95_limit_mps: float


@dataclass(frozen=True)
class CompletenessAudit:
    passed: bool
    n_pairs: int
    baseline_incomplete: int
    candidate_incomplete: int
    denominator_policy: str


@dataclass(frozen=True)
class MultiArmDesignAudit:
    passed: bool
    n_sessions: int
    arms: tuple[str, ...]
    duplicate_keys: tuple[tuple[str, int, str], ...]
    missing_keys: tuple[tuple[str, int, str], ...]
    reason: str


def audit_session_design(
    records: Iterable[PairedSessionRecord],
    *,
    expected_sessions: int = 8,
    expected_rolls_per_session: int = 3,
) -> SessionDesignAudit:
    records = tuple(records)
    counts: dict[str, set[int]] = {}
    seen: set[tuple[str, int]] = set()
    duplicates: list[tuple[str, int]] = []
    for record in records:
        key = (record.physical_session_id, record.roll_index)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
        counts.setdefault(record.physical_session_id, set()).add(record.roll_index)
    missing = []
    expected_rolls = set(range(expected_rolls_per_session))
    for session_id, rolls in counts.items():
        missing.extend(
            (session_id, roll) for roll in sorted(expected_rolls - rolls))
    n_sessions = len(counts)
    passed = (
        not duplicates
        and not missing
        and n_sessions == expected_sessions
        and all(len(rolls) == expected_rolls_per_session for rolls in counts.values())
    )
    return SessionDesignAudit(
        passed=passed,
        n_sessions=n_sessions,
        rolls_per_session=tuple(sorted(len(rolls) for rolls in counts.values())),
        duplicate_keys=tuple(sorted(duplicates)),
        missing_keys=tuple(sorted(missing)),
        reason="complete" if passed else "physical_session_design_mismatch",
    )


def audit_multi_arm_design(
    records: Iterable[MultiArmSessionRecord],
    *,
    expected_arms: Iterable[str],
    expected_sessions: int = 8,
    expected_rolls_per_session: int = 3,
) -> MultiArmDesignAudit:
    records = tuple(records)
    arms = tuple(sorted(set(expected_arms)))
    sessions = sorted({record.physical_session_id for record in records})
    seen = set()
    duplicates = []
    for record in records:
        key = (record.physical_session_id, record.roll_index, record.arm)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    expected = {
        (session, roll, arm)
        for session in sessions
        for roll in range(expected_rolls_per_session)
        for arm in arms
    }
    missing = sorted(expected - seen)
    unexpected_arms = sorted({record.arm for record in records} - set(arms))
    passed = (
        len(sessions) == expected_sessions
        and not duplicates
        and not missing
        and not unexpected_arms
        and len(seen) == len(expected)
    )
    return MultiArmDesignAudit(
        passed=passed,
        n_sessions=len(sessions),
        arms=arms,
        duplicate_keys=tuple(sorted(duplicates)),
        missing_keys=tuple(missing),
        reason="complete" if passed else "multi_arm_design_mismatch",
    )


def paired_contrast(
    records: Iterable[MultiArmSessionRecord],
    *,
    baseline_arm: str,
    candidate_arm: str,
) -> tuple[PairedSessionRecord, ...]:
    indexed = {
        (record.physical_session_id, record.roll_index, record.arm): record
        for record in records
    }
    keys = sorted({
        (session_id, roll_index)
        for session_id, roll_index, arm in indexed
        if arm in (baseline_arm, candidate_arm)
    })
    pairs = []
    for session_id, roll_index in keys:
        baseline = indexed.get((session_id, roll_index, baseline_arm))
        candidate = indexed.get((session_id, roll_index, candidate_arm))
        if baseline is None or candidate is None:
            continue
        pairs.append(PairedSessionRecord(
            physical_session_id=session_id,
            roll_index=roll_index,
            baseline_value=baseline.value,
            candidate_value=candidate.value,
            baseline_speed_mps=baseline.speed_mps,
            candidate_speed_mps=candidate.speed_mps,
            baseline_took_off=baseline.took_off,
            candidate_took_off=candidate.took_off,
            baseline_touched_down=baseline.touched_down,
            candidate_touched_down=candidate.touched_down,
            baseline_window_complete=baseline.window_complete,
            candidate_window_complete=candidate.window_complete,
        ))
    return tuple(pairs)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    n = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (n - index) * float(value)))
        adjusted[name] = running
    return adjusted


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires values")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def _statistic(values: Sequence[float], statistic: str) -> float:
    if statistic == "mean":
        return float(statistics.fmean(values))
    if statistic == "median":
        return float(statistics.median(values))
    raise ValueError("statistic must be mean or median")


def two_level_cluster_bootstrap(
    records: Iterable[PairedSessionRecord],
    *,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    seed: int = 20260811,
    alpha: float = 0.05,
) -> ClusterBootstrapResult:
    """Resample physical sessions, then paired rolls within each session."""
    records = tuple(records)
    grouped: dict[str, list[PairedSessionRecord]] = {}
    for record in records:
        grouped.setdefault(record.physical_session_id, []).append(record)
    if not grouped:
        raise ValueError("cluster bootstrap requires records")
    if n_bootstrap <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid bootstrap configuration")
    sessions = sorted(grouped)
    session_effects = [
        _statistic(
            [record.difference for record in grouped[session]], statistic)
        for session in sessions
    ]
    observed = _statistic(session_effects, statistic)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_bootstrap):
        sampled_session_effects = []
        for _session_draw in sessions:
            selected_session = rng.choice(sessions)
            rows = grouped[selected_session]
            sampled_rows = [
                rng.choice(rows).difference for _ in range(len(rows))]
            sampled_session_effects.append(
                _statistic(sampled_rows, statistic))
        draws.append(_statistic(sampled_session_effects, statistic))
    return ClusterBootstrapResult(
        estimate=observed,
        ci_lo=_quantile(draws, alpha / 2.0),
        ci_hi=_quantile(draws, 1.0 - alpha / 2.0),
        n_bootstrap=n_bootstrap,
        n_sessions=len(sessions),
        statistic=statistic,
        seed=seed,
    )


def session_level_exact_signflip(
    records: Iterable[PairedSessionRecord],
) -> SessionSignflipResult:
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.physical_session_id, []).append(record.difference)
    if not grouped:
        raise ValueError("session sign-flip requires records")
    effects = tuple(
        statistics.fmean(grouped[session_id]) for session_id in sorted(grouped))
    observed = abs(statistics.fmean(effects))
    n = len(effects)
    if n > 20:
        raise ValueError("exact sign-flip enumeration is capped at 20 sessions")
    extreme = 0
    total = 2 ** n
    for signs in product((-1.0, 1.0), repeat=n):
        value = abs(statistics.fmean(
            sign * effect for sign, effect in zip(signs, effects)))
        extreme += int(value >= observed - 1e-15)
    return SessionSignflipResult(
        estimate=statistics.fmean(effects),
        p_two_sided=extreme / total,
        n_sessions=n,
        session_effects=effects,
    )


def intraclass_correlation(
    records: Iterable[PairedSessionRecord],
) -> IntraclassCorrelationResult:
    """Descriptive one-way random-effects ICC for paired differences."""
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.physical_session_id, []).append(
            record.difference)
    n_sessions = len(grouped)
    n_total = sum(len(values) for values in grouped.values())
    mean_rolls = n_total / n_sessions if n_sessions else 0.0
    classification = (
        "not_estimable"
        if n_sessions < 2 or any(len(values) < 2 for values in grouped.values())
        else (
            "low_cluster_count_descriptive"
            if n_sessions < 8 else "estimable"
        )
    )
    if classification == "not_estimable":
        return IntraclassCorrelationResult(
            estimable=False,
            classification=classification,
            n_sessions=n_sessions,
            mean_rolls_per_session=mean_rolls,
            icc_1_1=None,
            icc_1_k=None,
            design_effect=None,
            effective_n=None,
        )

    session_means = {
        session: statistics.fmean(values)
        for session, values in grouped.items()
    }
    grand = statistics.fmean(session_means.values())
    ss_between = sum(
        len(grouped[session]) * (mean - grand) ** 2
        for session, mean in session_means.items()
    )
    ss_within = sum(
        sum((value - session_means[session]) ** 2 for value in values)
        for session, values in grouped.items()
    )
    ms_between = ss_between / (n_sessions - 1)
    ms_within = ss_within / (n_total - n_sessions)
    k = mean_rolls
    denominator = ms_between + (k - 1.0) * ms_within
    icc_1_1 = (
        (ms_between - ms_within) / denominator
        if denominator > 0.0 else 0.0
    )
    icc_1_k = (
        (ms_between - ms_within) / ms_between
        if ms_between > 0.0 else 0.0
    )
    design_effect = 1.0 + (k - 1.0) * max(0.0, icc_1_1)
    return IntraclassCorrelationResult(
        estimable=True,
        classification=classification,
        n_sessions=n_sessions,
        mean_rolls_per_session=mean_rolls,
        icc_1_1=icc_1_1,
        icc_1_k=icc_1_k,
        design_effect=design_effect,
        effective_n=n_total / design_effect,
    )


def leave_one_session_out(
    records: Iterable[PairedSessionRecord],
) -> LeaveOneSessionOutResult:
    """Report session leverage without treating LOSO as new inference."""
    grouped: dict[str, list[float]] = {}
    for record in records:
        grouped.setdefault(record.physical_session_id, []).append(
            record.difference)
    if len(grouped) < 2:
        raise ValueError("LOSO requires at least two physical sessions")
    effects = {
        session: statistics.fmean(values)
        for session, values in sorted(grouped.items())
    }
    estimate = statistics.fmean(effects.values())
    without = {
        excluded: statistics.fmean(
            effect for session, effect in effects.items()
            if session != excluded
        )
        for excluded in effects
    }

    def sign(value: float) -> int:
        return 1 if value > 0.0 else (-1 if value < 0.0 else 0)

    return LeaveOneSessionOutResult(
        estimate=estimate,
        n_sessions=len(effects),
        effects_by_session=effects,
        estimates_without_session=without,
        min_estimate=min(without.values()),
        max_estimate=max(without.values()),
        direction_flip=any(
            sign(value) != sign(estimate) for value in without.values()),
    )


def exact_mcnemar_p(
    baseline: Iterable[bool],
    candidate: Iterable[bool],
) -> tuple[int, int, float]:
    pairs = tuple(zip(baseline, candidate))
    baseline_only = sum(bool(base) and not bool(cand) for base, cand in pairs)
    candidate_only = sum(bool(cand) and not bool(base) for base, cand in pairs)
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return candidate_only, baseline_only, 1.0
    tail = min(candidate_only, baseline_only)
    p = min(
        1.0,
        2.0 * sum(comb(discordant, index) for index in range(tail + 1))
        / (2 ** discordant),
    )
    return candidate_only, baseline_only, p


def audit_speed_match(
    records: Iterable[PairedSessionRecord],
    *,
    mode: str,
) -> SpeedMatchAudit:
    records = tuple(records)
    gaps = [
        abs(record.candidate_speed_mps - record.baseline_speed_mps)
        for record in records
    ]
    if not gaps:
        raise ValueError("speed audit requires pairs")
    if mode == "timed":
        median_limit, q95_limit = 0.10, 0.20
    elif mode == "standard":
        median_limit, q95_limit = 0.20, 0.30
    else:
        raise ValueError("speed audit mode must be timed or standard")
    median_gap = float(statistics.median(gaps))
    q95_gap = _quantile(gaps, 0.95)
    return SpeedMatchAudit(
        passed=median_gap <= median_limit and q95_gap <= q95_limit,
        mode=mode,
        n_pairs=len(gaps),
        median_abs_gap_mps=median_gap,
        q95_abs_gap_mps=q95_gap,
        median_limit_mps=median_limit,
        q95_limit_mps=q95_limit,
    )


def audit_completeness(
    records: Iterable[PairedSessionRecord],
) -> CompletenessAudit:
    records = tuple(records)

    def incomplete(record: PairedSessionRecord, prefix: str) -> bool:
        return not all((
            getattr(record, f"{prefix}_took_off"),
            getattr(record, f"{prefix}_touched_down"),
            getattr(record, f"{prefix}_window_complete"),
        ))

    baseline_incomplete = sum(incomplete(record, "baseline") for record in records)
    candidate_incomplete = sum(incomplete(record, "candidate") for record in records)
    return CompletenessAudit(
        passed=baseline_incomplete == 0 and candidate_incomplete == 0,
        n_pairs=len(records),
        baseline_incomplete=baseline_incomplete,
        candidate_incomplete=candidate_incomplete,
        denominator_policy=(
            "all preregistered attempts retained; incomplete trajectories are "
            "endpoint failures, never exclusions"
        ),
    )


def cluster_analysis_payload(
    records: Iterable[PairedSessionRecord],
    *,
    speed_mode: str,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    seed: int = 20260811,
) -> dict:
    records = tuple(records)
    design = audit_session_design(records)
    bootstrap = two_level_cluster_bootstrap(
        records, statistic=statistic, n_bootstrap=n_bootstrap, seed=seed)
    signflip = session_level_exact_signflip(records)
    icc = intraclass_correlation(records)
    loso = leave_one_session_out(records)
    speed = audit_speed_match(records, mode=speed_mode)
    completeness = audit_completeness(records)
    return {
        "design": asdict(design),
        "cluster_bootstrap": asdict(bootstrap),
        "session_signflip": asdict(signflip),
        "icc": asdict(icc),
        "leave_one_session_out": asdict(loso),
        "speed_match": asdict(speed),
        "completeness": asdict(completeness),
        "decision_ready": (
            design.passed and speed.passed and completeness.passed),
    }


__all__ = [
    "PairedSessionRecord",
    "MultiArmSessionRecord",
    "SessionDesignAudit",
    "ClusterBootstrapResult",
    "SessionSignflipResult",
    "IntraclassCorrelationResult",
    "LeaveOneSessionOutResult",
    "SpeedMatchAudit",
    "CompletenessAudit",
    "MultiArmDesignAudit",
    "audit_session_design",
    "audit_multi_arm_design",
    "paired_contrast",
    "holm_adjust",
    "two_level_cluster_bootstrap",
    "session_level_exact_signflip",
    "intraclass_correlation",
    "leave_one_session_out",
    "exact_mcnemar_p",
    "audit_speed_match",
    "audit_completeness",
    "cluster_analysis_payload",
]
