"""Pure progressive-evidence gates for the DART-S Neural CalMap.

Evidence is local to ``(domain, action_kind, schedule_id)``.  The module does
not read artifacts or mutate a catalog: callers supply immutable audit facts,
and receive immutable assessments and promotion plans.  Invalid split reuse is
rejected at construction time so a leaked session can never earn a grade.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Iterable, Mapping


EVIDENCE_SCHEMA_VERSION = "darts-calmap-evidence/1"
EVIDENCE_FINGERPRINT_VERSION = "darts-calmap-evidence-fingerprint/1"
SESSION_ROLES = (
    "fit_session_ids",
    "calibration_session_ids",
    "holdout_session_ids",
    "ood_session_ids",
    "closed_loop_session_ids",
    "anchor_session_ids",
)


class EvidenceError(ValueError):
    """Base class for fail-closed evidence errors."""


class SessionIsolationError(EvidenceError):
    """A physical session was reused across evidence roles."""


class EvidencePromotionError(EvidenceError):
    """A caller requested more authority than the evidence earned."""


class EvidenceGrade(str, Enum):
    """Local evidence authority, ordered from unusable to deployable."""

    W0 = "W0"
    W1 = "W1"
    W2 = "W2"
    S1 = "S1"
    S2 = "S2"

    @property
    def rank(self) -> int:
        return _GRADE_RANK[self]

    @classmethod
    def coerce(cls, value: "EvidenceGrade | str") -> "EvidenceGrade":
        return value if isinstance(value, cls) else cls(str(value))


_GRADE_RANK = {
    EvidenceGrade.W0: 0,
    EvidenceGrade.W1: 1,
    EvidenceGrade.W2: 2,
    EvidenceGrade.S1: 3,
    EvidenceGrade.S2: 4,
}


def grade_at_least(
    actual: EvidenceGrade | str,
    required: EvidenceGrade | str,
) -> bool:
    """Return whether ``actual`` carries at least ``required`` authority."""

    return EvidenceGrade.coerce(actual).rank >= EvidenceGrade.coerce(required).rank


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EvidenceError(f"{label} must be a non-empty, trimmed string")
    return value


def _canonical_ids(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise EvidenceError(f"{label} must be a collection, not a string")
    result = tuple(_identifier(value, label) for value in values)
    if len(result) != len(set(result)):
        raise EvidenceError(f"{label} contains duplicate IDs")
    return tuple(sorted(result))


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_bool(value: object, label: str) -> None:
    if not isinstance(value, bool):
        raise EvidenceError(f"{label} must be bool")


@dataclass(frozen=True, order=True)
class EvidenceKey:
    """The smallest unit that may gain or lose evidence authority."""

    domain: str
    action_kind: str
    schedule_id: str

    def __post_init__(self) -> None:
        _identifier(self.domain, "domain")
        _identifier(self.action_kind, "action_kind")
        _identifier(self.schedule_id, "schedule_id")

    @property
    def token(self) -> str:
        return f"{self.domain}/{self.action_kind}/{self.schedule_id}"


@dataclass(frozen=True)
class SessionOverlap:
    session_id: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class SessionIsolationAudit:
    passed: bool
    session_count: int
    overlaps: tuple[SessionOverlap, ...]


def audit_session_isolation(
    sessions: "EvidenceSessions | Mapping[str, Iterable[str]]",
) -> SessionIsolationAudit:
    """Audit physical-session disjointness across every supplied role."""

    if isinstance(sessions, EvidenceSessions):
        role_map = sessions.as_role_map()
    elif isinstance(sessions, Mapping):
        role_map = {
            str(role): _canonical_ids(values, str(role))
            for role, values in sessions.items()
        }
    else:
        raise TypeError("sessions must be EvidenceSessions or a role mapping")

    memberships: dict[str, list[str]] = {}
    for role, session_ids in role_map.items():
        for session_id in session_ids:
            memberships.setdefault(session_id, []).append(role)
    overlaps = tuple(
        SessionOverlap(session_id, tuple(sorted(roles)))
        for session_id, roles in sorted(memberships.items())
        if len(roles) > 1
    )
    return SessionIsolationAudit(
        passed=not overlaps,
        session_count=len(memberships),
        overlaps=overlaps,
    )


def assert_session_isolation(sessions: "EvidenceSessions") -> None:
    """Raise if any physical session crosses an evidence split."""

    audit = audit_session_isolation(sessions)
    if audit.passed:
        return
    detail = "; ".join(
        f"{overlap.session_id}=>{','.join(overlap.roles)}"
        for overlap in audit.overlaps
    )
    raise SessionIsolationError(f"session leakage across evidence roles: {detail}")


@dataclass(frozen=True)
class EvidenceSessions:
    """Physical-session sets used by one local evidence key.

    All roles are pairwise disjoint.  This is stricter than the mandatory
    fit/cal/holdout rule and also preserves the independence claimed by OOD,
    anchor, and paired closed-loop evidence.
    """

    fit_session_ids: tuple[str, ...] = ()
    calibration_session_ids: tuple[str, ...] = ()
    holdout_session_ids: tuple[str, ...] = ()
    ood_session_ids: tuple[str, ...] = ()
    closed_loop_session_ids: tuple[str, ...] = ()
    anchor_session_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for role in SESSION_ROLES:
            object.__setattr__(
                self,
                role,
                _canonical_ids(getattr(self, role), role),
            )
        assert_session_isolation(self)

    def as_role_map(self) -> dict[str, tuple[str, ...]]:
        return {role: getattr(self, role) for role in SESSION_ROLES}

    @property
    def all_session_ids(self) -> tuple[str, ...]:
        return tuple(sorted({
            session_id
            for session_ids in self.as_role_map().values()
            for session_id in session_ids
        }))


@dataclass(frozen=True)
class EvidenceIntegrity:
    """Outcome-based audit of the fields needed for time-dependent evidence."""

    full_state_complete: bool
    session_metadata_complete: bool
    time_integrity_ok: bool
    session_identity_matches: bool = True
    missing_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "full_state_complete",
            "session_metadata_complete",
            "time_integrity_ok",
            "session_identity_matches",
        ):
            _require_bool(getattr(self, name), name)
        object.__setattr__(
            self,
            "missing_fields",
            _canonical_ids(self.missing_fields, "missing_fields"),
        )

    @classmethod
    def complete(cls) -> "EvidenceIntegrity":
        return cls(True, True, True)

    @property
    def passed(self) -> bool:
        return (
            self.full_state_complete
            and self.session_metadata_complete
            and self.time_integrity_ok
            and self.session_identity_matches
            and not self.missing_fields
        )

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.full_state_complete or self.missing_fields:
            suffix = (
                f"({','.join(self.missing_fields)})"
                if self.missing_fields
                else ""
            )
            reasons.append(f"full_state_incomplete{suffix}")
        if not self.session_metadata_complete:
            reasons.append("session_metadata_incomplete")
        if not self.session_identity_matches:
            reasons.append("params_protocol_session_mismatch")
        if not self.time_integrity_ok:
            reasons.append("time_integrity_failed")
        return tuple(reasons)


# A concise alias for callers that treat this object as a run-level audit.
RunIntegrityAudit = EvidenceIntegrity


@dataclass(frozen=True)
class EvidenceChecks:
    """Frozen gate outcomes for one key; false values never grant authority."""

    historical_or_discovery: bool = False
    joint_tube_frozen: bool = False
    holdout_untouched: bool = False
    holdout_coverage_passed: bool = False
    ood_abstention_passed: bool = False
    paired_closed_loop_passed: bool = False
    anchor_cohort_passed: bool = False

    def __post_init__(self) -> None:
        for field in fields(self):
            _require_bool(getattr(self, field.name), field.name)


@dataclass(frozen=True)
class ReplayAnchorEvidence:
    """Replay accounting for one old physical-session anchor."""

    anchor_id: str
    source_session_id: str
    replay_count: int
    expected_source_fingerprint: str | None
    actual_source_fingerprint: str | None
    time_integrity_ok: bool

    def __post_init__(self) -> None:
        _identifier(self.anchor_id, "anchor_id")
        _identifier(self.source_session_id, "source_session_id")
        if (
            isinstance(self.replay_count, bool)
            or not isinstance(self.replay_count, int)
            or self.replay_count < 0
        ):
            raise EvidenceError("replay_count must be a nonnegative integer")
        _require_bool(self.time_integrity_ok, "time_integrity_ok")
        for label, fingerprint in (
            ("expected_source_fingerprint", self.expected_source_fingerprint),
            ("actual_source_fingerprint", self.actual_source_fingerprint),
        ):
            if fingerprint is not None and not _is_sha256(fingerprint):
                raise EvidenceError(f"{label} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ReplayAnchorAudit:
    """Machine-readable replay balance, lineage, and isolation verdict."""

    passed: bool
    required_anchor_ids: tuple[str, ...]
    observed_anchor_ids: tuple[str, ...]
    replay_source_session_ids: tuple[str, ...]
    missing_anchor_ids: tuple[str, ...]
    unexpected_anchor_ids: tuple[str, ...]
    duplicate_anchor_ids: tuple[str, ...]
    duplicate_source_session_ids: tuple[str, ...]
    forbidden_session_overlaps: tuple[str, ...]
    min_replay_count: int | None
    max_replay_count: int | None
    max_allowed_count_spread: int
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_bool(self.passed, "passed")
        if self.passed != (not self.reasons):
            raise EvidenceError("replay anchor passed flag must agree with reasons")


def audit_replay_anchors(
    records: Iterable[ReplayAnchorEvidence],
    *,
    required_anchor_ids: Iterable[str],
    forbidden_session_ids: Iterable[str] = (),
    max_replay_count_spread: int = 1,
) -> ReplayAnchorAudit:
    """Audit session-balanced replay without touching training artifacts.

    A passing audit requires every required anchor exactly once, no extras,
    valid and matching source fingerprints, clean time provenance, positive
    replay counts, near-equal counts by physical session, and no reuse of an
    evaluation session.
    """

    if (
        isinstance(max_replay_count_spread, bool)
        or not isinstance(max_replay_count_spread, int)
        or max_replay_count_spread < 0
    ):
        raise EvidenceError("max_replay_count_spread must be nonnegative")
    required = _canonical_ids(required_anchor_ids, "required_anchor_ids")
    forbidden = set(_canonical_ids(forbidden_session_ids, "forbidden_session_ids"))
    materialized = tuple(records)

    anchor_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    reasons: list[str] = []
    for record in materialized:
        if not isinstance(record, ReplayAnchorEvidence):
            raise TypeError("records must contain ReplayAnchorEvidence")
        anchor_counts[record.anchor_id] = anchor_counts.get(record.anchor_id, 0) + 1
        source_counts[record.source_session_id] = (
            source_counts.get(record.source_session_id, 0) + 1
        )
        if record.replay_count <= 0:
            reasons.append(f"replay_count_not_positive({record.anchor_id})")
        if not record.time_integrity_ok:
            reasons.append(f"replay_time_integrity_failed({record.anchor_id})")
        if (
            record.expected_source_fingerprint is None
            or record.actual_source_fingerprint is None
        ):
            reasons.append(f"replay_fingerprint_missing({record.anchor_id})")
        elif (
            record.expected_source_fingerprint
            != record.actual_source_fingerprint
        ):
            reasons.append(f"replay_fingerprint_mismatch({record.anchor_id})")

    observed = set(anchor_counts)
    required_set = set(required)
    missing = tuple(sorted(required_set - observed))
    unexpected = tuple(sorted(observed - required_set))
    duplicate_anchors = tuple(sorted(
        anchor_id for anchor_id, count in anchor_counts.items() if count > 1
    ))
    duplicate_sources = tuple(sorted(
        session_id for session_id, count in source_counts.items() if count > 1
    ))
    forbidden_overlaps = tuple(sorted(set(source_counts) & forbidden))

    if not required:
        reasons.append("required_anchor_ids_empty")
    if missing:
        reasons.append(f"replay_anchors_missing({','.join(missing)})")
    if unexpected:
        reasons.append(f"unexpected_replay_anchors({','.join(unexpected)})")
    if duplicate_anchors:
        reasons.append(f"duplicate_replay_anchor_ids({','.join(duplicate_anchors)})")
    if duplicate_sources:
        reasons.append(
            f"reused_replay_source_sessions({','.join(duplicate_sources)})")
    if forbidden_overlaps:
        reasons.append(
            f"replay_evaluation_session_leakage({','.join(forbidden_overlaps)})")

    counts = [
        record.replay_count
        for record in materialized
        if record.anchor_id in required_set
    ]
    minimum = min(counts) if counts else None
    maximum = max(counts) if counts else None
    if (
        minimum is not None
        and maximum is not None
        and maximum - minimum > max_replay_count_spread
    ):
        reasons.append(
            f"replay_not_session_balanced(spread={maximum - minimum},"
            f"allowed={max_replay_count_spread})")

    unique_reasons = tuple(sorted(set(reasons)))
    return ReplayAnchorAudit(
        passed=not unique_reasons,
        required_anchor_ids=required,
        observed_anchor_ids=tuple(sorted(observed)),
        replay_source_session_ids=tuple(sorted(source_counts)),
        missing_anchor_ids=missing,
        unexpected_anchor_ids=unexpected,
        duplicate_anchor_ids=duplicate_anchors,
        duplicate_source_session_ids=duplicate_sources,
        forbidden_session_overlaps=forbidden_overlaps,
        min_replay_count=minimum,
        max_replay_count=maximum,
        max_allowed_count_spread=max_replay_count_spread,
        reasons=unique_reasons,
    )


@dataclass(frozen=True)
class EvidenceFailure:
    """A discovered failure that caps authority for this key."""

    failure_id: str
    reason: str
    maximum_grade: EvidenceGrade

    def __post_init__(self) -> None:
        _identifier(self.failure_id, "failure_id")
        _identifier(self.reason, "failure reason")
        object.__setattr__(
            self,
            "maximum_grade",
            EvidenceGrade.coerce(self.maximum_grade),
        )


@dataclass(frozen=True)
class LocalEvidence:
    """All immutable facts used to grade one local key."""

    key: EvidenceKey
    sessions: EvidenceSessions
    integrity: EvidenceIntegrity
    checks: EvidenceChecks
    model_fingerprint: str | None
    replay_anchor_audit: ReplayAnchorAudit | None = None
    failures: tuple[EvidenceFailure, ...] = ()

    def __post_init__(self) -> None:
        if self.model_fingerprint is not None and not _is_sha256(
            self.model_fingerprint
        ):
            raise EvidenceError("model_fingerprint must be a lowercase SHA-256 digest")
        if any(not isinstance(failure, EvidenceFailure) for failure in self.failures):
            raise TypeError("failures must contain EvidenceFailure")
        failure_ids = [failure.failure_id for failure in self.failures]
        if len(failure_ids) != len(set(failure_ids)):
            raise EvidenceError("failure IDs must be unique")
        object.__setattr__(
            self,
            "failures",
            tuple(sorted(self.failures, key=lambda failure: failure.failure_id)),
        )


def _primitive(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        converted: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceError("fingerprint mapping keys must be strings")
            converted[key] = _primitive(item)
        return converted
    if isinstance(value, (tuple, list)):
        return [_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_primitive(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ),
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise EvidenceError("non-finite values cannot be fingerprinted")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


@dataclass(frozen=True)
class VersionedFingerprint:
    """A digest whose schema and canonicalization version are explicit."""

    schema_version: str
    fingerprint_version: str
    algorithm: str
    digest: str

    def __post_init__(self) -> None:
        _identifier(self.schema_version, "schema_version")
        _identifier(self.fingerprint_version, "fingerprint_version")
        if self.algorithm != "sha256":
            raise EvidenceError("only sha256 fingerprints are supported")
        if not _is_sha256(self.digest):
            raise EvidenceError("digest must be a lowercase SHA-256 digest")

    @property
    def token(self) -> str:
        return (
            f"{self.schema_version}|{self.fingerprint_version}|"
            f"{self.algorithm}:{self.digest}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "schema_version": self.schema_version,
            "fingerprint_version": self.fingerprint_version,
            "algorithm": self.algorithm,
            "digest": self.digest,
            "token": self.token,
        }


def versioned_fingerprint(
    payload: object,
    *,
    schema_version: str = EVIDENCE_SCHEMA_VERSION,
    fingerprint_version: str = EVIDENCE_FINGERPRINT_VERSION,
) -> VersionedFingerprint:
    """Hash a canonical payload with version fields inside the hash envelope."""

    _identifier(schema_version, "schema_version")
    _identifier(fingerprint_version, "fingerprint_version")
    envelope = {
        "schema_version": schema_version,
        "fingerprint_version": fingerprint_version,
        "payload": _primitive(payload),
    }
    encoded = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return VersionedFingerprint(
        schema_version=schema_version,
        fingerprint_version=fingerprint_version,
        algorithm="sha256",
        digest=hashlib.sha256(encoded).hexdigest(),
    )


def local_evidence_fingerprint(
    evidence: LocalEvidence,
    *,
    fingerprint_version: str = EVIDENCE_FINGERPRINT_VERSION,
) -> VersionedFingerprint:
    return versioned_fingerprint(
        evidence,
        fingerprint_version=fingerprint_version,
    )


@dataclass(frozen=True)
class EvidenceAssessment:
    schema_version: str
    key: EvidenceKey
    uncapped_grade: EvidenceGrade
    grade: EvidenceGrade
    blocking_reasons: tuple[str, ...]
    candidate_eligible: bool
    shadow_eligible: bool
    deploy_eligible: bool
    model_fingerprint: str | None
    evidence_fingerprint: VersionedFingerprint

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "key": _primitive(self.key),
            "uncapped_grade": self.uncapped_grade.value,
            "grade": self.grade.value,
            "blocking_reasons": list(self.blocking_reasons),
            "candidate_eligible": self.candidate_eligible,
            "shadow_eligible": self.shadow_eligible,
            "deploy_eligible": self.deploy_eligible,
            "model_fingerprint": self.model_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint.as_dict(),
        }


def _replay_leakage(evidence: LocalEvidence) -> tuple[str, ...]:
    audit = evidence.replay_anchor_audit
    if audit is None:
        return ()
    evaluation_sessions = (
        set(evidence.sessions.calibration_session_ids)
        | set(evidence.sessions.holdout_session_ids)
        | set(evidence.sessions.ood_session_ids)
        | set(evidence.sessions.closed_loop_session_ids)
        | set(evidence.sessions.anchor_session_ids)
    )
    return tuple(sorted(
        set(audit.replay_source_session_ids) & evaluation_sessions
    ))


def assess_local_evidence(evidence: LocalEvidence) -> EvidenceAssessment:
    """Derive the highest defensible local grade, never a requested grade."""

    reasons: list[str] = list(evidence.integrity.failure_reasons)
    if not evidence.sessions.all_session_ids:
        reasons.append("physical_session_ids_missing")
    if evidence.model_fingerprint is None:
        reasons.append("model_fingerprint_missing")

    if reasons:
        uncapped = EvidenceGrade.W0
    else:
        uncapped = EvidenceGrade.W1
        if evidence.checks.historical_or_discovery:
            reasons.append("historical_or_discovery_evidence")
        elif len(evidence.sessions.all_session_ids) == 1:
            reasons.append("single_physical_session")
        else:
            w2_missing: list[str] = []
            if not evidence.sessions.fit_session_ids:
                w2_missing.append("fit_sessions_missing")
            if not evidence.sessions.calibration_session_ids:
                w2_missing.append("calibration_sessions_missing")
            if not evidence.checks.joint_tube_frozen:
                w2_missing.append("joint_tube_not_frozen")
            if w2_missing:
                reasons.extend(w2_missing)
            else:
                uncapped = EvidenceGrade.W2
                s1_missing: list[str] = []
                if not evidence.sessions.holdout_session_ids:
                    s1_missing.append("untouched_holdout_sessions_missing")
                if not evidence.checks.holdout_untouched:
                    s1_missing.append("holdout_not_untouched")
                if not evidence.checks.holdout_coverage_passed:
                    s1_missing.append("holdout_coverage_failed")
                if not evidence.sessions.ood_session_ids:
                    s1_missing.append("ood_sessions_missing")
                if not evidence.checks.ood_abstention_passed:
                    s1_missing.append("ood_abstention_failed")
                if s1_missing:
                    reasons.extend(s1_missing)
                else:
                    uncapped = EvidenceGrade.S1
                    s2_missing: list[str] = []
                    if not evidence.sessions.closed_loop_session_ids:
                        s2_missing.append("closed_loop_sessions_missing")
                    if not evidence.checks.paired_closed_loop_passed:
                        s2_missing.append("paired_closed_loop_failed")
                    if not evidence.sessions.anchor_session_ids:
                        s2_missing.append("anchor_sessions_missing")
                    if not evidence.checks.anchor_cohort_passed:
                        s2_missing.append("anchor_cohort_failed")
                    if evidence.replay_anchor_audit is None:
                        s2_missing.append("replay_anchor_audit_missing")
                    elif not evidence.replay_anchor_audit.passed:
                        s2_missing.extend(
                            f"replay_anchor_audit:{reason}"
                            for reason in evidence.replay_anchor_audit.reasons
                        )
                    replay_leakage = _replay_leakage(evidence)
                    if replay_leakage:
                        s2_missing.append(
                            "replay_evaluation_session_leakage"
                            f"({','.join(replay_leakage)})"
                        )
                    if s2_missing:
                        reasons.extend(s2_missing)
                    else:
                        uncapped = EvidenceGrade.S2

    grade = uncapped
    for failure in evidence.failures:
        if failure.maximum_grade.rank < grade.rank:
            grade = failure.maximum_grade
            reasons.append(
                f"discovered_failure:{failure.failure_id}:"
                f"cap={failure.maximum_grade.value}:{failure.reason}"
            )

    unique_reasons = tuple(dict.fromkeys(reasons))
    return EvidenceAssessment(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        key=evidence.key,
        uncapped_grade=uncapped,
        grade=grade,
        blocking_reasons=unique_reasons,
        candidate_eligible=grade_at_least(grade, EvidenceGrade.W2),
        shadow_eligible=grade_at_least(grade, EvidenceGrade.S1),
        deploy_eligible=grade is EvidenceGrade.S2,
        model_fingerprint=evidence.model_fingerprint,
        evidence_fingerprint=local_evidence_fingerprint(evidence),
    )


def assess_evidence_catalog(
    evidence_items: Iterable[LocalEvidence],
) -> tuple[EvidenceAssessment, ...]:
    """Assess keys independently and reject duplicate local identities."""

    assessments = tuple(
        sorted(
            (assess_local_evidence(item) for item in evidence_items),
            key=lambda assessment: assessment.key,
        )
    )
    keys = [assessment.key for assessment in assessments]
    if len(keys) != len(set(keys)):
        raise EvidenceError("evidence catalog contains duplicate EvidenceKey")
    return assessments


class PromotionAction(str, Enum):
    ADD = "add"
    PROMOTE = "promote"
    RETAIN = "retain"
    DOWNGRADE = "downgrade"
    WITHDRAW = "withdraw"


@dataclass(frozen=True)
class PromotionDecision:
    key: EvidenceKey
    action: PromotionAction
    previous_grade: EvidenceGrade | None
    candidate_grade: EvidenceGrade | None
    effective_grade: EvidenceGrade | None
    support_active: bool
    deploy_eligible: bool
    reason: str
    candidate_fingerprint: VersionedFingerprint | None


def decide_promotion(
    previous: EvidenceAssessment | None,
    candidate: EvidenceAssessment | None,
) -> PromotionDecision:
    """Plan one local catalog update.

    Lower grades and removal are accepted safety contractions, not errors.  An
    upward move is possible only because ``candidate.grade`` was derived from
    all prerequisite gates by :func:`assess_local_evidence`.
    """

    if previous is None and candidate is None:
        raise EvidencePromotionError("promotion requires previous or candidate evidence")
    if previous is not None and candidate is not None and previous.key != candidate.key:
        raise EvidencePromotionError("promotion cannot cross EvidenceKey boundaries")
    if candidate is not None:
        key = candidate.key
    elif previous is not None:
        key = previous.key
    else:  # Kept explicit so this path remains fail-closed if edited later.
        raise EvidencePromotionError("promotion key is unavailable")

    if candidate is None:
        return PromotionDecision(
            key=key,
            action=PromotionAction.WITHDRAW,
            previous_grade=previous.grade if previous is not None else None,
            candidate_grade=None,
            effective_grade=None,
            support_active=False,
            deploy_eligible=False,
            reason="key_removed_from_candidate_catalog; support_contracted",
            candidate_fingerprint=None,
        )

    if previous is None:
        action = PromotionAction.ADD
        reason = f"new_local_key_earned_{candidate.grade.value}"
    elif candidate.grade.rank > previous.grade.rank:
        action = PromotionAction.PROMOTE
        reason = (
            f"local_evidence_advanced_{previous.grade.value}_to_"
            f"{candidate.grade.value}"
        )
    elif candidate.grade.rank < previous.grade.rank:
        action = PromotionAction.DOWNGRADE
        reason = (
            f"new_evidence_reduced_{previous.grade.value}_to_"
            f"{candidate.grade.value}; support_may_contract"
        )
    else:
        action = PromotionAction.RETAIN
        reason = f"local_evidence_retained_{candidate.grade.value}"

    return PromotionDecision(
        key=key,
        action=action,
        previous_grade=previous.grade if previous is not None else None,
        candidate_grade=candidate.grade,
        effective_grade=candidate.grade,
        support_active=candidate.candidate_eligible,
        deploy_eligible=candidate.deploy_eligible,
        reason=reason,
        candidate_fingerprint=candidate.evidence_fingerprint,
    )


def assert_grade_authorized(
    assessment: EvidenceAssessment,
    requested_grade: EvidenceGrade | str,
) -> EvidenceGrade:
    """Fail closed when a catalog requests authority above earned evidence."""

    requested = EvidenceGrade.coerce(requested_grade)
    if requested.rank > assessment.grade.rank:
        raise EvidencePromotionError(
            f"{assessment.key.token} requested {requested.value} but earned "
            f"{assessment.grade.value}"
        )
    return requested


@dataclass(frozen=True)
class CatalogPromotionPlan:
    decisions: tuple[PromotionDecision, ...]
    candidate_catalog_fingerprint: VersionedFingerprint
    active_keys: tuple[EvidenceKey, ...]
    shadow_keys: tuple[EvidenceKey, ...]
    deploy_keys: tuple[EvidenceKey, ...]
    withdrawn_keys: tuple[EvidenceKey, ...]


def _assessment_map(
    assessments: Iterable[EvidenceAssessment],
    label: str,
) -> dict[EvidenceKey, EvidenceAssessment]:
    result: dict[EvidenceKey, EvidenceAssessment] = {}
    for assessment in assessments:
        if assessment.key in result:
            raise EvidencePromotionError(
                f"{label} catalog contains duplicate key {assessment.key.token}"
            )
        result[assessment.key] = assessment
    return result


def plan_catalog_update(
    previous: Iterable[EvidenceAssessment],
    candidate: Iterable[EvidenceAssessment],
    *,
    catalog_version: str,
    fingerprint_version: str = EVIDENCE_FINGERPRINT_VERSION,
) -> CatalogPromotionPlan:
    """Plan independent per-key promotion, downgrade, and support withdrawal."""

    _identifier(catalog_version, "catalog_version")
    previous_map = _assessment_map(previous, "previous")
    candidate_map = _assessment_map(candidate, "candidate")
    decisions = tuple(
        decide_promotion(previous_map.get(key), candidate_map.get(key))
        for key in sorted(set(previous_map) | set(candidate_map))
    )
    candidate_fingerprint = versioned_fingerprint(
        {
            "catalog_version": catalog_version,
            "entries": [
                {
                    "key": assessment.key,
                    "grade": assessment.grade,
                    "local_fingerprint": assessment.evidence_fingerprint,
                }
                for assessment in sorted(
                    candidate_map.values(), key=lambda item: item.key
                )
            ],
        },
        fingerprint_version=fingerprint_version,
    )
    return CatalogPromotionPlan(
        decisions=decisions,
        candidate_catalog_fingerprint=candidate_fingerprint,
        active_keys=tuple(sorted(
            key
            for key, assessment in candidate_map.items()
            if assessment.candidate_eligible
        )),
        shadow_keys=tuple(sorted(
            key
            for key, assessment in candidate_map.items()
            if assessment.shadow_eligible
        )),
        deploy_keys=tuple(sorted(
            key
            for key, assessment in candidate_map.items()
            if assessment.deploy_eligible
        )),
        withdrawn_keys=tuple(sorted(set(previous_map) - set(candidate_map))),
    )


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_FINGERPRINT_VERSION",
    "EvidenceError",
    "SessionIsolationError",
    "EvidencePromotionError",
    "EvidenceGrade",
    "grade_at_least",
    "EvidenceKey",
    "SessionOverlap",
    "SessionIsolationAudit",
    "EvidenceSessions",
    "audit_session_isolation",
    "assert_session_isolation",
    "EvidenceIntegrity",
    "RunIntegrityAudit",
    "EvidenceChecks",
    "ReplayAnchorEvidence",
    "ReplayAnchorAudit",
    "audit_replay_anchors",
    "EvidenceFailure",
    "LocalEvidence",
    "VersionedFingerprint",
    "versioned_fingerprint",
    "local_evidence_fingerprint",
    "EvidenceAssessment",
    "assess_local_evidence",
    "assess_evidence_catalog",
    "PromotionAction",
    "PromotionDecision",
    "decide_promotion",
    "assert_grade_authorized",
    "CatalogPromotionPlan",
    "plan_catalog_update",
]
