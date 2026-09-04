"""Simulator-specific touchdown capture model for DART-S certificates.

This module deliberately does not reuse the paper's task-outcome thresholds.
It stores a separately identified landing-state region whose provenance must
come from independent state-injection sessions. Until such a model is loaded,
certificate-first deployment must fail closed.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping

from control.darts.reachability import PitchTerminalRegion


CAPTURE_SCHEMA_VERSION = "darts-sim-landing-capture/1"


class LandingCaptureModelError(ValueError):
    pass


class UnfittedLandingCaptureModelError(LandingCaptureModelError):
    pass


@dataclass(frozen=True)
class LandingCaptureObservation:
    """State and resource variables at first landing contact."""

    pitch_relative_surface_rad: float
    pitch_rate_radps: float
    roll_rad: float
    roll_rate_radps: float
    normal_velocity_mps: float
    front_remaining_stroke_m: float
    rear_remaining_stroke_m: float
    impact_energy_per_mass_jpkg: float
    cfc10_peak_decel_mps2: float
    contact_order: str


@dataclass(frozen=True)
class SimulatorLandingCaptureRegion:
    """Frozen inner approximation of a simulator landing capture set."""

    model_id: str
    source_dataset_sha256: str
    fit_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    independent_state_injection: bool
    pitch_lo_rad: float
    pitch_hi_rad: float
    pitch_rate_lo_radps: float
    pitch_rate_hi_radps: float
    max_abs_roll_rad: float
    max_abs_roll_rate_radps: float
    normal_velocity_lo_mps: float
    normal_velocity_hi_mps: float
    min_front_remaining_stroke_m: float
    min_rear_remaining_stroke_m: float
    max_impact_energy_per_mass_jpkg: float
    max_cfc10_peak_decel_mps2: float
    allowed_contact_orders: tuple[str, ...]
    schema_version: str = CAPTURE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != CAPTURE_SCHEMA_VERSION:
            raise LandingCaptureModelError(
                f"unsupported capture schema {self.schema_version!r}")
        if not self.model_id.strip():
            raise LandingCaptureModelError("model_id is required")
        digest = self.source_dataset_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise LandingCaptureModelError(
                "source_dataset_sha256 must be a 64-character hexadecimal digest")
        fit = set(self.fit_session_ids)
        holdout = set(self.holdout_session_ids)
        if not fit or not holdout:
            raise LandingCaptureModelError(
                "fit and holdout physical-session IDs must both be non-empty")
        if fit & holdout:
            raise LandingCaptureModelError(
                f"fit/holdout session leakage: {sorted(fit & holdout)}")
        if not self.independent_state_injection:
            raise LandingCaptureModelError(
                "capture model must come from independent landing-state injection")
        finite_values = {
            "pitch_lo_rad": self.pitch_lo_rad,
            "pitch_hi_rad": self.pitch_hi_rad,
            "pitch_rate_lo_radps": self.pitch_rate_lo_radps,
            "pitch_rate_hi_radps": self.pitch_rate_hi_radps,
            "max_abs_roll_rad": self.max_abs_roll_rad,
            "max_abs_roll_rate_radps": self.max_abs_roll_rate_radps,
            "normal_velocity_lo_mps": self.normal_velocity_lo_mps,
            "normal_velocity_hi_mps": self.normal_velocity_hi_mps,
            "min_front_remaining_stroke_m": self.min_front_remaining_stroke_m,
            "min_rear_remaining_stroke_m": self.min_rear_remaining_stroke_m,
            "max_impact_energy_per_mass_jpkg":
                self.max_impact_energy_per_mass_jpkg,
            "max_cfc10_peak_decel_mps2": self.max_cfc10_peak_decel_mps2,
        }
        if not all(math.isfinite(float(value)) for value in finite_values.values()):
            raise LandingCaptureModelError("capture-region bounds must be finite")
        if self.pitch_lo_rad > self.pitch_hi_rad:
            raise LandingCaptureModelError("pitch bounds are reversed")
        if self.pitch_rate_lo_radps > self.pitch_rate_hi_radps:
            raise LandingCaptureModelError("pitch-rate bounds are reversed")
        if self.normal_velocity_lo_mps > self.normal_velocity_hi_mps:
            raise LandingCaptureModelError("normal-velocity bounds are reversed")
        nonnegative = (
            self.max_abs_roll_rad,
            self.max_abs_roll_rate_radps,
            self.min_front_remaining_stroke_m,
            self.min_rear_remaining_stroke_m,
            self.max_impact_energy_per_mass_jpkg,
            self.max_cfc10_peak_decel_mps2,
        )
        if any(value < 0.0 for value in nonnegative):
            raise LandingCaptureModelError(
                "roll, stroke, energy, and load bounds must be nonnegative")
        if not self.allowed_contact_orders:
            raise LandingCaptureModelError("allowed_contact_orders must be non-empty")

    def pitch_terminal_region(self) -> PitchTerminalRegion:
        """Project the independent capture set onto analytic pitch coordinates."""
        return PitchTerminalRegion(
            theta_lo=self.pitch_lo_rad,
            theta_hi=self.pitch_hi_rad,
            omega_lo=self.pitch_rate_lo_radps,
            omega_hi=self.pitch_rate_hi_radps,
        )

    def contains(self, observation: LandingCaptureObservation) -> bool:
        """Fail-closed membership in the full simulator capture region."""
        numeric = (
            observation.pitch_relative_surface_rad,
            observation.pitch_rate_radps,
            observation.roll_rad,
            observation.roll_rate_radps,
            observation.normal_velocity_mps,
            observation.front_remaining_stroke_m,
            observation.rear_remaining_stroke_m,
            observation.impact_energy_per_mass_jpkg,
            observation.cfc10_peak_decel_mps2,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            return False
        return (
            self.pitch_lo_rad
            <= observation.pitch_relative_surface_rad
            <= self.pitch_hi_rad
            and self.pitch_rate_lo_radps
            <= observation.pitch_rate_radps
            <= self.pitch_rate_hi_radps
            and abs(observation.roll_rad) <= self.max_abs_roll_rad
            and abs(observation.roll_rate_radps) <= self.max_abs_roll_rate_radps
            and self.normal_velocity_lo_mps
            <= observation.normal_velocity_mps
            <= self.normal_velocity_hi_mps
            and observation.front_remaining_stroke_m
            >= self.min_front_remaining_stroke_m
            and observation.rear_remaining_stroke_m
            >= self.min_rear_remaining_stroke_m
            and observation.impact_energy_per_mass_jpkg
            <= self.max_impact_energy_per_mass_jpkg
            and observation.cfc10_peak_decel_mps2
            <= self.max_cfc10_peak_decel_mps2
            and observation.contact_order in self.allowed_contact_orders
        )


@dataclass(frozen=True)
class LandingCaptureValidation:
    n: int
    predicted_inside: int
    observed_captured: int
    false_inclusion: int
    false_exclusion: int


@dataclass(frozen=True)
class CornerPrecontactInterval:
    """Robust last-airborne interval for one wheel corner."""

    wheel_name: str
    gap_lo_m: float
    gap_hi_m: float
    closing_speed_lo_mps: float
    closing_speed_hi_mps: float
    stroke_lo_m: float
    stroke_hi_m: float

    def __post_init__(self):
        values = (
            self.gap_lo_m,
            self.gap_hi_m,
            self.closing_speed_lo_mps,
            self.closing_speed_hi_mps,
            self.stroke_lo_m,
            self.stroke_hi_m,
        )
        if not self.wheel_name or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise LandingCaptureModelError(
                "corner precontact intervals require finite values and wheel name")
        if (
            self.gap_lo_m < 0.0
            or self.gap_lo_m > self.gap_hi_m
            or self.closing_speed_lo_mps < 0.0
            or self.closing_speed_lo_mps > self.closing_speed_hi_mps
            or self.stroke_lo_m <= 0.0
            or self.stroke_lo_m > self.stroke_hi_m
        ):
            raise LandingCaptureModelError(
                f"invalid precontact interval for {self.wheel_name}")


@dataclass(frozen=True)
class CaptureRiskBounds:
    risk_lower: float
    risk_upper: float
    demand_mean_lower: float
    demand_mean_upper: float
    demand_max_lower: float
    demand_max_upper: float
    contact_spread_min_s: float
    contact_spread_max_s: float


def _contact_time(gap_m: float, closing_speed_mps: float, gravity_mps2: float) -> float:
    return (
        math.sqrt(closing_speed_mps ** 2 + 2.0 * gravity_mps2 * gap_m)
        - closing_speed_mps
    ) / gravity_mps2


def capture_risk_bounds(
    corners: tuple[CornerPrecontactInterval, ...],
    *,
    alpha: float,
    gravity_mps2: float = 9.81,
    sync_scale_s: float = 0.05,
) -> CaptureRiskBounds:
    """Conservative interval bound for the precontact mechanical risk score."""
    if len(corners) != 4 or len({corner.wheel_name for corner in corners}) != 4:
        raise LandingCaptureModelError("capture risk requires four unique wheel corners")
    if not 0.0 <= alpha <= 1.0:
        raise LandingCaptureModelError("capture risk alpha must lie in [0,1]")
    if gravity_mps2 <= 0.0 or sync_scale_s <= 0.0:
        raise LandingCaptureModelError("gravity and sync scale must be positive")

    demand_lo = []
    demand_hi = []
    time_lo = []
    time_hi = []
    for corner in corners:
        demand_lo.append(
            (
                corner.closing_speed_lo_mps ** 2
                + 2.0 * gravity_mps2 * corner.gap_lo_m
            )
            / (2.0 * corner.stroke_hi_m)
        )
        demand_hi.append(
            (
                corner.closing_speed_hi_mps ** 2
                + 2.0 * gravity_mps2 * corner.gap_hi_m
            )
            / (2.0 * corner.stroke_lo_m)
        )
        time_lo.append(_contact_time(
            corner.gap_lo_m,
            corner.closing_speed_hi_mps,
            gravity_mps2,
        ))
        time_hi.append(_contact_time(
            corner.gap_hi_m,
            corner.closing_speed_lo_mps,
            gravity_mps2,
        ))

    mean_lo = sum(demand_lo) / 4.0
    mean_hi = sum(demand_hi) / 4.0
    max_lo = max(demand_lo)
    max_hi = max(demand_hi)
    spread_min = max(0.0, max(time_lo) - min(time_hi))
    spread_max = max(time_hi) - min(time_lo)
    sync_upper = math.exp(-((spread_min / sync_scale_s) ** 2))
    sync_lower = math.exp(-((spread_max / sync_scale_s) ** 2))
    heterogeneity_lower = max(0.0, max_lo - mean_hi)
    heterogeneity_upper = max(0.0, max_hi - mean_lo)
    return CaptureRiskBounds(
        risk_lower=(
            mean_lo * (1.0 + sync_lower)
            + alpha * heterogeneity_lower
        ),
        risk_upper=(
            mean_hi * (1.0 + sync_upper)
            + alpha * heterogeneity_upper
        ),
        demand_mean_lower=mean_lo,
        demand_mean_upper=mean_hi,
        demand_max_lower=max_lo,
        demand_max_upper=max_hi,
        contact_spread_min_s=spread_min,
        contact_spread_max_s=spread_max,
    )


@dataclass(frozen=True)
class LandingCaptureRiskModelV2:
    """Frozen simulator-specific precontact risk sublevel set."""

    model_id: str
    source_dataset_sha256: str
    fit_session_ids: tuple[str, ...]
    calibration_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    alpha: float
    threshold: float
    gravity_mps2: float = 9.81
    sync_scale_s: float = 0.05
    schema_version: str = "darts-sim-landing-capture-risk/2"

    def __post_init__(self):
        if self.schema_version != "darts-sim-landing-capture-risk/2":
            raise LandingCaptureModelError(
                f"unsupported capture risk schema {self.schema_version!r}")
        if not self.model_id or not math.isfinite(self.threshold):
            raise LandingCaptureModelError("capture risk model ID and threshold required")
        if not 0.0 <= self.alpha <= 1.0 or self.threshold <= 0.0:
            raise LandingCaptureModelError("capture risk alpha/threshold invalid")
        digest = self.source_dataset_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise LandingCaptureModelError(
                "capture risk source hash must be 64 lowercase hex")
        splits = (
            set(self.fit_session_ids),
            set(self.calibration_session_ids),
            set(self.holdout_session_ids),
        )
        if any(not split for split in splits):
            raise LandingCaptureModelError(
                "capture risk fit/calibration/holdout sessions required")
        if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
            raise LandingCaptureModelError("capture risk session leakage")

    def evaluate(
        self,
        corners: tuple[CornerPrecontactInterval, ...],
    ) -> tuple[bool, CaptureRiskBounds]:
        bounds = capture_risk_bounds(
            corners,
            alpha=self.alpha,
            gravity_mps2=self.gravity_mps2,
            sync_scale_s=self.sync_scale_s,
        )
        return bounds.risk_upper < self.threshold, bounds


def landing_capture_risk_fingerprint(model: LandingCaptureRiskModelV2) -> str:
    payload = json.dumps(
        asdict(model),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_landing_capture_risk_model_v2(
    payload: Mapping[str, object] | None,
    *,
    expected_fingerprint: str | None = None,
) -> LandingCaptureRiskModelV2:
    if payload is None:
        raise UnfittedLandingCaptureModelError(
            "landing capture risk v2 is not fitted")
    raw = payload.get("model", payload)
    if not isinstance(raw, Mapping):
        raise LandingCaptureModelError("capture risk model must be an object")
    values = dict(raw)
    for field in (
        "fit_session_ids",
        "calibration_session_ids",
        "holdout_session_ids",
    ):
        values[field] = tuple(values[field])
    try:
        model = LandingCaptureRiskModelV2(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise LandingCaptureModelError(
            f"invalid landing capture risk model: {exc}") from exc
    actual = landing_capture_risk_fingerprint(model)
    declared = payload.get("model_fingerprint")
    if declared is not None and declared != actual:
        raise LandingCaptureModelError(
            f"declared capture risk fingerprint mismatch: {declared} != {actual}")
    if expected_fingerprint is not None and expected_fingerprint != actual:
        raise LandingCaptureModelError(
            f"expected capture risk fingerprint mismatch: {expected_fingerprint} != {actual}")
    return model


def validate_capture_region(
    region: SimulatorLandingCaptureRegion,
    observations: tuple[LandingCaptureObservation, ...],
    captured: tuple[bool, ...],
) -> LandingCaptureValidation:
    """Evaluate a frozen region on an independent holdout; never refits bounds."""
    if len(observations) != len(captured):
        raise ValueError("observations and captured labels must have equal length")
    inside = tuple(region.contains(observation) for observation in observations)
    return LandingCaptureValidation(
        n=len(inside),
        predicted_inside=sum(inside),
        observed_captured=sum(bool(value) for value in captured),
        false_inclusion=sum(
            pred and not bool(label) for pred, label in zip(inside, captured)),
        false_exclusion=sum(
            (not pred) and bool(label) for pred, label in zip(inside, captured)),
    )


def landing_capture_fingerprint(region: SimulatorLandingCaptureRegion) -> str:
    payload = json.dumps(
        asdict(region),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_landing_capture_region(
    payload: Mapping[str, object] | None,
    *,
    expected_fingerprint: str | None = None,
) -> SimulatorLandingCaptureRegion:
    if payload is None:
        raise UnfittedLandingCaptureModelError(
            "independent simulator landing capture model is not fitted")
    region_payload = payload.get("region", payload)
    if not isinstance(region_payload, Mapping):
        raise LandingCaptureModelError("landing capture region must be an object")
    try:
        region = SimulatorLandingCaptureRegion(**dict(region_payload))
    except (TypeError, ValueError) as exc:
        raise LandingCaptureModelError(f"invalid landing capture model: {exc}") from exc
    actual = landing_capture_fingerprint(region)
    if expected_fingerprint is not None and actual != expected_fingerprint:
        raise LandingCaptureModelError(
            f"landing capture fingerprint mismatch: {actual} != "
            f"{expected_fingerprint}")
    return region


__all__ = [
    "CAPTURE_SCHEMA_VERSION",
    "LandingCaptureModelError",
    "UnfittedLandingCaptureModelError",
    "LandingCaptureObservation",
    "SimulatorLandingCaptureRegion",
    "LandingCaptureValidation",
    "CornerPrecontactInterval",
    "CaptureRiskBounds",
    "capture_risk_bounds",
    "LandingCaptureRiskModelV2",
    "validate_capture_region",
    "landing_capture_fingerprint",
    "landing_capture_risk_fingerprint",
    "load_landing_capture_region",
    "load_landing_capture_risk_model_v2",
]
