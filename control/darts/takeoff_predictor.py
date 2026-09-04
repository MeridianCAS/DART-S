"""Decision-state to neutral-takeoff predictor for DART-S strict_v2."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from control.darts.calibration_map import ActionConditionedDelta


NEUTRAL_PHYSICS_PRIOR_SCHEMA = "darts-neutral-takeoff-physics-prior/1"
_NEUTRAL_ACTION_KIND = "neutral"
_NEUTRAL_SCHEDULE_ID = "neutral"


def _canonical_mapping_fingerprint(
    payload: Mapping[str, Any],
    *,
    fingerprint_field: str,
) -> str:
    clean = dict(payload)
    clean.pop(fingerprint_field, None)
    return hashlib.sha256(json.dumps(
        clean,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionStateVector:
    state_id: str
    theta_rad: float
    omega_radps: float
    vx_mps: float
    vz_mps: float
    z_m: float
    wheels_forward_radps: tuple[float, float, float, float]

    def __post_init__(self):
        values = (
            self.theta_rad,
            self.omega_radps,
            self.vx_mps,
            self.vz_mps,
            self.z_m,
            *self.wheels_forward_radps,
        )
        if not self.state_id or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise ValueError("decision state must have an ID and finite values")


@dataclass(frozen=True)
class NeutralTransitionPoint:
    angle_deg: float
    speed_mps: float
    geometry_profile: str
    delta: ActionConditionedDelta
    flight_time_s: float
    train_session_ids: tuple[str, ...]
    source_id: str

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
            self.angle_deg, self.speed_mps, self.flight_time_s,
        )):
            raise ValueError("neutral transition point must be finite")
        if self.flight_time_s <= 0.0:
            raise ValueError("neutral flight time must be positive")
        if not (
            self.geometry_profile
            and self.train_session_ids
            and self.source_id
        ):
            raise ValueError("neutral transition provenance is required")


@dataclass(frozen=True)
class NeutralResidualTube:
    residual_set_id: str
    geometry_profile: str
    radius: ActionConditionedDelta
    flight_time_radius_s: float
    fit_session_ids: tuple[str, ...]
    calibration_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    method: str
    source_sha256: str

    def __post_init__(self):
        values = (
            self.radius.dtheta0_rad,
            self.radius.domega0_radps,
            self.radius.dvx_mps,
            self.radius.dvz_mps,
            self.radius.dz_m,
            *self.radius.dwheels_radps,
            self.radius.dT_s,
            self.flight_time_radius_s,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("neutral residual radii must be finite and nonnegative")
        splits = (
            set(self.fit_session_ids),
            set(self.calibration_session_ids),
            set(self.holdout_session_ids),
        )
        if any(not split for split in splits):
            raise ValueError("neutral fit/calibration/holdout splits are required")
        if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
            raise ValueError("neutral predictor session splits must be disjoint")
        if self.method not in ("split_conformal", "tolerance_bound"):
            raise ValueError("unsupported neutral residual method")
        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("neutral residual source hash must be 64 lowercase hex")


@dataclass(frozen=True)
class NeutralTakeoffPrediction:
    available: bool
    in_domain: bool
    reason: str
    state_id: str
    support_distance: float | None
    neighbor_count: int
    theta0_rad: float | None
    omega0_radps: float | None
    vx_mps: float | None
    vz_mps: float | None
    z_m: float | None
    wheels_forward_radps: tuple[float, float, float, float] | None
    T_s: float | None
    radius: ActionConditionedDelta | None
    flight_time_radius_s: float | None
    source_id: str | None
    residual_set_id: str | None
    train_session_ids: tuple[str, ...]
    calibration_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    backend: str = "table_v1"
    evidence_grade: str | None = None
    ood_reasons: tuple[str, ...] = ()
    model_fingerprint: str | None = None
    epistemic_score: float | None = None


class DecisionToTakeoffMap:
    def __init__(
        self,
        points: Iterable[NeutralTransitionPoint],
        residual_tubes: Iterable[NeutralResidualTube],
        *,
        angle_scale_deg: float = 5.0,
        speed_scale_mps: float = 1.0,
        max_support_distance: float = 1.5,
        k_neighbors: int = 4,
    ):
        self.points = tuple(points)
        tubes = tuple(residual_tubes)
        self.residual_tubes = {
            tube.geometry_profile: tube for tube in tubes
        }
        if len(self.residual_tubes) != len(tubes):
            raise ValueError("duplicate neutral residual geometry")
        self.angle_scale_deg = float(angle_scale_deg)
        self.speed_scale_mps = float(speed_scale_mps)
        self.max_support_distance = float(max_support_distance)
        self.k_neighbors = int(k_neighbors)
        if (
            self.angle_scale_deg <= 0.0
            or self.speed_scale_mps <= 0.0
            or self.max_support_distance <= 0.0
            or self.k_neighbors <= 0
        ):
            raise ValueError("neutral predictor scales and support must be positive")

    def _distance(
        self,
        point: NeutralTransitionPoint,
        angle_deg: float,
        speed_mps: float,
    ) -> float:
        return math.hypot(
            (point.angle_deg - angle_deg) / self.angle_scale_deg,
            (point.speed_mps - speed_mps) / self.speed_scale_mps,
        )

    def predict(
        self,
        state: DecisionStateVector,
        *,
        angle_deg: float,
        speed_mps: float,
        geometry_profile: str,
        decision_uncertainty: ActionConditionedDelta | None = None,
        context_features: Mapping[str, float] | None = None,
    ) -> NeutralTakeoffPrediction:
        del context_features  # v1 has no learned contextual feature schema.
        candidates = [
            point for point in self.points
            if point.geometry_profile == geometry_profile
        ]
        tube = self.residual_tubes.get(geometry_profile)
        if not candidates or tube is None:
            return NeutralTakeoffPrediction(
                False, False, "neutral_geometry_unsupported", state.state_id,
                None, 0, None, None, None, None, None, None, None, None, None,
                None, None, (), (), ())
        ranked = sorted(
            ((self._distance(point, angle_deg, speed_mps), point)
             for point in candidates),
            key=lambda item: item[0],
        )
        support = ranked[0][0]
        if len(candidates) == 1:
            in_box = support <= 1e-12
        else:
            in_box = (
                min(point.angle_deg for point in candidates)
                <= angle_deg
                <= max(point.angle_deg for point in candidates)
                and min(point.speed_mps for point in candidates)
                <= speed_mps
                <= max(point.speed_mps for point in candidates)
            )
        if support > self.max_support_distance or not in_box:
            return NeutralTakeoffPrediction(
                True, False, "neutral_outside_calibrated_support", state.state_id,
                support, 0, None, None, None, None, None, None, None, None, None,
                None, None, (), (), ())
        selected = ranked[:self.k_neighbors]
        if selected[0][0] <= 1e-12:
            selected = [selected[0]]
            weights = (1.0,)
        else:
            weights = tuple(
                1.0 / (distance * distance + 1e-9)
                for distance, _ in selected)
        selected_points = tuple(point for _, point in selected)
        train_sessions = tuple(sorted({
            session
            for point in selected_points
            for session in point.train_session_ids
        }))
        if not set(train_sessions).issubset(set(tube.fit_session_ids)):
            return NeutralTakeoffPrediction(
                True, False, "neutral_residual_fit_session_mismatch",
                state.state_id, support, 0, None, None, None, None, None, None,
                None, None, None, None, None, (), (), ())
        delta = ActionConditionedDelta.weighted_sum(
            tuple(point.delta for point in selected_points), weights)
        total_weight = sum(weights)
        flight_time = sum(
            weight * point.flight_time_s
            for weight, point in zip(weights, selected_points)
        ) / total_weight
        decision_radius = decision_uncertainty or ActionConditionedDelta.zeros()
        radius = ActionConditionedDelta(
            tube.radius.dtheta0_rad + decision_radius.dtheta0_rad,
            tube.radius.domega0_radps + decision_radius.domega0_radps,
            tube.radius.dvx_mps + decision_radius.dvx_mps,
            tube.radius.dvz_mps + decision_radius.dvz_mps,
            tube.radius.dz_m + decision_radius.dz_m,
            tuple(
                tube.radius.dwheels_radps[index]
                + decision_radius.dwheels_radps[index]
                for index in range(4)
            ),
            0.0,
        )
        return NeutralTakeoffPrediction(
            available=True,
            in_domain=True,
            reason="neutral_transition_with_frozen_residual_tube",
            state_id=state.state_id,
            support_distance=support,
            neighbor_count=len(selected_points),
            theta0_rad=state.theta_rad + delta.dtheta0_rad,
            omega0_radps=state.omega_radps + delta.domega0_radps,
            vx_mps=state.vx_mps + delta.dvx_mps,
            vz_mps=state.vz_mps + delta.dvz_mps,
            z_m=state.z_m + delta.dz_m,
            wheels_forward_radps=tuple(
                value + change for value, change in zip(
                    state.wheels_forward_radps, delta.dwheels_radps)
            ),
            T_s=flight_time,
            radius=radius,
            flight_time_radius_s=tube.flight_time_radius_s,
            source_id="+".join(point.source_id for point in selected_points),
            residual_set_id=tube.residual_set_id,
            train_session_ids=train_sessions,
            calibration_session_ids=tube.calibration_session_ids,
            holdout_session_ids=tube.holdout_session_ids,
        )


def neutral_predictor_fingerprint(
    points: Iterable[NeutralTransitionPoint],
    residual_tubes: Iterable[NeutralResidualTube],
    *,
    config: Mapping[str, object] | None = None,
) -> str:
    payload = {
        "points": [asdict(point) for point in points],
        "residual_tubes": [asdict(tube) for tube in residual_tubes],
        "config": dict(config or {}),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()


def load_decision_to_takeoff_map(
    payload: Mapping[str, object],
    *,
    expected_fingerprint: str | None = None,
) -> DecisionToTakeoffMap:
    schema_version = payload.get("schema_version")
    if schema_version != "darts-neutral-takeoff-predictor/1":
        raise ValueError("unsupported neutral takeoff predictor schema")
    raw_points = payload.get("points")
    raw_tubes = payload.get("residual_tubes")
    if not isinstance(raw_points, list) or not isinstance(raw_tubes, list):
        raise ValueError("neutral predictor arrays missing")
    points = []
    for raw in raw_points:
        values = dict(raw)
        values["delta"] = ActionConditionedDelta(**values["delta"])
        values["train_session_ids"] = tuple(values["train_session_ids"])
        points.append(NeutralTransitionPoint(**values))
    tubes = []
    for raw in raw_tubes:
        values = dict(raw)
        values["radius"] = ActionConditionedDelta(**values["radius"])
        for field in (
            "fit_session_ids", "calibration_session_ids", "holdout_session_ids",
        ):
            values[field] = tuple(values[field])
        tubes.append(NeutralResidualTube(**values))
    config = payload.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError("neutral predictor config must be an object")
    actual = neutral_predictor_fingerprint(points, tubes, config=config)
    if payload.get("model_fingerprint") not in (None, actual):
        raise ValueError("declared neutral predictor fingerprint mismatch")
    if expected_fingerprint is not None and expected_fingerprint != actual:
        raise ValueError("expected neutral predictor fingerprint mismatch")
    return DecisionToTakeoffMap(
        points,
        tubes,
        angle_scale_deg=float(config.get("angle_scale_deg", 5.0)),
        speed_scale_mps=float(config.get("speed_scale_mps", 1.0)),
        max_support_distance=float(config.get("max_support_distance", 1.5)),
        k_neighbors=int(config.get("k_neighbors", 4)),
    )


__all__ = [
    "DecisionStateVector",
    "NeutralTransitionPoint",
    "NeutralResidualTube",
    "NeutralTakeoffPrediction",
    "DecisionToTakeoffMap",
    "NEUTRAL_PHYSICS_PRIOR_SCHEMA",
    "neutral_predictor_fingerprint",
    "load_decision_to_takeoff_map",
]
