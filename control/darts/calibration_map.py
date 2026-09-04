"""Continuous, uncertainty-aware maps from ground action to takeoff state.

``ActionCalibrationMap`` preserves the legacy three-channel API. The strict
``ActionConditionedCalibrationMap`` predicts
``(dtheta0,domega0,dvx,dvz,dz,dOmega[1:4],dT)`` and requires an independently
calibrated residual tube. Neither map stores landing outcomes; those remain a
separate post-certificate qualifier.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class CalibrationPoint:
    angle_deg: float
    speed_mps: float
    action_kind: str
    dtheta0_deg: float
    domega_y0_dps: float
    dwheel_radps: float
    geometry_family: str = "tabletop_fillet"
    n: int = 0


@dataclass(frozen=True)
class CalibrationEstimate:
    available: bool
    in_domain: bool
    action_kind: str
    geometry_family: str
    support_distance: float | None
    neighbor_count: int
    dtheta0_deg: float | None
    domega_y0_dps: float | None
    dwheel_radps: float | None
    sigma_dtheta0_deg: float | None
    sigma_domega_y0_dps: float | None
    sigma_dwheel_radps: float | None
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


class ActionCalibrationMap:
    """Inverse-distance interpolation inside a declared calibration family."""

    def __init__(
        self,
        points: Iterable[CalibrationPoint],
        *,
        angle_scale_deg: float = 5.0,
        speed_scale_mps: float = 1.0,
        max_support_distance: float = 1.5,
        k_neighbors: int = 4,
    ):
        self.points = tuple(points)
        self.angle_scale_deg = float(angle_scale_deg)
        self.speed_scale_mps = float(speed_scale_mps)
        self.max_support_distance = float(max_support_distance)
        self.k_neighbors = int(k_neighbors)
        if self.angle_scale_deg <= 0.0 or self.speed_scale_mps <= 0.0:
            raise ValueError("calibration-map scales must be positive")
        if self.max_support_distance <= 0.0 or self.k_neighbors <= 0:
            raise ValueError("support distance and neighbor count must be positive")

    @classmethod
    def from_gate_entries(cls, entries: Iterable[dict]) -> "ActionCalibrationMap":
        points: list[CalibrationPoint] = []
        for entry in entries:
            common = {
                "angle_deg": float(entry["angle"]),
                "speed_mps": float(
                    entry.get("decision_speed_neutral_mps") or entry["v"]),
                "geometry_family": str(
                    entry.get("geometry_family", "tabletop_fillet")),
                "n": int(entry.get("n") or 0),
            }
            static_theta = entry.get("static_d_theta0", entry.get("d_theta0"))
            static_omega = entry.get("static_d_omega_y0", entry.get("d_omega_y0"))
            static_wheel = entry.get("static_d_wheel_w0", entry.get("d_wheel_w0"))
            if (
                static_theta is not None
                and static_omega is not None
                and static_wheel is not None
            ):
                points.append(CalibrationPoint(
                    action_kind="static_preset",
                    dtheta0_deg=float(static_theta),
                    domega_y0_dps=float(static_omega),
                    dwheel_radps=float(static_wheel),
                    **common,
                ))
            timed = entry.get("timed_effect") or {}
            if (
                timed.get("d_theta0") is not None
                and timed.get("d_omega_y0") is not None
                and timed.get("d_wheel_w0") is not None
            ):
                points.append(CalibrationPoint(
                    action_kind="timed_preset",
                    dtheta0_deg=float(timed["d_theta0"]),
                    domega_y0_dps=float(timed["d_omega_y0"]),
                    dwheel_radps=float(timed["d_wheel_w0"]),
                    **common,
                ))
        return cls(points)

    def _distance(self, point: CalibrationPoint, angle_deg: float, speed_mps: float) -> float:
        return math.hypot(
            (point.angle_deg - angle_deg) / self.angle_scale_deg,
            (point.speed_mps - speed_mps) / self.speed_scale_mps,
        )

    def query(
        self,
        angle_deg: float,
        speed_mps: float,
        *,
        action_kind: str,
        geometry_family: str = "tabletop_fillet",
        max_support_distance: float | None = None,
    ) -> CalibrationEstimate:
        candidates = [
            point for point in self.points
            if point.action_kind == action_kind
            and point.geometry_family == geometry_family
        ]
        if not candidates:
            return CalibrationEstimate(
                False, False, action_kind, geometry_family, None, 0,
                None, None, None, None, None, None,
                "no_matching_action_or_geometry_family",
            )
        ranked = sorted(
            ((self._distance(point, angle_deg, speed_mps), point)
             for point in candidates),
            key=lambda item: item[0],
        )
        support = ranked[0][0]
        limit = (
            self.max_support_distance
            if max_support_distance is None else float(max_support_distance)
        )
        in_box = (
            len(candidates) == 1
            or (
                min(p.angle_deg for p in candidates)
                <= angle_deg
                <= max(p.angle_deg for p in candidates)
                and min(p.speed_mps for p in candidates)
                <= speed_mps
                <= max(p.speed_mps for p in candidates)
            )
        )
        if support > limit or not in_box:
            return CalibrationEstimate(
                True, False, action_kind, geometry_family, support, 0,
                None, None, None, None, None, None,
                "outside_calibrated_support",
            )

        selected = ranked[: self.k_neighbors]
        if selected[0][0] <= 1e-12:
            selected = [selected[0]]
            weights = [1.0]
        else:
            weights = [1.0 / (distance * distance + 1e-9) for distance, _ in selected]
        weight_sum = sum(weights)

        def weighted(field: str) -> tuple[float, float]:
            values = [float(getattr(point, field)) for _, point in selected]
            mean = sum(w * value for w, value in zip(weights, values)) / weight_sum
            variance = sum(
                w * (value - mean) ** 2 for w, value in zip(weights, values)
            ) / weight_sum
            return mean, math.sqrt(max(0.0, variance))

        dtheta, s_theta = weighted("dtheta0_deg")
        domega, s_omega = weighted("domega_y0_dps")
        dwheel, s_wheel = weighted("dwheel_radps")
        return CalibrationEstimate(
            True, True, action_kind, geometry_family, support, len(selected),
            dtheta, domega, dwheel, s_theta, s_omega, s_wheel, "interpolated",
        )


@dataclass(frozen=True)
class ActionConditionedDelta:
    """Full takeoff-state shift in SI units for one ground action."""

    dtheta0_rad: float
    domega0_radps: float
    dvx_mps: float
    dvz_mps: float
    dz_m: float
    dwheels_radps: tuple[float, float, float, float]
    dT_s: float

    def __post_init__(self):
        if len(self.dwheels_radps) != 4:
            raise ValueError("dwheels_radps must contain exactly four wheels")
        values = (
            self.dtheta0_rad,
            self.domega0_radps,
            self.dvx_mps,
            self.dvz_mps,
            self.dz_m,
            *self.dwheels_radps,
            self.dT_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("action-conditioned deltas must be finite")

    @classmethod
    def zeros(cls) -> "ActionConditionedDelta":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0), 0.0)

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "ActionConditionedDelta":
        vector = tuple(float(value) for value in values)
        if len(vector) != 10:
            raise ValueError("action-conditioned delta array must have 10 values")
        return cls(
            dtheta0_rad=vector[0],
            domega0_radps=vector[1],
            dvx_mps=vector[2],
            dvz_mps=vector[3],
            dz_m=vector[4],
            dwheels_radps=(vector[5], vector[6], vector[7], vector[8]),
            dT_s=vector[9],
        )

    def as_array(self) -> tuple[float, ...]:
        return (
            self.dtheta0_rad,
            self.domega0_radps,
            self.dvx_mps,
            self.dvz_mps,
            self.dz_m,
            *self.dwheels_radps,
            self.dT_s,
        )

    @staticmethod
    def weighted_sum(
        others: tuple["ActionConditionedDelta", ...],
        weights: tuple[float, ...],
    ) -> "ActionConditionedDelta":
        if len(others) != len(weights) or not others:
            raise ValueError("delta/weight lengths must match and be non-empty")
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("delta weights must sum to a positive value")

        def mean(values: Iterable[float]) -> float:
            return sum(weight * value for weight, value in zip(weights, values)) / total

        return ActionConditionedDelta(
            dtheta0_rad=mean(delta.dtheta0_rad for delta in others),
            domega0_radps=mean(delta.domega0_radps for delta in others),
            dvx_mps=mean(delta.dvx_mps for delta in others),
            dvz_mps=mean(delta.dvz_mps for delta in others),
            dz_m=mean(delta.dz_m for delta in others),
            dwheels_radps=tuple(
                mean(delta.dwheels_radps[index] for delta in others)
                for index in range(4)
            ),
            dT_s=mean(delta.dT_s for delta in others),
        )


@dataclass(frozen=True)
class ActionConditionedCalibrationPoint:
    angle_deg: float
    speed_mps: float
    action_kind: str
    geometry_profile: str
    schedule_id: str
    delta: ActionConditionedDelta
    train_session_ids: tuple[str, ...]
    source_id: str

    def __post_init__(self):
        if not (
            math.isfinite(float(self.angle_deg))
            and math.isfinite(float(self.speed_mps))
        ):
            raise ValueError("calibration coordinates must be finite")
        if not all((
            self.action_kind.strip(),
            self.geometry_profile.strip(),
            self.schedule_id.strip(),
            self.source_id.strip(),
        )):
            raise ValueError(
                "action, geometry, schedule, and source IDs must be non-empty")
        if not self.train_session_ids:
            raise ValueError("calibration point requires train_session_ids")


@dataclass(frozen=True)
class FrozenResidualTube:
    """Independent residual bound for one action/geometry/schedule key."""

    residual_set_id: str
    action_kind: str
    geometry_profile: str
    schedule_id: str
    radius: ActionConditionedDelta
    fit_session_ids: tuple[str, ...]
    calibration_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    method: str
    target_coverage: float
    source_sha256: str

    def __post_init__(self):
        fields = (
            self.residual_set_id,
            self.action_kind,
            self.geometry_profile,
            self.schedule_id,
            self.method,
        )
        if not all(value.strip() for value in fields):
            raise ValueError("residual tube IDs and method must be non-empty")
        radius_values = (
            self.radius.dtheta0_rad,
            self.radius.domega0_radps,
            self.radius.dvx_mps,
            self.radius.dvz_mps,
            self.radius.dz_m,
            *self.radius.dwheels_radps,
            self.radius.dT_s,
        )
        if any(value < 0.0 for value in radius_values):
            raise ValueError("residual radii must be nonnegative")
        if self.method not in ("split_conformal", "tolerance_bound"):
            raise ValueError("residual method must be split_conformal or tolerance_bound")
        if not 0.0 < self.target_coverage < 1.0:
            raise ValueError("target_coverage must lie strictly between zero and one")
        digest = self.source_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("source_sha256 must be a 64-character hex digest")
        split_sets = (
            set(self.fit_session_ids),
            set(self.calibration_session_ids),
            set(self.holdout_session_ids),
        )
        if any(not split for split in split_sets):
            raise ValueError("fit/calibration/holdout sessions must be non-empty")
        if (
            split_sets[0] & split_sets[1]
            or split_sets[0] & split_sets[2]
            or split_sets[1] & split_sets[2]
        ):
            raise ValueError("residual-tube session splits must be disjoint")

    @property
    def key(self) -> tuple[str, str, str]:
        return self.action_kind, self.geometry_profile, self.schedule_id


@dataclass(frozen=True)
class ActionConditionedTube:
    nominal: ActionConditionedDelta
    radius: ActionConditionedDelta
    residual_set_id: str


@dataclass(frozen=True)
class ActionConditionedEstimate:
    available: bool
    in_domain: bool
    reason: str
    action_kind: str
    geometry_profile: str
    schedule_id: str
    support_distance: float | None
    neighbor_count: int
    tube: ActionConditionedTube | None
    train_session_ids: tuple[str, ...]
    calibration_session_ids: tuple[str, ...]
    holdout_session_ids: tuple[str, ...]
    backend: str = "table_v1"
    evidence_grade: str | None = None
    ood_reasons: tuple[str, ...] = ()
    model_fingerprint: str | None = None
    epistemic_score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class ActionConditionedCalibrationMap:
    """Strict full-state predictor with frozen, independently calibrated tubes."""

    def __init__(
        self,
        points: Iterable[ActionConditionedCalibrationPoint],
        residual_tubes: Iterable[FrozenResidualTube],
        *,
        angle_scale_deg: float = 5.0,
        speed_scale_mps: float = 1.0,
        max_support_distance: float = 1.5,
        k_neighbors: int = 4,
    ):
        self.points = tuple(points)
        tubes = tuple(residual_tubes)
        self.residual_tubes = {tube.key: tube for tube in tubes}
        if len(self.residual_tubes) != len(tubes):
            raise ValueError("duplicate residual-tube key")
        self.angle_scale_deg = float(angle_scale_deg)
        self.speed_scale_mps = float(speed_scale_mps)
        self.max_support_distance = float(max_support_distance)
        self.k_neighbors = int(k_neighbors)
        if self.angle_scale_deg <= 0.0 or self.speed_scale_mps <= 0.0:
            raise ValueError("calibration-map scales must be positive")
        if self.max_support_distance <= 0.0 or self.k_neighbors <= 0:
            raise ValueError("support distance and neighbor count must be positive")

    def _distance(
        self,
        point: ActionConditionedCalibrationPoint,
        angle_deg: float,
        speed_mps: float,
    ) -> float:
        return math.hypot(
            (point.angle_deg - angle_deg) / self.angle_scale_deg,
            (point.speed_mps - speed_mps) / self.speed_scale_mps,
        )

    @staticmethod
    def _unavailable(
        *,
        available: bool,
        reason: str,
        action_kind: str,
        geometry_profile: str,
        schedule_id: str,
        support_distance: float | None = None,
    ) -> ActionConditionedEstimate:
        return ActionConditionedEstimate(
            available=available,
            in_domain=False,
            reason=reason,
            action_kind=action_kind,
            geometry_profile=geometry_profile,
            schedule_id=schedule_id,
            support_distance=support_distance,
            neighbor_count=0,
            tube=None,
            train_session_ids=(),
            calibration_session_ids=(),
            holdout_session_ids=(),
        )

    def query(
        self,
        angle_deg: float,
        speed_mps: float,
        *,
        action_kind: str,
        geometry_profile: str,
        schedule_id: str,
        max_support_distance: float | None = None,
        context_features: Mapping[str, float] | None = None,
    ) -> ActionConditionedEstimate:
        if not (
            math.isfinite(float(angle_deg))
            and math.isfinite(float(speed_mps))
        ):
            return self._unavailable(
                available=False,
                reason="nonfinite_query",
                action_kind=action_kind,
                geometry_profile=geometry_profile,
                schedule_id=schedule_id,
            )
        key = (action_kind, geometry_profile, schedule_id)
        candidates = [
            point for point in self.points
            if (
                point.action_kind,
                point.geometry_profile,
                point.schedule_id,
            ) == key
        ]
        if not candidates:
            return self._unavailable(
                available=False,
                reason="no_matching_action_geometry_schedule",
                action_kind=action_kind,
                geometry_profile=geometry_profile,
                schedule_id=schedule_id,
            )
        tube = self.residual_tubes.get(key)
        if tube is None:
            return self._unavailable(
                available=True,
                reason="frozen_residual_tube_missing",
                action_kind=action_kind,
                geometry_profile=geometry_profile,
                schedule_id=schedule_id,
            )
        ranked = sorted(
            ((self._distance(point, angle_deg, speed_mps), point)
             for point in candidates),
            key=lambda item: item[0],
        )
        support = ranked[0][0]
        limit = (
            self.max_support_distance
            if max_support_distance is None
            else float(max_support_distance)
        )
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
        if support > limit or not in_box:
            return self._unavailable(
                available=True,
                reason="outside_calibrated_support",
                action_kind=action_kind,
                geometry_profile=geometry_profile,
                schedule_id=schedule_id,
                support_distance=support,
            )
        selected = ranked[: self.k_neighbors]
        if selected[0][0] <= 1e-12:
            selected = [selected[0]]
            weights = (1.0,)
        else:
            weights = tuple(
                1.0 / (distance * distance + 1e-9)
                for distance, _ in selected
            )
        points = tuple(point for _, point in selected)
        train_sessions = tuple(sorted({
            session_id
            for point in points
            for session_id in point.train_session_ids
        }))
        if not set(train_sessions).issubset(set(tube.fit_session_ids)):
            return self._unavailable(
                available=True,
                reason="predictor_residual_fit_session_mismatch",
                action_kind=action_kind,
                geometry_profile=geometry_profile,
                schedule_id=schedule_id,
                support_distance=support,
            )
        leaked = (
            set(train_sessions)
            & (set(tube.calibration_session_ids) | set(tube.holdout_session_ids))
        )
        if leaked:
            return self._unavailable(
                available=True,
                reason=f"predictor_session_leakage({sorted(leaked)})",
                action_kind=action_kind,
                geometry_profile=geometry_profile,
                schedule_id=schedule_id,
                support_distance=support,
            )
        nominal = ActionConditionedDelta.weighted_sum(
            tuple(point.delta for point in points), weights)
        return ActionConditionedEstimate(
            available=True,
            in_domain=True,
            reason="interpolated_with_frozen_residual_tube",
            action_kind=action_kind,
            geometry_profile=geometry_profile,
            schedule_id=schedule_id,
            support_distance=support,
            neighbor_count=len(points),
            tube=ActionConditionedTube(
                nominal=nominal,
                radius=tube.radius,
                residual_set_id=tube.residual_set_id,
            ),
            train_session_ids=train_sessions,
            calibration_session_ids=tuple(tube.calibration_session_ids),
            holdout_session_ids=tuple(tube.holdout_session_ids),
        )


def action_conditioned_calibration_fingerprint(
    points: Iterable[ActionConditionedCalibrationPoint],
    residual_tubes: Iterable[FrozenResidualTube],
    *,
    angle_scale_deg: float = 5.0,
    speed_scale_mps: float = 1.0,
    max_support_distance: float = 1.5,
    k_neighbors: int = 4,
) -> str:
    payload = {
        "points": [asdict(point) for point in points],
        "residual_tubes": [asdict(tube) for tube in residual_tubes],
        "config": {
            "angle_scale_deg": angle_scale_deg,
            "speed_scale_mps": speed_scale_mps,
            "max_support_distance": max_support_distance,
            "k_neighbors": k_neighbors,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_action_conditioned_calibration_map(
    payload: Mapping[str, object],
    *,
    expected_fingerprint: str | None = None,
) -> ActionConditionedCalibrationMap:
    """Load a strict CalMap artifact while rejecting schema/hash drift."""
    if payload.get("schema_version") != "darts-action-conditioned-calmap/1":
        raise ValueError("unsupported action-conditioned CalMap schema")
    raw_points = payload.get("points")
    raw_tubes = payload.get("residual_tubes")
    if not isinstance(raw_points, list) or not isinstance(raw_tubes, list):
        raise ValueError("CalMap points and residual_tubes must be arrays")
    points = []
    for raw in raw_points:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("delta"), Mapping):
            raise ValueError("invalid action-conditioned calibration point")
        values = dict(raw)
        values["delta"] = ActionConditionedDelta(**dict(values["delta"]))
        values["train_session_ids"] = tuple(values["train_session_ids"])
        points.append(ActionConditionedCalibrationPoint(**values))
    tubes = []
    for raw in raw_tubes:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("radius"), Mapping):
            raise ValueError("invalid frozen residual tube")
        values = dict(raw)
        values["radius"] = ActionConditionedDelta(**dict(values["radius"]))
        for field in (
            "fit_session_ids",
            "calibration_session_ids",
            "holdout_session_ids",
        ):
            values[field] = tuple(values[field])
        tubes.append(FrozenResidualTube(**values))
    config = payload.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError("CalMap config must be an object")
    kwargs = {
        "angle_scale_deg": float(config.get("angle_scale_deg", 5.0)),
        "speed_scale_mps": float(config.get("speed_scale_mps", 1.0)),
        "max_support_distance": float(
            config.get("max_support_distance", 1.5)),
        "k_neighbors": int(config.get("k_neighbors", 4)),
    }
    actual = action_conditioned_calibration_fingerprint(
        points, tubes, **kwargs)
    declared = payload.get("model_fingerprint")
    if declared is not None and declared != actual:
        raise ValueError(
            f"declared CalMap fingerprint mismatch: {declared} != {actual}")
    if expected_fingerprint is not None and expected_fingerprint != actual:
        raise ValueError(
            f"expected CalMap fingerprint mismatch: {expected_fingerprint} != {actual}")
    return ActionConditionedCalibrationMap(points, tubes, **kwargs)


__all__ = [
    "CalibrationPoint",
    "CalibrationEstimate",
    "ActionCalibrationMap",
    "ActionConditionedDelta",
    "ActionConditionedCalibrationPoint",
    "FrozenResidualTube",
    "ActionConditionedTube",
    "ActionConditionedEstimate",
    "ActionConditionedCalibrationMap",
    "action_conditioned_calibration_fingerprint",
    "load_action_conditioned_calibration_map",
]
