"""Conservative, explainable physics prior for the DART-S calibration map.

The prior predicts only an action-relative *takeoff-state* shift.  It never
contains a landing label, landing pass rate, capture result, or safety
endpoint.  The existing :class:`SuspPresetEffect` supports two takeoff
channels (pitch and pitch rate); every channel without such support is an
explicit zero prior.

All numerical arrays produced by this module are NumPy ``float64`` arrays.
There is deliberately no Torch dependency in the runtime path.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar, Iterable

import numpy as np

from control.darts.susp_precondition import SuspPresetEffect


# The frozen method specification enumerates these scalar outputs in this
# exact order.  Although prose elsewhere calls it "11-D", the enumerated
# vector has ten scalar components (5 body/translation + 4 wheels + time).
OUTPUT_ORDER = (
    "theta",
    "omega",
    "vx",
    "vz",
    "z",
    "wFL",
    "wFR",
    "wRL",
    "wRR",
    "T",
)
CALMAP_OUTPUT_ORDER = OUTPUT_ORDER
OUTPUT_DIM = len(OUTPUT_ORDER)
OUTPUT_UNITS = (
    "rad",
    "rad/s",
    "m/s",
    "m/s",
    "m",
    "rad/s",
    "rad/s",
    "rad/s",
    "rad/s",
    "s",
)
PHYSICS_PRIOR_SCHEMA_VERSION = "darts-calmap-physics-prior/1"

_ACTION_ALIASES = {
    "neutral": "neutral",
    "none": "neutral",
    "static": "static",
    "static_preset": "static",
    "timed": "timed",
    "timed_preset": "timed",
}


def _as_float64(value: object, *, name: str) -> np.float64:
    """Convert one scalar while rejecting booleans, NaN, and infinity."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar, not bool")
    try:
        converted = np.float64(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be convertible to float64") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class CalMapOutput:
    """Action-relative takeoff shift in the frozen output order."""

    theta: np.float64 = np.float64(0.0)
    omega: np.float64 = np.float64(0.0)
    vx: np.float64 = np.float64(0.0)
    vz: np.float64 = np.float64(0.0)
    z: np.float64 = np.float64(0.0)
    wFL: np.float64 = np.float64(0.0)
    wFR: np.float64 = np.float64(0.0)
    wRL: np.float64 = np.float64(0.0)
    wRR: np.float64 = np.float64(0.0)
    T: np.float64 = np.float64(0.0)

    order: ClassVar[tuple[str, ...]] = OUTPUT_ORDER
    units: ClassVar[tuple[str, ...]] = OUTPUT_UNITS

    def __post_init__(self) -> None:
        for name in OUTPUT_ORDER:
            object.__setattr__(
                self,
                name,
                _as_float64(getattr(self, name), name=name),
            )

    @classmethod
    def zeros(cls) -> "CalMapOutput":
        return cls()

    @classmethod
    def from_array(cls, values: Iterable[float]) -> "CalMapOutput":
        vector = np.asarray(tuple(values), dtype=np.float64)
        if vector.shape != (OUTPUT_DIM,):
            raise ValueError(
                f"CalMap output must have shape ({OUTPUT_DIM},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("CalMap output values must be finite")
        return cls(**dict(zip(OUTPUT_ORDER, vector, strict=True)))

    def as_array(self) -> np.ndarray:
        """Return a new, writable float64 vector in :data:`OUTPUT_ORDER`."""
        return np.asarray(
            [getattr(self, name) for name in OUTPUT_ORDER],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in OUTPUT_ORDER
        }


@dataclass(frozen=True)
class PhysicsPriorPrediction:
    """One explainable action-prior evaluation."""

    action_kind: str
    shift: CalMapOutput
    source: str
    assumptions: tuple[str, ...]
    prior_fingerprint: str

    def __post_init__(self) -> None:
        if self.action_kind not in ("neutral", "static", "timed"):
            raise ValueError("action_kind must be neutral, static, or timed")
        if not self.source:
            raise ValueError("physics-prior source must be non-empty")
        if not self.assumptions:
            raise ValueError("physics-prior assumptions must be explicit")
        if (
            len(self.prior_fingerprint) != 64
            or any(ch not in "0123456789abcdef"
                   for ch in self.prior_fingerprint)
        ):
            raise ValueError("prior_fingerprint must be a lowercase SHA-256")

    def as_array(self) -> np.ndarray:
        return self.shift.as_array()


def _canonical_action_kind(action_kind: str) -> str:
    if not isinstance(action_kind, str):
        raise TypeError("action_kind must be a string")
    try:
        return _ACTION_ALIASES[action_kind.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "action_kind must be neutral/static/timed "
            "(or the corresponding *_preset alias)") from exc


def _takeoff_effect_payload(
    effect: SuspPresetEffect | None,
) -> dict[str, float] | None:
    if effect is None:
        return None
    if not isinstance(effect, SuspPresetEffect):
        raise TypeError("effect must be SuspPresetEffect or None")
    theta = _as_float64(effect.dtheta0, name="effect.dtheta0")
    omega = _as_float64(effect.domega_y0, name="effect.domega_y0")
    # Validate but intentionally do not consume the landing-tolerance field.
    _as_float64(
        effect.dtheta_land_tol,
        name="effect.dtheta_land_tol",
    )
    return {
        "dtheta0_rad": float(theta),
        "domega_y0_radps": float(omega),
    }


@dataclass(frozen=True)
class CalMapPhysicsPrior:
    """Frozen prior configuration for neutral, static, and timed actions.

    ``SuspPresetEffect`` is the only empirical/physical effect consumed here.
    It supplies pitch and pitch-rate takeoff shifts.  Its landing-tolerance
    field is deliberately ignored, and unsupported state channels remain
    exactly zero.
    """

    static_effect: SuspPresetEffect | None = None
    timed_effect: SuspPresetEffect | None = None

    def __post_init__(self) -> None:
        _takeoff_effect_payload(self.static_effect)
        _takeoff_effect_payload(self.timed_effect)

    @property
    def fingerprint(self) -> str:
        return physics_prior_fingerprint(self)

    def predict(self, action_kind: str) -> PhysicsPriorPrediction:
        action = _canonical_action_kind(action_kind)
        effect = {
            "neutral": None,
            "static": self.static_effect,
            "timed": self.timed_effect,
        }[action]
        payload = _takeoff_effect_payload(effect)
        if payload is None:
            shift = CalMapOutput.zeros()
            source = "explicit_zero_prior"
            detail = (
                "neutral action has zero relative shift"
                if action == "neutral"
                else f"no validated SuspPresetEffect for {action} action"
            )
        else:
            shift = CalMapOutput(
                theta=payload["dtheta0_rad"],
                omega=payload["domega_y0_radps"],
            )
            source = "SuspPresetEffect.takeoff_channels"
            detail = (
                "theta/omega come from the supplied SuspPresetEffect; "
                "dtheta_land_tol is not a takeoff output"
            )
        return PhysicsPriorPrediction(
            action_kind=action,
            shift=shift,
            source=source,
            assumptions=(
                detail,
                "vx/vz/z/per-wheel/T are explicit zero priors",
                "no landing outcome or safety endpoint is represented",
            ),
            prior_fingerprint=self.fingerprint,
        )

    def as_dict(self) -> dict[str, object]:
        """Return the semantic configuration used by the fingerprint."""
        return _fingerprint_payload(self)


def _fingerprint_payload(prior: CalMapPhysicsPrior) -> dict[str, object]:
    return {
        "schema_version": PHYSICS_PRIOR_SCHEMA_VERSION,
        "output_order": list(OUTPUT_ORDER),
        "output_units": list(OUTPUT_UNITS),
        "semantics": {
            "target": "action_relative_takeoff_shift",
            "unsupported_channels": "explicit_zero",
            "landing_outcomes": "excluded",
            "susp_preset_fields_used": ["dtheta0", "domega_y0"],
        },
        "static_effect": _takeoff_effect_payload(prior.static_effect),
        "timed_effect": _takeoff_effect_payload(prior.timed_effect),
    }


def physics_prior_fingerprint(prior: CalMapPhysicsPrior) -> str:
    """Return a deterministic SHA-256 over prior semantics and parameters."""
    if not isinstance(prior, CalMapPhysicsPrior):
        raise TypeError("prior must be CalMapPhysicsPrior")
    encoded = json.dumps(
        _fingerprint_payload(prior),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def predict_physics_prior(
    action_kind: str,
    effect: SuspPresetEffect | None = None,
) -> PhysicsPriorPrediction:
    """Convenience entry point for one action and one optional effect."""
    action = _canonical_action_kind(action_kind)
    prior = CalMapPhysicsPrior(
        static_effect=effect if action == "static" else None,
        timed_effect=effect if action == "timed" else None,
    )
    return prior.predict(action)


# Descriptive aliases retained for call sites that prefer a prior-specific
# name over the shared CalMap output name.
PhysicsPriorOutput = CalMapOutput
PhysicsPrior = CalMapPhysicsPrior


__all__ = [
    "OUTPUT_ORDER",
    "CALMAP_OUTPUT_ORDER",
    "OUTPUT_DIM",
    "OUTPUT_UNITS",
    "PHYSICS_PRIOR_SCHEMA_VERSION",
    "CalMapOutput",
    "PhysicsPriorOutput",
    "PhysicsPriorPrediction",
    "CalMapPhysicsPrior",
    "PhysicsPrior",
    "physics_prior_fingerprint",
    "predict_physics_prior",
]
