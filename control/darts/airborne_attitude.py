"""Angle-PD airborne pitch inner loop, used as the PD comparison arm.

.. warning:: This is an angle-PD law, with torque proportional to the angle
   error, and it is *not* the DART law. DART tracks a saturated rate
   reference, ``omega_des = clip(K_theta * err, +/-omega_max)`` followed by
   ``dtau = K_d (omega_des - omega)``, and lives in the differential branch
   of ``scripts/_jump_geometry.py``. Angle-PD is precisely the overshoot
   counterexample the DART paper analyses: with wheels as momentum-bounded
   actuators, an angle error keeps commanding torque and the loop has to
   reverse to stop past the target. This module exists so that the PD arm of
   the A/B experiments is available; do not treat it as DART.

Physical basis
--------------
In the airborne phase, whole-vehicle throttle or brake exchanges angular
momentum with the wheels and reacts on the body as a pure pitch torque: with
the wheel spin axes along y, ``M = [0, -sum_i tau_i, 0]``, so pitch is the
only controllable axis. Throttle spins the wheels up and pitches the nose up;
braking or back-driving pitches it down. Roll and yaw would need per-wheel
differential torque and are out of scope here.

Control law
-----------
``u = theta_ddot_ref - K_d (omega_y - omega_y_ref) - K_p (theta - theta_ref)``
is the desired pitch acceleration in rad/s^2. It is resolved to nose-up
positive through ``nose_up_sign``, normalised by ``pitch_accel_max`` into
``[0, 1]``, and mapped to throttle for nose-up or brake for nose-down, which
are mutually exclusive.

Pure logic, no simulator or tensor dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PitchAttitudeController:
    """Pitch-only airborne inner loop mapping PD demand to throttle/brake.

    Parameters
    ----------
    kp, kd : pitch PD gains, in rad/s^2 per rad and per rad/s.
    pitch_accel_max : pitch-acceleration limit (rad/s^2) used to normalise
        ``u`` into ``[0, 1]``; see
        :func:`control.darts.reachability.pitch_accel_limit`.
    nose_up_sign : sign convention that makes nose-up positive, +1 or -1.
        Left as a parameter because the simulator's pitch sign is anchored by
        measurement rather than assumption.
    deadband : normalised magnitude below which nothing is commanded.
    """

    kp: float = 6.0
    kd: float = 3.0
    # Command normalisation only, not a physical limit; the measured limit is
    # 2.35 rad/s^2. Kept at the value the reported experiments ran with.
    pitch_accel_max: float = 1.85
    nose_up_sign: float = 1.0
    deadband: float = 1e-3

    def desired_pitch_accel(
        self,
        pitch: float,
        pitch_rate: float,
        pitch_ref: float,
        pitch_rate_ref: float = 0.0,
        pitch_accel_ref: float = 0.0,
    ) -> float:
        """Return the desired pitch acceleration ``u`` in rad/s^2."""
        return (
            pitch_accel_ref
            - self.kd * (pitch_rate - pitch_rate_ref)
            - self.kp * (pitch - pitch_ref)
        )

    def compute(
        self,
        pitch: float,
        pitch_rate: float,
        pitch_ref: float,
        pitch_rate_ref: float = 0.0,
        pitch_accel_ref: float = 0.0,
    ) -> tuple[float, float]:
        """Return a mutually exclusive ``(throttle, brake)`` pair in [0, 1].

        Nose-up demand goes to throttle and nose-down to brake, with
        magnitude ``|u| / pitch_accel_max`` clipped into ``[0, 1]``.
        """
        u = self.desired_pitch_accel(
            pitch, pitch_rate, pitch_ref, pitch_rate_ref, pitch_accel_ref
        )
        u_eff = self.nose_up_sign * u          # resolve to nose-up positive
        if self.pitch_accel_max <= 0:
            return (0.0, 0.0)
        cmd = max(-1.0, min(1.0, u_eff / self.pitch_accel_max))
        if abs(cmd) < self.deadband:
            return (0.0, 0.0)
        if cmd > 0.0:
            return (cmd, 0.0)        # nose-up demand -> throttle
        return (0.0, -cmd)           # nose-down demand -> brake


__all__ = ["PitchAttitudeController"]
