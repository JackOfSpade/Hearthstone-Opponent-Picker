"""Layer 4 - sensorimotor coherence (the highest-leverage Rev 2 addition).

Standard: *"A human tapping a handheld phone transfers a mechanical impulse to
the chassis ... A bench-mounted, ADB-driven phone shows anomalously flat
accelerometer/gyroscope data during 'human' touches - and fusing one motion
channel with touch roughly halves bot-detection error."* So for every touch we
must produce a coherent inertial side effect, and Layer 6 must fail closed if
touch and motion don't cohere.

Two implementation paths, chosen by threat model and posture:

* **handheld** - the phone is actually held (by a consenting tester) or on an
  actuating rig, so the real sensors naturally show the impulse. No sensor
  mocking, nothing to detect at the sensor API. This is the faithful path.
* **desk_mounted** - the phone lies flat, driven purely over ADB. We CANNOT
  inject sensors on a stock unrooted phone without an on-device helper (which
  Layer 0 forbids) or elevated privileges (which can itself trip integrity
  gates). The honest posture is therefore to **declare desk_mounted and not
  claim handheld realism** - exactly what the standard permits.

This module (a) *generates* the predicted impulse/torque for a gesture (used to
drive an actuation rig and to document what a handheld run should show), and
(b) provides the *validation* Layer 6 calls. The generation is pure; the
validation is parameterized by the declared posture so a desk-mounted run is
coherent-by-declaration while a handheld run with flat IMU fails closed.

Confidence: the physics here is MEDIUM (approximate impulse model); that a real
held device produces *some* coherent impulse is HIGH.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ..config import SensorConfig
from ..geometry import PanelGeometry
from ..touchstream import Gesture


class Posture(str, Enum):
    HANDHELD = "handheld"
    DESK_MOUNTED = "desk_mounted"


class CoherenceVerdict(str, Enum):
    COHERENT = "coherent"                       # motion matched the touch
    INCOHERENT = "incoherent"                   # handheld run, but IMU was flat -> HALT
    DECLARED_STATIC = "declared_static"         # desk_mounted: flatness expected, not claiming realism
    UNVERIFIABLE = "unverifiable"               # handheld declared but no sensor stream to check


@dataclass(frozen=True)
class ImuEvent:
    """A predicted inertial side effect, device frame.

    ``accel`` is in g (x=right, y=up, z=out of screen); ``gyro`` is deg/s about
    the same axes; ``t`` is seconds from gesture start.
    """

    t: float
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]


@dataclass(frozen=True)
class ImuSignature:
    events: list[ImuEvent]

    def peak_accel_g(self) -> float:
        return max((math.sqrt(sum(a * a for a in e.accel)) for e in self.events), default=0.0)

    def peak_gyro_dps(self) -> float:
        return max((math.sqrt(sum(g * g for g in e.gyro)) for e in self.events), default=0.0)


class SensorimotorModel:
    """Predicts the coupled inertial signal for a gesture, and validates it."""

    def __init__(self, cfg: SensorConfig, panel: PanelGeometry, posture: Posture):
        self.cfg = cfg
        self.panel = panel
        self.posture = posture

    # ── generation ──────────────────────────────────────────────────────────

    def predict(self, gesture: Gesture) -> ImuSignature:
        """Predicted IMU signature for a gesture.

        Tap: a localized acceleration spike into the screen (-z) at contact,
        plus a torque whose axis comes from the tap's offset from the device
        center (an off-center tap tilts the phone). Swipe: a rotational torque
        about the in-plane axis perpendicular to travel, scaled by speed. All
        sampled at ``poll_rate_hz`` so there are no gaps in the stream.
        """
        if gesture.kind in ("swipe", "scroll", "pinch"):
            return self._predict_swipe(gesture)
        return self._predict_tap(gesture)

    def _predict_tap(self, gesture: Gesture) -> ImuSignature:
        ex, ey = gesture.endpoint()
        cx, cy = self.panel.width_px / 2, self.panel.height_px / 2
        # normalized offset from center in [-1,1]; drives the tilt torque axis
        off_x = (ex - cx) / (self.panel.width_px / 2)
        off_y = (ey - cy) / (self.panel.height_px / 2)

        # impulse centered at the contact-down time; short ~30 ms half-sine
        t0 = gesture.samples[0].t if gesture.samples else 0.0
        dur = 0.035
        events: list[ImuEvent] = []
        steps = max(2, int(dur * self.cfg.poll_rate_hz))
        for i in range(steps + 1):
            t = t0 + i / self.cfg.poll_rate_hz
            env = math.sin(math.pi * min(1.0, (t - t0) / dur))  # half-sine impulse
            az = -self.cfg.tap_impulse_g * env                  # push into screen
            # tilt: off-center tap rotates about the orthogonal in-plane axis
            gx = -off_y * self.cfg.swipe_torque_dps * 0.6 * env
            gy = off_x * self.cfg.swipe_torque_dps * 0.6 * env
            events.append(ImuEvent(t=t, accel=(0.0, 0.0, az), gyro=(gx, gy, 0.0)))
        return ImuSignature(events)

    def _predict_swipe(self, gesture: Gesture) -> ImuSignature:
        events: list[ImuEvent] = []
        dt = 1.0 / self.cfg.poll_rate_hz
        n = max(2, int(gesture.duration / dt))
        # travel direction from endpoints
        if len(gesture.samples) >= 2:
            sx, sy = gesture.samples[0].x, gesture.samples[0].y
            ex, ey = gesture.endpoint()
            ang = math.atan2(ey - sy, ex - sx)
        else:
            ang = 0.0
        for i in range(n + 1):
            t = i * dt
            env = math.sin(math.pi * (i / n)) if n else 0.0  # ramp up/down
            # torque axis perpendicular to travel, in-plane
            wx = math.cos(ang + math.pi / 2) * self.cfg.swipe_torque_dps * env
            wy = math.sin(ang + math.pi / 2) * self.cfg.swipe_torque_dps * env
            ax = math.cos(ang) * self.cfg.tap_impulse_g * 0.3 * env
            ay = math.sin(ang) * self.cfg.tap_impulse_g * 0.3 * env
            events.append(ImuEvent(t=t, accel=(ax, ay, 0.0), gyro=(wx, wy, 0.0)))
        return ImuSignature(events)

    # ── validation (Layer 6 calls this) ─────────────────────────────────────

    def validate(
        self, gesture: Gesture, observed: ImuSignature | None
    ) -> CoherenceVerdict:
        """Return the coherence verdict for this gesture under the posture.

        * desk_mounted -> :attr:`DECLARED_STATIC` (flatness is expected; we do
          not pretend handheld realism, so this is not a failure).
        * handheld + no observed stream -> :attr:`UNVERIFIABLE` (the rig/tester
          provides real motion, but we have no host-side sensor read to confirm).
        * handheld + observed -> compare peaks: a flat observed signal during a
          real touch is :attr:`INCOHERENT` (Layer 6 HALTs); a matching impulse
          is :attr:`COHERENT`.
        """
        if self.posture == Posture.DESK_MOUNTED:
            return CoherenceVerdict.DECLARED_STATIC
        if observed is None:
            return CoherenceVerdict.UNVERIFIABLE
        predicted = self.predict(gesture)
        exp_a = predicted.peak_accel_g()
        obs_a = observed.peak_accel_g()
        obs_g = observed.peak_gyro_dps()
        # a real handheld touch is never perfectly flat; require a floor of motion
        if obs_a < 0.25 * max(exp_a, 1e-3) and obs_g < 0.25 * self.cfg.swipe_torque_dps:
            return CoherenceVerdict.INCOHERENT
        return CoherenceVerdict.COHERENT
