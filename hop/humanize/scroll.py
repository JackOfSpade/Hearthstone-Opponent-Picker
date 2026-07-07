"""Layer 3 - scrolling governed by Android fling physics.

Standard (Rev 2): *"A scroll is not an arbitrary Bezier drag. Android's
OverScroller/VelocityTracker model a spline-based fling with a fixed
deceleration profile; an app can compare an injected swipe's post-release
velocity and deceleration against what its own physics would produce. Generate
the release velocity and let the content decelerate on the app's terms."*

The Hearthstone mulligan flow does not scroll, so this is not on the hot path -
but it is implemented for completeness and for realistic idle behavior (e.g.
browsing collection/quests between sessions), because a raw Bezier drag on a
scrollable surface is a §9 anti-pattern. We synthesize the *finger* swipe with a
human release velocity via :func:`hop.humanize.motor.synth_swipe` and let the
app's own OverScroller decelerate the content; we never fake the content motion.
"""

from __future__ import annotations

import math
from random import Random

from ..config import MotorConfig, ScrollConfig
from ..geometry import PanelGeometry
from ..touchstream import Gesture, TouchSample
from .contact import ContactModel
from .state import HumanState

GRAVITY = 9.80665
INCH_PER_METER = 39.37


def _physical_coeff(ppi: float) -> float:
    # Mirrors AOSP OverScroller: gravity * inch/m * ppi * tuning(0.84)
    return GRAVITY * INCH_PER_METER * ppi * 0.84


def fling_deceleration(velocity_px_s: float, ppi: float, friction: float) -> float:
    """Approximate constant-equivalent deceleration (px/s^2) for a fling of the
    given release velocity, from the OverScroller friction model.

    AOSP uses a spline (non-constant) deceleration; we collapse it to an
    energy-equivalent constant so callers can size finger travel/duration. Good
    enough to *produce* a plausible release velocity - which is the observable
    the app scores - without claiming to reproduce the spline exactly.
    """
    v = abs(velocity_px_s)
    if v < 1e-3:
        return 1.0
    coeff = friction * _physical_coeff(ppi)
    # distance a spline fling would travel, then decel = v^2 / (2 d)
    # AOSP getSplineFlingDistance ~ coeff * exp(a/(a-1) * decel), simplified:
    decel_log = math.log(0.35 * v / coeff) if coeff > 0 else 0.0
    dist = coeff * math.exp(1.2 * max(0.0, decel_log)) if decel_log > 0 else v * 0.25
    dist = max(1.0, dist)
    return v * v / (2.0 * dist)


def synth_fling(
    rng: Random,
    start: tuple[float, float],
    direction: tuple[float, float],
    velocity_px_s: float,
    panel: PanelGeometry,
    motor_cfg: MotorConfig,
    scroll_cfg: ScrollConfig,
    contact: ContactModel,
    state: HumanState,
) -> Gesture:
    """Produce the finger swipe of a fling with the requested release velocity.

    Unlike a drag-to-target (which decelerates to rest via minimum jerk), a
    fling *lifts off while still moving fast*, so the finger path here uses an
    accelerating profile: position ~ travel * u^p with p>1, so instantaneous
    velocity is still rising at release. Travel is sized so the final-sample
    velocity equals ``velocity_px_s`` along ``direction``; the app then
    decelerates the content on its own OverScroller terms.
    """
    dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm

    duration = max(0.06, min(0.22, 0.12 * math.exp(rng.gauss(0.0, 0.25))))
    p = 2.0
    # for x(u)=travel*u^p, v(1)=travel*p/duration; solve travel for target speed
    travel = velocity_px_s * duration / p
    travel_dir = math.atan2(uy, ux)

    dt = 1.0 / motor_cfg.report_rate_hz
    n = max(4, int(round(duration / dt)))
    tremor_px = panel.mm_to_px(motor_cfg.tremor_amplitude_mm) * state.tremor_scale()
    ox = oy = 0.0
    samples: list[TouchSample] = []
    for i in range(n + 1):
        u = i / n
        d = travel * (u ** p)
        ox = ox - motor_cfg.tremor_theta * ox * dt + tremor_px * math.sqrt(dt) * rng.gauss(0, 1)
        oy = oy - motor_cfg.tremor_theta * oy * dt + tremor_px * math.sqrt(dt) * rng.gauss(0, 1)
        # tremor perpendicular to travel, fades as speed rises
        perp = (-uy, ux)
        jitter = ox * (1.0 - u)
        x = start[0] + ux * d + perp[0] * jitter
        y = start[1] + uy * d + perp[1] * jitter
        ch = contact.channels(u, travel_dir=travel_dir)
        samples.append(TouchSample(
            t=i * dt, x=x, y=y, pressure=ch.pressure, size=ch.size,
            major=ch.major, minor=ch.minor, orientation=ch.orientation, tip=True,
        ))
    a, b = samples[-2], samples[-1]
    ddt = max(1e-4, b.t - a.t)
    rv = ((b.x - a.x) / ddt, (b.y - a.y) / ddt)
    samples.append(TouchSample(
        t=b.t + dt, x=b.x, y=b.y, pressure=0.0, size=0.0, major=0.0, minor=0.0,
        orientation=b.orientation, tip=False,
    ))
    return Gesture(kind="scroll", samples=samples, target=(b.x, b.y), release_velocity=rv)


def reading_pause(rng: Random, scroll_cfg: ScrollConfig, state: HumanState) -> float:
    """A content-dependent pause between flings (seconds), longer when tired."""
    base = scroll_cfg.reading_pause_s * (1.0 + 0.5 * state.fatigue)
    return max(0.2, base * math.exp(rng.gauss(0.0, 0.3)))
