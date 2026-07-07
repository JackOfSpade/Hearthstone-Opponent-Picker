"""Layer 3 - multi-touch and incidental contact helpers.

Standard (Rev 2): *"Real phone use includes two-thumb typing, pinch/zoom,
multi-finger scroll, resting-thumb artifacts ... Model these where the app
normally elicits them - do not sprinkle random extra touches, because their
frequency and shape depend on grip, device size, layout, and one- vs two-handed
use."*

The Hearthstone opponent-picker's hot path (tap Play, tap mulligan cards, tap
concede) is single-finger, one-handed, and the game does **not** elicit
pinch/two-thumb during it - so, per the standard's explicit warning, we do NOT
inject spurious multi-touch there. These helpers exist so that any interaction
that *does* warrant it (e.g. a pinch-zoom on the collection, or modeling an
occasional resting-thumb contact during a long hold) has a correct pointer-ID
lifecycle available rather than being hand-rolled. They compose the same
per-sample synthesis with proper ACTION_POINTER_DOWN/UP identity.
"""

from __future__ import annotations

import math
from random import Random

from ..config import MotorConfig
from ..geometry import PanelGeometry
from ..touchstream import Gesture, TouchSample
from .contact import ContactModel
from .motor import synth_swipe
from .state import HumanState


def synth_pinch(
    rng: Random,
    center: tuple[float, float],
    start_gap_px: float,
    end_gap_px: float,
    panel: PanelGeometry,
    cfg: MotorConfig,
    contact: ContactModel,
    state: HumanState,
    angle_rad: float | None = None,
) -> Gesture:
    """Two-finger pinch/zoom about ``center`` with two distinct pointer IDs.

    Both contacts follow their own minimum-jerk path (via :func:`synth_swipe`)
    from the start gap to the end gap along ``angle_rad``; their samples are
    merged on a common timeline so the transport emits a coherent two-pointer
    ACTION_POINTER_DOWN ... ACTION_POINTER_UP lifecycle.
    """
    ang = angle_rad if angle_rad is not None else rng.uniform(0, math.pi)
    ux, uy = math.cos(ang), math.sin(ang)

    def endpoint(gap: float, sign: int) -> tuple[float, float]:
        return (center[0] + sign * ux * gap / 2, center[1] + sign * uy * gap / 2)

    g0 = synth_swipe(rng, endpoint(start_gap_px, +1), endpoint(end_gap_px, +1),
                     panel, cfg, contact, state, kind="pinch", pointer_id=0)
    g1 = synth_swipe(rng, endpoint(start_gap_px, -1), endpoint(end_gap_px, -1),
                     panel, cfg, contact, state, kind="pinch", pointer_id=1)
    merged = sorted([*g0.samples, *g1.samples], key=lambda s: (s.t, s.pointer_id))
    return Gesture(kind="pinch", samples=merged, target=center)


def resting_thumb_sample(
    x: float, y: float, cfg: MotorConfig, contact: ContactModel, t: float, pointer_id: int = 9
) -> TouchSample:
    """A single low-pressure incidental contact (a resting thumb/palm edge).

    Used only where a grip posture would plausibly produce one; never sprinkled
    randomly (§ anti-pattern). Low, roughly constant pressure with a larger,
    flatter contact ellipse than a deliberate tap.
    """
    ch = contact.channels(0.5, travel_dir=None)
    return TouchSample(
        t=t, x=x, y=y,
        pressure=min(0.35, ch.pressure), size=min(0.9, ch.size * 1.6),
        major=min(0.9, ch.major * 1.6), minor=ch.minor * 1.6,
        orientation=ch.orientation, tip=True, pointer_id=pointer_id,
    )
