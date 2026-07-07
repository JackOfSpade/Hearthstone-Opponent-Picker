"""The per-sample touch stream - the shared currency between motor synthesis
(Layer 3) and the transport (Layer 1).

Standard, §2: *"the goal on Android is ... to emit a per-sample touch stream
`(t, x, y, pressure, size, major/minor, orientation, tip, pointer_id)` that
reproduces all of the above."* This module defines that sample and a small
gesture container. Motor synthesis fills these in; the UHID backend turns them
into HID reports; the adb fallback discards everything but ``(x, y)`` and
records that fidelity was lost.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TouchSample:
    """One digitizer report for one contact.

    Coordinates are in device pixels (float during synthesis; the transport
    rounds at emit time). ``pressure``/``size``/``major``/``minor`` are
    normalized 0..1 and scaled to the descriptor's logical maxima by the
    transport. ``orientation`` is radians. ``tip`` is the Tip-Switch: True while
    the finger is in contact, False on the explicit release sample.
    """

    t: float          # seconds since gesture start (monotonic within a gesture)
    x: float          # device px
    y: float          # device px
    pressure: float   # normalized 0..1; never exactly 0 while tip is True
    size: float       # normalized 0..1 contact size (ABS_MT_TOUCH_MAJOR proxy)
    major: float      # normalized 0..1 touch-major
    minor: float      # normalized 0..1 touch-minor
    orientation: float  # radians; rotates toward travel on swipes
    tip: bool         # Tip Switch (in contact)
    pointer_id: int = 0  # stable across ACTION_POINTER_DOWN/UP within a gesture


@dataclass
class Gesture:
    """An ordered per-sample stream for one or more pointers.

    A tap is a single-pointer Gesture whose samples ramp pressure/size up then
    release. A swipe adds motion between. Multi-touch gestures interleave
    samples from multiple ``pointer_id``s (each with its own down/up lifecycle).
    """

    kind: str                       # "tap" | "swipe" | "scroll" | "long_press" | "pinch"
    samples: list[TouchSample] = field(default_factory=list)
    # Provenance/coherence metadata consumed by Layer 4 (sensorimotor) and
    # Layer 6 (coherence gates): the nominal target and dynamics of the gesture.
    target: tuple[float, float] | None = None      # intended endpoint (px)
    release_velocity: tuple[float, float] | None = None  # px/s at release (for fling)
    fidelity: str = "full"          # "full" (UHID) | "degraded" (adb input)

    @property
    def duration(self) -> float:
        return self.samples[-1].t - self.samples[0].t if self.samples else 0.0

    @property
    def pointer_ids(self) -> list[int]:
        seen: list[int] = []
        for s in self.samples:
            if s.pointer_id not in seen:
                seen.append(s.pointer_id)
        return seen

    def endpoint(self) -> tuple[float, float]:
        """Last in-contact position (the effective tap/release point)."""
        for s in reversed(self.samples):
            if s.tip:
                return (s.x, s.y)
        return (self.samples[-1].x, self.samples[-1].y) if self.samples else (0.0, 0.0)
