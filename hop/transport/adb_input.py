"""Degraded fallback transport: ``adb shell input``.

Standard: *"If /system/bin/hid is absent, raise a distinct 'UHID unavailable'
and fall back to the adb input transport - still functional, but pressure/size/
geometry are lost; record that fidelity dropped."* And §9: ``adb shell input``
is a §9 anti-pattern *as the primary transport* precisely because it is a
synthetic injection with no contact envelope. We keep it only as the explicit,
clearly-labeled degraded path, and stamp every gesture ``fidelity="degraded"``
so Layer 6 knows the contact channels were not delivered.

It still honors the gesture's *shape and timing* as far as ``input`` allows:
a tap uses ``input tap`` with the humanized endpoint and the dwell (via
``input swipe x y x y <dwell_ms>`` when a real down-time matters), and a swipe
uses ``input swipe`` with the synthesized duration. What is lost is per-sample
pressure/size/orientation and the historical sample structure - unavoidable
with this API.
"""

from __future__ import annotations

from ..geometry import PanelGeometry
from ..touchstream import Gesture
from .base import TouchBackend


class AdbInputBackend(TouchBackend):
    fidelity = "degraded"

    def __init__(self, adb, degraded_reason: str = "backend forced to adb"):
        self.adb = adb
        self.degraded_reason = degraded_reason
        self._panel: PanelGeometry | None = None

    def open(self, panel: PanelGeometry) -> None:
        self._panel = panel

    def emit(self, gesture: Gesture) -> None:
        gesture.fidelity = "degraded"
        pts = [(s.x, s.y) for s in gesture.samples if s.tip]
        if not pts:
            return
        if gesture.kind == "tap":
            x, y = gesture.endpoint()
            dwell_ms = int(round(gesture.duration * 1000)) or 60
            # a tap with a real down-time is a zero-length swipe of that duration
            self.adb.shell(f"input swipe {int(x)} {int(y)} {int(x)} {int(y)} {dwell_ms}")
        else:
            (x0, y0) = pts[0]
            (x1, y1) = gesture.endpoint()
            dur_ms = int(round(gesture.duration * 1000)) or 100
            self.adb.shell(f"input swipe {int(x0)} {int(y0)} {int(x1)} {int(y1)} {dur_ms}")

    def close(self) -> None:
        self._panel = None
