"""Physical-units helpers.

Standard, §8: tremor amplitude and finger precision must be *"specified in mm,
not px, and tied to panel density"* - a fixed pixel jitter means different
physical wobble on a 300 dpi phone vs a 500 dpi phone, which is a tell. Motor
synthesis works in millimeters and converts to device pixels through the live
panel density measured at connect time (``adb shell wm density`` / physical
size), so the same human motor model produces the correct physical motion on
any handset.
"""

from __future__ import annotations

from dataclasses import dataclass

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class PanelGeometry:
    """Live geometry of the target phone's display, measured at connect time."""

    width_px: int
    height_px: int
    dpi: float          # effective dots-per-inch (physical), NOT the density bucket

    @property
    def px_per_mm(self) -> float:
        return self.dpi / MM_PER_INCH

    def mm_to_px(self, mm: float) -> float:
        return mm * self.px_per_mm

    def px_to_mm(self, px: float) -> float:
        return px / self.px_per_mm

    def frac_to_px(self, xf: float, yf: float) -> tuple[float, float]:
        """Screen-fraction (0..1) to device pixels - for calibrated fallbacks."""
        return (xf * self.width_px, yf * self.height_px)

    def px_to_frac(self, x: float, y: float) -> tuple[float, float]:
        return (x / self.width_px, y / self.height_px)


# A conservative default used only before a real measurement is taken. Marked
# LIVE-VERIFY: the real dpi must come from the device (see hop.adb.measure_panel).
DEFAULT_PANEL = PanelGeometry(width_px=2400, height_px=1080, dpi=400.0)  # LIVE-VERIFY
