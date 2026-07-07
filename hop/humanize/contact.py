"""Layer 3 - contact-patch evolution.

Standard: *"Model the finger pad, not a point."* A real capacitive contact has
pressure that ramps→plateaus→decays, a size that co-evolves with pressure, an
orientation that rotates toward travel, and a centroid that micro-slips on
impact. Synthetic input that reports constant/zero pressure and a fixed circle
is *"physically impossible"* (§9 anti-pattern).

Critically, the standard also warns: **respect the device's pressure
semantics.** *"A fancy beta ramp on a device that reports binary pressure is
less realistic than a faithful flat signal."* So this model has three modes,
selected by the calibrated ``pressure_semantics`` config:

* ``ramp``      - the panel reports a real intensity: use the beta ramp.
* ``binary``    - the panel reports 1.0-while-down: report a faithful flat 1.0.
* ``amplitude`` - pressure is an amplitude proxy for contact size: tie them.

All outputs are normalized 0..1; the transport scales them to the HID
descriptor's logical maxima.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..config import ContactConfig


@dataclass(frozen=True)
class ContactChannels:
    pressure: float
    size: float
    major: float
    minor: float
    orientation: float


class ContactModel:
    """Produces the contact channels for a sample at gesture-progress ``u``."""

    def __init__(self, cfg: ContactConfig):
        self.cfg = cfg
        # location of the beta ramp's peak, for normalization
        a, b = cfg.pressure_alpha, cfg.pressure_beta
        self._u_star = a / (a + b) if (a + b) > 0 else 0.5
        self._raw_peak = self._raw_beta(self._u_star) or 1.0

    def _raw_beta(self, u: float) -> float:
        u = min(max(u, 1e-6), 1 - 1e-6)
        return (u ** self.cfg.pressure_alpha) * ((1 - u) ** self.cfg.pressure_beta)

    def pressure(self, u: float) -> float:
        """Normalized pressure at progress ``u`` in [0, 1], honoring semantics.

        Never returns 0 while in contact: the floor approximates the panel's
        detection threshold (a capacitive digitizer only reports contact above
        it). The explicit release sample is handled by the motor layer, which
        sets ``tip=False`` and pressure 0 there.
        """
        sem = self.cfg.pressure_semantics
        if sem == "binary":
            # Faithful flat signal - the honest choice for a binary panel.
            return 1.0
        norm = self._raw_beta(u) / self._raw_peak
        if sem == "amplitude":
            # Pressure IS the size proxy: a gentler curve tied to contact area.
            norm = norm ** 0.75
        return self.cfg.pressure_floor + (self.cfg.pressure_peak - self.cfg.pressure_floor) * norm

    def size(self, u: float) -> float:
        """Contact size co-evolves with pressure: grows on press, shrinks on
        release. Peaks slightly *after* pressure (the pad keeps spreading a hair
        as it settles)."""
        p = self.pressure(min(1.0, u + 0.05))
        base = self.cfg.contact_major_peak
        if self.cfg.pressure_semantics == "binary":
            return base  # flat, matching the pressure honesty
        # scale size by normalized pressure (0..1 within [floor,peak])
        span = max(1e-6, self.cfg.pressure_peak - self.cfg.pressure_floor)
        pnorm = (p - self.cfg.pressure_floor) / span
        return base * (0.55 + 0.45 * pnorm)

    def channels(self, u: float, travel_dir: float | None) -> ContactChannels:
        """All contact channels for progress ``u``.

        ``travel_dir`` (radians) is the instantaneous direction of motion for a
        swipe; on a tap it's None and orientation holds the grip bias. Touch-
        major/minor form an ellipse whose long axis rotates toward travel.
        """
        p = self.pressure(u)
        major = self.size(u)
        minor = major * self.cfg.contact_minor_ratio
        if travel_dir is None:
            orientation = self.cfg.orientation_bias_rad
        else:
            # rotate the pad toward travel, retaining a little grip bias
            orientation = 0.85 * travel_dir + 0.15 * self.cfg.orientation_bias_rad
        # normalize orientation to [-pi/2, pi/2] (touch-major axis is undirected)
        orientation = (orientation + math.pi / 2) % math.pi - math.pi / 2
        return ContactChannels(pressure=p, size=major, major=major, minor=minor, orientation=orientation)
