"""Transport interface + backend factory.

A :class:`TouchBackend` turns a synthesized :class:`~hop.touchstream.Gesture`
into real touch input on the phone. The engine holds exactly one backend for the
whole session (Layer 1's persistence rule), opens it once, streams every gesture
through it, and closes it at teardown.
"""

from __future__ import annotations

import abc

from ..geometry import PanelGeometry
from ..touchstream import Gesture


class TouchBackend(abc.ABC):
    """Abstract touch transport. Session-scoped: :meth:`open` once, many
    :meth:`emit`, :meth:`close` once."""

    #: "full" (UHID: pressure/size/geometry preserved) or "degraded" (adb input)
    fidelity: str = "full"

    @abc.abstractmethod
    def open(self, panel: PanelGeometry) -> None:
        """Bring the transport up for ``panel``. For UHID this registers the
        ONE persistent digitizer; it must not be re-created per gesture."""

    @abc.abstractmethod
    def emit(self, gesture: Gesture) -> None:
        """Deliver a gesture's per-sample stream to the device, respecting the
        samples' relative timestamps."""

    @abc.abstractmethod
    def close(self) -> None:
        """Tear the transport down (destroy the digitizer, only here)."""

    def __enter__(self) -> "TouchBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_backend(kind: str, adb, uhid_cfg=None) -> TouchBackend:
    """Create a backend for ``kind`` in {auto, uhid, adb}, given an Adb handle.

    ``auto`` prefers UHID and falls back to adb-input if ``/system/bin/hid`` is
    absent (probed here so the choice is explicit, per "graceful, explicit
    degradation"). The chosen backend records its fidelity for Layer 6.
    ``uhid_cfg`` is the :class:`~hop.config.UhidConfig` (required for uhid/auto).
    """
    from .adb_input import AdbInputBackend
    from .uhid import UhidBackend

    kind = kind.lower()
    if kind == "adb":
        return AdbInputBackend(adb)
    if kind == "uhid":
        if uhid_cfg is None:
            raise ValueError("uhid backend requires uhid_cfg")
        return UhidBackend(adb, uhid_cfg)
    if kind == "auto":
        if uhid_cfg is not None and UhidBackend.available(adb):
            return UhidBackend(adb, uhid_cfg)
        return AdbInputBackend(adb, degraded_reason="/system/bin/hid not present or no uhid_cfg")
    raise ValueError(f"Unknown touch_backend {kind!r}; choose auto|uhid|adb")
