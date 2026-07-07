"""Layer 1 - transport authenticity.

Delivers the per-sample touch stream to the phone. Two backends behind one
interface (:mod:`hop.transport.base`):

* :mod:`hop.transport.uhid`      - preferred: a persistent virtual HID digitizer
  driven through the stock ``/system/bin/hid`` tool over ``/dev/uhid``. Events
  flow through the genuine kernel input pipeline carrying pressure/size/geometry,
  are not injection-flagged, and the device is registered ONCE per session.
* :mod:`hop.transport.adb_input` - explicit degraded fallback: ``adb shell
  input``. Functional, but pressure/size/geometry are lost; it records that
  fidelity dropped so Layer 6 can account for it.

Selection is via the ``device.touch_backend = auto|uhid|adb`` config knob.
"""

from .base import TouchBackend, make_backend

__all__ = ["TouchBackend", "make_backend"]
