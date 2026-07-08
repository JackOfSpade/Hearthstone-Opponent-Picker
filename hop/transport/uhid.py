"""Preferred transport: a persistent virtual HID digitizer via ``/system/bin/hid``.

Standard, Layer 1 (the load-bearing Rev 2 fix): register **one** digitizer for
the whole session and stream every gesture through it; destroy it only at
teardown. Per-gesture register/destroy is a §9 anti-pattern - UHID destruction
is a *disconnect* observable via ``InputManager.InputDeviceListener``, and a
built-in panel never disconnects between taps.

Streaming approach ("resident host-side writer"): the AOSP ``hid`` tool reads a
stream of commands with a lazy JSON pull-parser and applies each as it arrives,
blocking for more when the stream is quiet. We keep **one** ``hid`` process
alive reading its **stdin** (fed by our long-lived ``adb shell`` pipe), send the
``register`` command once at session start, append ``report``/``delay`` commands
as gestures arrive, and close stdin at teardown (EOF -> ``hid`` exits -> device
destroyed). The device stays registered for exactly as long as we hold stdin
open - no add/remove churn.

On-device findings (Pixel 7a, Android 17 / SDK 37) that shaped this design:

* **No FIFO.** The original plan held a named pipe open on the device, but
  ``mkfifo``/``mknod`` in ``/data/local/tmp`` is SELinux-denied for the shell
  domain (``avc: denied { create } ... tclass=fifo_file``). stdin comes from
  ``adbd`` (an allowed context), so streaming to ``hid -`` sidesteps the denial
  entirely. Shell *is* in the ``uhid`` group, so ``/dev/uhid`` itself is
  reachable - the FIFO was the only blocker.
* **Object stream, not a JSON array.** This ``hid`` reads a *sequence of bare
  JSON objects* (``{...}\n{...}\n...``); handing it a ``[ ..., ... ]`` array
  fails with ``Expected BEGIN_OBJECT but was BEGIN_ARRAY``. We therefore emit
  one object per line with no enclosing brackets or commas.

Confirmed working: the descriptor enumerates as a 1080x2400 ``TOUCHSCREEN``
(``InputReader: Device added ... sources=TOUCHSCREEN``) and reports carry
pressure + contact size through the real kernel pipeline (pointer-location
overlay showed ``Prs``/``Size`` non-zero at the target coordinate).

Caveat still marked because it can only be confirmed on-device:
* Panel-matched identity (vid/pid/name) is generic by default - clone your real
  panel's values into ``[uhid]`` (``hop doctor`` prints them).
"""

from __future__ import annotations

import json
import time

from ..config import UhidConfig
from ..geometry import PanelGeometry
from ..orientation import display_to_native
from ..touchstream import Gesture, TouchSample
from . import hid_descriptor as hd
from .base import TouchBackend


def _q(v: int) -> int:
    return int(v)


class UhidBackend(TouchBackend):
    fidelity = "full"

    def __init__(self, adb, cfg: UhidConfig):
        self.adb = adb
        self.cfg = cfg
        self._panel: PanelGeometry | None = None
        self._proc = None        # long-lived `hid -` process; holds the device
        self._opened = False
        self._axes = hd.DEFAULT_AXES

    # ── availability probe ───────────────────────────────────────────────────

    @staticmethod
    def available(adb) -> bool:
        """True iff ``/system/bin/hid`` exists on the device."""
        try:
            out = adb.shell("[ -e /system/bin/hid ] && echo yes || echo no")
            return "yes" in out
        except Exception:
            return False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _detect_axes(self) -> hd.PanelAxes:
        """Clone the real panel's contact-channel ranges, not just its name.

        Android loads the panel's calibration by device NAME and applies it to our
        reports, so declaring different axis maxima makes the framework size our
        contact against the wrong scale. Measured consequence on a Pixel 7a: with a
        255-max TOUCH_MAJOR against the panel's real 2399, Hearthstone silently
        ignored every tap on a mulligan card while still honouring taps on buttons.
        """
        try:
            axes = hd.parse_panel_axes(self.adb.shell("getevent -pl 2>/dev/null"),
                                       self.cfg.device_name)
        except Exception:
            axes = None
        if axes is None:
            return hd.DEFAULT_AXES
        # explicit config overrides win; 0 means "take the panel's value"
        return hd.PanelAxes(
            touch_major_max=self.cfg.touch_major_max or axes.touch_major_max,
            touch_minor_max=self.cfg.touch_minor_max or axes.touch_minor_max,
            pressure_max=self.cfg.pressure_max or axes.pressure_max,
            orientation_max=self.cfg.orientation_max or axes.orientation_max,
        )

    def open(self, panel: PanelGeometry) -> None:
        if self._opened:
            return
        self._panel = panel
        self._axes = self._detect_axes()
        # One persistent reader of a bare-object stream on stdin (no FIFO: see
        # module docstring for the SELinux rationale).
        self._proc = self.adb.popen_shell("hid -")
        self._opened = True

        descriptor = hd.build_digitizer_descriptor(panel.width_px, panel.height_px,
                                                   hd.MAX_CONTACTS, self._axes)
        self._send({
            "id": 1,
            "command": "register",
            "name": self.cfg.device_name,
            "vid": _q(self.cfg.vendor_id),
            "pid": _q(self.cfg.product_id),
            "bus": self.cfg.bus,
            "descriptor": descriptor,
        })
        # Let the framework enumerate the new InputDevice before the first
        # gesture, so early reports aren't dropped before dispatch is wired up.
        if self.cfg.register_settle_ms > 0:
            time.sleep(self.cfg.register_settle_ms / 1000.0)

    def emit(self, gesture: Gesture) -> None:
        if not self._opened:
            raise RuntimeError("UhidBackend.emit before open()")
        panel = self._panel
        assert panel is not None
        # Gestures are synthesized in DISPLAY space; the panel is native. Rotate
        # each sample to native here, keyed on the live rotation (re-read per
        # gesture so a landscape flip mid-run is handled).
        rotation = self._current_rotation()
        active: dict[int, hd.ContactReport] = {}
        prev_t = gesture.samples[0].t if gesture.samples else 0.0

        for s in gesture.samples:
            dt_ms = int(round((s.t - prev_t) * 1000))
            floor = self.cfg.min_report_interval_ms
            if dt_ms >= floor:
                self._send({"id": 1, "command": "delay", "duration": dt_ms})
                prev_t = s.t

            rep = self._sample_to_report(s, panel, rotation)
            active[s.pointer_id] = rep
            self._send({
                "id": 1,
                "command": "report",
                "report": list(hd.encode_report(list(active.values()),
                                               hd.MAX_CONTACTS, self._axes)),
            })
            if not s.tip:
                active.pop(s.pointer_id, None)

    def close(self) -> None:
        if not self._opened:
            return
        try:
            if self._proc is not None and self._proc.stdin:
                self._proc.stdin.flush()
                self._proc.stdin.close()   # EOF -> hid exits -> device destroyed
        except Exception:
            pass
        try:
            if self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass
        self._opened = False
        self._proc = None

    # ── encoding ─────────────────────────────────────────────────────────────

    def _current_rotation(self) -> int:
        """Live display rotation (0..3); falls back to 0 if the adb handle can't
        report it (e.g. the transport unit tests' fake)."""
        try:
            return int(self.adb.get_rotation())
        except Exception:
            return 0

    def _sample_to_report(self, s: TouchSample, panel: PanelGeometry,
                          rotation: int) -> hd.ContactReport:
        nx, ny = display_to_native(s.x, s.y, rotation, panel.width_px, panel.height_px)
        axes = self._axes
        # major/minor are fractions of the MAJOR axis: they are lengths in the same
        # units, and the panel merely declares a smaller ceiling for the minor one.
        major_raw = int(round(max(0.0, min(1.0, s.major)) * axes.touch_major_max))
        minor_raw = int(round(max(0.0, min(1.0, s.minor)) * axes.touch_major_max))
        return hd.ContactReport(
            contact_id=s.pointer_id,
            x=max(0, min(panel.width_px - 1, int(round(nx)))),
            y=max(0, min(panel.height_px - 1, int(round(ny)))),
            pressure=int(round(max(0.0, min(1.0, s.pressure)) * axes.pressure_max)),
            major=min(axes.touch_major_max, major_raw),
            minor=min(axes.touch_minor_max, minor_raw),
            orientation=int(round(max(-1.0, min(1.0, s.orientation / (3.141592653589793 / 2)))
                                  * axes.orientation_max)),
            tip=s.tip,
            confidence=True,
        )

    # ── raw JSON streaming to the hid process stdin ─────────────────────────

    def _send(self, obj: dict) -> None:
        """Write one bare JSON object followed by a newline (the modern ``hid``
        tool reads a stream of objects, not a JSON array)."""
        self._raw_write(json.dumps(obj) + "\n")

    def _raw_write(self, text: str) -> None:
        p = self._proc
        if p is None or p.stdin is None:
            raise RuntimeError("UHID hid process not open")
        # adb.popen_shell opens stdin in binary mode; stream UTF-8 bytes.
        p.stdin.write(text.encode("utf-8"))
        p.stdin.flush()
