"""Preferred transport: a persistent virtual HID digitizer via ``/system/bin/hid``.

Standard, Layer 1 (the load-bearing Rev 2 fix): register **one** digitizer for
the whole session and stream every gesture through it; destroy it only at
teardown. Per-gesture register/destroy is a §9 anti-pattern - UHID destruction
is a *disconnect* observable via ``InputManager.InputDeviceListener``, and a
built-in panel never disconnects between taps.

Streaming approach (the standard's "resident host-side writer, a named pipe held
by an allowed context"): the AOSP ``hid`` tool reads a JSON *array* of commands
from a file with a streaming pull-parser, so we point it at a named pipe (FIFO)
on the device, open the array ``[`` with the ``register`` command at session
start, append ``report``/``delay`` commands as gestures arrive, and close the
array ``]`` at teardown. The device stays registered for exactly as long as the
FIFO is held open - no add/remove churn.

Caveats (marked because they can only be confirmed on-device):
* SELinux may deny the shell domain a held-open FIFO; the standard says to solve
  the plumbing rather than fall back to re-enumeration. If it can't be solved on
  a given phone, use ``touch_backend = adb`` and accept degraded fidelity.
* Panel-matched identity (vid/pid/name) is generic by default - clone your real
  panel's values into ``[uhid]`` (``hop doctor`` prints them).
"""

from __future__ import annotations

import json

from ..config import UhidConfig
from ..geometry import PanelGeometry
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
        self._reader = None      # long-lived `hid <fifo>` process (holds device)
        self._writer = None      # `cat > fifo` whose stdin we stream JSON into
        self._opened = False
        self._first_cmd = True

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

    def open(self, panel: PanelGeometry) -> None:
        if self._opened:
            return
        self._panel = panel
        fifo = self.cfg.fifo_path
        # (re)create the FIFO
        self.adb.shell(f"rm -f {fifo}; ( mkfifo {fifo} || mknod {fifo} p )")
        # reader: hid blocks reading the array from the FIFO for the whole session
        self._reader = self.adb.popen_shell(f"hid {fifo}")
        # writer: cat holds the FIFO open for writing; we feed it JSON
        self._writer = self.adb.popen_shell(f"cat > {fifo}")
        self._opened = True
        self._first_cmd = True

        descriptor = hd.build_digitizer_descriptor(panel.width_px, panel.height_px)
        self._raw_write("[\n")
        self._send({
            "id": 1,
            "command": "register",
            "name": self.cfg.device_name,
            "vid": _q(self.cfg.vendor_id),
            "pid": _q(self.cfg.product_id),
            "bus": self.cfg.bus,
            "descriptor": descriptor,
        })

    def emit(self, gesture: Gesture) -> None:
        if not self._opened:
            raise RuntimeError("UhidBackend.emit before open()")
        panel = self._panel
        assert panel is not None
        active: dict[int, hd.ContactReport] = {}
        prev_t = gesture.samples[0].t if gesture.samples else 0.0

        for s in gesture.samples:
            dt_ms = int(round((s.t - prev_t) * 1000))
            floor = self.cfg.min_report_interval_ms
            if dt_ms >= floor:
                self._send({"id": 1, "command": "delay", "duration": dt_ms})
            prev_t = s.t

            rep = self._sample_to_report(s, panel)
            active[s.pointer_id] = rep
            self._send({
                "id": 1,
                "command": "report",
                "report": list(hd.encode_report(list(active.values()))),
            })
            if not s.tip:
                active.pop(s.pointer_id, None)

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self._raw_write("\n]\n")
            if self._writer is not None and self._writer.stdin:
                self._writer.stdin.flush()
                self._writer.stdin.close()   # EOF -> cat exits -> hid exits -> device destroyed
        except Exception:
            pass
        for proc in (self._writer, self._reader):
            try:
                if proc is not None:
                    proc.terminate()
            except Exception:
                pass
        try:
            self.adb.shell(f"rm -f {self.cfg.fifo_path}")
        except Exception:
            pass
        self._opened = False

    # ── encoding ─────────────────────────────────────────────────────────────

    def _sample_to_report(self, s: TouchSample, panel: PanelGeometry) -> hd.ContactReport:
        return hd.ContactReport(
            contact_id=s.pointer_id,
            x=max(0, min(panel.width_px - 1, int(round(s.x)))),
            y=max(0, min(panel.height_px - 1, int(round(s.y)))),
            pressure=int(round(max(0.0, min(1.0, s.pressure)) * 255)),
            major=int(round(max(0.0, min(1.0, s.major)) * 255)),
            minor=int(round(max(0.0, min(1.0, s.minor)) * 255)),
            orientation=int(round(max(-1.0, min(1.0, s.orientation / (3.141592653589793 / 2))) * 127)),
            tip=s.tip,
            confidence=True,
        )

    # ── raw JSON streaming to the FIFO writer ───────────────────────────────

    def _send(self, obj: dict) -> None:
        prefix = "" if self._first_cmd else ",\n"
        self._first_cmd = False
        self._raw_write(prefix + json.dumps(obj))

    def _raw_write(self, text: str) -> None:
        w = self._writer
        if w is None or w.stdin is None:
            raise RuntimeError("UHID writer not open")
        # adb.popen_shell opens stdin in binary mode; stream UTF-8 bytes.
        w.stdin.write(text.encode("utf-8"))
        w.stdin.flush()
