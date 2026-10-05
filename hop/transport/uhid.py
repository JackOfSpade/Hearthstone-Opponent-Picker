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
* **Fixed-session orientation.** Display-to-panel rotation is sampled when the
  backend session opens and retained for that session, so gestures do not each
  issue a costly ``dumpsys input`` ADB request. Stream recovery is not a new
  orientation lifecycle and preserves the captured transform.

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

from ..adb import AdbError
from ..config import UhidConfig
from ..geometry import PanelGeometry
from ..orientation import display_to_native
from ..touchstream import Gesture, TouchSample
from . import hid_descriptor as hd
from .base import TouchBackend


def _q(v: int) -> int:
    return int(v)


class UhidTransportError(AdbError):
    """A resident ``hid -`` stream was lost or could not be (re)registered.

    It deliberately subclasses :class:`~hop.adb.AdbError`: losing this stream is
    an ADB-transport failure, and the engine already knows how to stop cleanly
    for that class of failure.  Crucially, callers must never replay the gesture
    that encountered it: a write can fail after an earlier report in that gesture
    reached the device.
    """


class UhidBackend(TouchBackend):
    fidelity = "full"

    def __init__(self, adb, cfg: UhidConfig):
        self.adb = adb
        self.cfg = cfg
        self._panel: PanelGeometry | None = None
        # The display orientation is a session property for the fixed phone this
        # backend drives.  Reading it uses `dumpsys input`, so retain the value
        # obtained at open rather than adding an ADB round trip to every tap.
        self._rotation: int | None = None
        # Kept after close solely for the terminal bug report. `_rotation` itself
        # is cleared on close so a later open samples a fresh display orientation.
        self._last_session_rotation: int | None = None
        self._proc = None        # long-lived `hid -` process; holds the device
        self._opened = False
        self._axes = hd.DEFAULT_AXES
        # Last successfully emitted contact endpoint in both coordinate spaces.
        # This is diagnostic metadata only; it never changes the input stream.
        self._last_display_endpoint: tuple[float, float] | None = None
        self._last_native_endpoint: tuple[int, int] | None = None
        # `adb disconnect` can kill this resident shell while one-shot ADB calls
        # later recover.  We remember the ADB epoch at registration and renew the
        # stream *before* the next gesture when it changes.
        self._stream_generation: int | None = None
        self._last_command: str | None = None
        self._last_write_at: float | None = None
        self._reopen_count = 0
        self._last_reopen_reason: str | None = None
        self._last_reopen_ok: bool | None = None
        self._last_failure: str | None = None
        # Keep terminal process facts after close() clears `_proc`: bug reports
        # are often taken after Engine.run() has reached its finally block.
        self._last_proc_status: dict[str, int | str | None] = {
            "pid": None,
            "returncode": None,
            "stderr_tail": "",
        }
        # Retain the most recently closed process object for a later, nonblocking
        # poll.  A BrokenPipe can arrive a few scheduler ticks before `poll()`
        # observes the exit; retaining it lets a subsequent bug report collect the
        # final return code/stderr even though the active stream was cleared.
        self._last_proc = None

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
        # A true close->open starts a new coordinate session.  Do not clear the
        # prior endpoint/failure facts until the replacement stream has actually
        # registered: a failed reopen must remain diagnosable as the old session.
        prior = (self._panel, self._rotation, self._last_session_rotation, self._axes)
        self._panel = panel
        self._rotation = self._read_rotation()
        self._axes = self._detect_axes()
        try:
            self._start_stream()
        except Exception:
            self._panel, self._rotation, self._last_session_rotation, self._axes = prior
            raise
        self._last_session_rotation = self._rotation
        self._last_display_endpoint = None
        self._last_native_endpoint = None
        self._last_failure = None
        self._last_proc = None
        self._last_proc_status = {"pid": None, "returncode": None, "stderr_tail": ""}

    def _start_stream(self) -> None:
        """Start and register one resident stream using the retained panel/axes.

        This is used for the first open and for a known-safe recovery at the
        *start* of a later gesture.  It intentionally never replays a gesture.
        """
        panel = self._panel
        assert panel is not None
        # One persistent reader of a bare-object stream on stdin (no FIFO: see
        # module docstring for the SELinux rationale).
        try:
            self._proc = self.adb.popen_shell("hid -")
        except Exception as e:
            self._record_failure("could not start UHID hid process", e)
            raise UhidTransportError(f"could not start UHID hid process: {e}") from e
        self._opened = True

        try:
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
        except UhidTransportError:
            # `_raw_write` has already separately closed a broken stdin, but
            # keep this defensive cleanup for a failure while building/registering.
            self._discard_stream()
            raise
        except Exception as e:
            self._record_failure("could not register UHID digitizer", e)
            self._discard_stream()
            raise UhidTransportError(f"could not register UHID digitizer: {e}") from e
        self._stream_generation = self._connection_generation()
        # Let the framework enumerate the new InputDevice before the first
        # gesture, so early reports aren't dropped before dispatch is wired up.
        if self.cfg.register_settle_ms > 0:
            time.sleep(self.cfg.register_settle_ms / 1000.0)

    def emit(self, gesture: Gesture) -> None:
        if not self._opened:
            if self._last_failure:
                raise UhidTransportError(f"UHID stream is unavailable: {self._last_failure}")
            raise RuntimeError("UhidBackend.emit before open()")
        # A stream re-register is safe only here, before any report from this
        # gesture has been sent.  A changed ADB epoch means `adb disconnect` or
        # a new endpoint may have severed the old shell.  An exited local process
        # is the same safe-before-gesture case.  A write error below is different:
        # it may be mid-gesture, so it is converted to a typed failure, never retried.
        self._renew_stream_before_gesture()
        panel = self._panel
        assert panel is not None
        # Gestures are synthesized in DISPLAY space; the panel is native.  The
        # orientation was captured with the session at open(), which avoids a
        # costly `dumpsys input` ADB request for every gesture.  A resident-HID
        # stream recovery stays within this same fixed-phone session, so it
        # intentionally retains that coordinate transform too.
        rotation = self._rotation
        assert rotation is not None
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
        # Only publish this after every report of the gesture has written.  A
        # mid-gesture transport error is deliberately ambiguous and must not be
        # represented as a delivered endpoint in a later bug report.
        dx, dy = gesture.endpoint()
        nx, ny = display_to_native(dx, dy, rotation, panel.width_px, panel.height_px)
        self._last_display_endpoint = (dx, dy)
        self._last_native_endpoint = (
            max(0, min(panel.width_px - 1, int(round(nx)))),
            max(0, min(panel.height_px - 1, int(round(ny)))),
        )

    def close(self) -> None:
        self._discard_stream()
        # A later open() is a new fixed-phone session and will sample rotation
        # again.  `_discard_stream()` itself is also used by stream recovery,
        # where the current session's cached transform must survive.
        self._rotation = None

    # ── persistent-stream lifecycle / diagnostics ─────────────────────────

    def _connection_generation(self) -> int | None:
        """Current ADB persistent-stream epoch, if the handle exposes one.

        Tiny fake ADBs and third-party callers can predate this contract; ``None``
        preserves their existing behavior while production :class:`Adb` supplies
        an integer that advances around disconnect/reconnect.
        """
        try:
            generation = getattr(self.adb, "connection_generation", None)
            return None if generation is None else int(generation)
        except Exception:
            return None

    @staticmethod
    def _poll(proc) -> int | None:
        try:
            value = proc.poll()
            return None if value is None else int(value)
        except Exception:
            return None

    def _read_stderr_tail(self, proc, returncode: int | None) -> str:
        """Read stderr only after exit, when EOF makes it non-blocking."""
        if returncode is None:
            return ""
        stream = getattr(proc, "stderr", None)
        if stream is None:
            return ""
        try:
            data = stream.read()
        except Exception:
            return ""
        if isinstance(data, bytes):
            data = data.decode(errors="replace")
        return str(data)[-1200:]

    def _snapshot_proc(self, proc) -> dict[str, int | str | None]:
        if proc is None:
            return dict(self._last_proc_status)
        returncode = self._poll(proc)
        status: dict[str, int | str | None] = {
            "pid": getattr(proc, "pid", None),
            "returncode": returncode,
            "stderr_tail": self._read_stderr_tail(proc, returncode),
        }
        # A fake/process wrapper could expose an unusual pid; retain only JSON-safe
        # primitives so status collection can never itself cause a dashboard crash.
        status["pid"] = status["pid"] if isinstance(status["pid"], int) else None
        return status

    def _remember_proc(self, proc) -> None:
        if proc is None:
            return
        status = self._snapshot_proc(proc)
        # Preserve a previously captured stderr tail when a later poll occurs after
        # its pipe has already been consumed.
        if not status["stderr_tail"]:
            status["stderr_tail"] = self._last_proc_status.get("stderr_tail", "")
        self._last_proc_status = status

    def transport_status(self) -> dict[str, int | float | str | bool | None]:
        """A serializable snapshot for controller/bug-report telemetry.

        It does not probe the device or block on a live process.  The stderr tail is
        read only once a process has exited, and remains available after ``close()``.
        """
        status = self._snapshot_proc(self._proc if self._proc is not None else self._last_proc)
        if not status["stderr_tail"]:
            status["stderr_tail"] = self._last_proc_status.get("stderr_tail", "")
        return {
            "kind": "uhid",
            "opened": self._opened,
            "pid": status["pid"],
            "returncode": status["returncode"],
            "last_command": self._last_command,
            "last_write_at": self._last_write_at,
            "stderr_tail": status["stderr_tail"],
            "failure": self._last_failure,
            "connection_generation": self._connection_generation(),
            "stream_generation": self._stream_generation,
            "reopen_count": self._reopen_count,
            "last_reopen_reason": self._last_reopen_reason,
            "last_reopen_ok": self._last_reopen_ok,
            # These values are captured once at backend open and describe the exact
            # display->native transform used by every later report.  They do not
            # trigger a device query when a dashboard or bug report asks for status.
            "rotation": (self._rotation if self._rotation is not None
                         else self._last_session_rotation),
            "panel_width_px": self._panel.width_px if self._panel else None,
            "panel_height_px": self._panel.height_px if self._panel else None,
            "axis_touch_major_max": self._axes.touch_major_max,
            "axis_touch_minor_max": self._axes.touch_minor_max,
            "axis_pressure_max": self._axes.pressure_max,
            "axis_orientation_max": self._axes.orientation_max,
            "last_display_x": (round(self._last_display_endpoint[0], 3)
                               if self._last_display_endpoint else None),
            "last_display_y": (round(self._last_display_endpoint[1], 3)
                               if self._last_display_endpoint else None),
            "last_native_x": (self._last_native_endpoint[0]
                              if self._last_native_endpoint else None),
            "last_native_y": (self._last_native_endpoint[1]
                              if self._last_native_endpoint else None),
        }

    def _record_failure(self, context: str, exc: Exception | None = None) -> None:
        detail = f"{type(exc).__name__}: {exc}" if exc is not None else ""
        self._last_failure = f"{context}{(': ' + detail) if detail else ''}"
        self._remember_proc(self._proc)

    def _discard_stream(self) -> None:
        """Best-effort teardown which closes stdin even when ``flush`` broke.

        A BufferedWriter whose flush saw EPIPE remains open; its later finalizer
        otherwise emits a noisy delayed BrokenPipe warning.  Keep flush and close
        in independent tries so close still marks that writer closed.
        """
        proc = self._proc
        if proc is not None:
            stdin = getattr(proc, "stdin", None)
            if stdin is not None:
                try:
                    stdin.flush()
                except Exception:
                    pass
                try:
                    stdin.close()  # EOF -> hid exits -> device is destroyed
                except Exception:
                    pass
            try:
                proc.terminate()
            except Exception:
                pass
            self._remember_proc(proc)
            self._last_proc = proc
        self._opened = False
        self._proc = None
        self._stream_generation = None

    def _reopen_stream(self, reason: str) -> None:
        """Renew the digitizer at a safe gesture boundary, preserving panel/axes."""
        self._reopen_count += 1
        self._last_reopen_reason = reason
        self._last_reopen_ok = False
        self._discard_stream()
        try:
            self._start_stream()
        except UhidTransportError:
            # `_start_stream` retained the failure/process facts for telemetry.
            raise
        self._last_reopen_ok = True

    def _renew_stream_before_gesture(self) -> None:
        proc = self._proc
        if proc is None:
            raise UhidTransportError("UHID stream disappeared before a gesture")
        connection_generation = self._connection_generation()
        if (connection_generation is not None
                and connection_generation != self._stream_generation):
            self._reopen_stream(
                f"ADB transport generation {self._stream_generation} -> {connection_generation}")
            return
        returncode = self._poll(proc)
        if returncode is not None:
            self._reopen_stream(f"hid process exited before gesture (status {returncode})")

    # ── encoding ─────────────────────────────────────────────────────────────

    def _read_rotation(self) -> int:
        """Session display rotation (0..3), with a test-friendly zero fallback.

        This is called once by :meth:`open`, rather than from :meth:`emit`.
        Reopening the HID process after a transport reconnect deliberately keeps
        the same cached transform: it is recovery within the same fixed-phone
        session, not a new display/orientation lifecycle.
        """
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
        self._last_command = str(obj.get("command", "json"))
        self._raw_write(json.dumps(obj) + "\n")

    def _raw_write(self, text: str) -> None:
        p = self._proc
        if p is None or p.stdin is None:
            raise UhidTransportError("UHID hid process is not open")
        # adb.popen_shell opens stdin in binary mode; stream UTF-8 bytes.
        try:
            p.stdin.write(text.encode("utf-8"))
            p.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            # Do not reopen and replay here: some earlier report of this gesture
            # may already have reached Android.  Tear down robustly and let the
            # engine's typed transport-failure path halt before another blind tap.
            command = self._last_command or "command"
            self._record_failure(f"UHID stream lost while sending {command}", e)
            self._discard_stream()
            raise UhidTransportError(
                f"UHID stream lost while sending {command}: {type(e).__name__}: {e}") from e
        self._last_write_at = time.time()
