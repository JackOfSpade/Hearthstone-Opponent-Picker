"""Off-device tests for the UHID streaming framing.

These pin the two device-dependent fixes discovered on the Pixel 7a
(Android 17): the modern `hid` tool reads a **stream of bare JSON objects**
(not a `[...]` array), and the device is fed via a single persistent process's
stdin (not a FIFO). We drive UhidBackend with a fake adb/process that records
exactly what bytes would go to `hid -`.
"""

import json
from io import BytesIO

import pytest

from hop.config import UhidConfig
from hop.geometry import PanelGeometry
from hop.transport.hid_descriptor import REPORT_ID
from hop.transport.uhid import UhidBackend, UhidTransportError
from hop.touchstream import Gesture, TouchSample


class FakeStdin:
    def __init__(self, *, fail_after=None, flush_error=None, close_error=None, on_fail=None,
                 on_flush_fail=None):
        self.buf = bytearray()
        self.closed = False
        self._fail_after = fail_after
        self._flush_error = flush_error
        self._close_error = close_error
        self._on_fail = on_fail
        self._on_flush_fail = on_flush_fail
        self.write_calls = 0

    def write(self, b):
        if self._fail_after is not None and self.write_calls >= self._fail_after:
            if self._on_fail is not None:
                self._on_fail()
            raise BrokenPipeError("simulated closed ADB shell")
        self.buf.extend(b)
        self.write_calls += 1

    def flush(self):
        if self._flush_error is not None:
            if self._on_flush_fail is not None:
                self._on_flush_fail()
            raise self._flush_error

    def close(self):
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class FakeProc:
    def __init__(self, *, stdin=None, pid=None, returncode=None, stderr=b""):
        self.stdin = stdin or FakeStdin()
        self.pid = pid
        self.returncode = returncode
        self.stderr = BytesIO(stderr)
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def poll(self):
        return self.returncode


class FakeAdb:
    def __init__(self, procs=None):
        self.popen_cmds = []
        self.procs = list(procs or [FakeProc(pid=100)])
        self.proc = self.procs[0]
        # Matches production Adb's persistent-shell epoch. Tests can advance it
        # to model either `_capture()` reconnecting implicitly or an explicit
        # capture-recovery `adb disconnect`/connect cycle.
        self.connection_generation = 0

    def shell(self, cmd):
        return "yes"

    def popen_shell(self, cmd):
        self.popen_cmds.append(cmd)
        i = len(self.popen_cmds) - 1
        self.proc = self.procs[i] if i < len(self.procs) else FakeProc(pid=100 + i)
        return self.proc


def _cfg(**over):
    base = dict(device_name="goodix_ts0", vendor_id=0x27C6, product_id=0x0100,
                bus="usb", min_report_interval_ms=2, register_settle_ms=0)
    base.update(over)
    return UhidConfig(**base)


def _tap_gesture(x=540, y=1200):
    return Gesture(kind="tap", target=(float(x), float(y)), samples=[
        TouchSample(t=0.00, x=x, y=y, pressure=0.3, size=0.1, major=0.2,
                    minor=0.16, orientation=0.0, tip=True),
        TouchSample(t=0.02, x=x, y=y, pressure=0.8, size=0.15, major=0.4,
                    minor=0.32, orientation=0.0, tip=True),
        TouchSample(t=0.10, x=x, y=y, pressure=0.0, size=0.0, major=0.0,
                    minor=0.0, orientation=0.0, tip=False),
    ])


def _run():
    adb = FakeAdb()
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))
    backend.emit(_tap_gesture())
    backend.close()
    text = bytes(adb.proc.stdin.buf).decode("utf-8")
    objs = [json.loads(line) for line in text.splitlines() if line.strip()]
    return adb, objs, text


def _objects(proc):
    return [json.loads(line) for line in bytes(proc.stdin.buf).decode("utf-8").splitlines()
            if line.strip()]


def test_uses_persistent_hid_stdin_process():
    adb, _objs, _text = _run()
    assert adb.popen_cmds == ["hid -"]  # one persistent reader, stdin-fed


def test_stream_is_bare_objects_not_an_array():
    _adb, objs, text = _run()
    # every non-empty line is a standalone JSON object...
    assert objs and all(isinstance(o, dict) for o in objs)
    # ...and the stream is not wrapped in a top-level array.
    assert text.lstrip()[0] == "{"
    assert "]" != text.rstrip()[-1]


def test_registers_exactly_once_first():
    _adb, objs, _text = _run()
    registers = [o for o in objs if o.get("command") == "register"]
    assert len(registers) == 1
    assert objs[0]["command"] == "register"  # before any report
    r = registers[0]
    assert r["name"] == "goodix_ts0"
    assert r["vid"] == 0x27C6 and r["pid"] == 0x0100
    assert isinstance(r["descriptor"], list) and r["descriptor"]


def test_emits_reports_and_delays_with_report_id():
    _adb, objs, _text = _run()
    reports = [o for o in objs if o.get("command") == "report"]
    delays = [o for o in objs if o.get("command") == "delay"]
    assert reports, "expected report commands"
    assert delays, "expected inter-sample delay commands"
    # each report payload starts with the descriptor's REPORT_ID
    assert all(o["report"][0] == REPORT_ID for o in reports)


def test_close_eofs_the_stream_and_terminates():
    adb, _objs, _text = _run()
    assert adb.proc.stdin.closed is True
    assert adb.proc.terminated is True


def test_transport_status_exposes_cached_transform_and_last_delivered_endpoint():
    adb = FakeAdb()
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))
    backend.emit(_tap_gesture())

    status = backend.transport_status()
    assert status["rotation"] == 0  # FakeAdb has no live rotation probe; backend caches fallback
    assert (status["panel_width_px"], status["panel_height_px"]) == (1080, 2400)
    assert status["axis_touch_major_max"] > 0 and status["axis_pressure_max"] > 0
    assert (status["last_display_x"], status["last_display_y"]) == (540.0, 1200.0)
    assert (status["last_native_x"], status["last_native_y"]) == (540, 1200)
    backend.close()
    # The backend clears the live transform for a later open, but the terminal
    # report must retain the transform that produced the last tap.
    assert backend.transport_status()["rotation"] == 0


# ── persistent-shell recovery: ADB reconnect must renew UHID before a gesture ──

def test_reconnect_generation_renews_the_stream_before_the_next_gesture():
    """The live failure: capture recovery reconnects ADB, which severs `hid -`.

    The next gesture must register a fresh digitizer first.  This is deliberately
    at a gesture boundary: a mid-gesture retry could duplicate a partial tap.
    """
    first = FakeProc(pid=101)
    second = FakeProc(pid=102)
    adb = FakeAdb([first, second])
    adb.connection_generation = 7
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))

    adb.connection_generation += 1  # `adb disconnect` / a new connection happened
    backend.emit(_tap_gesture(1200, 540))

    assert adb.popen_cmds == ["hid -", "hid -"]
    assert first.stdin.closed and first.terminated
    assert [o["command"] for o in _objects(first)] == ["register"]
    assert _objects(second)[0]["command"] == "register"
    assert any(o["command"] == "report" for o in _objects(second))
    status = backend.transport_status()
    assert status["connection_generation"] == status["stream_generation"] == 8
    assert status["reopen_count"] == 1 and status["last_reopen_ok"] is True
    assert "7 -> 8" in status["last_reopen_reason"]


def test_exited_hid_process_renews_only_before_a_new_gesture():
    """A dead resident process found before a gesture is safe to replace."""
    first = FakeProc(pid=101)
    second = FakeProc(pid=102)
    adb = FakeAdb([first, second])
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))
    first.returncode = 1

    backend.emit(_tap_gesture())

    assert adb.popen_cmds == ["hid -", "hid -"]
    assert _objects(second)[0]["command"] == "register"
    assert backend.transport_status()["last_reopen_ok"] is True


def test_midgesture_broken_pipe_is_typed_closed_and_never_replayed():
    """The pasted failure mode: report write succeeds but its flush gets EPIPE."""
    stdin = FakeStdin()
    proc = FakeProc(stdin=stdin, pid=101, stderr=b"remote shell gone\n")
    adb = FakeAdb([proc])
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))
    # Initial register was flushed successfully; the next report's *flush* is
    # exactly where the live traceback raised BrokenPipeError.
    stdin._flush_error = BrokenPipeError("simulated closed ADB shell")
    stdin._on_flush_fail = lambda: setattr(proc, "returncode", 1)

    with pytest.raises(UhidTransportError, match="stream lost while sending report"):
        backend.emit(_tap_gesture())

    assert adb.popen_cmds == ["hid -"]       # no unsafe retry/replay
    assert stdin.closed and proc.terminated    # cleanup survives the failed flush/write
    status = backend.transport_status()
    assert status["opened"] is False
    assert status["last_command"] == "report"
    assert "BrokenPipeError" in status["failure"]
    assert "remote shell gone" in status["stderr_tail"]


def test_register_failure_cleans_up_the_just_started_process():
    """`open()` must not leak a Popen if its initial register write breaks."""
    stdin = FakeStdin(fail_after=0)
    proc = FakeProc(stdin=stdin, pid=101, returncode=1)
    adb = FakeAdb([proc])
    backend = UhidBackend(adb, _cfg())

    with pytest.raises(UhidTransportError, match="stream lost while sending register"):
        backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))

    assert stdin.closed and proc.terminated
    assert backend.transport_status()["opened"] is False


def test_close_attempts_stdin_close_even_if_flush_broke():
    """Buffered stdin only marks closed when close is attempted separately from flush."""
    proc = FakeProc(pid=101)
    adb = FakeAdb([proc])
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))
    proc.stdin._flush_error = BrokenPipeError("late flush failure")

    backend.close()

    assert proc.stdin.closed and proc.terminated


# ── rotation: display-space samples must be mapped to native panel px ─────────

class FakeAdbRot(FakeAdb):
    def __init__(self, rot, procs=None):
        super().__init__(procs)
        self._rot = rot
        self.rotation_queries = 0

    def get_rotation(self):
        self.rotation_queries += 1
        return self._rot


def _emit_single(rot, x, y):
    """Emit one explicit display-space sample; return decoded native (x,y)."""
    adb = FakeAdbRot(rot)
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))  # native
    backend.emit(Gesture(kind="tap", samples=[
        TouchSample(t=0.0, x=x, y=y, pressure=0.5, size=0.2, major=0.3,
                    minor=0.24, orientation=0.0, tip=True)]))
    backend.close()
    text = bytes(adb.proc.stdin.buf).decode("utf-8")
    rep = [json.loads(l) for l in text.splitlines()
           if l.strip() and json.loads(l).get("command") == "report"][0]["report"]
    # payload: [id][flags][cid][x_lo][x_hi][y_lo][y_hi]...
    return (rep[3] | (rep[4] << 8), rep[5] | (rep[6] << 8))


def test_rot270_display_center_maps_to_native_center():
    # landscape center (1199,540) -> native (540, 2399-1199=1200) [measured truth]
    assert _emit_single(3, 1199, 540) == (540, 1200)


def test_rot0_is_identity():
    assert _emit_single(0, 300, 900) == (300, 900)


def test_rotation_is_read_once_per_open_and_reused_for_all_gestures():
    adb = FakeAdbRot(3)
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))

    backend.emit(_tap_gesture())
    backend.emit(_tap_gesture())

    assert adb.rotation_queries == 1


def test_status_reports_the_last_endpoint_in_both_spaces_under_rotation():
    adb = FakeAdbRot(3)
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))
    backend.emit(_tap_gesture(1200, 540))

    status = backend.transport_status()
    assert (status["last_display_x"], status["last_display_y"]) == (1200.0, 540.0)
    # ROTATION_270 maps display (x,y) to native (y, native_h - 1 - x).
    assert (status["last_native_x"], status["last_native_y"]) == (540, 1199)


def test_rotation_is_refreshed_after_a_true_close_then_open_session():
    adb = FakeAdbRot(3)
    backend = UhidBackend(adb, _cfg())
    panel = PanelGeometry(width_px=1080, height_px=2400, dpi=420.0)
    backend.open(panel)
    backend.emit(_tap_gesture())
    backend.close()
    # Model the closed session's terminal diagnostics; a successfully registered
    # new session must not attach them to its new rotation/panel metadata.
    backend._last_failure = "old session failure"
    backend._last_proc_status["stderr_tail"] = "old session stderr"
    adb._rot = 1

    backend.open(panel)

    assert adb.rotation_queries == 2
    status = backend.transport_status()
    assert status["rotation"] == 1
    assert status["failure"] is None and status["stderr_tail"] == ""
    assert status["last_display_x"] is None and status["last_display_y"] is None
    assert status["last_native_x"] is None and status["last_native_y"] is None


def test_failed_true_reopen_retains_prior_endpoint_and_transform_evidence():
    first = FakeProc(pid=101)
    failed = FakeProc(stdin=FakeStdin(fail_after=0), pid=102)
    adb = FakeAdbRot(3, [first, failed])
    backend = UhidBackend(adb, _cfg())
    panel = PanelGeometry(width_px=1080, height_px=2400, dpi=420.0)
    backend.open(panel)
    backend.emit(_tap_gesture(1200, 540))
    backend.close()
    adb._rot = 1

    with pytest.raises(UhidTransportError):
        backend.open(panel)

    status = backend.transport_status()
    assert status["rotation"] == 3
    assert (status["last_display_x"], status["last_display_y"]) == (1200.0, 540.0)
    assert (status["last_native_x"], status["last_native_y"]) == (540, 1199)
    assert "UHID stream lost" in status["failure"]


def test_stream_recovery_keeps_the_session_rotation_cache():
    first = FakeProc(pid=101)
    second = FakeProc(pid=102)
    adb = FakeAdbRot(3, [first, second])
    backend = UhidBackend(adb, _cfg())
    backend.open(PanelGeometry(width_px=1080, height_px=2400, dpi=420.0))

    adb.connection_generation += 1
    backend.emit(_tap_gesture())

    assert adb.rotation_queries == 1


def test_missing_get_rotation_falls_back_to_identity():
    # base FakeAdb has no get_rotation -> rotation 0 -> unchanged coordinates
    _adb, objs, _text = _run()
    reports = [o for o in objs if o.get("command") == "report"]
    x = reports[0]["report"][3] | (reports[0]["report"][4] << 8)
    assert x == 540  # the gesture's first sample x, unrotated
