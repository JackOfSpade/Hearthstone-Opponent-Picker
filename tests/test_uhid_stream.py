"""Off-device tests for the UHID streaming framing.

These pin the two device-dependent fixes discovered on the Pixel 7a
(Android 17): the modern `hid` tool reads a **stream of bare JSON objects**
(not a `[...]` array), and the device is fed via a single persistent process's
stdin (not a FIFO). We drive UhidBackend with a fake adb/process that records
exactly what bytes would go to `hid -`.
"""

import json

from hop.config import UhidConfig
from hop.geometry import PanelGeometry
from hop.transport.hid_descriptor import REPORT_ID
from hop.transport.uhid import UhidBackend
from hop.touchstream import Gesture, TouchSample


class FakeStdin:
    def __init__(self):
        self.buf = bytearray()
        self.closed = False

    def write(self, b):
        self.buf.extend(b)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self):
        self.stdin = FakeStdin()
        self.terminated = False

    def terminate(self):
        self.terminated = True


class FakeAdb:
    def __init__(self):
        self.popen_cmds = []
        self.proc = FakeProc()

    def shell(self, cmd):
        return "yes"

    def popen_shell(self, cmd):
        self.popen_cmds.append(cmd)
        return self.proc


def _cfg(**over):
    base = dict(device_name="goodix_ts0", vendor_id=0x27C6, product_id=0x0100,
                bus="usb", min_report_interval_ms=2, register_settle_ms=0)
    base.update(over)
    return UhidConfig(**base)


def _tap_gesture():
    return Gesture(kind="tap", target=(540.0, 1200.0), samples=[
        TouchSample(t=0.00, x=540, y=1200, pressure=0.3, size=0.1, major=0.2,
                    minor=0.16, orientation=0.0, tip=True),
        TouchSample(t=0.02, x=540, y=1200, pressure=0.8, size=0.15, major=0.4,
                    minor=0.32, orientation=0.0, tip=True),
        TouchSample(t=0.10, x=540, y=1200, pressure=0.0, size=0.0, major=0.0,
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


# ── rotation: display-space samples must be mapped to native panel px ─────────

class FakeAdbRot(FakeAdb):
    def __init__(self, rot):
        super().__init__()
        self._rot = rot

    def get_rotation(self):
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


def test_missing_get_rotation_falls_back_to_identity():
    # base FakeAdb has no get_rotation -> rotation 0 -> unchanged coordinates
    _adb, objs, _text = _run()
    reports = [o for o in objs if o.get("command") == "report"]
    x = reports[0]["report"][3] | (reports[0]["report"][4] << 8)
    assert x == 540  # the gesture's first sample x, unrotated
