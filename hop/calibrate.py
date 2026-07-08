"""Pure parsers for the §10 calibration measurements.

These are separated from :mod:`hop.cli` so they can be tested against captured
``getevent`` output without a phone attached. Everything here is a pure function
from device text to a number; the CLI owns the ADB calls.

The touch **report rate** is the one §10 constant that cannot be read from any
sysfs node or ``dumpsys`` dump: only a real finger makes the panel emit reports.
:func:`estimate_report_rate_hz` turns a ``getevent -lt`` capture of a human swipe
into that rate.
"""

from __future__ import annotations

import re
import statistics

_TABLE_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")


def _key_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*{re.escape(key)}\s*=")


def upsert_toml_scalar(text: str, table: str, key: str, value: str,
                       comment: str | None = None) -> str:
    """Set ``key = value`` under ``[table]``, creating or replacing in place.

    ``hop calibrate`` used to *append* its results, which emits a second
    ``[motor]`` header on the second run - and duplicate tables are invalid TOML,
    so re-calibrating would leave a config that no ``hop`` command could load.
    This keeps the write idempotent.

    Deliberately line-based rather than a full TOML round-trip: it preserves the
    user's comments and provenance notes, which a parse/re-emit would discard.
    """
    line = f"{key} = {value}" + (f"    # {comment}" if comment else "")
    lines = text.splitlines()

    start = None
    for i, raw in enumerate(lines):
        m = _TABLE_RE.match(raw)
        if m and m.group(1).strip() == table:
            start = i
            break

    if start is None:
        prefix = lines + ([""] if lines and lines[-1].strip() else [])
        return "\n".join([*prefix, f"[{table}]", line, ""])

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _TABLE_RE.match(lines[j]):
            end = j
            break

    pat = _key_re(key)
    for j in range(start + 1, end):
        if pat.match(lines[j]):
            lines[j] = line
            return "\n".join(lines) + "\n"

    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines.insert(insert, line)
    return "\n".join(lines) + "\n"

# "[  410604.936265] EV_ABS       ABS_MT_POSITION_X    000001b3"
_EVENT_RE = re.compile(r"\[\s*([\d.]+)\]\s+(\S+)\s+(\S+)\s+(\S+)")

# Frames closer than this are the same report; anything longer is a gap between
# gestures (or a stalled panel) and must not enter the rate estimate.
MAX_FRAME_GAP_S = 0.05
MIN_FRAMES = 12


_ADD_DEVICE_RE = re.compile(r"\s*add device \d+: (/dev/input/event\d+)")
_SECTION_RE = re.compile(r"\s*([A-Z]+) \(([0-9a-f]{4})\):")
_ABS_MT_POSITION_X = "0035"  # linux/input-event-codes.h


def parse_touch_event_node(getevent_p_output: str) -> str | None:
    """Return the ``/dev/input/eventN`` node of the multitouch panel.

    A phone has several input devices - fingerprint reader, haptics, power keys -
    and any of them can emit events during a sample window. Only the touchscreen
    reports ``ABS_MT_POSITION_X``, so we key on that capability rather than on a
    device name, which varies per vendor.

    Accepts output from ``getevent -pl`` (axes labelled) or ``getevent -pi`` (raw
    hex codes); the axis code is only trusted inside the device's ``ABS`` section,
    since the same number can appear as a ``KEY`` scancode.
    """
    node: str | None = None
    section: str | None = None
    for line in getevent_p_output.splitlines():
        m = _ADD_DEVICE_RE.match(line)
        if m:
            node, section = m.group(1), None
            continue
        if node is None:
            continue
        s = _SECTION_RE.match(line)
        if s:
            section = s.group(1)
            head: str = line.split(":", 1)[1]
        elif line.startswith(" " * 8):
            head = line  # continuation of the current section
        else:
            section = None
            continue
        if section != "ABS":
            continue
        if "ABS_MT_POSITION_X" in head:
            return node
        if re.search(rf"(?<![0-9a-f]){_ABS_MT_POSITION_X}(?![0-9a-f])", head.split(":", 1)[0]):
            return node
    return None


def _frame_timestamps(getevent_lt_output: str) -> tuple[list[float], list[tuple[float, float]]]:
    """Extract SYN_REPORT times and (down, up) gesture spans from a capture."""
    syn: list[float] = []
    gestures: list[tuple[float, float]] = []
    down: float | None = None
    for line in getevent_lt_output.splitlines():
        m = _EVENT_RE.match(line)
        if not m:
            continue
        t, code, value = float(m.group(1)), m.group(3), m.group(4)
        if code == "SYN_REPORT":
            syn.append(t)
        elif code == "BTN_TOUCH":
            if value == "DOWN":
                down = t
            elif value == "UP" and down is not None:
                gestures.append((down, t))
                down = None
    return syn, gestures


def estimate_report_rate_hz(getevent_lt_output: str) -> float | None:
    """Estimate the panel's touch report rate (Hz) from a ``getevent -lt`` capture.

    Counts **SYN_REPORT** frames, not ``ABS_MT_POSITION_*`` events. That
    distinction is the whole game:

    * a stationary finger emits a frame carrying only pressure/size and **no**
      position axis at all, so position-counting silently drops frames, and
    * a moving finger emits *both* ``_X`` and ``_Y`` in one frame, so a substring
      match on ``ABS_MT_POSITION`` double-counts it.

    ``SYN_REPORT`` is emitted exactly once per report, whatever the finger did.

    The estimate is the **median inter-frame interval within a gesture**, not
    ``(n-1)/span``: idle time between swipes would otherwise drag the mean down
    (a real capture read 64 Hz that way against a true 183 Hz), and a single
    dropped frame would drag it up. The median is immune to both.
    """
    syn, gestures = _frame_timestamps(getevent_lt_output)
    if len(syn) < MIN_FRAMES:
        return None

    if gestures:
        # Keep only frames that fall inside a real finger-down span.
        frames = [t for t in syn if any(a <= t <= b for a, b in gestures)]
    else:
        frames = syn
    if len(frames) < MIN_FRAMES:
        frames = syn

    deltas = [b - a for a, b in zip(frames, frames[1:]) if 0.0 < b - a < MAX_FRAME_GAP_S]
    if len(deltas) < MIN_FRAMES - 1:
        return None
    median = statistics.median(deltas)
    if median <= 0:
        return None
    return 1.0 / median
