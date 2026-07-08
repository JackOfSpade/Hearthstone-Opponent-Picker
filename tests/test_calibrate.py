"""Tests for the pure §10 calibration parsers.

The two `getevent` fixtures are **real captures** from the Pixel 7a (goodix_ts0,
2026-07-07): a human swiping, and `getevent -pi`. The golden rate below (183 Hz)
was cross-checked against the per-gesture medians of 24 separate swipes.
"""

from pathlib import Path

import pytest

from hop.calibrate import estimate_report_rate_hz, parse_touch_event_node
from hop.tomledit import upsert_toml_scalar

DATA = Path(__file__).parent / "data"


@pytest.fixture
def swipe() -> str:
    return (DATA / "getevent_swipe.txt").read_text()


@pytest.mark.parametrize("fixture", ["getevent_pl.txt", "getevent_pi.txt"])
def test_parse_touch_event_node_picks_the_multitouch_panel(fixture):
    """Not the fingerprint reader, not the haptics, not the power keys.

    ``-pl`` labels the axes; ``-pi`` prints raw hex codes. Both must resolve to
    the same node - the phone in question lists five input devices, of which the
    fingerprint reader is also a ``goodix`` device.
    """
    out = (DATA / fixture).read_text()
    assert parse_touch_event_node(out) == "/dev/input/event3"


def test_parse_touch_event_node_none_when_no_multitouch():
    assert parse_touch_event_node("add device 1: /dev/input/event0\n  KEY (0001)\n") is None


def test_parse_touch_event_node_ignores_a_key_scancode_0035():
    """0x35 is ABS_MT_POSITION_X but also KEY_SLASH; only the ABS block counts."""
    text = (
        "add device 1: /dev/input/event0\n"
        "    KEY (0001): 0035  0036 \n"
        "add device 2: /dev/input/event9\n"
        "    ABS (0003): 0035  : value 0, min 0, max 1079\n"
    )
    assert parse_touch_event_node(text) == "/dev/input/event9"


def test_estimate_report_rate_matches_the_measured_panel(swipe):
    rate = estimate_report_rate_hz(swipe)
    assert rate is not None
    # goodix_ts0 reports at ~183 Hz; allow a little slack for fixture slicing.
    assert 175.0 < rate < 192.0


def test_rate_is_not_fooled_by_position_axis_counting(swipe):
    """Regression: the old parser matched ``ABS_MT_POSITION`` substrings.

    A moving finger emits both ``_X`` and ``_Y`` per frame (double count) while a
    stationary one emits neither (dropped frame), so the position axes are not a
    frame counter. SYN_REPORT is. Assert the fixture actually exercises both
    hazards, so this test would fail if someone reverted the counting rule.
    """
    lines = swipe.splitlines()
    syn = sum(1 for line in lines if "SYN_REPORT" in line)
    pos = sum(1 for line in lines if "ABS_MT_POSITION" in line)
    assert pos != syn, "fixture must not have a 1:1 position/frame ratio"

    frames_without_position = 0
    seen_pos = False
    for line in lines:
        if "ABS_MT_POSITION" in line:
            seen_pos = True
        elif "SYN_REPORT" in line:
            if not seen_pos:
                frames_without_position += 1
            seen_pos = False
    assert frames_without_position > 0, "fixture must contain a stationary-finger frame"


def test_rate_ignores_idle_gaps_between_gestures():
    """``(n-1)/span`` over a capture with idle time reads far too low."""
    lines = []
    t = 100.0
    for burst in range(2):
        for _ in range(30):
            lines.append(f"[{t:15.6f}] EV_SYN       SYN_REPORT           00000000")
            t += 1 / 200.0
        t += 5.0  # a long human pause between swipes
    rate = estimate_report_rate_hz("\n".join(lines))
    assert rate == pytest.approx(200.0, rel=0.02)


def test_rate_survives_a_dropped_frame():
    lines = []
    t = 0.0
    for i in range(60):
        lines.append(f"[{t:15.6f}] EV_SYN       SYN_REPORT           00000000")
        t += (2 / 180.0) if i == 30 else (1 / 180.0)  # one skipped report
    assert estimate_report_rate_hz("\n".join(lines)) == pytest.approx(180.0, rel=0.02)


def test_rate_none_without_a_swipe():
    assert estimate_report_rate_hz("") is None
    assert estimate_report_rate_hz("[  1.0] EV_SYN SYN_REPORT 00000000\n") is None


# --- upsert_toml_scalar ------------------------------------------------------

def test_upsert_creates_missing_table():
    out = upsert_toml_scalar("[criteria]\nmode = \"casual\"\n", "motor", "report_rate_hz", "183")
    assert "[motor]" in out and "report_rate_hz = 183" in out


def test_upsert_replaces_in_place_and_stays_valid_toml():
    """Regression: `hop calibrate` used to append, emitting a duplicate table."""
    import tomllib

    text = "[motor]\nreport_rate_hz = 180\n\n[criteria]\nmode = \"casual\"\n"
    once = upsert_toml_scalar(text, "motor", "report_rate_hz", "183")
    twice = upsert_toml_scalar(once, "motor", "report_rate_hz", "183")
    assert once == twice                       # idempotent
    assert once.count("[motor]") == 1          # never a second header
    parsed = tomllib.loads(twice)
    assert parsed["motor"]["report_rate_hz"] == 183
    assert parsed["criteria"]["mode"] == "casual"   # neighbours untouched


def test_upsert_preserves_comments_and_appends_provenance():
    text = "[contact]\n# measured by hand\npressure_semantics = \"ramp\"\n"
    out = upsert_toml_scalar(text, "contact", "contact_major_peak", "0.5", comment="LIVE-VERIFIED")
    assert "# measured by hand" in out
    assert "contact_major_peak = 0.5    # LIVE-VERIFIED" in out


def test_upsert_adds_key_to_existing_table_before_next_table():
    import tomllib

    text = "[motor]\nreport_rate_hz = 180\n\n[criteria]\nmode = \"casual\"\n"
    out = upsert_toml_scalar(text, "motor", "tap_dwell_median_s", "0.08")
    parsed = tomllib.loads(out)
    assert parsed["motor"]["tap_dwell_median_s"] == 0.08
    assert parsed["criteria"]["mode"] == "casual"
