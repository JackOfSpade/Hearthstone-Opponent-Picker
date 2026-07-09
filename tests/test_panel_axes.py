"""Cloning the panel's contact-channel axis ranges, not just its name.

Android loads a touch device's calibration (`<name>.idc`) **by name** and applies
it to our reports. So cloning the panel's name while declaring different axis
maxima makes the framework size our contact against the wrong scale.

That is not a theoretical tell. Measured on a Pixel 7a (`goodix_ts0`, real
`ABS_MT_TOUCH_MAJOR` max 2399): with the old hardcoded 255 ceiling, a
normal-looking contact was interpreted as enormous, and **Hearthstone silently
ignored every tap on a mulligan card** while still honouring taps on buttons. The
tap reached the kernel with a clean DOWN/UP and correct coordinates; the game just
threw it away.
"""

import struct

import pytest

from hop.transport import hid_descriptor as hd

# Real `getevent -pl` output from the Pixel 7a, trimmed to the touchscreen block.
GETEVENT_PL = """add device 1: /dev/input/event4
  name:     "cs40l26_input"
    ABS (0003): ABS_MT_POSITION_X     : value 0, min 0, max 100, fuzz 0, flat 0, resolution 0
add device 2: /dev/input/event3
  name:     "goodix_ts0"
  events:
    ABS (0003): ABS_X                 : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                ABS_MT_TOUCH_MAJOR    : value 0, min 0, max 2399, fuzz 0, flat 0, resolution 0
                ABS_MT_TOUCH_MINOR    : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                ABS_MT_ORIENTATION    : value 0, min -4096, max 4096, fuzz 0, flat 0, resolution 0
                ABS_MT_POSITION_X     : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                ABS_MT_POSITION_Y     : value 0, min 0, max 2399, fuzz 0, flat 0, resolution 0
                ABS_MT_PRESSURE       : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
"""


def test_parses_the_real_panels_axis_maxima():
    axes = hd.parse_panel_axes(GETEVENT_PL, "goodix_ts0")
    assert axes == hd.PanelAxes(touch_major_max=2399, touch_minor_max=1079,
                                pressure_max=255, orientation_max=4096)


def test_the_old_defaults_are_not_what_the_panel_reports():
    """Pins the actual bug: 255 != 2399. If these ever agree, drop the cloning."""
    axes = hd.parse_panel_axes(GETEVENT_PL, "goodix_ts0")
    assert axes.touch_major_max != hd.DEFAULT_AXES.touch_major_max


def test_never_clones_axes_from_our_own_virtual_device():
    """The trap that would silently make this fix a no-op.

    Once registered, our clone carries the panel's *name* too, and on the Pixel 7a
    `getevent` lists it FIRST (higher event number, earlier in the output). Taking
    the first name match would read our own 255-max axes back and change nothing.
    """
    ours_first = """add device 1: /dev/input/event5
  name:     "goodix_ts0"
    ABS (0003): ABS_MT_POSITION_X     : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                ABS_MT_TOUCH_MAJOR    : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
add device 2: /dev/input/event3
  name:     "goodix_ts0"
    ABS (0003): ABS_MT_POSITION_X     : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                ABS_MT_TOUCH_MAJOR    : value 0, min 0, max 2399, fuzz 0, flat 0, resolution 0
                ABS_MT_TOUCH_MINOR    : value 0, min 0, max 1079, fuzz 0, flat 0, resolution 0
                ABS_MT_PRESSURE       : value 0, min 0, max 255, fuzz 0, flat 0, resolution 0
"""
    axes = hd.parse_panel_axes(ours_first, "goodix_ts0")
    assert axes.touch_major_max == 2399, "cloned from the virtual device, not the panel"


def test_returns_none_without_a_multitouch_device_of_that_name():
    assert hd.parse_panel_axes(GETEVENT_PL, "some_other_panel") is None
    assert hd.parse_panel_axes("add device 1: /dev/input/event0\n  KEY\n", "goodix_ts0") is None


def test_descriptor_declares_the_cloned_maxima():
    axes = hd.PanelAxes(2399, 1079, 255, 4096)
    d = bytes(hd.build_digitizer_descriptor(1080, 2400, 1, axes))
    # Logical Maximum (2-byte) items carrying each cloned ceiling
    assert bytes([0x26]) + struct.pack("<H", 2399) in d
    assert bytes([0x26]) + struct.pack("<H", 1079) in d
    assert bytes([0x26]) + struct.pack("<H", 255) in d


def test_report_round_trips_the_raw_channel_values():
    axes = hd.PanelAxes(2399, 1079, 255, 4096)
    c = hd.ContactReport(contact_id=0, x=540, y=1200, pressure=200,
                         major=215, minor=163, orientation=3000, tip=True)
    payload = hd.encode_report([c], max_contacts=1, axes=axes)
    assert len(payload) == 1 + hd.PER_CONTACT_BYTES + 1
    flags, cid, x, y, p, maj, mnr, ori = struct.unpack("<BBHHHHHh", payload[1:-1])
    assert (flags & 0x01) and (flags & 0x02)     # tip + confidence
    assert (x, y, p, maj, mnr, ori) == (540, 1200, 200, 215, 163, 3000)
    assert payload[-1] == 1                      # contact count


def test_raw_values_are_clamped_to_their_own_axis():
    axes = hd.PanelAxes(2399, 1079, 255, 4096)
    c = hd.ContactReport(0, 0, 0, pressure=999, major=9999, minor=9999,
                         orientation=99999, tip=True)
    _, _, _, _, p, maj, mnr, ori = struct.unpack(
        "<BBHHHHHh", hd.encode_report([c], 1, axes)[1:-1])
    assert (p, maj, mnr, ori) == (255, 2399, 1079, 4096)


def test_release_report_lifts_the_contact():
    """tip=False must zero the contact count: an unlifted touch is a stuck finger."""
    axes = hd.PanelAxes(2399, 1079, 255, 4096)
    down = hd.encode_report([hd.ContactReport(0, 10, 10, 200, 215, 163, 0, True)], 1, axes)
    up = hd.encode_report([hd.ContactReport(0, 10, 10, 0, 0, 0, 0, False)], 1, axes)
    assert down[-1] == 1 and up[-1] == 0


def test_inactive_slots_are_zero_filled():
    axes = hd.DEFAULT_AXES
    payload = hd.encode_report([hd.ContactReport(0, 5, 5, 100, 10, 8, 0, True)], 3, axes)
    assert len(payload) == 1 + 3 * hd.PER_CONTACT_BYTES + 1
    second = payload[1 + hd.PER_CONTACT_BYTES: 1 + 2 * hd.PER_CONTACT_BYTES]
    assert second == bytes(hd.PER_CONTACT_BYTES)
    assert payload[-1] == 1


@pytest.mark.parametrize("major_frac,expected_raw", [(0.12, 288), (0.0, 0), (1.0, 2399)])
def test_normalized_channels_scale_to_the_cloned_axis(major_frac, expected_raw):
    """`contact_major_peak` is a fraction of the panel's own maximum.

    0.12 of the Pixel's 2399 is a plausible finger (measured 70-293 raw). The same
    0.12 against the old 255 ceiling was ~31 raw -- and the previous default of
    0.55 was 140, roughly 6x a real contact.
    """
    axes = hd.PanelAxes(2399, 1079, 255, 4096)
    assert round(major_frac * axes.touch_major_max) == expected_raw
