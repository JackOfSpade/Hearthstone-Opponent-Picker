"""Tests for `hop doctor`'s touchscreen-identity parser (CALIBRATION.md §4).

The parser feeds panel-matched [uhid] cloning. The format below is the real
`dumpsys input` layout on a Pixel 7a / Android 17 (SDK 37): per-device blocks
headed by `<n>: <name>`, a `Classes:` line, and an `Identifier:` line with
`vendor=0x..`/`product=0x..`. The pre-2.0.1 parser grepped for a single line
containing capitalized `Vendor`+`Product` and found nothing here.
"""

from hop.cli import parse_touch_identity

# Trimmed but structurally faithful capture from the real device.
PIXEL_7A_DUMP = """\
INPUT MANAGER (dumpsys input)

Input Reader State:
  Devices:
    0: cs40l26_input
      Classes: KEYBOARD | VIBRATOR
      Identifier: bus=0x0000, vendor=0x0000, product=0x0000, version=0x0000, bluetoothAddress=<not set>
    2: goodix_ts0
      Classes: KEYBOARD | TOUCH | TOUCH_MT
      Path: /dev/input/event3
      Enabled: true
      Descriptor: dfe9ddb6d53e7403f4ca6bdde7c57ae645841fc3
      Location: goodix_ts0
      ControllerNumber: 0
      UniqueId: google_touchscreen
      Identifier: bus=0x0001, vendor=0x27c6, product=0x0100, version=0x0100, bluetoothAddress=<not set>
    5: goodix_fingerprint
      Classes: KEYBOARD
      Identifier: bus=0x0018, vendor=0x0001, product=0x0001, version=0x0100, bluetoothAddress=<not set>
"""


def test_parses_pixel_7a_touchscreen_identity():
    ident = parse_touch_identity(PIXEL_7A_DUMP)
    assert ident is not None
    assert ident["name"] == "goodix_ts0"
    assert ident["vendor"] == 0x27C6
    assert ident["product"] == 0x0100
    assert ident["version"] == 0x0100
    assert ident["bus"] == 0x0001
    assert ident["multitouch"] is True


def test_ignores_non_touch_and_fingerprint_devices():
    """A KEYBOARD-only vibrator and a fingerprint 'sensor' must not win."""
    ident = parse_touch_identity(PIXEL_7A_DUMP)
    assert ident["name"] == "goodix_ts0"  # not cs40l26_input, not goodix_fingerprint


def test_prefers_multitouch_nonzero_vendor():
    """Given two TOUCH devices, prefer TOUCH_MT with a real (non-zero) vendor."""
    dump = """\
  Devices:
    1: virtual_touch
      Classes: TOUCH
      Identifier: bus=0x0000, vendor=0x0000, product=0x0000, version=0x0000
    2: real_panel
      Classes: TOUCH | TOUCH_MT
      Identifier: bus=0x0018, vendor=0x2a94, product=0x00c1, version=0x0100
"""
    ident = parse_touch_identity(dump)
    assert ident["name"] == "real_panel"
    assert ident["vendor"] == 0x2A94


def test_legacy_single_line_format_still_parsed():
    """Older Android printed Vendor/Product capitalized on one line."""
    dump = "    Touchscreen: Vendor: 0x0596 Product: 0x0501 blah\n"
    ident = parse_touch_identity(dump)
    assert ident is not None
    assert ident["vendor"] == 0x0596
    assert ident["product"] == 0x0501


def test_returns_none_when_no_touchscreen():
    dump = """\
  Devices:
    0: some_keyboard
      Classes: KEYBOARD | ALPHAKEY
      Identifier: bus=0x0000, vendor=0x0000, product=0x0000, version=0x0000
"""
    assert parse_touch_identity(dump) is None
