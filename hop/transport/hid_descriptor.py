"""HID report descriptor + input-report encoding for a multitouch digitizer.

Standard, Layer 1: *"The descriptor must carry the full contact envelope -
single-finger-capable multitouch digitizer with Tip Switch / Confidence /
Contact ID / X / Y / Tip Pressure (0..255) / Touch-Major / Touch-Minor /
Orientation / Contact Count. Build X/Y logical maxima from the live screen size
so device coordinates map 1:1 to pixels. Omit the Contact-Count-Maximum feature
report - it triggers a kernel GET_FEATURE the HID stream can't answer and kills
the device mid-enumeration."*

The descriptor and per-contact report layout defined here are kept strictly in
sync (the byte offsets in :func:`encode_report` match the field order the
descriptor declares), so the encoder is fully unit-testable off-device. The
exact descriptor a given kernel accepts is best confirmed on the target phone
(the ``hid`` tool rejects a malformed descriptor at enumeration) - this is
marked LIVE-VERIFY, but the layout below is a conventional Windows-Precision-
Touchpad-style digitizer that most kernels accept.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass

# HID usage pages / usages we reference.
_UP_GENERIC = 0x01       # Generic Desktop
_UP_DIGITIZER = 0x0D     # Digitizers

MAX_CONTACTS = 5         # single-finger-capable, but supports incidental contacts
REPORT_ID = 0x01


@dataclass(frozen=True)
class PanelAxes:
    """Logical maxima of the contact channels, cloned from the real panel.

    Cloning the panel's *name* is not cosmetic: Android loads its input-device
    calibration (``<name>.idc``) by name and applies it to **our** reports. If we
    then declare different axis ranges, the framework sizes our contact against
    the wrong scale.

    That is not a theoretical tell. With a 255-max TOUCH_MAJOR against the Pixel
    7a's real 2399, a normal-looking contact was interpreted as enormous and
    Hearthstone silently ignored every tap on a mulligan card -- while still
    honouring taps on buttons. Clone the ranges too, and emit the raw values a
    real finger emits.
    """

    touch_major_max: int = 255
    touch_minor_max: int = 255
    pressure_max: int = 255
    orientation_max: int = 127     # symmetric; logical min is -orientation_max


DEFAULT_AXES = PanelAxes()

_ADD_DEVICE = re.compile(r"\s*add device \d+: (/dev/input/event\d+)")
_AXIS = re.compile(r"\s*(ABS_MT_\w+)\s*:.*?max (-?\d+)")


def parse_panel_axes(getevent_pl_text: str, device_name: str) -> PanelAxes | None:
    """Read the real panel's contact-channel maxima out of ``getevent -pl``.

    A device qualifies only if it carries ``device_name`` *and* reports
    ``ABS_MT_POSITION_X`` - a phone lists several input devices, and on this one the
    fingerprint reader is also a "goodix".

    Among qualifiers we take the **lowest event number**. Once our virtual clone is
    registered it shares the panel's name, and it enumerates with a higher event
    number - yet is *listed first* by ``getevent`` on this phone. Taking the first
    match would clone our own (wrong) axes back onto ourselves, quietly turning this
    whole fix into a no-op.
    """
    blocks: list[tuple[int, list[str]]] = []
    index: int | None = None
    block: list[str] = []
    for line in getevent_pl_text.splitlines():
        m = _ADD_DEVICE.match(line)
        if m:
            if index is not None:
                blocks.append((index, block))
            index, block = int(re.sub(r"\D", "", m.group(1))), [line]
        elif index is not None:
            block.append(line)
    if index is not None:
        blocks.append((index, block))

    def qualifies(lines: list[str]) -> bool:
        return (any("ABS_MT_POSITION_X" in ln for ln in lines)
                and (not device_name or any(f'"{device_name}"' in ln for ln in lines)))

    candidates = [(i, b) for i, b in blocks if qualifies(b)]
    if not candidates:
        return None
    _, lines = min(candidates, key=lambda ib: ib[0])

    found: dict[str, int] = {}
    for line in lines:
        m = _AXIS.match(line)
        if m:
            found[m.group(1)] = int(m.group(2))
    if "ABS_MT_TOUCH_MAJOR" not in found:
        return None
    return PanelAxes(
        touch_major_max=found.get("ABS_MT_TOUCH_MAJOR", 255),
        touch_minor_max=found.get("ABS_MT_TOUCH_MINOR", found.get("ABS_MT_TOUCH_MAJOR", 255)),
        pressure_max=found.get("ABS_MT_PRESSURE", 255),
        orientation_max=abs(found.get("ABS_MT_ORIENTATION", 127)) or 127,
    )


@dataclass(frozen=True)
class ContactReport:
    """One contact's state for a single input report."""

    contact_id: int
    x: int              # logical (== device pixel), 0..width-1
    y: int              # 0..height-1
    pressure: int       # raw, 0..axes.pressure_max
    major: int          # raw, 0..axes.touch_major_max
    minor: int          # raw, 0..axes.touch_minor_max
    orientation: int    # raw, -axes.orientation_max..+axes.orientation_max
    tip: bool
    confidence: bool = True


def build_digitizer_descriptor(width_px: int, height_px: int, max_contacts: int = MAX_CONTACTS,
                               axes: PanelAxes = DEFAULT_AXES) -> list[int]:
    """Return the HID report descriptor as a list of unsigned bytes.

    One top-level Digitizer/Touch Screen application collection containing
    ``max_contacts`` finger collections (each: Tip Switch, Confidence, Contact
    Id, X, Y, Tip Pressure, Touch Major, Touch Minor, Orientation) plus a
    trailing Contact Count byte. **No Contact-Count-Maximum feature report** is
    emitted, per the standard, so enumeration never blocks on a GET_FEATURE.
    """

    def item(tag_type: int, data: int | None = None, size: int = 1) -> list[int]:
        # short item: prefix byte = (tag<<4)|(type<<2)|size_code
        if data is None:
            return [tag_type]
        if size == 1:
            return [tag_type | 0x01, data & 0xFF]
        if size == 2:
            return [tag_type | 0x02, data & 0xFF, (data >> 8) & 0xFF]
        raise ValueError(size)

    d: list[int] = []
    # Usage Page (Digitizers), Usage (Touch Screen), Collection (Application)
    d += [0x05, _UP_DIGITIZER]
    d += [0x09, 0x04]           # Usage (Touch Screen)
    d += [0xA1, 0x01]           # Collection (Application)
    d += [0x85, REPORT_ID]      # Report ID

    for _ in range(max_contacts):
        d += [0x09, 0x22]       # Usage (Finger)
        d += [0xA1, 0x02]       # Collection (Logical)
        # Tip Switch + Confidence (2 x 1-bit)
        d += [0x09, 0x42]       # Usage (Tip Switch)
        d += [0x09, 0x47]       # Usage (Confidence)
        d += [0x15, 0x00]       # Logical Min 0
        d += [0x25, 0x01]       # Logical Max 1
        d += [0x75, 0x01]       # Report Size 1
        d += [0x95, 0x02]       # Report Count 2
        d += [0x81, 0x02]       # Input (Data,Var,Abs)
        # padding 6 bits
        d += [0x75, 0x06]       # Report Size 6
        d += [0x95, 0x01]       # Report Count 1
        d += [0x81, 0x03]       # Input (Const)
        # Contact Identifier (1 byte)
        d += [0x09, 0x51]       # Usage (Contact Identifier)
        d += [0x25, 0x1F]       # Logical Max 31
        d += [0x75, 0x08]       # Report Size 8
        d += [0x95, 0x01]       # Report Count 1
        d += [0x81, 0x02]       # Input
        # X, Y (2 bytes each), logical max = panel size - 1
        d += [0x05, _UP_GENERIC]
        d += [0x15, 0x00]       # Logical Min 0
        d += [0x09, 0x30]       # Usage (X)
        d += [0x26] + list(struct.pack("<H", max(1, width_px - 1)))   # Logical Max (2 byte)
        d += [0x75, 0x10]       # Report Size 16
        d += [0x95, 0x01]
        d += [0x81, 0x02]
        d += [0x09, 0x31]       # Usage (Y)
        d += [0x26] + list(struct.pack("<H", max(1, height_px - 1)))
        d += [0x81, 0x02]
        # back to Digitizers for pressure/size/orientation
        # Pressure / major / minor / orientation are all 16-bit so the panel's real
        # logical maxima (e.g. TOUCH_MAJOR 2399) fit; see PanelAxes.
        d += [0x05, _UP_DIGITIZER]
        d += [0x09, 0x30]       # Usage (Tip Pressure)
        d += [0x15, 0x00]
        d += [0x26] + list(struct.pack("<H", axes.pressure_max))
        d += [0x75, 0x10]       # Report Size 16
        d += [0x95, 0x01]
        d += [0x81, 0x02]
        d += [0x09, 0x48]       # Usage (Width / Touch Major)
        d += [0x26] + list(struct.pack("<H", axes.touch_major_max))
        d += [0x95, 0x01]
        d += [0x81, 0x02]
        d += [0x09, 0x49]       # Usage (Height / Touch Minor)
        d += [0x26] + list(struct.pack("<H", axes.touch_minor_max))
        d += [0x95, 0x01]
        d += [0x81, 0x02]
        d += [0x09, 0x3F]       # Usage (Azimuth / Orientation)
        d += [0x16] + list(struct.pack("<h", -axes.orientation_max))   # Logical Min
        d += [0x26] + list(struct.pack("<H", axes.orientation_max))    # Logical Max
        d += [0x95, 0x01]
        d += [0x81, 0x02]
        d += [0xC0]             # End Collection (finger)

    # Contact count (1 byte) at report end
    d += [0x09, 0x54]           # Usage (Contact Count)
    d += [0x15, 0x00]
    d += [0x25, max_contacts]
    d += [0x75, 0x08]
    d += [0x95, 0x01]
    d += [0x81, 0x02]
    d += [0xC0]                 # End Collection (application)
    return d


def _per_contact_bytes(c: ContactReport | None, axes: PanelAxes = DEFAULT_AXES) -> bytes:
    """14 bytes per contact, in the descriptor's field order:
    ``[flags:1][id:1][x:2][y:2][pressure:2][major:2][minor:2][orient:2]``
    where flags packs Tip Switch (bit0) + Confidence (bit1) + 6 pad bits.
    An inactive slot is 14 zero bytes."""
    if c is None:
        return bytes(PER_CONTACT_BYTES)
    flags = (0x01 if c.tip else 0x00) | (0x02 if c.confidence else 0x00)
    return struct.pack(
        "<BBHHHHHh",
        flags,
        c.contact_id & 0xFF,
        max(0, min(0xFFFF, c.x)),
        max(0, min(0xFFFF, c.y)),
        max(0, min(axes.pressure_max, c.pressure)),
        max(0, min(axes.touch_major_max, c.major)),
        max(0, min(axes.touch_minor_max, c.minor)),
        max(-axes.orientation_max, min(axes.orientation_max, c.orientation)),
    )


# per-contact fixed width used by encode_report (must match _per_contact_bytes)
PER_CONTACT_BYTES = 14


def encode_report(contacts: list[ContactReport], max_contacts: int = MAX_CONTACTS,
                  axes: PanelAxes = DEFAULT_AXES) -> bytes:
    """Encode one input report: report id, N contact slots, contact count.

    Inactive slots are zero-filled. The trailing byte is the number of active
    (tip-down) contacts, matching the Contact Count usage in the descriptor.
    """
    active = [c for c in contacts if c is not None]
    payload = bytes([REPORT_ID])
    for i in range(max_contacts):
        c = contacts[i] if i < len(contacts) else None
        payload += _per_contact_bytes(c, axes)
    payload += bytes([min(max_contacts, sum(1 for c in active if c.tip))])
    return payload
