"""Display-rotation transforms between two coordinate spaces.

Hearthstone runs **landscape** while the phone's panel is native **portrait**,
so screencap frames (what perception + ``GameLayout`` fractions use) are in the
rotated *display* space, while the persistent UHID digitizer reports raw
coordinates in the panel's *native* space. The Android input framework rotates
our virtual touchscreen's raw coordinates by the live display rotation before
dispatch, so to land a tap at a display target we must send the *native* point
that rotates back to it.

We therefore synthesize gestures in display space (the space the "human" sees
and aims in) and map each sample to native panel pixels here, keyed on the
current rotation. This is robust to BOTH landscape orientations (ROTATION_90 and
ROTATION_270), which Hearthstone can switch between.

Rotation constants follow ``android.view.Surface``: 0, 1 (=90 CW), 2 (=180),
3 (=270). Verified on-device (Pixel 7a, ROTATION_270): a native tap at
(540, 1200) lands at display (1199, 540) - i.e. native (Nx,Ny) maps to display
(Lx,Ly) with ``Lx = (native_h-1) - Ny`` and ``Ly = Nx``; this module is the
inverse (display -> native).
"""

from __future__ import annotations

ROT_0, ROT_90, ROT_180, ROT_270 = 0, 1, 2, 3
_LANDSCAPE = (ROT_90, ROT_270)


def display_size(native_w: int, native_h: int, rotation: int) -> tuple[int, int]:
    """Display (width, height) for a native panel of ``native_w x native_h`` at
    ``rotation``. Landscape rotations swap the axes."""
    return (native_h, native_w) if rotation in _LANDSCAPE else (native_w, native_h)


def display_to_native(x: float, y: float, rotation: int,
                      native_w: int, native_h: int) -> tuple[float, float]:
    """Map a DISPLAY-space point ``(x, y)`` to NATIVE panel pixels.

    ``native_w``/``native_h`` are the panel's native (rotation-0) dimensions.
    The inverse of the framework's raw->display rotation, so a gesture aimed at
    a display coordinate is delivered there once the framework rotates our
    virtual device's report.
    """
    if rotation == ROT_90:
        return (native_w - 1 - y, x)
    if rotation == ROT_180:
        return (native_w - 1 - x, native_h - 1 - y)
    if rotation == ROT_270:
        return (y, native_h - 1 - x)
    return (x, y)  # ROT_0 (or unknown): identity
