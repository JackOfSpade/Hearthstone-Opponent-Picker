"""Region-split diffing.

Standard: *"Region-split diffing distinguishes kinds of change: a bottom sheet
sliding up (bottom changes, top stays) vs a full transition (top changes too).
This is how observe mode infers the human's tap-vs-scroll-vs-transition from
pixels alone."*

Given two frames and a set of named regions, this returns the mean-abs-diff per
region so the caller can classify the *kind* of change, not just its presence.
"""

from __future__ import annotations

from .image import Frame, mean_abs_diff
from .templates import Region


def region_diffs(prev: Frame, cur: Frame, regions: dict[str, Region]) -> dict[str, float]:
    """Mean absolute grayscale difference within each named region."""
    out: dict[str, float] = {}
    for name, reg in regions.items():
        px, py, pw, ph = reg.to_px(prev)
        cx, cy, cw, ch = reg.to_px(cur)
        w, h = min(pw, cw), min(ph, ch)
        a = prev.crop(px, py, w, h)
        b = cur.crop(cx, cy, w, h)
        out[name] = mean_abs_diff(a, b)
    return out


# The canonical top/middle/bottom split used to classify change kinds.
THIRDS: dict[str, Region] = {
    "top": Region(0.0, 0.0, 1.0, 0.33),
    "middle": Region(0.0, 0.33, 1.0, 0.34),
    "bottom": Region(0.0, 0.67, 1.0, 0.33),
}


def classify_change(prev: Frame, cur: Frame, threshold: float) -> str:
    """Coarse change classification from thirds.

    Returns one of: ``none`` (nothing moved), ``bottom_sheet`` (only the bottom
    changed), ``top_banner`` (only the top), or ``full_transition`` (top+middle
    moved). Used in observe mode and to sanity-check that a tap produced the
    *kind* of change we expected (Layer 6 semantic check).
    """
    d = region_diffs(prev, cur, THIRDS)
    top, mid, bot = d["top"] > threshold, d["middle"] > threshold, d["bottom"] > threshold
    if not (top or mid or bot):
        return "none"
    if bot and not top and not mid:
        return "bottom_sheet"
    if top and not mid and not bot:
        return "top_banner"
    if top and mid:
        return "full_transition"
    return "partial"
