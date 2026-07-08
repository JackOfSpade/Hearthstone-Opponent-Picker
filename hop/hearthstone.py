"""Hearthstone-specific perception glue and screen coordinates.

Everything that "knows about Hearthstone" lives here: where the opponent's class
label sits, how to count mulligan cards, and the (calibratable) screen-fraction
locations of the buttons the engine taps. Per the standard, none of these are
hardcoded pixel coordinates in the engine - they are fractions here, overridable
by ``hop calibrate`` and always resolved against the live panel size, so the
tool is resolution-independent. Every default fraction is marked LIVE-VERIFY
because it was estimated from mobile mulligan screenshots, not measured on your
exact device.

Class detection (primary signal): OCR the bottom-left class label and snap to 11
(:mod:`hop.perception.ocr`). Going-second (secondary signal): **count mulligan
cards** - 3 = we go first, 4 = we go second - never by finding the Coin (the
Coin isn't in the mulligan hand, and coin skins vary). Card counting is pure
pixel math (mana-gem NCC count, with a brightness peak-count fallback): no ML.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import PanelGeometry
from .hero_classes import HeroClass
from .perception.image import Frame
from .perception.ocr import ClassRead, ClassReader
from .perception.templates import Match, Region, Template, match_all


@dataclass(frozen=True)
class Point:
    """A calibratable action location: center + hit radius, in screen fractions."""

    xf: float
    yf: float
    radius_f: float = 0.02   # hit radius as a fraction of screen width

    def to_px(self, panel: PanelGeometry) -> tuple[float, float, float]:
        return (self.xf * panel.width_px, self.yf * panel.height_px, self.radius_f * panel.width_px)


@dataclass(frozen=True)
class GameLayout:
    """Screen-fraction geometry of the Hearthstone mobile UI. LIVE-VERIFY all."""

    # opponent is bottom-left; class label is the line under the portrait.
    # LIVE-VERIFIED: reads the OPPONENT's class (you are bottom-right).
    opponent_class_region: Region = Region(0.03, 0.90, 0.20, 0.07)
    # the band across the mulligan card row where the mana gems sit (top of cards)
    # LIVE-VERIFY (2400x1080): gems occupy y ~300-400.
    mana_gem_row: Region = Region(0.12, 0.275, 0.76, 0.10)
    # a single mana gem's width as a fraction of screen width (~86px @ 2400 wide).
    # Used to reject blue *card art* (far wider) from the gem mask. LIVE-VERIFY.
    gem_width_f: float = 0.0358
    # the whole card row, for the brightness fallback
    card_row: Region = Region(0.12, 0.24, 0.76, 0.45)

    # action points (center + hit radius, screen fractions)
    # LIVE-VERIFY (Pixel 7a, 2400x1080 landscape): the Play button lives on the
    # deck-detail screen (ScreenState.PLAY_SCREEN) at the lower right of the deck
    # panel - not centered at the bottom. Hearthstone returns here after a game.
    play_button: Point = Point(0.728, 0.85, 0.025)
    # All LIVE-VERIFIED on a Pixel 7a (2400x1080 landscape). Hit radii are set from
    # the control's SMALLER half-dimension, because the FFitts endpoint spread is
    # isotropic and truncated to 0.9*radius - an over-large radius throws taps off
    # short, wide buttons (which is how the old values missed).
    # measured from the Confirm button's blue glow centroid on three real mulligan
    # frames (both 3-card and 4-card hands agree: 0.5035, 0.876 +/- 0.002)
    mulligan_confirm: Point = Point(0.5035, 0.876, 0.0125)
    gear_button: Point = Point(0.935, 0.037, 0.0125)
    # In the in-game "Game Menu" the order is Concede / Options / Quit. The old
    # y=0.42 landed between Options and Quit; Concede is the TOP entry at y~0.196.
    concede_button: Point = Point(0.5025, 0.196, 0.014)
    # This client concedes immediately with no confirmation dialog; _concede() only
    # taps this if a concede menu is still classified afterwards.
    concede_confirm: Point = Point(0.50, 0.56, 0.06)
    end_dismiss: Point = Point(0.50, 0.90, 0.10)
    pass_turn_button: Point = Point(0.80, 0.497, 0.012)
    # Recovery: the first deck on the deck-select list, and the OK button of
    # Hearthstone's "There was an error starting your game." dialog (a frequent,
    # transient network blip - dismiss and requeue; no long backoff needed).
    deck_slot: Point = Point(0.286, 0.289, 0.02)
    error_ok: Point = Point(0.495, 0.667, 0.02)
    # "You are currently offline" - Hearthstone shuts down an idle connection.
    # LIVE-MEASURED from the gold-button mask on a real dialog: the Reconnect and
    # Cancel buttons are each 300x92 px, centered at x=982 / x=1394 of 2400.
    # Tap Reconnect (LEFT). Cancel leaves the client offline, and from there every
    # subsequent tap silently does nothing - the worst possible failure mode.
    reconnect_button: Point = Point(0.4092, 0.8380, 0.0190)

    def card_slot(self, slot: int, num_cards: int, panel: PanelGeometry) -> Point:
        """Center of mulligan card ``slot`` given the layout has ``num_cards``.

        3-card and 4-card mulligans are centered differently; we lay slots out
        evenly across the card row so a keep/replace tap lands on the right card
        regardless of count.
        """
        left, right = 0.20, 0.80
        if num_cards <= 1:
            xf = 0.5
        else:
            span = right - left
            xf = left + span * (slot / (num_cards - 1))
        return Point(xf, 0.45, 0.05)


@dataclass(frozen=True)
class MulliganRead:
    """The two signals extracted from a mulligan frame."""

    opponent_class: HeroClass | None
    we_go_second: bool
    num_cards: int
    class_confidence: float
    class_raw: str
    method: str

    @property
    def usable(self) -> bool:
        return self.opponent_class is not None and self.num_cards in (3, 4)


def _count_mana_gems(frame: Frame, layout: GameLayout, vision) -> int | None:
    """Count blue mana gems in the gem row; ``None`` if colour is unavailable.

    Each mulligan card carries exactly one blue mana gem and gems never merge, so
    they are the reliable count. We mask pixels whose blue channel dominates red
    and green, project the mask onto columns, and keep only runs whose width
    matches a gem. That width filter is what makes this robust: blue *card art*
    also masks blue, but produces runs several times wider (measured 254/269px
    vs a gem's 86px), and highlights produce slivers.

    NCC-matching a gem *template* does not work here - the gem's digit differs
    per card (3/2/9/7), which destroys the correlation.
    """
    rgb = getattr(frame, "rgb", None)
    if rgb is None:
        return None
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a vision extra
        return None

    rx, ry, rw, rh = layout.mana_gem_row.to_px(frame)
    band = rgb[ry:ry + rh, rx:rx + rw]
    if band.size == 0:
        return None
    r = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    b = band[:, :, 2].astype(np.int16)
    bias = vision.gem_blue_bias
    mask = (b > r + bias) & (b > g + bias) & (b > vision.gem_min_blue)

    col = mask.sum(axis=0)
    peak = int(col.max()) if col.size else 0
    if peak <= 0:
        return 0
    thresh = max(1, int(peak * vision.gem_col_min_frac))

    expected = layout.gem_width_f * frame.width
    lo = expected * (1.0 - vision.gem_width_tolerance)
    hi = expected * (1.0 + vision.gem_width_tolerance)

    count = run = 0
    for v in col:
        if v >= thresh:
            run += 1
        else:
            if lo <= run <= hi:
                count += 1
            run = 0
    if lo <= run <= hi:
        count += 1
    return count


def count_mulligan_cards(
    frame: Frame,
    layout: GameLayout,
    vision=None,
    gem_template: Template | None = None,
) -> int:
    """Return the number of mulligan cards (expected 3 or 4).

    Primary: count the blue mana gems (:func:`_count_mana_gems`) - one per card,
    never merging. Fallback: count brightness "humps" across the card row. The
    fallback is **unreliable** (a dark-art card can sit entirely below the
    brightness threshold and be skipped, which is how a 4-card hand was once
    counted as 3), so it is used only when colour is unavailable or the gem count
    is implausible; an implausible count then fails closed via ``MulliganRead``.
    """
    if gem_template is not None:
        matches: list[Match] = match_all(frame, gem_template, stride=2)
        return len(matches)
    if vision is not None:
        n = _count_mana_gems(frame, layout, vision)
        if n in (3, 4):
            return n
    return _peak_count_cards(frame, layout.card_row)


def _peak_count_cards(frame: Frame, region: Region) -> int:
    """Count cards by projecting brightness across the row and counting humps."""
    rx, ry, rw, rh = region.to_px(frame)
    band = frame.crop(rx, ry, rw, rh)
    # column-mean brightness profile
    cols: list[float] = []
    for x in range(band.width):
        s = 0
        for y in range(0, band.height, 3):  # subsample rows for speed
            s += band.get(x, y)
        cols.append(s / max(1, (band.height + 2) // 3))
    if not cols:
        return 0
    lo, hi = min(cols), max(cols)
    if hi - lo < 8:
        return 0
    thresh = lo + (hi - lo) * 0.45
    # count contiguous runs above threshold wider than a min card width
    min_run = max(4, band.width // 12)
    runs = 0
    run_len = 0
    for v in cols:
        if v >= thresh:
            run_len += 1
        else:
            if run_len >= min_run:
                runs += 1
            run_len = 0
    if run_len >= min_run:
        runs += 1
    return runs


def read_mulligan(
    frame: Frame,
    layout: GameLayout,
    reader: ClassReader,
    vision=None,
    gem_template: Template | None = None,
) -> MulliganRead:
    """Extract opponent class + going-second from a mulligan frame.

    ``vision`` is the :class:`~hop.config.VisionConfig`; it carries the mana-gem
    thresholds used to count cards. Without it we fall back to the unreliable
    brightness counter, so the engine always passes it.
    """
    rx, ry, rw, rh = layout.opponent_class_region.to_px(frame)
    label = frame.crop(rx, ry, rw, rh)
    cr: ClassRead = reader.read(label)
    num = count_mulligan_cards(frame, layout, vision, gem_template)
    return MulliganRead(
        opponent_class=cr.hero_class,
        we_go_second=(num >= 4),
        num_cards=num,
        class_confidence=cr.confidence,
        class_raw=cr.raw_text,
        method=cr.method,
    )
