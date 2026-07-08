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

    # opponent is bottom-left; class label is the line under the portrait
    opponent_class_region: Region = Region(0.03, 0.90, 0.20, 0.07)
    # the band across the mulligan card row where the mana gems sit (top of cards)
    mana_gem_row: Region = Region(0.14, 0.24, 0.72, 0.10)
    # the whole card row, for the peak-count fallback
    card_row: Region = Region(0.12, 0.24, 0.76, 0.45)

    # action points (center + hit radius, screen fractions)
    # LIVE-VERIFY (Pixel 7a, 2400x1080 landscape): the Play button lives on the
    # deck-detail screen (ScreenState.PLAY_SCREEN) at the lower right of the deck
    # panel - not centered at the bottom. Hearthstone returns here after a game.
    play_button: Point = Point(0.728, 0.85, 0.025)
    # LIVE-VERIFY (Pixel 7a, 2400x1080 landscape): the mulligan "Confirm" button
    # sits at ~y0.85, not 0.92 (measured from a real Starting-Hand screen).
    mulligan_confirm: Point = Point(0.50, 0.85, 0.05)
    gear_button: Point = Point(0.965, 0.05, 0.03)
    concede_button: Point = Point(0.50, 0.42, 0.06)
    concede_confirm: Point = Point(0.50, 0.56, 0.06)
    end_dismiss: Point = Point(0.50, 0.90, 0.10)
    pass_turn_button: Point = Point(0.92, 0.50, 0.05)

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


def count_mulligan_cards(
    frame: Frame,
    layout: GameLayout,
    gem_template: Template | None = None,
) -> int:
    """Return the number of mulligan cards (expected 3 or 4).

    Primary: count mana-gem matches across the gem-row band (each card has
    exactly one blue mana gem; gems never merge). Fallback (no template): count
    brightness "humps" across the card row - cards are bright/colorful, the gaps
    between them and the table beyond are dark.
    """
    if gem_template is not None:
        matches: list[Match] = match_all(frame, gem_template, stride=2)
        return len(matches)
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
    gem_template: Template | None = None,
) -> MulliganRead:
    """Extract opponent class + going-second from a mulligan frame."""
    rx, ry, rw, rh = layout.opponent_class_region.to_px(frame)
    label = frame.crop(rx, ry, rw, rh)
    cr: ClassRead = reader.read(label)
    num = count_mulligan_cards(frame, layout, gem_template)
    return MulliganRead(
        opponent_class=cr.hero_class,
        we_go_second=(num >= 4),
        num_cards=num,
        class_confidence=cr.confidence,
        class_raw=cr.raw_text,
        method=cr.method,
    )
