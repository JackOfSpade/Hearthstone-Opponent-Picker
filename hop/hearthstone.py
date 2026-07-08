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
Coin isn't in the mulligan hand, and coin skins vary). Cards are counted by their
green **keep-glow strips** (N cards -> N+1 strips), cross-checked against the card
pitch: pure pixel math, no ML. The glow is UI chrome, so unlike the mana gems or a
brightness profile it cannot be confused by a card's artwork - see
:func:`count_mulligan_cards` for the two methods this replaced and why both were
wrong on real frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import PanelGeometry
from .hero_classes import HeroClass
from .perception.image import Frame
from .perception.ocr import ClassRead, ClassReader
from .perception.templates import Region


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
    # the whole mulligan card row: the band the keep-glow strips are counted in
    card_row: Region = Region(0.12, 0.24, 0.76, 0.45)
    # Width of a card's interior - the span between its left and right keep-glow -
    # as a fraction of screen width. LIVE-MEASURED on five real hands (one 3-card,
    # four 4-card, 2400x1080): 344-382 px, i.e. 0.143-0.159 W. Card size does not
    # depend on hand size. The background gap between two cards is 0.054 W, so the
    # two are separated by a factor of ~2.7 in width (and ~3 in brightness).
    card_inner_w_f: float = 0.147

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


def glow_strip_runs(frame: Frame, layout: GameLayout, vision) -> list[tuple[int, int]] | None:
    """``(start, end)`` columns of the keep-glow strips; ``None`` without colour.

    Every mulligan card is outlined by a saturated green "keep" glow. The glow is
    UI chrome, not artwork, which is the entire point: no card picture can imitate
    it. A strip is accepted only where the green mask covers most of the band's
    *height*, because the glow runs a card's full height and a patch of green art
    does not.

    Columns are relative to the card row's left edge, which is what
    :func:`count_mulligan_cards` needs to slice the band.

    Note that a strip is **not** the same thing as a card boundary. When cards sit
    close together their glows merge into one wide strip; when the hand is spread
    out each card contributes two separate strips. The strip *count* therefore does
    not determine the card count - see :func:`count_mulligan_cards`.
    """
    rgb = getattr(frame, "rgb", None)
    if rgb is None:
        return None
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a vision extra
        return None

    rx, ry, rw, rh = layout.card_row.to_px(frame)
    band = rgb[ry:ry + rh, rx:rx + rw]
    if band.size == 0:
        return None
    r = band[:, :, 0].astype(np.int16)
    g = band[:, :, 1].astype(np.int16)
    b = band[:, :, 2].astype(np.int16)
    bias = vision.glow_green_bias
    mask = (g > r + bias) & (g > b + bias) & (g > vision.glow_min_green)

    col = mask.sum(axis=0)
    row_thresh = int(vision.glow_col_min_frac * rh)
    min_width = max(4, int(vision.glow_min_strip_frac * frame.width))

    runs: list[tuple[int, int]] = []
    run = start = 0
    for i, v in enumerate(col):
        if v >= row_thresh:
            if run == 0:
                start = i
            run += 1
        else:
            if run >= min_width:
                runs.append((start, i))
            run = 0
    if run >= min_width:
        runs.append((start, len(col)))
    return runs


def count_mulligan_cards(frame: Frame, layout: GameLayout, vision) -> int:
    """Number of mulligan cards, or ``0`` meaning *unreadable* (fail closed).

    Counts **card interiors**: the spans between consecutive keep-glow strips
    (:func:`glow_strip_runs`) that are both as wide as a card and as bright as a
    card. On five real hands the two kinds of span separate cleanly:

    ============  ==================  ============
    span          width               mean gray
    ============  ==================  ============
    card interior 0.143 - 0.159 W     96 - 118
    gap between   0.054 W             33 - 35
    ============  ==================  ============

    The two tests are orthogonal - a wide dark span is not a card, nor is a narrow
    bright one - and each has a margin of roughly 3x. An implausible total (not 3
    or 4) returns 0, which makes :attr:`MulliganRead.usable` false so the engine
    fails closed rather than guess at the signal that decides whether to concede.

    Do **not** be tempted to count the strips instead: a strip is one card edge
    when the hand is spread out and two merged edges when it is packed, so 3 cards
    produce 6 strips and 4 cards produce 5. Counting strips gives the right answer
    on 4-card hands and the wrong one on 3-card hands - which is exactly the half
    of the signal that decides ``we_go_second``.

    Three earlier methods are gone because each was wrong on real frames:

    * **Strip counting** (N+1 strips), above.
    * **Mana-gem colour masking.** A gem's white digit punches a hole through the
      middle of it, splitting one gem into fragments, while blue *card art* forms
      runs of its own. Colour cannot even separate them - a real gem measured
      ``b-g=22`` against a card's blue sky at ``b-g=21``.
    * **Brightness "humps".** A dark-art card sits entirely below the threshold.

    The last two each reported 3 on a real 4-card hand, and each had passing unit
    tests built on *synthetic* frames. The tests for this function use real captures.
    """
    # A keep-glow strip is only ~1.2% of the screen wide. Below a few hundred pixels
    # a strip drops under `glow_min_strip_frac` and vanishes, which merges two card
    # interiors into one over-wide span and *undercounts* rather than failing. That
    # is the one thing this function must never do, so refuse outright. (Measured:
    # correct down to 400px; at 300px a 4-card hand reads 3. Real captures are 2400.)
    if frame.width < vision.min_count_frame_width:
        return 0

    runs = glow_strip_runs(frame, layout, vision)
    if not runs or len(runs) < 2:
        return 0

    rgb = frame.rgb
    rx, ry, rw, rh = layout.card_row.to_px(frame)
    band = rgb[ry:ry + rh, rx:rx + rw]

    expected = layout.card_inner_w_f * frame.width
    tol = vision.card_width_tolerance

    cards = 0
    for (_, end), (start, _) in zip(runs, runs[1:]):
        width = start - end
        if abs(width - expected) > tol * expected:
            continue                       # a background gap, not a card
        segment = band[:, end:start]
        if segment.size == 0:
            continue
        if float(segment.mean()) < vision.card_min_gray:
            continue                       # wide but dark: not a card
        cards += 1
    return cards if cards in (3, 4) else 0


def read_mulligan(
    frame: Frame,
    layout: GameLayout,
    reader: ClassReader,
    vision,
) -> MulliganRead:
    """Extract opponent class + going-second from a mulligan frame.

    ``vision`` is the :class:`~hop.config.VisionConfig`; it carries the keep-glow
    thresholds used to count cards. A frame captured without colour, or one whose
    glow strips don't cohere, yields ``num_cards == 0`` and an unusable read.
    """
    rx, ry, rw, rh = layout.opponent_class_region.to_px(frame)
    label = frame.crop(rx, ry, rw, rh)
    cr: ClassRead = reader.read(label)
    num = count_mulligan_cards(frame, layout, vision)
    return MulliganRead(
        opponent_class=cr.hero_class,
        we_go_second=(num >= 4),
        num_cards=num,
        class_confidence=cr.confidence,
        class_raw=cr.raw_text,
        method=cr.method,
    )
