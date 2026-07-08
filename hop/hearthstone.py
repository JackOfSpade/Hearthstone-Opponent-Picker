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
    # The hand is centred on the screen. LIVE-MEASURED: the mean of the card-interior
    # centres is 0.497 +/- 0.001 across five real hands of both sizes. This is what
    # catches a hand whose OUTERMOST card has been marked for replacement (see
    # :func:`card_interiors`) - the survivors stay evenly spaced but shift off centre.
    hand_center_xf: float = 0.4975

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

    def card_point(self, center_xf: float) -> Point:
        """A tap target on the mulligan card whose interior is centred at ``center_xf``.

        Centres are **measured from the frame** (:func:`mulligan_card_centers_f`),
        not laid out by formula. A formula got this wrong: evenly spacing slots
        across 0.20..0.80 puts the outer cards at 0.20/0.80 when the real ones sit
        at 0.241/0.752 (4-card) and 0.270/0.723 (3-card). That is ~100px off on a
        345px-wide card, so the FFitts endpoint spread can push a tap off its edge -
        and we already detect the exact interiors in order to count them.

        The hit radius stays well inside the card's smaller half-dimension (172px),
        per the rule that an over-large radius throws taps off their target.
        """
        return Point(center_xf, self.card_row.yf + self.card_row.hf / 2, 0.05)

    def card_region(self, center_xf: float) -> Region:
        """The card's own rectangle, for scoping a tap's verification to it."""
        half = self.card_inner_w_f / 2
        return Region(max(0.0, center_xf - half), self.card_row.yf,
                      self.card_inner_w_f, self.card_row.hf)


@dataclass(frozen=True)
class MulliganRead:
    """The two signals extracted from a mulligan frame."""

    opponent_class: HeroClass | None
    we_go_second: bool
    num_cards: int
    class_confidence: float
    class_raw: str
    method: str
    #: measured x centres of the cards, for the reject journey's replace taps
    card_centers_f: tuple[float, ...] = ()

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


def card_interiors(frame: Frame, layout: GameLayout, vision) -> list[tuple[int, int]]:
    """Absolute x spans of the mulligan cards' interiors, left to right.

    A span between two keep-glow strips is a card when it is both card-WIDE and
    card-BRIGHT. See :func:`count_mulligan_cards` for the measurements behind
    those two tests, and for the three methods this replaced.
    """
    # A keep-glow strip is only ~1.2% of the screen wide. Below a few hundred pixels
    # a strip drops under `glow_min_strip_frac` and vanishes, which merges two card
    # interiors into one over-wide span and *undercounts* rather than failing. That
    # is the one thing this must never do, so refuse outright. (Measured: correct
    # down to 400px; at 300px a 4-card hand reads 3. Real captures are 2400.)
    if frame.width < vision.min_count_frame_width:
        return []

    runs = glow_strip_runs(frame, layout, vision)
    if not runs or len(runs) < 2:
        return []

    rgb = frame.rgb
    rx, ry, rw, rh = layout.card_row.to_px(frame)
    band = rgb[ry:ry + rh, rx:rx + rw]

    expected = layout.card_inner_w_f * frame.width
    tol = vision.card_width_tolerance

    out: list[tuple[int, int]] = []
    for (_, end), (start, _) in zip(runs, runs[1:]):
        width = start - end
        if abs(width - expected) > tol * expected:
            continue                       # a background gap, not a card
        segment = band[:, end:start]
        if segment.size == 0:
            continue
        if float(segment.mean()) < vision.card_min_gray:
            continue                       # wide but dark: not a card
        out.append((rx + end, rx + start))

    return out if hand_is_coherent(out, layout, vision, frame.width) else []


def hand_is_coherent(interiors: list[tuple[int, int]], layout: GameLayout,
                     vision, frame_width: int) -> bool:
    """Do these card interiors look like a whole, untouched mulligan hand?

    A card MARKED FOR REPLACEMENT loses its keep-glow, so its interior vanishes
    from the detection. On a 4-card hand that leaves **three** interiors - a
    perfectly plausible reading that silently inverts ``we_go_second`` and makes
    the loop concede exactly the games it was told to keep. The count is therefore
    only meaningful *before* any replace tap, and this is the guard that enforces
    it rather than leaving it as an unwritten invariant.

    Two structural facts about a real hand catch a missing card, and both are cheap:

    1. **Contiguity.** Adjacent cards are separated only by their glow strips and a
       thin background gap (measured 0.014-0.082 W). A missing *inner* card leaves a
       hole a whole card wide (measured 0.197 W).
    2. **Symmetry.** The hand is centred (measured mean interior centre 0.497 +/-
       0.001 across five real hands of both sizes). A missing *outer* card keeps the
       survivors evenly spaced -- contiguity sees nothing -- but drags their mean
       centre off by ~0.085 W.
    """
    if len(interiors) < 2:
        return True   # too few to be a hand at all; the count check rejects it

    max_sep = vision.card_max_separation_f * frame_width
    if any(b[0] - a[1] > max_sep for a, b in zip(interiors, interiors[1:])):
        return False

    mean_center = sum((a + b) / 2 for a, b in interiors) / len(interiors) / frame_width
    return abs(mean_center - layout.hand_center_xf) <= vision.hand_center_tolerance_f


def mulligan_card_centers_f(frame: Frame, layout: GameLayout, vision) -> tuple[float, ...]:
    """Screen-fraction x centres of the mulligan cards, left to right."""
    return tuple((a + b) / 2 / frame.width for a, b in card_interiors(frame, layout, vision))


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
    cards = len(card_interiors(frame, layout, vision))
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
    centers = mulligan_card_centers_f(frame, layout, vision)
    num = len(centers) if len(centers) in (3, 4) else 0
    return MulliganRead(
        opponent_class=cr.hero_class,
        we_go_second=(num >= 4),
        num_cards=num,
        class_confidence=cr.confidence,
        class_raw=cr.raw_text,
        method=cr.method,
        card_centers_f=centers,
    )
