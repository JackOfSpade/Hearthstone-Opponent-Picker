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
    # Distance between adjacent keep-glow strips, as a fraction of screen width.
    # LIVE-VERIFIED on four real 4-card hands (2400x1080): 400.5-402.5 px.
    # Card size is independent of hand size, so this is the same for 3 and 4 cards
    # and gives an independent cross-check on the strip count.
    card_pitch_f: float = 0.1672

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


def glow_strip_centers(frame: Frame, layout: GameLayout, vision) -> list[int] | None:
    """x centres (px) of the mulligan keep-glow strips; ``None`` without colour.

    Every mulligan card is outlined by a saturated green "keep" glow. Between two
    adjacent cards the two glows form one bright vertical strip, and the outer
    edges of the hand contribute one each - so **N cards produce N+1 strips**.

    The glow is UI chrome, not artwork, which is the entire point: it cannot be
    imitated by a card's picture. A strip is accepted only if the green mask
    covers most of the band's *height*, because the glow runs the full height of a
    card while a patch of green art does not.
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

    centers: list[int] = []
    run = start = 0
    for i, v in enumerate(col):
        if v >= row_thresh:
            if run == 0:
                start = i
            run += 1
        else:
            if run >= min_width:
                centers.append(rx + start + run // 2)
            run = 0
    if run >= min_width:
        centers.append(rx + start + run // 2)
    return centers


def count_mulligan_cards(frame: Frame, layout: GameLayout, vision) -> int:
    """Number of mulligan cards, or ``0`` meaning *unreadable* (fail closed).

    Counts the keep-glow strips (:func:`glow_strip_centers`): N cards -> N+1
    strips. The strip count is then **cross-checked against the card pitch**, which
    is a fixed fraction of the screen regardless of hand size; a spurious or
    missing strip changes the implied pitch and is rejected. Disagreement returns
    0, which makes :attr:`MulliganRead.usable` false and the engine fail closed.

    Two earlier methods are gone because both were wrong on real frames:

    * **Mana-gem colour masking.** A gem's white digit punches a hole through the
      middle of it, splitting one gem into fragments, while blue *card art* forms
      its own runs. Colour cannot even separate them - a real gem measured
      ``b-g=22`` against a card's blue sky at ``b-g=21``. This reported 3 on a real
      4-card hand, which silently inverts ``we_go_second``.
    * **Brightness "humps".** A dark-art card sits entirely below the threshold and
      is skipped.

    Both passed their unit tests, which drew *synthetic* gems and humps. Only real
    frames caught them, so the tests for this function use real captures.
    """
    centers = glow_strip_centers(frame, layout, vision)
    if not centers or len(centers) < 3:
        return 0

    n = len(centers) - 1
    span = centers[-1] - centers[0]
    if n <= 0 or span <= 0:
        return 0

    expected = layout.card_pitch_f * frame.width
    tol = vision.card_pitch_tolerance
    if abs(span / n - expected) > tol * expected:
        return 0
    # the strips must also be evenly spaced: an extra strip inside the hand would
    # keep the mean pitch plausible while making one gap conspicuously short.
    gaps = [b - a for a, b in zip(centers, centers[1:])]
    if any(abs(gap - expected) > tol * expected for gap in gaps):
        return 0
    return n


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
