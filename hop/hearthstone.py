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
    #
    # One conservative crop, anchored at the screen edge and sized for the LONGEST class label.
    # In the CASUAL mulligan (the mode the hunt runs in) there is no rank medallion, so the class
    # word butts the left edge -- LIVE-MEASURED at x 0.008..0.050 W for "PRIEST". The longest
    # labels, "DEMON HUNTER"/"DEATH KNIGHT" (12 glyph-units), reach ~0.094 W at the measured
    # ~0.0072 W casual pitch, so x 0.0 width 0.18 (to 0.18 W) clears the longest label with margin.
    # Deliberately generous: over-reaching into the dark nameplate gutter is harmless (the crop
    # still OCRs "PRIEST" cleanly out to width 0.25), whereas the OLD Region(0.03, .., 0.20, ..)
    # started 0.022 W INSIDE the word and CLIPPED its left glyphs -- "PRIEST" -> "EST" (snapped,
    # dist 3), and the reported bug, a "DEMON HUNTER" whose whole first word was cut off, leaving
    # just "HUNTER" (a valid class the fail-closed snap then accepted, conf 0.25). CLIPPING is the
    # catastrophic failure (a lost word is unrecoverable -- "HUNTER" is deliberately Hunter's own
    # word, never a Demon Hunter fragment, see hero_classes._PARTIAL_LABELS); a too-wide crop is not.
    #
    # RANKED draws a league medallion in exactly this left band (x 0.009..0.060 W, measured), which
    # OCRs as junk leading glyphs ("BSY PRIEST", dist 3) -- so this single crop is tuned for CASUAL,
    # and a ranked two-word class could still mis-snap. If the hunt is ever run RANKED, add a
    # mode-specific crop starting ~0.065 W (past the medallion). Card geometry is mode-shared.
    opponent_class_region: Region = Region(0.0, 0.90, 0.18, 0.07)
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
    # The Black Market early-concede warning can appear immediately after Concede
    # when a quest is active.  Its gold "Concede Now" button measured
    # x=829..1172, y=681..769 on this Pixel 7a (2400x1080 landscape), hence the
    # centre below.  The reject fast path sends this point exactly once whether
    # that modal appears or not; without it, this coordinate is inert on this
    # calibrated phone.  This is deliberately a distinct point from the Game
    # Menu's Concede control, not a generic confirmation location.
    concede_now_button: Point = Point(0.4167, 0.6713, 0.014)
    # NOTE: there is deliberately NO generic/retry `concede_confirm` point for the
    # Game Menu. It reads Concede / Options / Quit, and the old (0.50, 0.56, 0.06)
    # "confirm" landed dead centre on **Quit** (its 0.9*144 px truncation disc lies
    # almost entirely on the Quit plate). So the recovery path for "the Concede tap
    # was ignored" was "quit Hearthstone". _concede() re-sends only the top Concede
    # coordinate while that menu is named. The separately calibrated Black Market
    # `concede_now_button` above is a different modal control, used once only in the
    # direct reject trace; it is not a Game Menu retry or a generic confirmation.
    #
    # Dismisses victory/defeat/rewards/quest. LIVE-VERIFIED on a victory screen
    # (a tap at 1208,941 advanced to the rewards screen first try) and on the
    # rewards screen. Radius is deliberately small: radius_f is a fraction of the
    # screen's WIDTH (2400), so the old 0.10 meant a 240 px disc on a 1080 px tall
    # screen centred 108 px from the bottom -- 13.1% of sampled endpoints fell OFF
    # the panel and 39.8% into Android's bottom mandatorySystemGestures inset
    # (y >= 996, read from `dumpsys window`). 0.0125*2400 = 30 px keeps the whole
    # 0.9r disc inside y 913..967: on the panel, clear of the gesture inset, and
    # clear of the deck list's "My Collection" plate at y >= 990.
    end_dismiss: Point = Point(0.50, 0.87, 0.0125)
    pass_turn_button: Point = Point(0.80, 0.497, 0.012)
    # The Collection's back arrow (bottom-right). hop is never meant to be in the
    # Collection - the end_dismiss geometry and the END_SCREENS whitelist keep it off
    # the deck list's "My Collection" plate - but a stray navigation must be
    # recoverable, not a halt: this returns to the deck list. LIVE-MEASURED (Pixel 7a):
    # the button interior is x 1930-2130, y 975-1050. It is wide and short, so the hit
    # radius comes from the smaller (height) half-dimension. Its centre is below
    # Android's y=996 gesture inset, but discrete taps there are delivered on this
    # device (measured: `input tap 1200 1020` navigated fine).
    collection_back: Point = Point(0.846, 0.9370, 0.0154)
    # The "No" button of the INCOMPLETE_DECK dialog ("Complete deck automatically?").
    # hop taps this, NEVER "Yes" - auto-completing a deck spends the user's dust/cards.
    # LIVE-MEASURED: the gold No button spans x 1200-1540, y 645-715; radius from its
    # smaller (height) half. Yes is the LEFT button (~x 600-1000), well clear of this.
    deck_decline: Point = Point(0.5708, 0.6315, 0.014)
    # OK button of Hearthstone's "There was an error starting your game." dialog (a
    # frequent, transient network blip - dismiss and requeue; no long backoff needed).
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


def mulligan_card_count_diagnostics(frame: Frame, layout: GameLayout, vision) -> dict:
    """Return bounded, JSON-safe evidence for a mulligan card-count miss.

    This deliberately mirrors :func:`card_interiors` without changing the hot
    path.  It is for the exceptional ``cards == 0`` path, where knowing whether
    the fault was no colour, missing glow strips, a width/brightness rejection,
    or the marked-card coherence guard matters much more than saving a few
    vector operations.  It contains measurements rather than image pixels; the
    caller can retain the bounded anomaly frame separately.

    Glow runs are row-relative (as returned by :func:`glow_strip_runs`), while
    candidate interiors are absolute screen x spans.  Lists are capped so a
    pathological image cannot make a bug report enormous.
    """
    max_spans = 24
    rgb = getattr(frame, "rgb", None)
    rx, ry, rw, rh = layout.card_row.to_px(frame)
    expected = layout.card_inner_w_f * frame.width
    tolerance = vision.card_width_tolerance * expected
    result: dict = {
        "frame": {
            "width": int(frame.width),
            "height": int(frame.height),
            "has_rgb": rgb is not None,
        },
        "card_row": {"x": int(rx), "y": int(ry), "width": int(rw), "height": int(rh)},
        "thresholds": {
            "glow_green_bias": int(vision.glow_green_bias),
            "glow_min_green": int(vision.glow_min_green),
            "glow_col_min_frac": float(vision.glow_col_min_frac),
            "glow_column_min_rows": int(vision.glow_col_min_frac * rh),
            "glow_min_strip_width_px": max(
                4, int(vision.glow_min_strip_frac * frame.width)
            ),
            "expected_width_px": round(expected, 1),
            "width_tolerance_px": round(tolerance, 1),
            "min_mean_rgb": float(vision.card_min_gray),
            "min_frame_width": int(vision.min_count_frame_width),
        },
        "glow_run_count": 0,
        "glow_runs": [],
        "glow_runs_truncated": False,
        "candidates": [],
        "candidates_truncated": False,
        "candidate_interiors": [],
        "coherent": None,
        "count": 0,
        "rejection": "",
    }
    if rgb is None:
        result["rejection"] = "no_rgb"
        return result
    if frame.width < vision.min_count_frame_width:
        result["rejection"] = "frame_too_narrow"
        return result

    # ``glow_strip_runs`` returns ``None`` both for a missing colour frame and
    # an empty crop.  Colour was checked above, so distinguish the latter here.
    band = rgb[ry:ry + rh, rx:rx + rw]
    if getattr(band, "size", 0) == 0:
        result["rejection"] = "empty_card_row"
        return result
    runs = glow_strip_runs(frame, layout, vision) or []
    result["glow_run_count"] = len(runs)
    result["glow_runs"] = [[int(start), int(end)] for start, end in runs[:max_spans]]
    result["glow_runs_truncated"] = len(runs) > max_spans
    if len(runs) < 2:
        result["rejection"] = "insufficient_glow_strips"
        return result

    interiors: list[tuple[int, int]] = []
    candidates: list[dict] = []
    for (_, left_end), (right_start, _) in zip(runs, runs[1:]):
        width = right_start - left_end
        width_ok = abs(width - expected) <= tolerance
        segment = band[:, left_end:right_start]
        mean_rgb = float(segment.mean()) if getattr(segment, "size", 0) else None
        brightness_ok = mean_rgb is not None and mean_rgb >= vision.card_min_gray
        selected = width_ok and brightness_ok
        if len(candidates) < max_spans:
            candidates.append({
                "start": int(rx + left_end),
                "end": int(rx + right_start),
                "width_px": int(width),
                "width_ok": bool(width_ok),
                "mean_rgb": round(mean_rgb, 1) if mean_rgb is not None else None,
                "brightness_ok": bool(brightness_ok),
                "selected": bool(selected),
            })
        if selected:
            interiors.append((rx + left_end, rx + right_start))
    result["candidates"] = candidates
    result["candidates_truncated"] = len(runs) - 1 > max_spans
    result["candidate_interiors"] = [[int(start), int(end)] for start, end in interiors[:max_spans]]

    if not interiors:
        result["rejection"] = "no_candidate_interiors"
        return result
    coherent = hand_is_coherent(interiors, layout, vision, frame.width)
    result["coherent"] = bool(coherent)
    if not coherent:
        result["rejection"] = "incoherent_hand"
        return result
    if len(interiors) not in (3, 4):
        result["rejection"] = "implausible_card_total"
        return result
    result["count"] = len(interiors)
    result["rejection"] = "ok"
    return result


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
    or 4) returns 0, which makes :attr:`MulliganRead.usable` false.  The engine
    requires that full two-signal read whenever the configured criteria depend
    on the coin; a class-only criterion can still safely decide from a known
    opponent class without inventing a turn result.

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
    glow strips don't cohere, yields ``num_cards == 0`` and an incomplete
    two-signal read.  The engine retains it as colour evidence and only permits
    a class-only decision when its criteria do not need the turn.
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
