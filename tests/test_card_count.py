"""Mulligan card counting: 3 cards = we go first, 4 = we go second.

A wrong count silently inverts ``require_second``, so this is the highest-stakes
pure function in the project. It is tested against **real captured frames**.

That is not a stylistic preference. Every implementation that was tested only on
synthetic frames shipped a bug:

* Counting brightness "humps" -- a dark-art card sits below the threshold.
* Masking the blue mana gems -- a gem's own white digit splits it into fragments,
  and colour cannot separate a gem (``b-g=22``) from a card's blue sky (``b-g=21``).
* Counting the keep-glow *strips* as N+1 -- true only when the hand is packed tight
  enough for adjacent glows to merge.

The first two each returned **3 on a real 4-card hand**. The third was right on
every 4-card hand and wrong on the first real 3-card hand it ever saw, returning
an unreadable 0 -- which is what fail-closed is for, but it was still wrong.

Fixtures below are real captures (downscaled to 600px; the counter works in screen
fractions, so it is resolution-independent):

* ``mulligan_3card``  - Y'Shaarj / Pipsi Painthoof / Colossus of the Moon.
  Three cards, spread out: each card's glow stays separate, so **6 strips**.
* ``mulligan_4card_a`` - Knickknack Shack / Bob the Bartender / 2x Shield Battery.
* ``mulligan_4card_b`` - Kologarn / Scrapyard Colossus / Colossus of the Moon /
  Darkmoon Rabbit. Four cards, packed: adjacent glows merge, so **5 strips**.
  This is the hand that broke gem-masking.
"""

import io
from pathlib import Path

import pytest

from hop.hearthstone import GameLayout, count_mulligan_cards, glow_strip_runs
from hop.perception.image import Frame

FRAMES = Path(__file__).parent / "data" / "frames"
HANDS = [("mulligan_3card", 3), ("mulligan_4card_a", 4), ("mulligan_4card_b", 4)]


@pytest.fixture
def layout():
    return GameLayout()


def _frame(name: str) -> Frame:
    return Frame.from_png((FRAMES / f"{name}.png").read_bytes(), keep_rgb=True)


@pytest.mark.parametrize("name,expected", HANDS)
def test_counts_real_hands(name, expected, layout, cfg):
    assert count_mulligan_cards(_frame(name), layout, cfg.vision) == expected


def test_three_card_hand_means_we_go_first(layout, cfg):
    """The whole point of the count. 3 -> first, 4 -> second."""
    assert count_mulligan_cards(_frame("mulligan_3card"), layout, cfg.vision) < 4
    assert count_mulligan_cards(_frame("mulligan_4card_b"), layout, cfg.vision) >= 4


def test_strip_count_alone_does_not_determine_card_count(layout, cfg):
    """Regression: the reason we count interiors and not strips.

    A spread-out 3-card hand keeps each card's glow separate (2 strips per card);
    a packed 4-card hand merges adjacent glows (N+1 strips). So *fewer* cards can
    produce *more* strips, and any rule of the form ``cards = f(len(strips))`` is
    wrong. If this ever stops holding, the counter can be simplified -- but only
    then.
    """
    three = len(glow_strip_runs(_frame("mulligan_3card"), layout, cfg.vision))
    four = len(glow_strip_runs(_frame("mulligan_4card_b"), layout, cfg.vision))
    assert three == 6 and four == 5
    assert three > four, "3 cards produce MORE glow strips than 4 -- do not count strips"


@pytest.mark.parametrize("name", ["defeat", "play_screen"])
def test_non_mulligan_screens_are_unreadable_not_guessed(name, layout, cfg):
    """A screen with no hand yields 0 (fail closed), never a plausible 3 or 4."""
    assert count_mulligan_cards(_frame(name), layout, cfg.vision) == 0


def test_grayscale_frame_is_unreadable(layout, cfg):
    """The glow is a *colour*; a frame captured without RGB cannot be counted."""
    gray = Frame.from_gray_bytes(60, 27, bytes([40]) * (60 * 27))
    assert glow_strip_runs(gray, layout, cfg.vision) is None
    assert count_mulligan_cards(gray, layout, cfg.vision) == 0


def _rescaled(name: str, width: int) -> Frame:
    from PIL import Image

    im = Image.open(FRAMES / f"{name}.png").convert("RGB")
    im = im.resize((width, int(width * im.height / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return Frame.from_png(buf.getvalue(), keep_rgb=True)


@pytest.mark.parametrize("name,expected", HANDS)
@pytest.mark.parametrize("width", [600, 480, 400])
def test_counter_is_resolution_independent(name, expected, width, layout, cfg):
    """Fractions, not pixels -- down to the documented floor."""
    assert count_mulligan_cards(_rescaled(name, width), layout, cfg.vision) == expected


@pytest.mark.parametrize("name,expected", HANDS)
def test_below_the_resolution_floor_it_refuses_rather_than_undercounts(
    name, expected, layout, cfg
):
    """The one failure this function must never have.

    A keep-glow strip is ~1.2% of the screen wide. Shrink the frame enough and a
    strip drops below `glow_min_strip_frac` and vanishes, merging two card
    interiors into one over-wide span -- which reads 3 on a 4-card hand and so
    inverts we_go_second. Verified: at 300px, `mulligan_4card_a` really does read
    3 without the guard. Refuse instead.
    """
    below = cfg.vision.min_count_frame_width - 100
    assert count_mulligan_cards(_rescaled(name, below), layout, cfg.vision) == 0


def test_a_wide_but_dark_span_is_not_a_card(layout, cfg, monkeypatch):
    """Width alone is not enough: the interior must also be card-bright.

    Fake two strips straddling the dark background left of the hand. It is as wide
    as a card and contains no card.
    """
    import hop.hearthstone as hs

    frame = _frame("mulligan_3card")
    inner = int(layout.card_inner_w_f * frame.width)
    monkeypatch.setattr(hs, "glow_strip_runs", lambda *a, **k: [(0, 4), (4 + inner, 8 + inner)])
    assert hs.count_mulligan_cards(frame, layout, cfg.vision) == 0


def test_a_narrow_bright_span_is_not_a_card(layout, cfg, monkeypatch):
    """...and brightness alone is not enough either: a gap between cards is bright-ish."""
    import hop.hearthstone as hs

    frame = _frame("mulligan_3card")
    runs = glow_strip_runs(frame, layout, cfg.vision)
    # keep only the strips bounding the narrow background gaps
    monkeypatch.setattr(hs, "glow_strip_runs", lambda *a, **k: runs[1:3])
    assert hs.count_mulligan_cards(frame, layout, cfg.vision) == 0


def test_implausible_totals_fail_closed(layout, cfg, monkeypatch):
    """2 or 5 cards cannot happen in a mulligan; refuse rather than report them."""
    import hop.hearthstone as hs

    frame = _frame("mulligan_4card_b")
    runs = glow_strip_runs(frame, layout, cfg.vision)
    monkeypatch.setattr(hs, "glow_strip_runs", lambda *a, **k: runs[:3])   # -> 2 interiors
    assert hs.count_mulligan_cards(frame, layout, cfg.vision) == 0


def test_too_few_strips_is_unreadable(layout, cfg, monkeypatch):
    import hop.hearthstone as hs

    monkeypatch.setattr(hs, "glow_strip_runs", lambda *a, **k: [(10, 20)])
    assert hs.count_mulligan_cards(_frame("mulligan_4card_b"), layout, cfg.vision) == 0
