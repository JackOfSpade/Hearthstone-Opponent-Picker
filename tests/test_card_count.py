"""Mulligan card counting: 3 cards = we go first, 4 = we go second.

A wrong count silently inverts ``require_second``, so this is the highest-stakes
pure function in the project. It is tested against **real captured frames**.

That is not a stylistic preference. The two previous implementations - masking the
blue mana gems, and counting brightness "humps" - each passed a full suite of
synthetic unit tests that drew idealised gems and bars, and each was then observed
to return **3 on a real 4-card hand**. Synthetic fixtures tested the implementation
against its own assumptions. The frames below came off the phone:

* ``mulligan_4card_a`` - Knickknack Shack / Bob the Bartender / 2x Shield Battery
* ``mulligan_4card_b`` - Kologarn / Scrapyard Colossus / Colossus of the Moon /
  Darkmoon Rabbit. This is the hand that broke gem-masking: the "10" digits punch
  holes through their own gems, and the Colossus's blue sky (``b-g=21``) is
  indistinguishable by colour from a real mana gem (``b-g=22``).

Frames are downscaled to 600px wide; the counter works in screen fractions, so it
is resolution-independent and the small fixtures exercise the same code path.
"""

import io
from pathlib import Path

import pytest

from hop.hearthstone import GameLayout, count_mulligan_cards, glow_strip_centers
from hop.perception.image import Frame

FRAMES = Path(__file__).parent / "data" / "frames"
FOUR_CARD = ["mulligan_4card_a", "mulligan_4card_b"]


@pytest.fixture
def layout():
    return GameLayout()


def _frame(name: str) -> Frame:
    return Frame.from_png((FRAMES / f"{name}.png").read_bytes(), keep_rgb=True)


@pytest.mark.parametrize("name", FOUR_CARD)
def test_counts_a_real_four_card_hand(name, layout, cfg):
    assert count_mulligan_cards(_frame(name), layout, cfg.vision) == 4


@pytest.mark.parametrize("name", FOUR_CARD)
def test_four_cards_produce_five_glow_strips(name, layout, cfg):
    """Adjacent cards share one glow strip, and the hand's edges add one each."""
    assert len(glow_strip_centers(_frame(name), layout, cfg.vision)) == 5


@pytest.mark.parametrize("name", FOUR_CARD)
def test_glow_strips_are_evenly_pitched(name, layout, cfg):
    frame = _frame(name)
    centers = glow_strip_centers(frame, layout, cfg.vision)
    expected = layout.card_pitch_f * frame.width
    for a, b in zip(centers, centers[1:]):
        assert (b - a) == pytest.approx(expected, rel=cfg.vision.card_pitch_tolerance)


@pytest.mark.parametrize("name", ["defeat", "play_screen"])
def test_non_mulligan_screens_are_unreadable_not_guessed(name, layout, cfg):
    """A screen with no hand yields 0 (fail closed), never a plausible 3 or 4."""
    assert count_mulligan_cards(_frame(name), layout, cfg.vision) == 0


def test_grayscale_frame_is_unreadable(layout, cfg):
    """The glow is a *colour*; a frame captured without RGB cannot be counted."""
    gray = Frame.from_gray_bytes(60, 27, bytes([40]) * (60 * 27))
    assert glow_strip_centers(gray, layout, cfg.vision) is None
    assert count_mulligan_cards(gray, layout, cfg.vision) == 0


def test_counter_is_resolution_independent(layout, cfg):
    """Fractions, not pixels: the same hand at half size must still read 4."""
    from PIL import Image

    im = Image.open(FRAMES / "mulligan_4card_b.png").convert("RGB")
    small = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, "PNG")
    frame = Frame.from_png(buf.getvalue(), keep_rgb=True)
    assert count_mulligan_cards(frame, layout, cfg.vision) == 4


def test_a_spurious_strip_inside_the_hand_fails_closed(layout, cfg, monkeypatch):
    """Green card art faking a strip must not become a fifth card.

    Six strips where one gap is half-pitch is not a hand; refuse to answer rather
    than report 5 -- or, worse, silently accept a count that flips we_go_second.
    """
    import hop.hearthstone as hs

    frame = _frame("mulligan_4card_b")
    real = glow_strip_centers(frame, layout, cfg.vision)
    spurious = sorted(real + [(real[1] + real[2]) // 2])
    monkeypatch.setattr(hs, "glow_strip_centers", lambda *a, **k: spurious)
    assert hs.count_mulligan_cards(frame, layout, cfg.vision) == 0


def test_too_few_strips_is_unreadable(layout, cfg, monkeypatch):
    import hop.hearthstone as hs

    monkeypatch.setattr(hs, "glow_strip_centers", lambda *a, **k: [100, 200])
    assert hs.count_mulligan_cards(_frame("mulligan_4card_b"), layout, cfg.vision) == 0


def test_a_dropped_edge_strip_reads_as_three_not_unreadable(layout, cfg, monkeypatch):
    """Documented limit, pinned so nobody assumes more safety than exists.

    Four strips at the true pitch is a *self-consistent* 3-card reading, so neither
    the pitch check nor the evenness check can catch a dropped outer strip. What
    protects us is that the glow is high-contrast UI chrome spanning a card's full
    height -- it is present or absent, never half-detected. If that ever stops
    holding, this test is where the assumption is written down.
    """
    import hop.hearthstone as hs

    frame = _frame("mulligan_4card_b")
    real = glow_strip_centers(frame, layout, cfg.vision)
    monkeypatch.setattr(hs, "glow_strip_centers", lambda *a, **k: real[:-1])
    assert hs.count_mulligan_cards(frame, layout, cfg.vision) == 3
