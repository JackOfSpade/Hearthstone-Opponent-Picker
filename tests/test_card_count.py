"""Tests for mulligan card counting via blue mana gems.

Live capture caught the old brightness peak-count reporting **3** on a real
4-card hand (a dark-art card never crossed the brightness threshold), which would
invert the going-second signal. The gem counter masks blue pixels in the gem row
and keeps only column-runs of gem width, so blue *card art* (much wider) and
specular slivers are rejected. These tests pin that behaviour on synthetic frames
with the same geometry as a 2400-wide landscape screen.
"""

import numpy as np
import pytest

from hop.config import load_config
from hop.hearthstone import GameLayout, _count_mana_gems, count_mulligan_cards
from hop.perception.image import Frame

W, H = 2400, 200          # gem band lands inside this height
GEM_W = 86                # ~ gem_width_f * 2400
BLUE = (30, 60, 200)      # blue-dominant
GREY = (120, 120, 120)


@pytest.fixture
def vision():
    return load_config().vision


def _frame(runs):
    """Build a frame whose gem band contains blue rects: runs = [(x, width), ...]"""
    layout = GameLayout()
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :] = GREY
    y0 = int(0.275 * H)
    y1 = y0 + int(0.10 * H)
    for x, w in runs:
        rgb[y0:y1, x:x + w] = BLUE
    gray = rgb.mean(axis=2).astype(np.uint8)
    return Frame(W, H, gray, rgb), layout


def test_counts_four_gems(vision):
    f, layout = _frame([(400, GEM_W), (800, GEM_W), (1200, GEM_W), (1600, GEM_W)])
    assert _count_mana_gems(f, layout, vision) == 4
    assert count_mulligan_cards(f, layout, vision) == 4


def test_counts_three_gems(vision):
    f, layout = _frame([(600, GEM_W), (1100, GEM_W), (1600, GEM_W)])
    assert _count_mana_gems(f, layout, vision) == 3
    assert count_mulligan_cards(f, layout, vision) == 3


def test_rejects_wide_blue_card_art(vision):
    """Blue art masks blue too, but is several gem-widths wide."""
    f, layout = _frame([(400, GEM_W), (800, GEM_W), (1300, 260)])
    assert _count_mana_gems(f, layout, vision) == 2


def test_rejects_thin_specular_slivers(vision):
    f, layout = _frame([(400, GEM_W), (800, GEM_W), (1200, 8), (1300, 14)])
    assert _count_mana_gems(f, layout, vision) == 2


def test_no_blue_means_zero_gems(vision):
    f, layout = _frame([])
    assert _count_mana_gems(f, layout, vision) == 0


def test_falls_back_when_colour_unavailable(vision):
    """Grayscale-only frames (no rgb) must not crash; gems return None."""
    f, layout = _frame([(400, GEM_W), (800, GEM_W)])
    grayscale_only = Frame(f.width, f.height, f.data, None)
    assert _count_mana_gems(grayscale_only, layout, vision) is None
    # count_mulligan_cards then uses the (unreliable) brightness fallback
    assert isinstance(count_mulligan_cards(grayscale_only, layout, vision), int)


def test_implausible_gem_count_falls_back(vision):
    """5 gems is impossible; fall through to the brightness counter rather than
    trusting a nonsense colour read."""
    f, layout = _frame([(400, GEM_W), (700, GEM_W), (1000, GEM_W),
                        (1300, GEM_W), (1600, GEM_W)])
    assert _count_mana_gems(f, layout, vision) == 5
    assert count_mulligan_cards(f, layout, vision) != 5  # fell back
