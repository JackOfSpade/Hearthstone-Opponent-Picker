"""Tests for `hop capture`'s glyph->search-region expansion.

best_match() slides the template inside its search region and bails when the
template is wider/taller than the region, so a captured anchor's search region
must strictly contain the glyph box with margin. These pin that invariant.
"""

from hop.cli import _expand_box


def test_expands_glyph_box_by_margin():
    box = [0.40, 0.50, 0.20, 0.10]
    x, y, w, h = _expand_box(box, margin=0.5)
    assert w > box[2] and h > box[3]          # strictly larger => template fits
    assert x <= box[0] and y <= box[1]        # and it contains the glyph
    assert x + w >= box[0] + box[2]
    assert y + h >= box[1] + box[3]


def test_clamps_to_frame_bounds():
    # a glyph hard against the top-left corner must not produce negative origin
    x, y, w, h = _expand_box([0.0, 0.0, 0.2, 0.1], margin=1.0)
    assert x == 0.0 and y == 0.0
    assert 0.0 <= x + w <= 1.0 and 0.0 <= y + h <= 1.0


def test_clamps_to_far_edge():
    x, y, w, h = _expand_box([0.9, 0.95, 0.1, 0.05], margin=1.0)
    assert x >= 0.0 and y >= 0.0
    assert abs((x + w) - 1.0) < 1e-9
    assert abs((y + h) - 1.0) < 1e-9


def test_full_frame_glyph_stays_full_frame():
    assert _expand_box([0.0, 0.0, 1.0, 1.0]) == [0.0, 0.0, 1.0, 1.0]
