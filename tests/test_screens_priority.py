"""A modal dialog must beat the screen it is drawn over.

Hearthstone's "There was an error starting your game." dialog does not hide the
deck-select screen beneath it, so both anchors clear their thresholds on the same
frame. Without priority the classifier picked whichever scored higher and the
engine would act on the occluded screen (tapping a deck behind a modal).
"""

import numpy as np

from hop.perception.image import Frame
from hop.perception.screens import Anchor, ScreenClassifier, ScreenState
from hop.perception.templates import Region, Template


def _textured_frame(w=60, h=20):
    # deterministic non-uniform data (uniform frames give an undefined NCC)
    data = ((np.arange(w * h) * 37) % 251).astype("uint8").tobytes()
    return Frame.from_gray_bytes(w, h, data)


def _two_matching_anchors(prio_a, prio_b):
    frame = _textured_frame()
    a = Template("a", frame.crop(0, 0, 10, 10), Region(0.0, 0.0, 0.5, 1.0), 0.5)
    b = Template("b", frame.crop(40, 0, 10, 10), Region(0.6, 0.0, 0.4, 1.0), 0.5)
    anchors = [Anchor(ScreenState.DECK_SELECT, a, prio_a),
               Anchor(ScreenState.ERROR_DIALOG, b, prio_b)]
    return ScreenClassifier(anchors), frame


def test_higher_priority_anchor_wins():
    clf, frame = _two_matching_anchors(prio_a=0, prio_b=10)
    assert clf.classify(frame).state == ScreenState.ERROR_DIALOG


def test_priority_is_respected_in_both_directions():
    clf, frame = _two_matching_anchors(prio_a=10, prio_b=0)
    assert clf.classify(frame).state == ScreenState.DECK_SELECT


def test_equal_priority_falls_back_to_score():
    clf, frame = _two_matching_anchors(prio_a=0, prio_b=0)
    # both score ~1.0; whichever wins, it must be a real state, not UNKNOWN
    assert clf.classify(frame).state in (ScreenState.DECK_SELECT, ScreenState.ERROR_DIALOG)


def _noisy_like(patch: Frame) -> Frame:
    """A template that still correlates with ``patch`` but not perfectly.

    NCC is invariant to scale and offset, so `a*k + c` would score exactly 1.0 -
    the perturbation has to be per-pixel.
    """
    arr = np.asarray(patch.data, dtype=int).reshape(patch.height, patch.width).copy()
    arr += (np.arange(arr.size) % 5).reshape(arr.shape) * 9
    return Frame.from_gray_bytes(patch.width, patch.height,
                                 bytes(np.clip(arr, 0, 255).astype("uint8").ravel()))


def test_confidence_is_the_winning_anchors_score_not_the_highest_score():
    """A higher-priority anchor can win with a LOWER score.

    The concede menu is drawn over the board, so `in_game` still matches through it
    and scores higher. Reporting that score as the confidence of CONCEDE_MENU hands
    `HumanState.observe_confidence` the certainty of a screen we did not pick.
    """
    frame = _textured_frame()
    exact = Template("exact", frame.crop(0, 0, 10, 10), Region(0.0, 0.0, 0.5, 1.0), 0.5)
    weak = Template("weak", _noisy_like(frame.crop(40, 0, 10, 10)),
                    Region(0.6, 0.0, 0.4, 1.0), 0.5)

    clf = ScreenClassifier([Anchor(ScreenState.IN_GAME, exact, 0),
                            Anchor(ScreenState.CONCEDE_MENU, weak, 10)])
    c = clf.classify(frame)
    assert c.state == ScreenState.CONCEDE_MENU        # priority decides the state...
    assert c.confidence < 0.999                        # ...so ITS score is reported

    # the loser really did score higher - that is the whole point
    solo = ScreenClassifier([Anchor(ScreenState.IN_GAME, exact, 0)])
    assert solo.classify(frame).confidence > c.confidence


def test_unknown_confidence_is_zero_not_a_sub_threshold_score():
    """`best_match` returns None below threshold, so no sub-threshold score exists."""
    frame = _textured_frame()
    bad = Frame.from_gray_bytes(10, 10, bytes((np.arange(100) % 7).astype("uint8")))
    clf = ScreenClassifier([Anchor(ScreenState.MENU, Template("x", bad, Region(), 0.99), 0)])
    c = clf.classify(frame)
    assert c.state == ScreenState.UNKNOWN
    assert c.confidence == 0.0
    assert c.at is None


def test_classification_carries_the_winning_match_location():
    """The anchor glyph is drawn on the thing it identifies."""
    clf, frame = _two_matching_anchors(prio_a=0, prio_b=10)
    c = clf.classify(frame)
    assert c.state == ScreenState.ERROR_DIALOG
    assert c.at is not None
    assert c.at[0] >= int(0.6 * frame.width)     # inside anchor b's search region


def test_no_match_is_unknown():
    frame = _textured_frame()
    other = _textured_frame(w=60, h=20)
    # a template that cannot clear a very high threshold
    t = Template("x", other.crop(0, 0, 10, 10), Region(0.0, 0.0, 1.0, 1.0), 0.999999)
    clf = ScreenClassifier([Anchor(ScreenState.MENU, t, 0)])
    # identical data would score 1.0, so perturb the template
    import numpy as _np
    bad = Frame.from_gray_bytes(10, 10, bytes((_np.arange(100) % 7).astype("uint8")))
    clf2 = ScreenClassifier([Anchor(ScreenState.MENU, Template("x", bad, Region(), 0.99), 0)])
    assert clf2.classify(frame).state == ScreenState.UNKNOWN
