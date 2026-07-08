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
