"""Scoped classification (`classify_expected`) must be RESULT-IDENTICAL to a full
`classify` on every frame the engine can feed it -- that is the whole basis for scoping
being a pure speed-up that leaves the loop's decisions and RNG stream bit-identical.

The divergences all live on frames where TWO anchors clear at once (a modal over a
screen; a cross-fade where the leaving and arriving screens both match). Plain
single-state fakes never exercise those, so these fixtures build the co-clear frames by
hand with the real classifier.
"""

import numpy as np

from hop.perception.image import Frame
from hop.perception.screens import (INTERRUPT_FLOOR, Anchor, ScreenClassifier,
                                     ScreenState, Classification)
from hop.perception.templates import Region, Template


def _frame(w=90, h=30):
    data = ((np.arange(w * h) * 37) % 251).astype("uint8").tobytes()
    return Frame.from_gray_bytes(w, h, data)


def _noisy(patch: Frame) -> Frame:
    # correlates but < 1.0 (NCC is offset/scale invariant, so perturb per-pixel)
    arr = np.asarray(patch.data, dtype=int).reshape(patch.height, patch.width).copy()
    arr += (np.arange(arr.size) % 5).reshape(arr.shape) * 9
    return Frame.from_gray_bytes(patch.width, patch.height,
                                 bytes(np.clip(arr, 0, 255).astype("uint8").ravel()))


# three disjoint search regions, each with an exact-cropped glyph inside it
_REGIONS = {"r1": (0.0, 0.0, 0.3, 1.0), "r2": (0.35, 0.0, 0.3, 1.0), "r3": (0.7, 0.0, 0.3, 1.0)}
_BOXES = {"r1": (0, 0, 10, 10), "r2": (35, 0, 10, 10), "r3": (65, 0, 10, 10)}


def _anchor(state, frame, slot, *, prio=0, thr=0.5, noisy=False):
    patch = frame.crop(*_BOXES[slot])
    if noisy:
        patch = _noisy(patch)
    return Anchor(state, Template(state.value, patch, Region(*_REGIONS[slot]), thr), prio)


def _assert_identical(clf, frame, expected):
    """classify_expected(expected) must return the SAME Classification as classify()."""
    scoped = clf.classify_expected(frame, expected)
    full = clf.classify(frame)
    assert isinstance(scoped, Classification)
    assert scoped.state == full.state, f"{scoped.state} != {full.state} for expected={expected}"
    assert scoped.confidence == full.confidence
    assert scoped.at == full.at


def test_floor_is_a_frozenset_of_the_overlay_states():
    # the fixed set that closes the fatal blind-tap; membership is load-bearing
    assert ScreenState.RECONNECT_DIALOG in INTERRUPT_FLOOR
    assert ScreenState.CONCEDE_WARNING in INTERRUPT_FLOOR
    assert ScreenState.CONCEDE_MENU in INTERRUPT_FLOOR
    assert ScreenState.RANK_PROGRESS in INTERRUPT_FLOOR
    assert ScreenState.QUEUE not in INTERRUPT_FLOOR      # base screens are not overlays


def test_cross_fade_two_base_anchors_scoped_matches_full():
    """mulligan (leaving) and in_game (arriving) both clear on the cross-fade; the leaving
    screen scores higher. With BOTH in `expected` (as `_wait_until_screen_leaves` guarantees
    by keeping the from-state) scoped picks the same winner full does."""
    f = _frame()
    clf = ScreenClassifier([
        _anchor(ScreenState.MULLIGAN, f, "r1"),                 # exact -> ~1.0 (higher)
        _anchor(ScreenState.IN_GAME, f, "r2", noisy=True),      # clears, but lower
    ])
    assert clf.classify(f).state == ScreenState.MULLIGAN        # higher score wins (both prio 0)
    _assert_identical(clf, f, {ScreenState.MULLIGAN, ScreenState.IN_GAME})


def test_the_fatal_case_floor_catches_reconnect_over_the_concede_menu():
    """concede_menu (top-priority, the ONLY thing scoped for the gear wait in the naive
    design) with a reconnect dialog co-drawn over it. The naive 'scan only higher priority'
    rule scanned nothing extra and would have tapped Concede into the dialog. The fixed
    floor scans reconnect_dialog, so scoped == full == the dialog (a fallback, no blind tap)."""
    f = _frame()
    clf = ScreenClassifier([
        _anchor(ScreenState.RECONNECT_DIALOG, f, "r1", prio=10),           # dialog (exact, wins)
        _anchor(ScreenState.CONCEDE_MENU, f, "r2", prio=10, noisy=True),   # menu (lower score)
        _anchor(ScreenState.IN_GAME, f, "r3"),                             # board behind (prio 0)
    ])
    assert clf.classify(f).state == ScreenState.RECONNECT_DIALOG
    # gear-wait scopes to {IN_GAME, CONCEDE_MENU}; the floor must still surface the dialog
    _assert_identical(clf, f, {ScreenState.IN_GAME, ScreenState.CONCEDE_MENU})


def test_end_screen_cross_fade_defeat_over_dissolving_board():
    """Just after a concede, in_game (End-Turn housing still drawn) and defeat both clear.
    `_clear_end_screens` scopes with in_game included so it routes to wait-it-out, never a
    blind end_dismiss into the board."""
    f = _frame()
    clf = ScreenClassifier([
        _anchor(ScreenState.IN_GAME, f, "r1"),                  # board (higher)
        _anchor(ScreenState.DEFEAT, f, "r2", noisy=True),       # banner fading in
    ])
    assert clf.classify(f).state == ScreenState.IN_GAME
    _assert_identical(clf, f, {ScreenState.DEFEAT, ScreenState.IN_GAME, ScreenState.PLAY_SCREEN})


def test_single_match_is_identical_even_when_expected_misses_it():
    """The common case: exactly one anchor clears. Scoped == full whether or not `expected`
    names it -- a miss just falls back to the full scan on the same frame."""
    f = _frame()
    clf = ScreenClassifier([_anchor(ScreenState.QUEUE, f, "r1")])
    _assert_identical(clf, f, {ScreenState.QUEUE})              # named
    _assert_identical(clf, f, {ScreenState.MULLIGAN})           # wrong guess -> fallback
    _assert_identical(clf, f, set())                            # empty -> plain classify


def test_unknown_frame_is_identical():
    """Nothing clears (animation frame): scoped must also report UNKNOWN, via fallback."""
    f = _frame()
    other = _frame()
    bad = _anchor(ScreenState.MENU, other, "r1", thr=0.999999, noisy=True)
    clf = ScreenClassifier([bad])
    assert clf.classify(f).state == ScreenState.UNKNOWN
    _assert_identical(clf, f, {ScreenState.MENU})


def test_scoped_never_diverges_across_a_grid_of_expected_sets():
    """Exhaustive small check: for a frame where three anchors co-clear at mixed priority,
    classify_expected equals classify for every `expected` that contains the true winner
    (and for the rest, falls back to it)."""
    f = _frame()
    clf = ScreenClassifier([
        _anchor(ScreenState.IN_GAME, f, "r1"),
        _anchor(ScreenState.MULLIGAN, f, "r2", noisy=True),
        _anchor(ScreenState.CONCEDE_MENU, f, "r3", prio=10, noisy=True),
    ])
    winner = clf.classify(f).state                              # concede_menu (priority 10)
    assert winner == ScreenState.CONCEDE_MENU
    from itertools import combinations
    states = [ScreenState.IN_GAME, ScreenState.MULLIGAN, ScreenState.CONCEDE_MENU,
              ScreenState.QUEUE, ScreenState.DEFEAT]
    for r in range(0, 4):
        for combo in combinations(states, r):
            _assert_identical(clf, f, set(combo))
