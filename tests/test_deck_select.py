"""The deck list: hop pauses there, and never auto-completes a deck.

hop reaches the deck grid only on recovery (an error dropped it back there). It does NOT
reopen a deck: it cannot reliably tell which deck was in play (the grid names render too
soft/stylised to OCR; any other pick could open the wrong deck or an incomplete one), and
re-selecting a *different* deck than the user chose is worse than stopping. So it pauses
(halts + alerts) and the user re-selects. The one thing it still does on the deck screens
is refuse to auto-complete: on "Complete deck automatically?" it taps No, never Yes.
"""

from random import Random

import pytest

from hop.engine import Engine
from hop.geometry import PanelGeometry
from hop.hearthstone import GameLayout
from hop.perception.screens import ScreenState
from hop.verify import Halt

from conftest import FakeAdb, FakeBackend, FakeClassifier, gray_frame

PANEL = PanelGeometry(80, 40, 400.0)


class _AlternatingCapturer:
    """Every capture differs from the last, so every _tap verifies as changed."""

    def __init__(self):
        self.n = 0

    def capture(self):
        self.n += 1
        return gray_frame(80, 40, 20 if self.n % 2 else 220)


def _engine(cfg, states):
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PANEL, FakeClassifier(states), reader=None,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(4),
                 layout=GameLayout(), capturer=_AlternatingCapturer())
    return eng, backend


# ── the deck list pauses ─────────────────────────────────────────────────────

def test_deck_select_pauses_and_never_taps_a_deck(cfg):
    """hop cannot tell which deck was in play, so it must not open one -- opening a
    different deck than the user chose is worse than stopping."""
    eng, backend = _engine(cfg, [ScreenState.DECK_SELECT])
    frame = gray_frame(80, 40)
    with pytest.raises(Halt, match="[Rr]e-select your deck"):
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert backend.gestures == []          # never a deck tap


# ── never auto-complete ──────────────────────────────────────────────────────

def test_decline_is_the_no_button_not_yes():
    """No is the RIGHT button; Yes is the left. Tapping Yes auto-completes the deck."""
    d = GameLayout().deck_decline
    assert d.xf > 0.5, "decline must be right-of-centre (No), not Yes"
    assert d.xf - d.radius_f * 0.9 > 0.5   # the 0.9r spread stays off the Yes button


def test_incomplete_deck_dialog_is_declined_never_confirmed(cfg):
    """On "Complete deck automatically?", tap No -- Yes spends the user's dust/cards."""
    eng, backend = _engine(cfg, [ScreenState.INCOMPLETE_DECK, ScreenState.DECK_SELECT])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert len(backend.gestures) == 1      # the decline (No), and nothing else
    tapped = backend.gestures[0]
    assert max(s.x for s in tapped.samples) > PANEL.width_px / 2   # right button = No


def test_decline_waits_for_the_dialog_to_actually_leave(cfg):
    """An unreadable frame is not proof the dialog closed -> fail closed."""
    eng, _backend = _engine(cfg, [ScreenState.INCOMPLETE_DECK] * 12)
    with pytest.raises(Halt):
        eng._decline_incomplete_deck()
