"""Deck-select recovery: reopen only a user-named complete deck; never auto-complete.

hop reaches the deck list only on recovery (an error dropped it back there). It CANNOT
safely guess which deck is complete: on a real account the first slots are often
incomplete ("27/30 - Missing Cards"), and both automatic oracles fail on-device - the
"Complete deck automatically?" dialog is session-suppressed after one decline, and the
badge's red/gold colour is faked by deck art (measured). So it reopens only the COMPLETE
decks named in `[deck] recovery_slots`; if none are configured it halts with guidance
rather than open an unplayable deck. If a configured deck turns out incomplete it
declines the dialog (No) - never Yes, which would spend the user's dust/cards.
"""

from dataclasses import replace
from random import Random

import pytest

from hop.engine import Engine
from hop.geometry import PanelGeometry
from hop.hearthstone import GameLayout
from hop.perception.screens import ScreenState
from hop.verify import Halt

from conftest import FakeAdb, FakeBackend, FakeClassifier, gray_frame

PANEL = PanelGeometry(80, 40, 400.0)
ALL_SLOTS = tuple(range(1, 10))


class _AlternatingCapturer:
    """Every capture differs from the last, so every _tap verifies as changed."""

    def __init__(self):
        self.n = 0

    def capture(self):
        self.n += 1
        return gray_frame(80, 40, 20 if self.n % 2 else 220)


def _engine(cfg, states, recovery_slots=ALL_SLOTS):
    cfg = replace(cfg, deck=replace(cfg.deck, recovery_slots=tuple(recovery_slots)))
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PANEL, FakeClassifier(states), reader=None,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(4),
                 layout=GameLayout(), capturer=_AlternatingCapturer())
    return eng, backend


# ── geometry ─────────────────────────────────────────────────────────────────

def test_deck_grid_is_nine_slots_in_reading_order():
    slots = GameLayout().deck_slots()
    assert len(slots) == 9
    # row-major: the first three share a row (same yf), then the y increases
    assert slots[0].yf == slots[1].yf == slots[2].yf
    assert slots[0].yf < slots[3].yf < slots[6].yf
    assert slots[0].xf < slots[1].xf < slots[2].xf


def test_every_deck_slot_disc_stays_on_the_panel():
    panel = PanelGeometry(2400, 1080, 420.0)
    for p in GameLayout().deck_slots():
        x, y, r = p.to_px(panel)
        assert 0 <= x - r * 0.9 and x + r * 0.9 <= panel.width_px - 1
        assert 0 <= y - r * 0.9 and y + r * 0.9 <= panel.height_px - 1


def test_decline_is_the_no_button_not_yes():
    """No is the RIGHT button; Yes is the left. Tapping Yes auto-completes the deck."""
    d = GameLayout().deck_decline
    assert d.xf > 0.5, "decline must be right-of-centre (No), not Yes"
    # the 0.9r spread must not cross back toward the Yes button (~xf 0.33)
    assert d.xf - d.radius_f * 0.9 > 0.5


# ── behaviour ────────────────────────────────────────────────────────────────

def test_unconfigured_recovery_halts_rather_than_guessing(cfg):
    """With no recovery deck named, hop must NOT guess -- guessing could open an
    incomplete deck. Halt with guidance instead."""
    eng, backend = _engine(cfg, [ScreenState.DECK_SELECT], recovery_slots=())
    with pytest.raises(Halt, match="no complete deck is configured"):
        eng._handle_deck_select()
    assert backend.gestures == []          # never a blind tap


def test_the_configured_deck_is_opened(cfg):
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], recovery_slots=(4,))
    eng._handle_deck_select()
    assert len(backend.gestures) == 1      # one tap on the configured slot -> Play
    # it tapped slot 4 (index 3): row 2, col 1 -> left column, middle row
    tapped_x = backend.gestures[0].samples[0].x / PANEL.width_px
    assert tapped_x == pytest.approx(GameLayout().deck_slots()[3].xf, abs=0.05)


def test_only_configured_slots_are_tried(cfg):
    """Slots outside recovery_slots are never touched, even the first one."""
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], recovery_slots=(6,))
    eng._handle_deck_select()
    tapped_x = backend.gestures[0].samples[0].x / PANEL.width_px
    assert tapped_x == pytest.approx(GameLayout().deck_slots()[5].xf, abs=0.05)  # slot 6


def test_a_configured_deck_that_is_incomplete_is_declined_then_next_tried(cfg):
    # slot A -> the dialog; No -> back to list; slot B -> Play
    eng, backend = _engine(cfg, [ScreenState.INCOMPLETE_DECK,
                                 ScreenState.DECK_SELECT,
                                 ScreenState.PLAY_SCREEN], recovery_slots=(4, 6))
    eng._handle_deck_select()
    assert len(backend.gestures) == 3      # slot4, decline(No), slot6
    decline = backend.gestures[1]
    assert min(s.x for s in decline.samples) > 0.45 * PANEL.width_px, "decline drifted toward Yes"


def test_all_configured_decks_incomplete_halts(cfg):
    states = [ScreenState.INCOMPLETE_DECK, ScreenState.DECK_SELECT] * 3
    eng, backend = _engine(cfg, states, recovery_slots=(1, 2, 3))
    with pytest.raises(Halt, match="none of the configured recovery decks"):
        eng._handle_deck_select()
    assert len(backend.gestures) == 6      # 3 slot taps + 3 declines, never a Yes


def test_top_level_incomplete_dialog_is_declined(cfg):
    """If hop starts on the dialog, dispatch declines it (never auto-completes)."""
    eng, backend = _engine(cfg, [ScreenState.INCOMPLETE_DECK, ScreenState.DECK_SELECT])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert len(backend.gestures) == 1      # the decline (No)
    assert max(s.x for s in backend.gestures[0].samples) > PANEL.width_px / 2


def test_an_unreadable_frame_is_not_proof_the_dialog_closed(cfg):
    # after No, the classifier never leaves the dialog -> fail closed
    eng, backend = _engine(cfg, [ScreenState.INCOMPLETE_DECK] * 12)
    with pytest.raises(Halt):
        eng._decline_incomplete_deck()
