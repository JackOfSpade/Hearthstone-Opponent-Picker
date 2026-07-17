"""The deck list: hop pauses there, and never auto-completes a deck.

hop reaches the deck grid only on recovery (an error dropped it back there). It does NOT
reopen a deck: it cannot reliably tell which deck was in play (the grid names render too
soft/stylised to OCR; any other pick could open the wrong deck or an incomplete one), and
re-selecting a *different* deck than the user chose is worse than stopping. So it PAUSES --
alerts and waits, never tapping the list -- and resumes the hunt by itself when the user is
back on a deck's Play screen; only if nobody re-selects within the bounded wait does it fail
closed. The one thing it still does on the deck screens is refuse to auto-complete: on
"Complete deck automatically?" it taps No, never Yes.
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

def test_deck_select_pauses_without_halting_and_never_taps_a_deck(cfg):
    """A single look at the deck list must NOT end the hunt: hop pauses (waits, never
    tapping) so it can resume when the user re-selects. The invariant is unchanged -- it
    never opens a deck, since opening a different one than the user chose is worse."""
    eng, backend = _engine(cfg, [ScreenState.DECK_SELECT])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)   # pauses, does not raise
    assert backend.gestures == []          # never a deck tap
    assert eng._deck_select_polls == 1     # counting toward the bounded wait


def test_deck_select_resumes_when_the_user_returns_to_a_play_screen(cfg):
    """The whole point of pausing: after the user re-selects their deck, the next look is a
    Play screen and the hunt continues on its own (the pause counter resets, Play is tapped)."""
    eng, backend = _engine(cfg, [ScreenState.DECK_SELECT, ScreenState.PLAY_SCREEN])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)   # deck list: pause
    assert backend.gestures == []
    eng._dispatch(eng.classifier.classify(frame), frame)   # play screen: resume -> tap Play
    assert eng._deck_select_polls == 0                     # pause cleared on leaving the list
    assert len(backend.gestures) == 1                      # the Play tap fired


def test_deck_select_halts_only_after_the_wait_budget(cfg):
    """A deck list nobody re-selects is bounded: after the budget it fails closed, so a
    walked-away session still stops cleanly rather than pausing forever -- and never taps."""
    eng, backend = _engine(cfg, [ScreenState.DECK_SELECT])
    frame = gray_frame(80, 40)
    for _ in range(cfg.vision.deck_select_wait_attempts):   # exhaust the budget (each pauses)
        eng._dispatch(eng.classifier.classify(frame), frame)
    with pytest.raises(Halt, match="[Rr]e-select your deck"):
        eng._dispatch(eng.classifier.classify(frame), frame)   # one past the budget -> halt
    assert backend.gestures == []          # still never tapped a deck


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


def test_decline_does_not_blame_a_dropped_tap_when_the_wait_only_saw_unknown(cfg):
    """Regression: breaking early because the look came back UNKNOWN (an unresolved
    transition, not proof of a drop) used to still claim "after {attempts} tap(s) ...
    not registering (a dropped committing tap)" even though only ONE tap was ever sent
    and the real cause is unidentified, not a drop. INCOMPLETE_DECK persisting is the
    drop signature; UNKNOWN persisting is not, and must never be retapped into."""
    eng, backend = _engine(cfg, [ScreenState.UNKNOWN] * 20)
    with pytest.raises(Halt) as e:
        eng._decline_incomplete_deck()
    assert "not registering (a dropped committing tap" not in str(e.value)
    assert "after 1 tap(s)" in str(e.value)
    assert len(backend.gestures) == 1   # never blind-retapped into an unrecognised screen


class _StuckDeckCapturer:
    """Every capture is identical: a Decline tap that never registers a pixel change --
    unlike Concede's board, this dialog has no ambient animation to mask that."""

    def capture(self):
        return gray_frame(80, 40, 60)


class _DeckStuckUntilNthTap:
    """Every capture is identical until the Nth physical gesture has landed."""

    def __init__(self, backend, land_on_attempt):
        self.backend = backend
        self.land_on_attempt = land_on_attempt

    def capture(self):
        if len(self.backend.gestures) >= self.land_on_attempt:
            return gray_frame(80, 40, 220)
        return gray_frame(80, 40, 60)


def test_a_dropped_decline_tap_makes_taps_own_verify_raise_and_still_recovers(cfg):
    """Regression: unlike Concede's animated board, a dropped Decline tap here is a
    genuinely UNCHANGING frame, so `_tap` itself raises `Halt.NO_CHANGE` (not just a
    `_wait_until` timeout) -- an earlier version of this fix let that exception escape
    uncaught, skipping the retry loop entirely. The last of N taps lands and a fresh
    look confirms we left; must not raise."""
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PANEL, FakeClassifier([ScreenState.DECK_SELECT]),
                 reader=None, sleep=lambda s: None, clock=lambda: 0.0, rng=Random(4),
                 layout=GameLayout(),
                 capturer=_DeckStuckUntilNthTap(backend, land_on_attempt=cfg.vision.deck_decline_tap_attempts))
    assert cfg.vision.deck_decline_tap_attempts >= 2
    eng._decline_incomplete_deck()   # must not raise
    assert len(backend.gestures) == cfg.vision.deck_decline_tap_attempts


def test_decline_exhausts_its_retry_budget_then_fails_closed(cfg):
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PANEL, FakeClassifier([ScreenState.INCOMPLETE_DECK]),
                 reader=None, sleep=lambda s: None, clock=lambda: 0.0, rng=Random(4),
                 layout=GameLayout(), capturer=_StuckDeckCapturer())
    with pytest.raises(Halt) as e:
        eng._decline_incomplete_deck()
    assert "not registering (a dropped committing tap" in str(e.value)
    assert len(backend.gestures) == cfg.vision.deck_decline_tap_attempts


def test_only_a_no_change_halt_is_retried_for_decline(cfg):
    """A coherence/non-repetition failure means the automation is wrong, not the
    client -- it must not be retried or folded into the dropped-tap message."""
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PANEL,
                 FakeClassifier([ScreenState.INCOMPLETE_DECK]), reader=None,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(4),
                 layout=GameLayout(), capturer=_StuckDeckCapturer())

    def wrong_change(*a, **k):
        raise Halt("screen changed but not as expected", Halt.WRONG_CHANGE)

    eng.verifier.verify = wrong_change
    with pytest.raises(Halt) as e:
        eng._decline_incomplete_deck()
    assert e.value.kind == Halt.WRONG_CHANGE
    assert len(eng.backend.gestures) == 1
