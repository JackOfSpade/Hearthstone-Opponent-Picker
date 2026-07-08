"""The post-game path, and the concede that leads into it.

Neither `_concede` nor `_clear_end_screens` had a single test, which is how a tap on
the Game Menu's **Quit** entry survived in the tree: `concede_confirm = Point(0.50,
0.56, 0.06)` was the recovery move for "the Concede tap was ignored", and on this
client the Game Menu reads Concede / Options / Quit -- so that point is dead centre
on Quit. Its 0.9*144 px truncation disc lies almost entirely on the Quit plate.

`end_dismiss` had the mirror problem: a fixed point tapped on *any* state outside a
five-case whitelist. At (1200, 972) with a 216 px disc it is 27 px from the mulligan's
Confirm, 205 px from the reconnect dialog's *Cancel*, and on top of the deck list's
"My Collection" plate.
"""

from random import Random

import pytest

from hop.engine import Engine
from hop.hearthstone import GameLayout, Point
from hop.geometry import PanelGeometry
from hop.perception.screens import ScreenState
from hop.verify import Halt

from conftest import FakeAdb, FakeBackend, FakeClassifier, gray_frame


class _AlternatingCapturer:
    """Every capture differs from the last, so every `_tap` verifies as changed."""

    def __init__(self):
        self.n = 0

    def capture(self):
        self.n += 1
        return gray_frame(80, 40, 20 if self.n % 2 else 220)


class _RecordingDebug:
    def __init__(self):
        self.unknowns = []
        self.records = []

    def record(self, kind, **detail):
        self.records.append((kind, detail))

    def anomaly(self, reason, before=None, after=None, **ctx):
        self.records.append(("anomaly", {"reason": reason}))

    def unknown_screen(self, frame, *, where, **ctx):
        self.unknowns.append((where, frame, ctx))
        return None


def _engine(cfg, states, *, debug=None):
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier(states), reader=None, debug=debug,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_AlternatingCapturer())
    return eng, backend


# ── the concede must never reach Quit ────────────────────────────────────────

def test_there_is_no_concede_confirm_point():
    """The Game Menu's second and third entries are Options and Quit.

    No fixed point below Concede is safe, so the layout must not offer one.
    """
    assert not hasattr(GameLayout(), "concede_confirm")


def test_engine_never_taps_a_concede_confirm():
    import inspect

    import hop.engine as engine

    # the name may appear in prose explaining why it is gone; it must not be tapped
    assert "layout.concede_confirm" not in inspect.getsource(engine.Engine)


def test_concede_halts_rather_than_tapping_below_the_concede_button(cfg):
    """If Concede is ignored, the menu is still up -- stop, never tap again.

    The tap below Concede is Quit. Exactly two gestures may leave the engine here:
    the gear, and Concede itself.
    """
    eng, backend = _engine(cfg, [ScreenState.CONCEDE_MENU])
    with pytest.raises(Halt) as e:
        eng._concede()
    assert "Quit" in str(e.value)
    assert len(backend.gestures) == 2       # gear, concede -- and nothing else


def test_concede_returns_once_the_menu_is_gone(cfg):
    eng, backend = _engine(cfg, [ScreenState.IN_GAME])
    eng._concede()                           # board dissolving == menu left
    assert len(backend.gestures) == 2


def test_concede_does_not_accept_unknown_as_proof_the_menu_left(cfg):
    """An unreadable frame is not an observation that we left the menu."""
    debug = _RecordingDebug()
    eng, backend = _engine(cfg, [ScreenState.UNKNOWN], debug=debug)
    with pytest.raises(Halt):
        eng._concede()
    assert len(backend.gestures) == 2
    assert [w for w, _f, _c in debug.unknowns] == ["concede"]


# ── end_dismiss may only land on a screen we named ───────────────────────────

def test_deck_select_is_a_home_screen(cfg):
    """Hearthstone drops back to the deck LIST after a game.

    If DECK_SELECT is not terminal here, the loop taps end_dismiss on the deck list --
    i.e. on "My Collection" and on the deck boxes (which would queue the wrong deck).
    """
    assert ScreenState.DECK_SELECT in Engine.HOME_SCREENS
    eng, backend = _engine(cfg, [ScreenState.DECK_SELECT])
    eng._clear_end_screens()
    assert backend.gestures == []


@pytest.mark.parametrize("state", [
    ScreenState.RECONNECT_DIALOG,   # end_dismiss's disc reaches Cancel (205 px)
    ScreenState.MULLIGAN,           # ...and the mulligan's Confirm (27 px)
    ScreenState.CONCEDE_MENU,
    ScreenState.ERROR_DIALOG,
    ScreenState.RECONNECTING,
])
def test_unexpected_screens_halt_instead_of_being_blind_tapped(cfg, state):
    eng, backend = _engine(cfg, [state])
    with pytest.raises(Halt) as e:
        eng._clear_end_screens()
    assert "refusing to blind-tap" in str(e.value)
    assert backend.gestures == []


def test_end_screens_are_dismissed_until_a_home_screen(cfg):
    eng, backend = _engine(cfg, [ScreenState.VICTORY, ScreenState.REWARDS,
                                 ScreenState.DECK_SELECT])
    eng._clear_end_screens()
    assert len(backend.gestures) == 2


def test_exhausting_the_tap_budget_halts_rather_than_returning_quietly(cfg):
    """A silent return here books a completed game and requeues into a stuck client."""
    eng, backend = _engine(cfg, [ScreenState.VICTORY])
    with pytest.raises(Halt) as e:
        eng._clear_end_screens(max_taps=2)
    assert "did not clear" in str(e.value)
    assert len(backend.gestures) == 2


def test_waiting_out_the_board_fade_does_not_spend_the_tap_budget(cfg):
    """The dissolving board is a wait, not a tap; it used to consume `max_taps`."""
    states = [ScreenState.IN_GAME] * 5 + [ScreenState.VICTORY, ScreenState.DECK_SELECT]
    eng, backend = _engine(cfg, states)
    eng._clear_end_screens(max_taps=1)       # one tap is enough, despite 5 waits
    assert len(backend.gestures) == 1


def test_a_board_that_never_dissolves_halts(cfg):
    eng, backend = _engine(cfg, [ScreenState.IN_GAME])
    with pytest.raises(Halt) as e:
        eng._clear_end_screens()
    assert "never finished dissolving" in str(e.value)
    assert backend.gestures == []


def test_unknown_while_clearing_keeps_the_pixels(cfg):
    """The one place most likely to meet a screen nobody anticipated.

    A "Ban Notice" modal over the deck list is what actually halted a live run; the
    old code raised without saving the frame, so the anchor could never be built.
    """
    debug = _RecordingDebug()
    eng, backend = _engine(cfg, [ScreenState.UNKNOWN], debug=debug)
    with pytest.raises(Halt):
        eng._clear_end_screens()
    assert [w for w, _f, _c in debug.unknowns] == ["clear_end_screens"]
    assert backend.gestures == []


# ── end_dismiss geometry ─────────────────────────────────────────────────────

def _disc(p: Point, panel: PanelGeometry):
    x, y, r = p.to_px(panel)
    return x, y, r * 0.9          # motor truncates the endpoint spread to 0.9r


def test_end_dismiss_disc_stays_on_the_panel_and_off_the_gesture_inset():
    """radius_f is a fraction of the WIDTH, applied on a screen 1080 px tall.

    The old (0.50, 0.90, 0.10) meant a 240 px disc centred 108 px from the bottom:
    13.1% of sampled endpoints fell off the panel, 39.8% into Android's bottom
    `mandatorySystemGestures` inset (y >= 996 on this device).
    """
    panel = PanelGeometry(2400, 1080, 420.0)
    x, y, lim = _disc(GameLayout().end_dismiss, panel)
    assert 0 <= x - lim and x + lim <= panel.width_px - 1
    assert 0 <= y - lim and y + lim <= panel.height_px - 1
    assert y + lim < 996, "tap can reach Android's bottom system-gesture inset"
    assert y + lim < 990, "tap can reach the deck list's 'My Collection' plate"


def test_end_dismiss_overlaps_the_mulligan_confirm_which_is_why_the_whitelist_exists():
    """These two points are ~11 px apart. That is not a bug -- it is a landmine.

    Both controls live at the bottom centre of their own screen, so a fixed dismiss
    point necessarily sits on the mulligan's Confirm. Nothing about the coordinates
    can fix that; the only thing standing between them is
    :attr:`Engine.END_SCREENS` -- `_clear_end_screens` refuses to tap anywhere it has
    not positively identified as an end screen. Pin the overlap here so that if
    someone ever loosens that whitelist, this test explains what they just armed.
    """
    import math

    panel = PanelGeometry(2400, 1080, 420.0)
    layout = GameLayout()
    x, y, lim = _disc(layout.end_dismiss, panel)
    cx, cy, _ = layout.mulligan_confirm.to_px(panel)
    assert math.hypot(cx - x, cy - y) < lim          # they DO overlap
    assert ScreenState.MULLIGAN not in Engine.END_SCREENS
    assert ScreenState.MULLIGAN not in Engine.HOME_SCREENS   # => Halt, never tap
