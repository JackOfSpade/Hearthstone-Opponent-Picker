from dataclasses import replace
from random import Random

import pytest

from hop.engine import Engine, evaluate_matchup
from hop.hearthstone import GameLayout, MulliganRead
from hop.hero_classes import HeroClass
from hop.humanize.limiter import CapReached, Limiter
from hop.perception.screens import ScreenState

from conftest import FakeAdb, FakeBackend, FakeClassifier, ScriptedCapturer, gray_frame


def _engine(cfg, states, *, target_read=None, capturer=None, alerter=None):
    adb = FakeAdb()
    backend = FakeBackend()
    from hop.geometry import PanelGeometry
    panel = PanelGeometry(800, 400, 400.0)
    classifier = FakeClassifier(states)
    eng = Engine(
        cfg, adb, backend, panel, classifier, reader=None,
        alerts=alerter, sleep=lambda s: None, clock=lambda: 0.0,
        rng=Random(1), layout=GameLayout(),
        capturer=capturer or ScriptedCapturer([gray_frame(80, 40)]),
    )
    if target_read is not None:
        eng._read_mulligan = lambda *a, **k: target_read
    return eng, backend


def test_evaluate_matchup_pure(cfg):
    keep = MulliganRead(HeroClass.MAGE, True, 4, 1.0, "MAGE", "tesseract")
    assert evaluate_matchup(keep, cfg) == "keep"  # no filter -> keep
    unusable = MulliganRead(None, False, 0, 0.0, "??", "tesseract")
    assert evaluate_matchup(unusable, cfg) == "unusable"


def test_target_found_alerts_and_stops(cfg):
    read = MulliganRead(HeroClass.MAGE, True, 4, 0.98, "MAGE", "tesseract")

    class RecordingAlerter:
        found = None
        def target_found(self, name, second): self.found = (name, second)
        def info(self, m): pass
        def halt(self, m): pass

    alerter = RecordingAlerter()
    eng, backend = _engine(cfg, [ScreenState.MULLIGAN], target_read=read, alerter=alerter)
    stats = eng.run(max_iterations=3)
    assert stats.target_found is True
    assert alerter.found == ("Mage", True)
    # the engine must NOT tap the game once a target is found
    assert backend.gestures == []


def test_unknown_screen_halts(cfg):
    """A *persistently* unrecognized screen must fail closed."""
    eng, backend = _engine(cfg, [ScreenState.UNKNOWN])
    stats = eng.run(max_iterations=2)
    assert stats.stop_reason.startswith("halt")
    assert backend.gestures == []  # never blind-tap an unknown screen


def test_transient_unknown_is_tolerated(cfg):
    """Hearthstone animates between screens; one blurred frame is not 'lost'.

    The classifier yields UNKNOWN once, then MULLIGAN. The engine must re-look
    (without tapping) rather than halt.
    """
    read = MulliganRead(HeroClass.MAGE, True, 4, 0.98, "MAGE", "tesseract")
    eng, backend = _engine(cfg, [ScreenState.UNKNOWN, ScreenState.MULLIGAN],
                           target_read=read)
    stats = eng.run(max_iterations=2)
    assert not stats.stop_reason.startswith("halt")
    assert stats.target_found is True


def test_settle_retries_are_bounded(cfg):
    """More consecutive UNKNOWNs than the settle budget -> still halts."""
    n = cfg.vision.unknown_settle_attempts
    states = [ScreenState.UNKNOWN] * (n + 2)
    eng, backend = _engine(cfg, states)
    stats = eng.run(max_iterations=1)
    assert stats.stop_reason.startswith("halt")


def test_no_templates_refuses(cfg):
    adb = FakeAdb()
    from hop.geometry import PanelGeometry

    class Empty(FakeClassifier):
        @property
        def has_templates(self):
            return False

    eng = Engine(cfg, adb, FakeBackend(), PanelGeometry(800, 400, 400.0),
                 Empty([ScreenState.MENU]), reader=None, sleep=lambda s: None,
                 capturer=ScriptedCapturer([gray_frame(80, 40)]))
    stats = eng.run(max_iterations=1)
    assert "halt" in stats.stop_reason


def test_committing_cap_stops_run(cfg):
    # Force a tiny committing cap and drive a concede-menu tap repeatedly.
    lim = Limiter(cfg.caps, scale=1.0)
    lim.commits_this_run = lim.committing_action_cap  # already at cap
    adb = FakeAdb()
    from hop.geometry import PanelGeometry
    panel = PanelGeometry(800, 400, 400.0)
    # before/after frames differ (full transition) so verify would pass if reached
    before = gray_frame(80, 40, 20)
    after = gray_frame(80, 40, 220)
    eng = Engine(cfg, adb, FakeBackend(), panel,
                 FakeClassifier([ScreenState.CONCEDE_MENU]), reader=None,
                 sleep=lambda s: None, rng=Random(1), limiter=lim,
                 capturer=ScriptedCapturer([before, after]))
    stats = eng.run(max_iterations=2)
    assert stats.stop_reason == "cap:committing"


def test_tap_emits_and_registers(cfg):
    # A Play-screen tap: before/after differ as a full transition so verify passes.
    adb = FakeAdb()
    backend = FakeBackend()
    from hop.geometry import PanelGeometry
    panel = PanelGeometry(80, 40, 400.0)
    before = gray_frame(80, 40, 20)
    after = gray_frame(80, 40, 220)
    eng = Engine(cfg, adb, backend, panel, FakeClassifier([ScreenState.PLAY_SCREEN]),
                 reader=None, sleep=lambda s: None, rng=Random(1),
                 capturer=ScriptedCapturer([before, after]))
    # one dispatch of the play screen should emit exactly one Play tap
    eng._dispatch(eng.classifier.classify(before), before)
    assert len(backend.gestures) == 1
    assert eng.limiter.actions_this_run == 1


def test_queue_screen_never_taps(cfg):
    """Tapping while Hearthstone searches for an opponent CANCELS the queue."""
    eng, backend = _engine(cfg, [ScreenState.QUEUE])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert backend.gestures == []
    assert eng.limiter.actions_this_run == 0


def _dispatch_once(cfg, state):
    """Dispatch one screen state with before/after frames that verify as changed."""
    adb = FakeAdb()
    backend = FakeBackend()
    from hop.geometry import PanelGeometry
    before = gray_frame(80, 40, 20)
    after = gray_frame(80, 40, 220)
    eng = Engine(cfg, adb, backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([state]), reader=None, sleep=lambda s: None,
                 rng=Random(1), capturer=ScriptedCapturer([before, after]))
    eng._dispatch(eng.classifier.classify(before), before)
    return eng, backend


def test_error_dialog_is_dismissed_not_halted(cfg):
    """HS throws a transient 'error starting your game'; dismiss and requeue."""
    eng, backend = _dispatch_once(cfg, ScreenState.ERROR_DIALOG)
    assert len(backend.gestures) == 1          # tapped OK
    # dismissing an error is not a committing action (it is not a concede)
    assert eng.limiter.commits_this_run == 0


def test_deck_select_reopens_the_deck(cfg):
    eng, backend = _dispatch_once(cfg, ScreenState.DECK_SELECT)
    assert len(backend.gestures) == 1          # tapped the deck slot


def _bottom_sheet_frame(w=80, h=40, base=20, bottom=220) -> "object":
    """A frame differing from ``gray_frame(base)`` only in its bottom third.

    This reproduces what tapping Reconnect *actually* does: the dialog's body text
    swaps to "Reconnecting..." in place, so `classify_change` reports
    ``bottom_sheet``, not ``full_transition``.
    """
    from hop.perception.image import Frame

    rows = [bytes([base]) * w] * (2 * h // 3) + [bytes([bottom]) * w] * (h - 2 * h // 3)
    return Frame.from_gray_bytes(w, h, b"".join(rows))


class _AlternatingCapturer:
    """Alternates two frames that differ only in the bottom third.

    ScriptedCapturer repeats its final frame, which makes the *second* tap in a
    test look like a missed tap; _reconnect can tap several times across a run.
    Using a bottom-only change also keeps this test honest: with a whole-frame
    change, the old ``expected_change="full_transition"`` would have passed too.
    """

    def __init__(self):
        self.frames = [gray_frame(80, 40, 20), _bottom_sheet_frame()]
        self.i = 0

    def capture(self):
        f = self.frames[self.i % 2]
        self.i += 1
        return f


def _reconnect_engine(cfg, states):
    """Engine whose captures always look 'changed', driving _reconnect's poll."""
    adb = FakeAdb()
    backend = FakeBackend()
    from hop.geometry import PanelGeometry
    eng = Engine(cfg, adb, backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier(states), reader=None, sleep=lambda s: None,
                 rng=Random(1), capturer=_AlternatingCapturer())
    return eng, backend


def test_reconnect_dialog_is_accepted(cfg):
    """HS shuts down idle connections; the loop's own pacing provokes this."""
    eng, backend = _reconnect_engine(
        cfg, [ScreenState.RECONNECT_DIALOG, ScreenState.RECONNECTING, ScreenState.PLAY_SCREEN])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert len(backend.gestures) == 1          # exactly one Reconnect tap
    assert eng.limiter.commits_this_run == 0   # reconnecting is not a concede


def test_reconnecting_screen_never_taps(cfg):
    """Mid-reconnect the buttons are removed; a tap would hit dead space."""
    eng, backend = _dispatch_once(cfg, ScreenState.RECONNECTING)
    assert backend.gestures == []
    assert eng.limiter.actions_this_run == 0


def test_reconnect_does_not_retap_while_reconnecting(cfg):
    """Regression: the single-correction retap fired during 'Reconnecting...'.

    The reconnect is asynchronous - its immediate effect is an in-place body swap,
    not a screen transition. Demanding ``full_transition`` made verify fail, and
    the correction then tapped a dialog whose buttons had been removed.
    """
    eng, backend = _reconnect_engine(
        cfg, [ScreenState.RECONNECT_DIALOG] + [ScreenState.RECONNECTING] * 3
             + [ScreenState.PLAY_SCREEN])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert len(backend.gestures) == 1, "must not emit a correction tap mid-reconnect"


def test_reconnect_attempts_are_capped(cfg):
    """A reconnect that keeps failing needs a human, so fail closed."""
    from hop.verify import Halt

    cap = cfg.vision.reconnect_attempt_cap
    # RECONNECT_DIALOG every time = the attempt resolves back to the button dialog
    eng, backend = _reconnect_engine(cfg, [ScreenState.RECONNECT_DIALOG])
    frame = gray_frame(80, 40)
    for _ in range(cap):
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert len(backend.gestures) == cap
    with pytest.raises(Halt, match="failed to reconnect"):
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert len(backend.gestures) == cap, "no tap once the cap is reached"


def test_stuck_on_reconnecting_halts(cfg):
    """If 'Reconnecting...' never resolves, halt rather than wait forever."""
    from hop.verify import Halt

    eng, _ = _reconnect_engine(cfg, [ScreenState.RECONNECT_DIALOG, ScreenState.RECONNECTING])
    frame = gray_frame(80, 40)
    with pytest.raises(Halt, match="Reconnecting"):
        eng._dispatch(eng.classifier.classify(frame), frame)


def test_reconnect_taps_reconnect_and_never_cancel(cfg):
    """Cancel leaves the client offline, after which every tap is a silent no-op.

    Reconnect and Cancel sit side by side on the same row, so the only thing
    separating "recovered" from "wedged forever" is the x fraction. Pin it.
    """
    layout = GameLayout()
    assert layout.reconnect_button.yf == pytest.approx(0.838, abs=0.005)
    # Reconnect is the LEFT button: it must sit left of screen center, and the
    # FFitts spread (truncated to 0.9*radius) must not reach Cancel at x=0.5808.
    assert layout.reconnect_button.xf < 0.5
    reach = layout.reconnect_button.xf + layout.reconnect_button.radius_f
    assert reach < 0.5808 - 0.019, "tap spread could land on Cancel"

    eng, backend = _dispatch_once(cfg, ScreenState.RECONNECT_DIALOG)
    (gesture,) = backend.gestures
    xs = [s.x for s in gesture.samples]
    panel_w = 80  # _dispatch_once uses an 80x40 panel
    assert max(xs) / panel_w < 0.5, "tap drifted past center toward Cancel"


def test_main_menu_halts_with_guidance(cfg):
    """The loop queues from the deck's Play screen; the main menu is not it."""
    from hop.verify import Halt
    eng, backend = _engine(cfg, [ScreenState.MENU])
    frame = gray_frame(80, 40)
    with pytest.raises(Halt):
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert backend.gestures == []
