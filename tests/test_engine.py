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


def test_main_menu_halts_with_guidance(cfg):
    """The loop queues from the deck's Play screen; the main menu is not it."""
    from hop.verify import Halt
    eng, backend = _engine(cfg, [ScreenState.MENU])
    frame = gray_frame(80, 40)
    with pytest.raises(Halt):
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert backend.gestures == []
