from random import Random

import pytest

from hop.engine import Engine, evaluate_matchup
from hop.hearthstone import GameLayout, MulliganRead
from hop.hero_classes import HeroClass
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


def test_phase_label_is_friendly_and_total():
    from hop.engine import _phase_label
    assert "queue" in _phase_label(ScreenState.QUEUE)
    assert "mulligan" in _phase_label(ScreenState.MULLIGAN).lower()
    # every state maps to a non-empty label (unmapped ones fall back to the value)
    for st in ScreenState:
        assert _phase_label(st)


def test_interruptible_sleep_raises_when_stopped(cfg):
    from hop.engine import StopRequested
    eng, _ = _engine(cfg, [ScreenState.MENU])
    eng.request_stop()
    with pytest.raises(StopRequested):
        eng.sleep(5.0)          # a long wait is abandoned immediately, not after 5 s


def test_interruptible_sleep_is_a_noop_when_not_stopped(cfg):
    eng, _ = _engine(cfg, [ScreenState.MENU])
    eng.sleep(0.01)             # returns normally; no raise


def test_run_exits_with_user_stop_when_stopped(cfg):
    eng, _ = _engine(cfg, [ScreenState.MENU])
    eng.request_stop()
    stats = eng.run()
    assert stats.stop_reason == "user_stop"


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


def test_target_records_concedes_until_target(cfg):
    """The hunt stops on a target, so `concedes` at that moment is 'concedes until target'."""
    read = MulliganRead(HeroClass.MAGE, True, 4, 0.98, "MAGE", "tesseract")
    eng, _ = _engine(cfg, [ScreenState.MULLIGAN], target_read=read)
    eng.stats.concedes = 7                      # pretend 7 non-targets were conceded first
    frame = gray_frame(80, 40)
    eng._handle_mulligan(frame)
    assert eng.stats.target_found is True
    assert eng.stats.concedes_until_target == 7


def test_mulligan_read_accumulates_the_class_distribution(cfg):
    """Every game (target or not) is counted into the observed class histogram + coin split."""
    from dataclasses import replace
    cfg = replace(cfg, criteria=replace(cfg.criteria, target_classes=(HeroClass.PALADIN,)))
    eng, _ = _engine(cfg, [ScreenState.MULLIGAN])   # Mage/Warrior are rejects vs a Paladin target
    eng._execute_reject = lambda read: None     # skip the full concede journey
    frame = gray_frame(80, 40)

    for cls, second in [(HeroClass.MAGE, False), (HeroClass.MAGE, True), (HeroClass.WARRIOR, False)]:
        eng._read_mulligan = lambda *a, r=MulliganRead(cls, second, 4 if second else 3, 0.9, "X", "t"), **k: r
        eng._handle_mulligan(frame)

    assert eng.stats.class_distribution == {"Mage": 2, "Warrior": 1}
    assert eng.stats.going_first == 2 and eng.stats.going_second == 1
    assert eng.stats.concedes_until_target is None   # no target yet


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


def test_a_soft_locked_queue_halts_instead_of_spinning_forever(cfg):
    """Tapping the queue cancels it, so the only safe move is to wait -- which made
    this the loop's one reachable infinite spin. No tap, so no cap; no cap, so no stop.

    (The attempt count, not the frozen test clock, is what terminates this.)
    """
    from hop.verify import Halt

    eng, backend = _engine(cfg, [ScreenState.QUEUE])
    frame = gray_frame(80, 40)
    for _ in range(cfg.vision.queue_wait_attempts):
        eng._dispatch(eng.classifier.classify(frame), frame)
    with pytest.raises(Halt) as e:
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert "matchmaking never matched" in str(e.value)
    assert backend.gestures == []          # never, ever tap the queue


def test_finding_a_match_resets_the_queue_patience(cfg):
    eng, _backend = _engine(cfg, [ScreenState.QUEUE, ScreenState.VS_SPLASH])
    frame = gray_frame(80, 40)
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert eng._queue_polls == 1
    eng._dispatch(eng.classifier.classify(frame), frame)
    assert eng._queue_polls == 0


def test_session_time_is_accrued_as_wall_clock(cfg):
    """Session time is wall clock, accrued once per loop iteration (not only inside
    `_tap`). It is informational now - no cap - so a tapless loop like a stuck queue
    just keeps polling until the queue-poll cap Halts it, never a session cap."""
    eng, backend = _engine(cfg, [ScreenState.QUEUE])
    ticks = iter([float(i) for i in range(200)])
    eng.clock = lambda: next(ticks)
    stats = eng.run(max_iterations=5)
    assert eng.limiter.session_seconds > 0        # time was accrued
    assert not (stats.stop_reason or "").startswith("cap")  # never a volume/time cap
    assert backend.gestures == []                 # a stuck queue still taps nothing


def test_error_dialog_is_dismissed_not_halted(cfg):
    """HS throws a transient 'error starting your game'; dismiss and requeue."""
    eng, backend = _dispatch_once(cfg, ScreenState.ERROR_DIALOG)
    assert len(backend.gestures) == 1          # tapped OK
    # dismissing an error is not a committing action (it is not a concede)
    assert eng.limiter.commits_this_run == 0


# deck-select behaviour lives in tests/test_deck_select.py (the cycler needs a
# scripted classifier, not the single-state _dispatch_once helper).


def test_collection_is_backed_out_of_not_halted(cfg):
    """hop is never meant to be in the Collection, but a stray navigation there must be
    recoverable: tap the back arrow to the deck list, don't halt.

    (Live-verified on a Pixel 7a: collection 0.92 -> back -> deck_select 0.91.)
    """
    eng, backend = _dispatch_once(cfg, ScreenState.COLLECTION)
    assert len(backend.gestures) == 1          # tapped the back arrow
    assert eng.limiter.commits_this_run == 0   # backing out is not a committing action


def test_collection_back_tap_stays_on_the_button_not_off_the_bottom():
    """Wide, short button: the hit radius must come from the smaller (height) half.

    Its centre is below Android's y=996 gesture inset (taps there are delivered on this
    device), but the 0.9r disc must not spill off the bottom of a 1080px panel.
    """
    from hop.hearthstone import GameLayout
    from hop.geometry import PanelGeometry

    panel = PanelGeometry(2400, 1080, 420.0)
    x, y, r = GameLayout().collection_back.to_px(panel)
    assert y + r * 0.9 <= panel.height_px - 1
    assert 1930 <= x <= 2130                    # within the button interior
    assert r < 45                               # from the ~37px height half, not width


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


def test_a_successful_reconnect_resets_the_attempt_counter(cfg):
    """Regression: the attempt counter was a lifetime tally, never reset on success.

    Idle disconnects are expected traffic the loop's own pacing provokes; each is
    individually recovered. The cap is meant to bound *consecutive failures* within one
    episode ('needs a human'), so more successful reconnects than the cap must NOT halt.
    """
    from hop.verify import Halt

    cap = cfg.vision.reconnect_attempt_cap
    # the poll resolves to PLAY_SCREEN on the first look -> a clean, successful reconnect
    eng, backend = _reconnect_engine(cfg, [ScreenState.PLAY_SCREEN])
    for _ in range(cap + 3):        # more successful episodes than the failure cap
        eng._reconnect()            # would raise Halt on the (cap+1)th without the reset
        assert eng._reconnect_attempts == 0     # each success zeroes the counter
    assert len(backend.gestures) == cap + 3     # one Reconnect tap per episode, no Halt


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


class _StuckCardCapturer:
    """Captures never change: every card tap looks ignored."""

    def capture(self):
        return gray_frame(80, 40, 60)


def _mulligan_engine(cfg, capturer):
    from hop.geometry import PanelGeometry
    return Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                  FakeClassifier([ScreenState.MULLIGAN]), reader=None,
                  sleep=lambda s: None, rng=Random(2), capturer=capturer)


def test_an_ignored_card_tap_retries_then_keeps_the_card(cfg):
    """Hearthstone accepts card taps only intermittently; never halt the hunt on one.

    We are still on the mulligan and know exactly where we are, so this is not the
    unknown state that fail-closed exists for. Retry, then keep the card.
    """
    eng = _mulligan_engine(cfg, _StuckCardCapturer())
    took = eng._replace_card(slot=0, center_xf=0.25, decision_type="reject")
    assert took is False
    assert eng.stats.ignored_card_taps == 1
    assert len(eng.backend.gestures) == cfg.vision.mulligan_card_tap_attempts


def test_a_card_tap_that_takes_stops_retrying(cfg):
    class _TogglesOnFirstTap:
        def __init__(self):
            self.n = 0

        def capture(self):
            self.n += 1
            # before, then a clearly-changed after
            return gray_frame(80, 40, 20 if self.n % 2 else 220)

    eng = _mulligan_engine(cfg, _TogglesOnFirstTap())
    assert eng._replace_card(slot=1, center_xf=0.5, decision_type="reject") is True
    assert eng.stats.ignored_card_taps == 0
    assert len(eng.backend.gestures) == 1


def test_only_a_no_change_halt_counts_as_an_ignored_card_tap(cfg):
    """`stats.ignored_card_taps` is the statistic CALIBRATION.md chases the ~67%
    card-tap mystery with. Folding coherence or non-repetition failures into it both
    hides a real fault and poisons the evidence that would reveal it.

    Only "the screen did not move" means the game ignored the tap.
    """
    from hop.verify import Halt

    eng = _mulligan_engine(cfg, _StuckCardCapturer())

    def wrong_change(*a, **k):
        raise Halt("screen changed but not as expected", Halt.WRONG_CHANGE)

    eng.verifier.verify = wrong_change
    with pytest.raises(Halt) as e:
        eng._replace_card(slot=0, center_xf=0.25, decision_type="reject")
    assert e.value.kind == Halt.WRONG_CHANGE
    assert eng.stats.ignored_card_taps == 0        # not our statistic to inflate
    assert len(eng.backend.gestures) == 1          # and not retried three times


def test_card_taps_are_verified_against_the_card_not_the_screen(cfg):
    """Regression: a whole-frame check called a real toggle 'no screen change'.

    Measured on the phone: marking a card moves the whole frame by 4.80 (threshold
    9.0) and the card's own rectangle by 24.24. Only the scoped check can see it.
    """
    from hop.hearthstone import GameLayout

    layout = GameLayout()
    region = layout.card_region(0.25)
    assert region.wf == pytest.approx(layout.card_inner_w_f)
    assert region.hf == pytest.approx(layout.card_row.hf)
    # centred on the card, not on the screen
    assert region.xf + region.wf / 2 == pytest.approx(0.25, abs=1e-6)


def test_main_menu_halts_with_guidance(cfg):
    """The loop queues from the deck's Play screen; the main menu is not it."""
    from hop.verify import Halt
    eng, backend = _engine(cfg, [ScreenState.MENU])
    frame = gray_frame(80, 40)
    with pytest.raises(Halt):
        eng._dispatch(eng.classifier.classify(frame), frame)
    assert backend.gestures == []


def test_mulligan_confirm_waits_for_the_mulligan_to_leave(cfg):
    """Confirming is asynchronous: within the settle window only the bottom moves.

    Regression: demanding `full_transition` halted on a confirm that had worked
    (measured `bottom_sheet`), and the single-correction retap then fired *after* the
    mulligan was already confirmed -- a blind tap into a live game.
    """
    eng, backend = _reconnect_engine(
        cfg, [ScreenState.MULLIGAN, ScreenState.IN_GAME])
    eng._confirm_mulligan()
    assert len(backend.gestures) == 1, "must not emit a correction tap into a live game"


def test_mulligan_confirm_halts_if_the_mulligan_never_leaves(cfg):
    from hop.verify import Halt

    eng, _ = _reconnect_engine(cfg, [ScreenState.MULLIGAN])
    with pytest.raises(Halt, match="did not dismiss"):
        eng._confirm_mulligan()


def test_every_tap_site_is_named_for_the_journal(cfg):
    """A tap logged as `?` can't be attributed when a run halts.

    The one that slipped through was the *concede* -- the committing action, the
    single most important thing to be able to identify in a journal.
    """
    import inspect
    import re

    import hop.engine as engine

    src = inspect.getsource(engine.Engine)
    for m in re.finditer(r"self\._tap\((.{0,320}?)\)\n", src, re.DOTALL):
        call = m.group(1)
        assert "what=" in call, f"unnamed tap site: {call.splitlines()[0].strip()}"


def test_tap_records_timing_evidence_for_delay_reports(cfg):
    from hop.geometry import PanelGeometry

    class RecordingDebug:
        def __init__(self):
            self.events = []

        def record(self, kind, /, **detail):
            self.events.append((kind, detail))

    dbg = RecordingDebug()
    before = gray_frame(80, 40, 20)
    after = gray_frame(80, 40, 220)
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.PLAY_SCREEN]), reader=None,
                 sleep=lambda s: None, rng=Random(1), debug=dbg,
                 capturer=ScriptedCapturer([before, after]))
    eng._tap(GameLayout().play_button, committing=False, decision_type="commit",
             expected_change="full_transition", what="play")

    kinds = [k for k, _ in dbg.events]
    assert "tap_timing" in kinds
    assert any(k == "sleep" and d["reason"] == "tap_think" and d["what"] == "play"
               for k, d in dbg.events)
