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

from dataclasses import replace
import json
import math
from random import Random

import pytest

from hop.engine import Engine
from hop.hearthstone import GameLayout, Point
from hop.geometry import PanelGeometry
from hop.perception.screens import Classification, ScreenState
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
        self.terminal_frames = []

    def record(self, kind, **detail):
        self.records.append((kind, detail))

    def anomaly(self, reason, before=None, after=None, **ctx):
        self.records.append(("anomaly", {"reason": reason}))

    def unknown_screen(self, frame, *, where, **ctx):
        self.unknowns.append((where, frame, ctx))
        return None

    def terminal_screen(self, reason, frame=None, **ctx):
        self.terminal_frames.append(frame)
        self.records.append(("terminal_screen", {"reason": reason, **ctx}))


def _engine(cfg, states, *, debug=None):
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier(states), reader=None, debug=debug,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_AlternatingCapturer())
    return eng, backend


class _CountingCapturer:
    def __init__(self):
        self.calls = 0
        self.frames = []

    def capture(self):
        self.calls += 1
        frame = gray_frame(80, 40, 60)
        self.frames.append(frame)
        return frame


def test_reject_uses_one_bounded_open_loop_play_clickthrough(cfg):
    """The usual rejected mulligan never classifies the individual end screens.

    The only capture is the semantic boundary after the full fixed-location burst
    and its queue cooldown.  This locks the safety property that the burst ends
    before the engine is permitted to touch a future mulligan.
    """
    debug = _RecordingDebug()
    backend = FakeBackend()
    capturer = _CountingCapturer()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.QUEUE]), reader=None, debug=debug,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=capturer)

    eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)

    assert capturer.calls == 1
    assert eng._deferred_screen is not None
    assert eng._deferred_screen[0].state == ScreenState.QUEUE

    taps = [(d["what"], d["point"]) for kind, d in debug.records if kind == "tap"]
    gear_pt = tuple(round(v) for v in eng.layout.gear_button.to_px(eng.panel)[:2])
    concede_pt = tuple(round(v) for v in eng.layout.concede_button.to_px(eng.panel)[:2])
    concede_now_pt = tuple(round(v) for v in eng.layout.concede_now_button.to_px(eng.panel)[:2])
    play_pt = tuple(round(v) for v in eng.layout.play_button.to_px(eng.panel)[:2])
    assert taps.count(("gear", gear_pt)) == 1
    assert taps.count(("concede", concede_pt)) == cfg.vision.concede_tap_attempts
    assert taps.count(("concede_now", concede_now_pt)) == 1
    assert taps.count(("post_concede_play", play_pt)) == cfg.timing.post_concede_click_count
    assert len(taps) == 2 + cfg.vision.concede_tap_attempts + cfg.timing.post_concede_click_count
    assert not {what for what, _pt in taps} & {"end_dismiss", "mulligan_confirm", "pass"}

    sleeps = [d for kind, d in debug.records if kind == "sleep"]
    click_gaps = [d for d in sleeps if d["reason"] == "post_concede_click_cadence"]
    assert len(click_gaps) == cfg.timing.post_concede_click_count - 1
    assert all(cfg.timing.post_concede_click_interval_s <= d["seconds"]
               <= cfg.timing.post_concede_click_interval_max_s for d in click_gaps)
    assert sum(d["seconds"] for d in click_gaps) <= (
        (cfg.timing.post_concede_click_count - 1)
        * cfg.timing.post_concede_click_interval_max_s
    ) < cfg.timing.play_to_mulligan_observed_min_s
    # The two known UI transitions are deliberate half-second opening pauses:
    # gear -> Game Menu, then Concede -> Black Market confirmation.  They must
    # stay short and occur before the tap that depends on each newly opened UI.
    gear = next(i for i, (kind, d) in enumerate(debug.records)
                if kind == "tap" and d["what"] == "gear")
    gear_wait, gear_wait_detail = next(
        (i, d) for i, (kind, d) in enumerate(debug.records)
        if kind == "sleep" and d["reason"] == "gear_menu_cooldown")
    first_concede = next(i for i, (kind, d) in enumerate(debug.records)
                         if kind == "tap" and d["what"] == "concede")
    last_concede = max(i for i, (kind, d) in enumerate(debug.records)
                       if kind == "tap" and d["what"] == "concede")
    concede_now = next(i for i, (kind, d) in enumerate(debug.records)
                       if kind == "tap" and d["what"] == "concede_now")
    start_wait, start_wait_detail = next(
        (i, d) for i, (kind, d) in enumerate(debug.records)
        if kind == "sleep" and d["reason"] == "post_concede_start_cooldown")
    first_burst = next(i for i, (kind, d) in enumerate(debug.records)
                       if kind == "tap" and d["what"] == "post_concede_play")
    assert gear < gear_wait < first_concede
    assert gear_wait_detail["seconds"] == 0.5
    assert last_concede < start_wait < concede_now < first_burst
    assert start_wait_detail["seconds"] == 0.5
    assert not [d for kind, d in debug.records
                if kind == "sleep" and d.get("what") == "concede_now"]
    # The last burst event precedes the queue cooldown and the sole capture.
    last_burst = max(i for i, (kind, d) in enumerate(debug.records)
                     if kind == "tap" and d["what"] == "post_concede_play")
    queue_wait = next(i for i, (kind, d) in enumerate(debug.records)
                      if kind == "sleep" and d["reason"] == "post_concede_queue_cooldown")
    capture = next(i for i, (kind, _d) in enumerate(debug.records) if kind == "capture")
    assert last_burst < queue_wait < capture


@pytest.mark.parametrize("boundary, recovery_tap", [
    (ScreenState.CONCEDE_WARNING, "concede_now_recovery"),
    (ScreenState.CONCEDE_MENU, "concede_recovery"),
])
def test_post_concede_warning_or_menu_gets_one_bounded_recovery(cfg, boundary, recovery_tap):
    """The named boundary is the sole authority for one safe retry, never recursion."""
    debug = _RecordingDebug()
    backend = FakeBackend()
    capturer = _CountingCapturer()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([boundary, ScreenState.QUEUE]), reader=None, debug=debug,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=capturer)

    result = eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)

    assert result.state == ScreenState.QUEUE
    assert capturer.calls == 2
    taps = [d["what"] for kind, d in debug.records if kind == "tap"]
    assert taps.count(recovery_tap) == 1
    assert sum(what.endswith("recovery") for what in taps) == (
        1 if boundary == ScreenState.CONCEDE_WARNING else 2)
    normal_count = 2 + cfg.vision.concede_tap_attempts + cfg.timing.post_concede_click_count
    assert len(backend.gestures) == normal_count + (
        1 + cfg.timing.post_concede_click_count
        if boundary == ScreenState.CONCEDE_WARNING else 2 + cfg.timing.post_concede_click_count)
    boundaries = [d for kind, d in debug.records if kind == "post_concede_boundary"]
    assert [(d["attempt"], d["state"]) for d in boundaries] == [
        (0, boundary.value), (1, ScreenState.QUEUE.value)]
    bursts = [d for kind, d in debug.records if kind == "post_concede_burst_complete"]
    assert [(d["attempt"], d["recovery"]) for d in bursts] == [(0, "normal"),
                                                                    (1, "warning" if boundary == ScreenState.CONCEDE_WARNING else "menu")]


@pytest.mark.parametrize("second_boundary", [ScreenState.CONCEDE_WARNING, ScreenState.CONCEDE_MENU])
def test_post_concede_recovery_terminal_keeps_frame_and_attempt_context(cfg, second_boundary):
    debug = _RecordingDebug()
    capturer = _CountingCapturer()
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.CONCEDE_WARNING, second_boundary]),
                 reader=None, debug=debug, sleep=lambda _s: None, clock=lambda: 0.0,
                 rng=Random(3), layout=GameLayout(), capturer=capturer)

    with pytest.raises(Halt, match=f"unexpected '{second_boundary.value}'"):
        eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)

    assert capturer.calls == 2  # one boundary per bounded attempt, no evidence recapture
    evidence = [d for kind, d in debug.records if kind == "terminal_screen"]
    assert len(evidence) == 1
    assert evidence[0]["state"] == second_boundary.value
    assert evidence[0]["attempt"] == 1
    assert evidence[0]["recovery"] == "warning"
    assert evidence[0]["initial_state"] == ScreenState.CONCEDE_WARNING.value
    assert evidence[0]["source"] == ScreenState.MULLIGAN.value
    assert evidence[0]["concede_now_taps"] == 2
    assert "queue" in evidence[0]["accepted_states"]
    assert len(debug.terminal_frames) == 1
    assert debug.terminal_frames[0] is capturer.frames[-1]


def test_generic_dispatch_never_blind_taps_an_early_concede_warning(cfg):
    eng, backend = _engine(cfg, [ScreenState.QUEUE])

    with pytest.raises(Halt, match="outside the owned post-concede trace"):
        eng._dispatch(Classification(ScreenState.CONCEDE_WARNING, 1.0), gray_frame(80, 40))

    assert backend.gestures == []


def test_post_concede_terminal_context_attributes_an_in_game_source(cfg):
    debug = _RecordingDebug()
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.IN_GAME]), reader=None, debug=debug,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())

    with pytest.raises(Halt, match="unexpected 'in_game'"):
        eng._post_concede_clickthrough(source=ScreenState.IN_GAME)

    terminal = next(d for kind, d in debug.records if kind == "terminal_screen")
    assert terminal["source"] == ScreenState.IN_GAME.value


def test_post_concede_terminal_records_classifier_and_timing_evidence(cfg):
    """A future report can distinguish a real full-scan board from a scope hit."""
    debug = _RecordingDebug()

    class AnchoredBoardClassifier:
        has_templates = True

        @staticmethod
        def classify(_frame):
            return Classification(ScreenState.IN_GAME, 0.774, (1187, 214))

        @staticmethod
        def classify_expected(_frame, _expected):
            return Classification(ScreenState.IN_GAME, 0.774, (1187, 214))

    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 AnchoredBoardClassifier(), reader=None, debug=debug,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())

    with pytest.raises(Halt, match="unexpected 'in_game'"):
        eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)

    boundary = next(d for kind, d in debug.records if kind == "post_concede_boundary")
    terminal = next(d for kind, d in debug.records if kind == "terminal_screen")
    for detail in (boundary, terminal):
        assert detail["classification_scan"] == "full_fallback"
        assert detail["anchor_at"] == [1187, 214]
        assert detail["mulligan_floor_s"] == cfg.timing.play_to_mulligan_observed_min_s
        assert detail["mulligan_deadline_exceeded"] is False


def test_reject_books_the_finished_game_but_owns_the_queued_successor(cfg):
    """`_book_game` closes the concede before the burst's new Play ownership is set."""
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.QUEUE]), reader=None,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())
    from hop.hearthstone import MulliganRead
    from hop.hero_classes import HeroClass

    eng._execute_reject(MulliganRead(HeroClass.MAGE, False, 3, 1.0, "MAGE", "test"))

    assert eng.stats.games == 1 and eng.stats.concedes == 1
    assert eng._own_game is True


def test_reject_does_not_claim_a_successor_until_the_boundary_names_one(cfg):
    """A returned Play screen is named, but it has not yet queued the next game."""
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.PLAY_SCREEN]), reader=None,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())
    from hop.hearthstone import MulliganRead
    from hop.hero_classes import HeroClass

    eng._execute_reject(MulliganRead(HeroClass.MAGE, False, 3, 1.0, "MAGE", "test"))

    assert eng.stats.games == 1 and eng.stats.concedes == 1
    assert eng._own_game is False


def test_post_burst_mulligan_boundary_uses_the_class_gate_and_books_only_the_old_game(cfg):
    """A very fast successor can already be on mulligan at the one burst boundary.

    Its class must be read from that deferred frame before any new game is counted
    or rejected; the just-conceded game remains booked exactly once.
    """
    from dataclasses import replace
    from hop.hearthstone import MulliganRead
    from hop.hero_classes import HeroClass

    cfg = replace(cfg, criteria=replace(cfg.criteria, target_classes=(HeroClass.MAGE,)))
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.MULLIGAN]), reader=None,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())
    eng._read_mulligan = lambda *_a: MulliganRead(
        HeroClass.MAGE, False, 3, 1.0, "MAGE", "test")

    eng._execute_reject(MulliganRead(HeroClass.WARRIOR, False, 3, 1.0, "WARRIOR", "test"))
    stats = eng.run(max_iterations=3)

    assert stats.target_found is True
    assert stats.games == 1 and stats.concedes == 1
    assert stats.class_distribution == {"Mage": 1}


def test_reject_unknown_boundary_books_once_then_watches_hands_off_until_class(cfg):
    """An unanchored quiet boundary is an in-progress search, not a failed concede.

    The old game is booked exactly once. The following UNKNOWN watch frame has no
    wait or tap, and the readable mulligan frame continues with ordinary logic.
    """
    debug = _RecordingDebug()
    backend = FakeBackend()
    from hop.hearthstone import MulliganRead
    from hop.hero_classes import HeroClass

    cfg = replace(cfg, criteria=replace(cfg.criteria, target_classes=(HeroClass.MAGE,)))
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.UNKNOWN, ScreenState.UNKNOWN,
                                 ScreenState.MULLIGAN]), reader=None, debug=debug,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())
    eng._read_mulligan = lambda *_a: MulliganRead(
        HeroClass.MAGE, False, 3, 1.0, "MAGE", "test")

    eng._execute_reject(MulliganRead(HeroClass.WARRIOR, False, 3, 1.0, "WARRIOR", "test"))

    burst_gestures = 2 + cfg.vision.concede_tap_attempts + cfg.timing.post_concede_click_count
    assert eng.stats.games == 1 and eng.stats.concedes == 1
    assert eng._own_game is False
    assert eng._awaiting_match_class is True
    assert len(backend.gestures) == burst_gestures
    sleeps_before_watch = len([r for r in debug.records if r[0] == "sleep"])

    stats = eng.run(max_iterations=3)

    assert stats.target_found is True
    assert stats.games == 1 and stats.concedes == 1
    assert stats.class_distribution == {"Mage": 1}
    assert eng.capturer.calls == 3  # quiet boundary, UNKNOWN watch, resolved mulligan
    assert len(backend.gestures) == burst_gestures
    assert len([r for r in debug.records if r[0] == "sleep"]) == sleeps_before_watch
    assert [where for where, _frame, _ctx in debug.unknowns] == []


def test_unknown_boundary_is_not_deferred_after_match_watch_is_interrupted(cfg):
    """A watch interruption must force a fresh capture, never replay old UNKNOWN."""
    debug = _RecordingDebug()
    capturer = _CountingCapturer()
    eng = Engine(cfg, FakeAdb(), FakeBackend(), PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.UNKNOWN, ScreenState.ERROR_DIALOG,
                                 ScreenState.PLAY_SCREEN, ScreenState.QUEUE]),
                 reader=None, debug=debug, sleep=lambda _s: None, clock=lambda: 0.0,
                 rng=Random(3), layout=GameLayout(), capturer=capturer)

    boundary = eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)
    assert boundary.state == ScreenState.UNKNOWN
    assert eng._awaiting_match_class is True
    assert eng._deferred_screen is None

    stats = eng.run(max_iterations=2)

    assert stats.stop_reason == "max_iterations"
    # Boundary UNKNOWN, fresh ERROR watch observation, fresh PLAY observation, and
    # Play's semantic boundary. If old UNKNOWN had been deferred, iteration two
    # would halt before the fresh Play capture.
    assert capturer.calls == 4
    assert eng._deferred_screen is not None
    assert eng._deferred_screen[0].state == ScreenState.QUEUE


def test_named_end_boundary_is_deferred_without_a_second_burst(cfg):
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.VICTORY]), reader=None,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())
    from hop.hearthstone import MulliganRead
    from hop.hero_classes import HeroClass

    eng._execute_reject(MulliganRead(HeroClass.MAGE, False, 3, 1.0, "MAGE", "test"))

    assert eng.stats.games == 1 and eng.stats.concedes == 1
    assert eng._deferred_screen is not None
    assert eng._deferred_screen[0].state == ScreenState.VICTORY
    assert len(backend.gestures) == (
        2 + cfg.vision.concede_tap_attempts
        + cfg.timing.post_concede_click_count
    )


def test_post_concede_tap_starts_stop_at_the_hard_elapsed_budget(cfg):
    """Slow gesture emission may shorten the burst, never extend its tap-start window."""
    clock = [0.0]

    class SlowBackend(FakeBackend):
        def emit(self, gesture):
            super().emit(gesture)
            clock[0] += 0.45

    timing = replace(cfg.timing, post_concede_click_count=4,
                     post_concede_click_interval_s=0.75,
                     post_concede_click_interval_max_s=0.75,
                     post_concede_burst_max_s=2.4)
    eng = Engine(replace(cfg, timing=timing), FakeAdb(), SlowBackend(),
                 PanelGeometry(80, 40, 400.0), FakeClassifier([ScreenState.QUEUE]),
                 reader=None, sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
                 clock=lambda: clock[0], rng=Random(3), layout=GameLayout(),
                 capturer=_CountingCapturer())

    eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)

    # The gear/Concede/Concede Now emissions happen before the measured window; inside it,
    # 0.45s emits plus one 0.75s gap allow exactly two starts before 2.4s.
    burst = [g for g in eng.backend.gestures]
    assert len(burst) == 2 + cfg.vision.concede_tap_attempts + 2


def test_open_loop_tap_checks_a_stop_immediately_before_emitting(cfg):
    eng, backend = _engine(cfg, [ScreenState.QUEUE])
    eng.request_stop()
    from hop.engine import StopRequested

    with pytest.raises(StopRequested):
        eng._tap_open_loop(eng.layout.play_button, committing=False, what="post_concede_play")
    assert backend.gestures == []


def test_open_loop_tap_does_not_emit_when_stop_races_during_synthesis(cfg):
    eng, backend = _engine(cfg, [ScreenState.QUEUE])
    real_synth = eng._synth_non_repeating_tap

    def synth_then_stop(*args, **kwargs):
        gesture = real_synth(*args, **kwargs)
        eng.request_stop()
        return gesture

    eng._synth_non_repeating_tap = synth_then_stop
    from hop.engine import StopRequested

    with pytest.raises(StopRequested):
        eng._tap_open_loop(eng.layout.play_button, committing=False, what="post_concede_play")
    assert backend.gestures == []


def test_tap_journal_keeps_nominal_and_actual_endpoint_provenance(cfg):
    debug = _RecordingDebug()
    eng, _backend = _engine(cfg, [ScreenState.QUEUE], debug=debug)

    eng._tap_open_loop(eng.layout.play_button, committing=False, what="endpoint_probe")

    detail = next(d for kind, d in debug.records if kind == "tap")
    assert detail["point"] == detail["nominal_point"]  # compatibility + explicit meaning
    assert len(detail["actual_endpoint"]) == 2
    assert detail["radius_px"] > 0
    assert detail["target_radius_px"] > 0
    assert detail["duration_s"] > 0
    assert math.dist(detail["nominal_point"], detail["actual_endpoint"]) <= detail["radius_px"]
    json.dumps(detail)  # journal fields are JSON-safe without a DebugLog coercion pass


@pytest.mark.parametrize("state", [
    ScreenState.CONCEDE_MENU, ScreenState.IN_GAME,
    ScreenState.RECONNECT_DIALOG, ScreenState.RECONNECTING, ScreenState.MENU,
])
def test_post_concede_rejects_every_unnamed_boundary_without_deferring(cfg, state):
    """Only explicit successor/end states may book or resume the main loop."""
    timing = replace(cfg.timing, post_concede_click_count=1,
                     post_concede_burst_max_s=1.0)
    eng = Engine(replace(cfg, timing=timing), FakeAdb(), FakeBackend(),
                 PanelGeometry(80, 40, 400.0), FakeClassifier([state]), reader=None,
                 sleep=lambda _s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_CountingCapturer())
    with pytest.raises(Halt, match="unexpected"):
        eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)
    assert eng._deferred_screen is None


def test_post_concede_observes_before_the_fastest_successor_mulligan_can_appear(cfg):
    """The direct exit must not blind-wait past the fastest measured Play->Mulligan.

    The reported failure was not an ``IN_GAME`` state that should be accepted: a
    new match had already gone from its unseen mulligan to the board before the
    one post-burst capture.  Model the reported 2.247 s Wi-Fi capture and
    0.528 s classification here.  A boundary completed before the retained
    30.270 s floor is still QUEUE; one completed after it is the unsafe,
    unclassified board from the report.

    This pins the whole blind interval -- burst *and* quiet wait *and* capture
    latency -- rather than only the burst's tap-start deadline.
    """
    clock = [0.0]
    capture_s = 2.247  # report: final boundary screencap, Pixel 7a over Wi-Fi
    classify_s = 0.528  # report: final boundary NCC classification
    emit_s = 0.45
    first_play_at = []
    play_taps = []
    capture_started = []
    boundary_elapsed = []
    floor_s = cfg.timing.play_to_mulligan_observed_min_s

    class TimedCapturer:
        def capture(self):
            capture_started.append(clock[0])
            clock[0] += capture_s
            return gray_frame(80, 40, 60)

    class SlowBackend(FakeBackend):
        def emit(self, gesture):
            super().emit(gesture)
            clock[0] += emit_s

    class FastSuccessorClassifier:
        has_templates = True

        def _boundary(self):
            clock[0] += classify_s
            elapsed = clock[0] - first_play_at[0]
            boundary_elapsed.append(elapsed)
            # The boundary read gets one of the last safe queue frames only if it
            # completes before the fastest measured successor mulligan can pass.
            state = ScreenState.QUEUE if elapsed < floor_s else ScreenState.IN_GAME
            return Classification(state, 0.95)

        def classify(self, _frame):
            return self._boundary()

        def classify_expected(self, _frame, _expected):
            return self._boundary()

    eng = Engine(cfg, FakeAdb(), SlowBackend(), PanelGeometry(80, 40, 400.0),
                 FastSuccessorClassifier(), reader=None,
                 sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
                 clock=lambda: clock[0], rng=Random(3), layout=GameLayout(),
                 capturer=TimedCapturer())
    # Drive a slow but still in-budget burst, like the reported 20 s sequence.
    eng._burst_interval = lambda: cfg.timing.post_concede_click_interval_max_s
    tap_open_loop = eng._tap_open_loop

    def record_open_loop(point, *, committing, what):
        if what == "post_concede_play":
            play_taps.append(clock[0])
            if not first_play_at:
                first_play_at.append(clock[0])
        tap_open_loop(point, committing=committing, what=what)

    eng._tap_open_loop = record_open_loop
    boundary = eng._post_concede_clickthrough(source=ScreenState.MULLIGAN)

    assert boundary.state == ScreenState.QUEUE
    assert first_play_at and play_taps and capture_started
    assert boundary_elapsed
    # No open-loop Play tap, nor the final semantic observation, may arrive once
    # the retained fast-mulligan floor has passed.
    assert max(play_taps) - first_play_at[0] < floor_s
    assert max(play_taps) - first_play_at[0] >= cfg.timing.post_concede_queue_cooldown_s
    # Once the burst has already spent that due time, the boundary capture must
    # start directly after its final input emission -- no second blind cooldown.
    assert capture_started[0] - max(play_taps) == pytest.approx(emit_s)
    assert boundary_elapsed[0] < floor_s


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


def test_concede_retries_the_concede_button_then_halts_never_tapping_below_it(cfg):
    """A dropped Concede leaves the menu up; retry Concede's OWN coordinate, then fail closed.

    A Concede button tap can be silently dropped (a congested link), and unlike the old
    single-tap-then-halt, the engine re-sends it a bounded number of times -- but ONLY the
    Concede button, never a point below it (Options/Quit). If the menu never leaves it halts.
    The safety invariant is not "never tap again"; it is "never tap below Concede": every
    retap is on the Concede coordinate itself.
    """
    debug = _RecordingDebug()
    eng, backend = _engine(cfg, [ScreenState.IN_GAME, ScreenState.CONCEDE_MENU], debug=debug)
    with pytest.raises(Halt) as e:
        eng._concede()
    assert "Quit" not in str(e.value)                        # the stale, wrong warning is gone
    assert "not registering" in str(e.value) or "still up" in str(e.value)
    # gear + one Concede tap per attempt, and NOTHING else
    assert len(backend.gestures) == 1 + cfg.vision.concede_tap_attempts
    # every Concede retap landed on the Concede button's own point -- never below it
    concede = eng.layout.concede_button.to_px(eng.panel)
    concede_pt = (round(concede[0]), round(concede[1]))
    taps = [d["point"] for kind, d in debug.records if kind == "tap" and d["what"] == "concede"]
    assert len(taps) == cfg.vision.concede_tap_attempts
    assert all(pt == concede_pt for pt in taps)              # exactly Concede, never Options/Quit
    # the TOP entry sits well above Options (y~0.42) and Quit (y~0.56): no retap can reach them
    assert concede_pt[1] < 40 * 0.42
    # each ignored tap is journalled so the dropped-button rate is observable
    ignored = [d for kind, d in debug.records if kind == "concede_tap_ignored"]
    assert len(ignored) == cfg.vision.concede_tap_attempts - 1


def test_concede_retries_a_dropped_tap_then_succeeds(cfg):
    """The recovery the halt used to deny: the first Concede is dropped (the menu persists the
    whole first leave-wait), the retap lands, and the board dissolves -- a completed concede.

    The first attempt keeps the FULL screen_wait budget before it decides to retap, so the
    menu must stay up for that entire wait to force one retry; then the next look leaves.
    """
    debug = _RecordingDebug()
    states = [ScreenState.IN_GAME, ScreenState.CONCEDE_MENU, ScreenState.VICTORY]
    eng, backend = _engine(cfg, states, debug=debug)
    assert eng._concede() is True
    # fixed open-loop gear + Concede; one phase-boundary capture proves success
    assert len(backend.gestures) == 3
    concede = eng.layout.concede_button.to_px(eng.panel)
    concede_pt = (round(concede[0]), round(concede[1]))
    taps = [d["point"] for kind, d in debug.records if kind == "tap" and d["what"] == "concede"]
    assert taps == [concede_pt, concede_pt]


def test_concede_open_loop_pair_is_bounded_when_destination_never_arrives(cfg):
    """`full_transition` after the gear does not mean the Game Menu opened.

    `_compatible()` admits `top_banner` and `partial` for an expected
    `full_transition`, so an opponent's turn animating behind a *dropped* gear tap
    satisfies it. The concede is the loop's committing action; look before you tap.
    """
    eng, backend = _engine(cfg, [ScreenState.IN_GAME])   # gear never opened the menu
    with pytest.raises(Halt) as e:
        eng._concede()
    assert "did not dismiss" in str(e.value)
    assert len(backend.gestures) >= 2


def test_concede_returns_once_the_menu_is_gone(cfg):
    eng, backend = _engine(cfg, [ScreenState.IN_GAME, ScreenState.VICTORY])
    assert eng._concede() is True            # menu opened, then the board dissolved
    assert len(backend.gestures) == 2


def test_concede_does_not_accept_unknown_as_proof_the_menu_left(cfg):
    """An unreadable frame is not an observation that we left the menu."""
    debug = _RecordingDebug()
    eng, backend = _engine(cfg, [ScreenState.IN_GAME, ScreenState.UNKNOWN], debug=debug)
    with pytest.raises(Halt):
        eng._concede()
    assert len(backend.gestures) == 2
    assert [w for w, _f, _c in debug.unknowns] == ["concede"]


def test_concede_is_skipped_when_the_game_already_ended(cfg):
    """The opponent can concede first, or a lethal can land during _play_beats.

    The gear icon is drawn on the victory screen too, so tapping it there opens
    something we never anchored -- and booking a concede that never happened inflates
    the concede/commit ratio the caps exist to keep human.
    """
    eng, backend = _engine(cfg, [ScreenState.VICTORY])
    assert eng._concede() is False
    assert backend.gestures == []


def test_concede_refuses_to_run_from_a_screen_that_is_not_a_live_game(cfg):
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN])
    with pytest.raises(Halt) as e:
        eng._concede()
    assert "not a live game" in str(e.value)
    assert backend.gestures == []


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
    # Exceptional closed-loop recovery still uses one fixed dismiss per named
    # end screen; the ordinary reject path above bypasses this loop entirely.
    assert len(backend.gestures) == 2


def test_rank_progress_overlay_is_a_dismissible_end_screen(cfg):
    """A ranked medal overlays a dimmed board, so it must beat IN_GAME and dismiss safely."""
    assert ScreenState.RANK_PROGRESS in Engine.END_SCREENS
    eng, backend = _engine(cfg, [ScreenState.RANK_PROGRESS, ScreenState.PLAY_SCREEN])
    eng._clear_end_screens()
    assert len(backend.gestures) == 1


def test_postgame_timeout_keeps_named_terminal_screen_evidence(cfg):
    debug = _RecordingDebug()
    eng, backend = _engine(cfg, [ScreenState.IN_GAME], debug=debug)
    with pytest.raises(Halt, match="board never finished dissolving"):
        eng._clear_end_screens()
    assert backend.gestures == []
    evidence = [d for kind, d in debug.records if kind == "terminal_screen"]
    assert len(evidence) == 1
    assert evidence[0]["state"] == ScreenState.IN_GAME.value
    assert evidence[0]["waits"] == cfg.vision.screen_wait_attempts


class _EndDismissStuckUntilNthTap:
    """A static end screen until the Nth physical dismissal gesture lands."""

    def __init__(self, backend, land_on_attempt):
        self.backend = backend
        self.land_on_attempt = land_on_attempt

    def capture(self):
        if len(self.backend.gestures) >= self.land_on_attempt:
            return gray_frame(80, 40, 220)
        return gray_frame(80, 40, 60)


def test_end_dismiss_retries_a_dropped_tap_only_on_the_same_named_end_screen(cfg):
    """The report's failure: an ignored Defeat dismiss must not end the hunt.

    The retry is not a blind second tap.  Every dropped attempt is followed by a fresh
    positive Defeat classification; only then may the exact same dismiss coordinate be
    sent again.  The final gesture changes the screen and the next state is home.
    """
    debug = _RecordingDebug()
    backend = FakeBackend()
    attempts = cfg.vision.end_dismiss_tap_attempts
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.DEFEAT] * attempts + [ScreenState.PLAY_SCREEN]),
                 reader=None, debug=debug, sleep=lambda s: None, clock=lambda: 0.0,
                 rng=Random(3), layout=GameLayout(),
                 capturer=_EndDismissStuckUntilNthTap(backend, land_on_attempt=attempts))

    assert attempts >= 2
    eng._clear_end_screens()
    assert len(backend.gestures) == attempts
    dismiss = eng.layout.end_dismiss.to_px(eng.panel)
    dismiss_pt = (round(dismiss[0]), round(dismiss[1]))
    taps = [d["point"] for kind, d in debug.records
            if kind == "tap" and d["what"] == "end_dismiss"]
    assert taps == [dismiss_pt] * attempts
    ignored = [d for kind, d in debug.records if kind == "end_dismiss_tap_ignored"]
    assert len(ignored) == attempts - 1
    assert all(d["screen"] == ScreenState.DEFEAT.value for d in ignored)


def test_end_dismiss_exhausts_its_own_retry_budget_then_fails_closed(cfg):
    """Static pixels plus a positively persistent end screen are the drop signature."""
    debug = _RecordingDebug()
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.VICTORY]), reader=None, debug=debug,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_EndDismissStuckUntilNthTap(backend, 999))
    with pytest.raises(Halt) as e:
        eng._clear_end_screens()
    assert "semantic destination" in str(e.value)
    assert len(backend.gestures) == cfg.vision.end_dismiss_tap_attempts
    ignored = [d for kind, d in debug.records if kind == "end_dismiss_tap_ignored"]
    assert len(ignored) == cfg.vision.end_dismiss_tap_attempts - 1


def test_end_dismiss_static_retry_respects_the_outer_stack_tap_budget(cfg):
    """The per-screen retry budget must never exceed ``_clear_end_screens(max_taps)``."""
    assert cfg.vision.end_dismiss_tap_attempts == 3
    debug = _RecordingDebug()
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.VICTORY]), reader=None, debug=debug,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_EndDismissStuckUntilNthTap(backend, 999))

    with pytest.raises(Halt) as e:
        eng._clear_end_screens(max_taps=2)

    assert "did not clear after 2 dismiss taps" in str(e.value)
    assert len(backend.gestures) == 2
    ignored = [d for kind, d in debug.records if kind == "end_dismiss_tap_ignored"]
    assert len(ignored) == 1


def test_end_dismiss_never_retries_when_the_post_failure_look_is_not_the_same_screen(cfg):
    """Its point overlaps Mulligan Confirm, so a differing screen is a hard stop, not a retap."""
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier([ScreenState.VICTORY, ScreenState.MULLIGAN]), reader=None,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(3),
                 layout=GameLayout(), capturer=_EndDismissStuckUntilNthTap(backend, 999))
    with pytest.raises(Halt) as e:
        eng._clear_end_screens()
    assert "refusing to blind-tap end_dismiss" in str(e.value)
    assert len(backend.gestures) == 1


def test_end_dismiss_does_not_run_pixel_verification(cfg):
    """A changed-but-wrong screen is not evidence for another fixed-point tap."""
    eng, backend = _engine(cfg, [ScreenState.VICTORY])

    def wrong_change(*_args, **_kwargs):
        raise Halt("screen changed but not as expected", Halt.WRONG_CHANGE)

    eng.verifier.verify = wrong_change
    with pytest.raises(Halt):
        eng._clear_end_screens()
    assert len(backend.gestures) == cfg.vision.end_dismiss_tap_attempts


def test_exhausting_the_tap_budget_halts_rather_than_returning_quietly(cfg):
    """A silent return here books a completed game and requeues into a stuck client."""
    eng, backend = _engine(cfg, [ScreenState.VICTORY])
    with pytest.raises(Halt) as e:
        eng._clear_end_screens(max_taps=2)
    assert "did not clear after 2 dismiss taps" in str(e.value)
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


# ── a live board is not a place to wait forever ──────────────────────────────

def test_a_board_at_the_top_of_the_loop_is_polled_then_abandoned(cfg):
    """`_execute_reject` can only start from the mulligan, so a board reached any other
    way had no matchup read and, before this, no exit at all: dispatch slept and
    re-looped while a game hop queued roped out turn by turn.

    Conceding is not special to the mulligan -- the gear is on every board.
    """
    n = cfg.vision.in_game_wait_attempts
    states = (
        [ScreenState.IN_GAME] * (n + 1)    # n patient polls, then the one that acts
            + [ScreenState.DEFEAT]             # fixed pair -> named end state
        + [ScreenState.DEFEAT,             # _clear_end_screens: dismiss...
           ScreenState.PLAY_SCREEN]        # ...and home
    )
    eng, backend = _engine(cfg, states)
    eng._own_game = True                                # hop tapped Play; this is ours

    for _ in range(n):                                  # patient first
        cls, frame = eng._classify_settled()
        eng._dispatch(cls, frame)
        assert backend.gestures == []

    cls, frame = eng._classify_settled()                # ...then it acts
    eng._dispatch(cls, frame)
    assert eng.stats.games == 1
    assert eng.stats.concedes == 1
    assert [g.kind for g in backend.gestures] == ["tap", "tap", "tap", "tap"]
    assert eng._own_game is False


def test_hop_never_concedes_a_game_it_did_not_start(cfg):
    """Restarting the hunt during the user's game -- quite possibly the target game it
    alerted them about -- must not throw it away."""
    n = cfg.vision.in_game_wait_attempts
    eng, backend = _engine(cfg, [ScreenState.IN_GAME])
    assert eng._own_game is False
    for _ in range(n):
        cls, frame = eng._classify_settled()
        eng._dispatch(cls, frame)
    cls, frame = eng._classify_settled()
    with pytest.raises(Halt) as e:
        eng._dispatch(cls, frame)
    assert "did not start" in str(e.value)
    assert backend.gestures == []


def test_tapping_play_makes_the_next_game_ours(cfg):
    eng, _backend = _engine(cfg, [ScreenState.PLAY_SCREEN])
    assert eng._own_game is False
    cls, frame = eng._classify_settled()
    eng._dispatch(cls, frame)
    assert eng._own_game is True


def test_leaving_the_board_resets_the_patience_counter(cfg):
    eng, _backend = _engine(cfg, [ScreenState.IN_GAME, ScreenState.QUEUE])
    cls, frame = eng._classify_settled()
    eng._dispatch(cls, frame)
    assert eng._in_game_polls == 1
    cls, frame = eng._classify_settled()
    eng._dispatch(cls, frame)
    assert eng._in_game_polls == 0


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
