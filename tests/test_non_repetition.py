"""Non-repetition is a PRE-action gate, and correction is not a blind retry.

Two coupled defects lived here. The gate was implemented as a post-action check --
synthesize, test, emit anyway, and let `Verifier` raise afterwards -- so it could never
prevent a duplicate, only stop the run after sending one. And it fired constantly:
measured over the real per-game tap sequence (300 seeded runs x 30 actions), 14.2% of
taps near-duplicate a prior one and 293/300 runs flag at least once, median tap #13.

The engine only survived that because `_tap`'s correction retap re-emitted with
`near_duplicate=False` hardcoded. So the blind retap was load-bearing for the loop not
halting, and gating the correction on the Halt reason -- obviously correct on its own --
would have turned 98% of runs into a Halt. They have to move together.
"""

from random import Random

import pytest

from hop.engine import Engine
from hop.geometry import PanelGeometry
from hop.hearthstone import GameLayout
from hop.humanize.contact import ContactModel
from hop.humanize.limiter import (
    TrajectoryMemory,
    trajectory_fingerprint,
)
from hop.humanize.motor import synth_tap
from hop.humanize.state import HumanState
from hop.perception.screens import ScreenState
from hop.verify import Halt

from conftest import FakeAdb, FakeBackend, FakeClassifier, gray_frame

PANEL = PanelGeometry(2400, 1080, 420.0)


def _game_sequence(layout: GameLayout):
    return [layout.play_button, layout.card_point(0.35), layout.card_point(0.55),
            layout.mulligan_confirm, layout.gear_button, layout.concede_button,
            layout.end_dismiss, layout.end_dismiss]


# ── the fingerprint ──────────────────────────────────────────────────────────

def test_duration_is_an_exact_function_of_the_sample_count(cfg):
    """`dur` and `n_samples` are one degree of freedom, not two.

    dur == (n - 1) / report_rate_hz exactly, so `_normalized_distance` weighs the same
    information twice. Recorded because the obvious remedy -- drop the collinear
    feature -- makes the flag rate WORSE (15.5% vs 14.2%), and the real fix is to
    resample instead of to retune this vector.
    """
    rng = Random(7)
    layout = GameLayout()
    contact = ContactModel(cfg.contact)
    for i, p in enumerate(_game_sequence(layout)):
        x, y, r = p.to_px(PANEL)
        g = synth_tap(rng, (x, y), r, PANEL, cfg.motor, contact, HumanState())
        dur, n = trajectory_fingerprint(g)[0], trajectory_fingerprint(g)[1]
        assert dur == pytest.approx((n - 1) / cfg.motor.report_rate_hz, abs=1e-12)


def test_the_gate_really_does_saturate_without_resampling(cfg):
    """Pin the measurement the resample budget was chosen from."""
    layout = GameLayout()
    seq = _game_sequence(layout)
    flags = taps = 0
    for seed in range(40):
        rng = Random(seed)
        mem = TrajectoryMemory()
        contact = ContactModel(cfg.contact)
        state = HumanState()
        for i in range(30):        # a representative run's worth of taps
            x, y, r = seq[i % len(seq)].to_px(PANEL)
            g = synth_tap(rng, (x, y), r, PANEL, cfg.motor, contact, state)
            fp = trajectory_fingerprint(g)
            taps += 1
            flags += mem.is_near_duplicate(fp, cfg.caps.non_repetition_threshold)
            mem.add(fp)
    assert flags / taps > 0.05, "a single draw used to duplicate ~14% of the time"


# ── the pre-emit gate ────────────────────────────────────────────────────────

class _AlternatingCapturer:
    def __init__(self):
        self.n = 0

    def capture(self):
        self.n += 1
        return gray_frame(80, 40, 20 if self.n % 2 else 220)


def _engine(cfg, states, *, limiter=None, capturer=None):
    backend = FakeBackend()
    eng = Engine(cfg, FakeAdb(), backend, PanelGeometry(80, 40, 400.0),
                 FakeClassifier(states), reader=None, limiter=limiter,
                 sleep=lambda s: None, clock=lambda: 0.0, rng=Random(5),
                 layout=GameLayout(), capturer=capturer or _AlternatingCapturer())
    return eng, backend


def test_a_near_duplicate_is_resampled_and_never_reaches_the_backend(cfg):
    """The whole point of a pre-action gate: the duplicate is not emitted."""
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN])

    calls = {"n": 0}
    real = eng.limiter.is_near_duplicate

    def flag_the_first_two(gesture):
        calls["n"] += 1
        return calls["n"] <= 2      # first two draws are duplicates, third is not

    eng.limiter.is_near_duplicate = flag_the_first_two
    eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
             expected_change="full_transition", what="play")
    assert calls["n"] == 3          # two rejected draws, one accepted
    assert len(backend.gestures) == 1
    assert real is not None


def test_a_degenerate_generator_fails_closed(cfg):
    """If every draw collides, the motor has collapsed. Halt, do not emit."""
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN])
    eng.limiter.is_near_duplicate = lambda g: True
    with pytest.raises(Halt) as e:
        eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
                 expected_change="full_transition", what="play")
    assert e.value.kind == Halt.NEAR_DUPLICATE
    assert backend.gestures == []   # nothing on the wire


def test_only_the_emitted_gesture_is_remembered(cfg):
    """Remembering rejected draws poisons the memory against future draws."""
    eng, _backend = _engine(cfg, [ScreenState.PLAY_SCREEN])
    calls = {"n": 0}
    eng.limiter.is_near_duplicate = lambda g: (calls.__setitem__("n", calls["n"] + 1)
                                               or calls["n"] <= 2)
    eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
             expected_change="full_transition", what="play")
    assert len(eng.limiter._traj.fingerprints) == 1


def test_a_full_game_of_taps_never_halts_on_non_repetition(cfg):
    """The regression that the shipped code would have hit at ~tap #13."""
    layout = GameLayout()
    seq = _game_sequence(layout)
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN])
    eng.panel = PANEL
    for i in range(30):        # a full run's worth of taps
        eng._tap(seq[i % len(seq)], committing=False, decision_type="commit",
                 expected_change="full_transition", allow_correction=False, what="x")
    assert len(backend.gestures) == 30


# ── the correction is not a blind retry ──────────────────────────────────────

class _NeverChangesCapturer:
    def capture(self):
        return gray_frame(80, 40, 30)


class _WrongChangeCapturer:
    """Bottom third only: a `bottom_sheet`, which is not a `full_transition`."""

    def __init__(self):
        self.n = 0

    def capture(self):
        import numpy as np

        from hop.perception.image import Frame
        self.n += 1
        arr = np.full((40, 80), 20, dtype="uint8")
        if self.n % 2 == 0:
            arr[27:, :] = 220
        return Frame(80, 40, arr)


def test_a_missed_tap_is_corrected_exactly_once(cfg):
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], capturer=_NeverChangesCapturer())
    with pytest.raises(Halt) as e:
        eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
                 expected_change="full_transition", what="play")
    assert e.value.kind == Halt.NO_CHANGE
    assert len(backend.gestures) == 2       # original + one correction


def test_a_wrong_change_is_never_corrected(cfg):
    """The screen already moved. Re-tapping the old coordinates is the blind tap."""
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], capturer=_WrongChangeCapturer())
    with pytest.raises(Halt) as e:
        eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
                 expected_change="full_transition", what="play")
    assert e.value.kind == Halt.WRONG_CHANGE
    assert len(backend.gestures) == 1       # no correction


def test_a_size_mismatch_is_never_corrected(cfg):
    """The display rotated: the coordinates we would retap no longer exist."""

    class _RotatingCapturer:
        def __init__(self):
            self.n = 0

        def capture(self):
            self.n += 1
            return gray_frame(80, 40, 20) if self.n == 1 else gray_frame(40, 80, 220)

    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], capturer=_RotatingCapturer())
    with pytest.raises(Halt) as e:
        eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
                 expected_change="full_transition", what="play")
    assert e.value.kind == Halt.SIZE_MISMATCH
    assert len(backend.gestures) == 1


# ── looking again beats tapping again ────────────────────────────────────────

def test_a_slow_transition_is_waited_out_rather_than_retapped(cfg):
    """"Has not moved" and "has not moved YET" are the same picture in one frame.

    A fixed settle turns the second into the first, and the remedy for a missed tap is
    another tap. Look again instead.
    """

    class _SlowCapturer:
        """before, then two unchanged looks, then the transition lands."""

        def __init__(self):
            self.n = 0

        def capture(self):
            self.n += 1
            return gray_frame(80, 40, 20 if self.n <= 3 else 220)

    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], capturer=_SlowCapturer())
    eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
             expected_change="full_transition", what="play")
    assert len(backend.gestures) == 1       # no correction: we saw it arrive


def test_a_genuinely_missed_tap_still_halts_after_the_extra_looks(cfg):
    eng, backend = _engine(cfg, [ScreenState.PLAY_SCREEN], capturer=_NeverChangesCapturer())
    with pytest.raises(Halt) as e:
        eng._tap(eng.layout.play_button, committing=False, decision_type="commit",
                 expected_change="full_transition", allow_correction=False, what="play")
    assert e.value.kind == Halt.NO_CHANGE


def test_scoped_card_taps_do_not_pay_for_the_motion_wait(cfg):
    """A card toggle moves the whole frame by 4.80 against a 9.0 threshold, so the
    signature cannot see it. Polling for whole-screen motion on every card tap would
    burn ~3 captures x 3 attempts x 2 cards per mulligan and find nothing."""
    eng, _backend = _engine(cfg, [ScreenState.MULLIGAN], capturer=_NeverChangesCapturer())
    looks = {"n": 0}
    real = eng._await_screen_motion
    eng._await_screen_motion = lambda b, a: (looks.__setitem__("n", looks["n"] + 1), a)[1]
    eng._replace_card(slot=0, center_xf=0.25, decision_type="reject")
    assert looks["n"] == 0
    assert real is not None
