from random import Random

import pytest

from hop.humanize import journey
from hop.humanize.contact import ContactModel
from hop.humanize.limiter import CapReached, Limiter, trajectory_fingerprint
from hop.humanize.motor import synth_tap
from hop.humanize.state import HumanState


def test_committing_cap_raises(cfg):
    lim = Limiter(cfg.caps, scale=1.0)
    # allow plenty of non-committing headroom, tighten commits
    lim.commits_this_run = lim.committing_action_cap
    with pytest.raises(CapReached) as e:
        lim.check_before_action(committing=True)
    assert e.value.which == "committing"


def test_actions_per_run_cap(cfg):
    lim = Limiter(cfg.caps, scale=1.0)
    lim.actions_this_run = lim.max_actions_per_run
    with pytest.raises(CapReached):
        lim.check_before_action(committing=False)


def test_break_cadence(cfg):
    lim = Limiter(cfg.caps, scale=1.0)
    rng = Random(1)
    every = cfg.caps.mandatory_break_every_games
    for _ in range(every - 1):
        lim.register_game()
    assert lim.needs_break(rng) is None
    lim.register_game()
    brk = lim.needs_break(rng)
    assert brk is not None and brk > 0


def test_committing_ratio(cfg):
    lim = Limiter(cfg.caps, scale=1.0)
    lim.actions_this_run = 10
    lim.commits_this_run = 3
    assert lim.committing_ratio() == 0.3


def test_non_repetition_detects_replay(cfg, panel):
    lim = Limiter(cfg.caps, scale=1.0)
    cm = ContactModel(cfg.contact)
    g = synth_tap(Random(42), (1200, 540), 40, panel, cfg.motor, cm, HumanState())
    assert not lim.is_near_duplicate(g)
    lim.remember_trajectory(g)
    # exact same gesture (verbatim replay) is a near-duplicate
    assert lim.is_near_duplicate(g)
    # a different gesture is not
    g2 = synth_tap(Random(99), (300, 200), 40, panel, cfg.motor, cm, HumanState())
    assert not lim.is_near_duplicate(g2)


def test_limiter_persistence_roundtrip(cfg):
    lim = Limiter(cfg.caps, scale=1.0)
    lim.actions_today = 12
    lim.commits_today = 4
    data = lim.to_dict()
    lim2 = Limiter.from_dict(cfg.caps, 1.0, data)
    assert lim2.actions_today == 12 and lim2.commits_today == 4


def test_mulligan_plan_always_interacts():
    for seed in range(50):
        decs = journey.plan_mulligan(Random(seed), 4, keeping=False)
        assert len(decs) == 4
        # keeps at least one card (never a zero-touch that looks robotic-perfect)
        assert sum(1 for d in decs if not d.replace) >= 1


def test_concede_point_shifts_with_min_play_turns():
    cautious = [journey.choose_concede_point(Random(s), 1.0) for s in range(400)]
    aggressive = [journey.choose_concede_point(Random(s), 0.0) for s in range(400)]
    # aggressive bails at mulligan more often than cautious
    assert aggressive.count("mulligan") > cautious.count("mulligan")


def test_reject_plan_shape():
    plan = journey.plan_reject(Random(3), 4, min_play_turns=1.0)
    assert plan.concede_point in ("mulligan", "turn1", "turn2")
    assert len(plan.mulligan) == 4
