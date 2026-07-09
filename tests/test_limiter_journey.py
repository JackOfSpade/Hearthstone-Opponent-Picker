from random import Random

from hop.humanize import journey
from hop.humanize.contact import ContactModel
from hop.humanize.limiter import Limiter
from hop.humanize.motor import synth_tap
from hop.humanize.state import HumanState


def test_no_volume_cap_gates_a_run(cfg):
    """Volume/session caps and breaks were removed: registering far more actions,
    concedes and games than any old ceiling never raises or blocks, and the whole
    cap-gate surface is gone. A hunt stops only on target / error / user-stop."""
    lim = Limiter(cfg.caps)
    for _ in range(1000):        # 1000 >> every old cap (30 actions / 15 concedes / 60 games)
        lim.register_action(committing=True)
        lim.register_game()
        lim.register_time(60)
    assert lim.actions_this_run == 1000
    assert lim.commits_this_run == 1000
    assert lim.games_this_session == 1000
    for gone in ("check_before_action", "check_elapsed_caps", "needs_break"):
        assert not hasattr(lim, gone)


def test_cap_reached_exception_is_gone():
    import hop.humanize.limiter as limiter_mod
    assert not hasattr(limiter_mod, "CapReached")


def test_committing_ratio(cfg):
    lim = Limiter(cfg.caps)
    lim.actions_this_run = 10
    lim.commits_this_run = 3
    assert lim.committing_ratio() == 0.3


def test_non_repetition_detects_replay(cfg, panel):
    lim = Limiter(cfg.caps)
    cm = ContactModel(cfg.contact)
    g = synth_tap(Random(42), (1200, 540), 40, panel, cfg.motor, cm, HumanState())
    assert not lim.is_near_duplicate(g)
    lim.remember_trajectory(g)
    # exact same gesture (verbatim replay) is a near-duplicate
    assert lim.is_near_duplicate(g)
    # a different gesture is not
    g2 = synth_tap(Random(99), (300, 200), 40, panel, cfg.motor, cm, HumanState())
    assert not lim.is_near_duplicate(g2)


def test_mulligan_plan_always_interacts():
    for seed in range(50):
        decs = journey.plan_mulligan(Random(seed), 4, keeping=False)
        assert len(decs) == 4
        # keeps at least one card (never a zero-touch that looks robotic-perfect)
        assert sum(1 for d in decs if not d.replace) >= 1


def test_concede_point_favours_playing_into_the_game():
    """The anti-barcode invariant: a normal user rarely bails at the mulligan, so
    the concede almost never lines up with the class reveal."""
    points = [journey.choose_concede_point(Random(s)) for s in range(400)]
    assert points.count("mulligan") < points.count("turn1")
    assert points.count("mulligan") < points.count("turn2")


def test_reject_plan_shape():
    plan = journey.plan_reject(Random(3), 4)
    assert plan.concede_point in ("mulligan", "turn1", "turn2")
    assert len(plan.mulligan) == 4


def test_reject_plan_extra_reads_are_bounded():
    """A normal user can play a beat, but should not stack several idle reads first."""
    for seed in range(200):
        plan = journey.plan_reject(Random(seed), 4)
        assert plan.extra_reads <= 1
