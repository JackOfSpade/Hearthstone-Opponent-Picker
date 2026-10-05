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


def test_target_mulligan_plan_keeps_its_existing_card_decisions():
    for seed in range(50):
        decs = journey.plan_keep(Random(seed), 4)
        assert len(decs) == 4
        # A target matchup retains the careful, normal mulligan behavior.
        assert sum(1 for d in decs if not d.replace) >= 1


def test_reject_concede_point_is_always_immediate_post_mulligan():
    assert {journey.choose_concede_point(Random(seed)) for seed in range(400)} == {"mulligan"}


def test_reject_plan_has_no_performative_interactions():
    for seed in range(200):
        plan = journey.plan_reject(Random(seed), 4)
        assert plan.concede_point == "mulligan"
        assert plan.mulligan == []
        assert plan.hesitate_before_concede is False
        assert plan.extra_reads == 0
