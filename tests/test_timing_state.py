from random import Random

from hop.humanize import timing
from hop.humanize.state import HumanState


def _mean(fn, n=300):
    return sum(fn(Random(s)) for s in range(n)) / n


def test_commit_faster_than_reject(cfg):
    commit = _mean(lambda r: timing.think_time(r, "commit", cfg.timing, HumanState()))
    reject = _mean(lambda r: timing.think_time(r, "reject", cfg.timing, HumanState()))
    assert reject > commit


def test_think_time_never_negative(cfg):
    for s in range(500):
        assert timing.think_time(Random(s), "reject", cfg.timing, HumanState()) >= 0


def test_fatigue_lengthens_think(cfg):
    tired = HumanState(); tired.fatigue = 0.9
    a = timing.think_time(Random(2), "commit", cfg.timing, tired)
    b = timing.think_time(Random(2), "commit", cfg.timing, HumanState())
    assert a > b


def test_familiarity_shortens_think(cfg):
    fam = HumanState(); fam.familiarity = 0.9
    a = timing.think_time(Random(3), "commit", cfg.timing, fam)
    b = timing.think_time(Random(3), "commit", cfg.timing, HumanState())
    assert a < b


def test_human_delay_positive(cfg):
    for s in range(200):
        assert timing.human_delay(Random(s), 3.0, cfg.timing) >= 0


def test_cooldown_never_below_anchor(cfg):
    for s in range(200):
        assert timing.human_cooldown(Random(s), 5.0, cfg.timing) >= 5.0


def test_state_fatigue_accumulates_and_familiarity_grows():
    st = HumanState()
    rng = Random(7)
    f0, fam0 = st.fatigue, st.familiarity
    for _ in range(50):
        st.tick(rng, dt=30.0)
    assert st.fatigue > f0
    assert st.familiarity > fam0
    assert 0.0 <= st.fatigue <= 1.0


def test_think_time_scale_bounds():
    st = HumanState()
    st.fatigue, st.confidence, st.urgency = 1.0, 0.0, 0.0
    assert 0.35 <= st.think_time_scale() <= 3.0


def test_confidence_follows_observation():
    st = HumanState()
    st.observe_confidence(0.1)
    low = st.confidence
    st.observe_confidence(0.95)
    st.observe_confidence(0.95)
    assert st.confidence > low


def test_decision_gates_on_confidence():
    st = HumanState(); st.confidence = 0.9
    assert st.decision(Random(1)) == "commit"
    st.confidence = 0.6
    assert st.decision(Random(1)) == "inspect"
