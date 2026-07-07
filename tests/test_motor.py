import math
from random import Random

from hop.humanize.contact import ContactModel
from hop.humanize.motor import (
    ffitts_movement_time,
    fitts_index_of_difficulty,
    synth_swipe,
    synth_tap,
    tap_dwell,
)
from hop.humanize.state import HumanState


def test_tap_endpoint_within_hit_area(cfg, panel):
    cm = ContactModel(cfg.contact)
    for seed in range(50):
        g = synth_tap(Random(seed), (1200, 540), 40, panel, cfg.motor, cm, HumanState())
        ex, ey = g.endpoint()
        assert math.hypot(ex - 1200, ey - 540) <= 40  # lands inside the control


def test_tap_pressure_never_zero_in_contact(cfg, panel):
    cm = ContactModel(cfg.contact)
    g = synth_tap(Random(0), (1200, 540), 40, panel, cfg.motor, cm, HumanState())
    in_contact = [s.pressure for s in g.samples if s.tip]
    assert min(in_contact) > 0.0
    assert g.samples[-1].tip is False and g.samples[-1].pressure == 0.0  # explicit release


def test_tap_dwell_clamped(cfg):
    for seed in range(200):
        d = tap_dwell(Random(seed), cfg.motor, HumanState())
        assert cfg.motor.tap_dwell_min_s <= d <= cfg.motor.tap_dwell_max_s


def test_swipe_endpoint_exact(cfg, panel):
    cm = ContactModel(cfg.contact)
    for seed in range(30):
        g = synth_swipe(Random(seed), (100, 100), (2000, 1000), panel, cfg.motor, cm, HumanState())
        ex, ey = g.endpoint()
        assert abs(ex - 2000) < 1e-6 and abs(ey - 1000) < 1e-6


def test_ffitts_monotonic_in_distance(cfg):
    st = HumanState()
    mts = [ffitts_movement_time(Random(1), d, 180, cfg.motor, st) for d in (50, 300, 1200, 2400)]
    assert mts == sorted(mts)


def test_index_of_difficulty():
    assert fitts_index_of_difficulty(0, 180) == 0.0
    assert fitts_index_of_difficulty(180, 180) == 1.0  # log2(2)


def test_fatigue_slows_movement(cfg):
    tired = HumanState(); tired.fatigue = 0.95
    fresh = HumanState()
    mt_tired = ffitts_movement_time(Random(5), 800, 180, cfg.motor, tired)
    mt_fresh = ffitts_movement_time(Random(5), 800, 180, cfg.motor, fresh)
    assert mt_tired > mt_fresh
