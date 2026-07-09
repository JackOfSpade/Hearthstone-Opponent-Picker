from dataclasses import replace

from hop.config import load_config
from hop.hero_classes import HeroClass


def test_defaults_load():
    cfg = load_config()
    assert cfg.device.opponent_corner == "bottom_left"
    assert cfg.motor.tap_dwell_min_s < cfg.motor.tap_dwell_max_s


def test_criteria_accepts_no_filter():
    cfg = load_config()
    assert cfg.criteria.accepts(HeroClass.MAGE, we_go_second=True)
    assert cfg.criteria.accepts(HeroClass.WARRIOR, we_go_second=False)


def test_criteria_target_and_second():
    cfg = load_config()
    crit = replace(cfg.criteria, target_classes=(HeroClass.MAGE,), require_second=True)
    assert crit.accepts(HeroClass.MAGE, True)
    assert not crit.accepts(HeroClass.MAGE, False)   # wrong coin
    assert not crit.accepts(HeroClass.WARLOCK, True)  # wrong class


def test_avoid_classes():
    cfg = load_config()
    crit = replace(cfg.criteria, target_classes=(), avoid_classes=(HeroClass.PRIEST,))
    assert not crit.accepts(HeroClass.PRIEST, True)
    assert crit.accepts(HeroClass.MAGE, True)


def test_pass_rate_estimate():
    cfg = load_config()
    # no filter -> ~1.0
    assert cfg.criteria.pass_rate_estimate() == 1.0
    one_class = replace(cfg.criteria, target_classes=(HeroClass.MAGE,))
    assert 0.0 < one_class.pass_rate_estimate() < 0.2
    one_class_second = replace(one_class, require_second=True)
    assert one_class_second.pass_rate_estimate() < one_class.pass_rate_estimate()
