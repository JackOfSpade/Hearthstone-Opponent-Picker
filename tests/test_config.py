from dataclasses import replace

from hop.config import clear_target_classes, load_config, save_criteria
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


def test_second_only_across_none_some_and_all_class_selections():
    """'Only when going 2nd' with the class filter empty / some / ALL selected -- the three
    configurations the dashboard exposes:
      * none selected -> ANY class, but only when going 2nd
      * some selected -> only those classes, and only when going 2nd
      * ALL selected  -> IDENTICAL to none (opponent in {all 11} is always true)
    and every one of them concedes every going-FIRST game (the require_second gate)."""
    from hop.hero_classes import ALL_CLASSES

    base = load_config().criteria
    none_ = replace(base, target_classes=(), avoid_classes=(), require_second=True)
    some = replace(base, target_classes=(HeroClass.PALADIN, HeroClass.MAGE),
                   avoid_classes=(), require_second=True)
    allc = replace(base, target_classes=tuple(ALL_CLASSES), avoid_classes=(), require_second=True)

    for c in ALL_CLASSES:
        # going 2nd: none and all keep every class; some keeps only the two picked
        assert none_.accepts(c, we_go_second=True)
        assert allc.accepts(c, we_go_second=True)            # all-selected == none (any class)
        assert some.accepts(c, we_go_second=True) == (c in (HeroClass.PALADIN, HeroClass.MAGE))
        # going 1st: every configuration concedes, whatever the class
        assert not none_.accepts(c, we_go_second=False)
        assert not some.accepts(c, we_go_second=False)
        assert not allc.accepts(c, we_go_second=False)


def test_pass_rate_estimate():
    cfg = load_config()
    # no filter -> ~1.0
    assert cfg.criteria.pass_rate_estimate() == 1.0
    one_class = replace(cfg.criteria, target_classes=(HeroClass.MAGE,))
    assert 0.0 < one_class.pass_rate_estimate() < 0.2
    one_class_second = replace(one_class, require_second=True)
    assert one_class_second.pass_rate_estimate() < one_class.pass_rate_estimate()


# ── target classes are session-only: cleared on app launch ───────────────────

def test_clear_target_classes_wipes_targets_but_keeps_other_criteria(tmp_path):
    """The app clears the persisted target picks on launch (nothing selected on start),
    while require_second and the file's comments survive."""
    p = tmp_path / "config.toml"
    save_criteria(target_classes=(HeroClass.PALADIN,), require_second=True,
                  mode="casual", path=p)
    assert load_config(p).criteria.target_classes == (HeroClass.PALADIN,)

    clear_target_classes(p)

    crit = load_config(p).criteria
    assert crit.target_classes == ()          # nothing selected after a launch
    assert crit.require_second is True         # the other picks are untouched
    assert crit.mode == "casual"


def test_clear_target_classes_preserves_provenance_comments(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        "[criteria]\n"
        "# keep me: a provenance note\n"
        'target_classes = ["PALADIN"]\n'
        "require_second = false\n"
    )
    clear_target_classes(p)
    text = p.read_text()
    assert "# keep me: a provenance note" in text   # comments survive the rewrite
    assert "target_classes = []" in text


def test_clear_target_classes_no_file_is_a_noop(tmp_path):
    clear_target_classes(tmp_path / "does_not_exist.toml")   # must not raise


def test_clear_target_classes_never_corrupts_a_multiline_array(tmp_path):
    """Regression: the config is meant to be hand-edited, and a hand-written MULTI-LINE
    target_classes array is valid TOML. tomledit is line-based, so a naive rewrite would
    replace only the first line and orphan the rest -> invalid TOML. Since this now runs
    automatically on every app launch, a corrupted write would brick every later load. The
    file must stay parseable; the clean slate is still applied in memory by the caller."""
    import tomllib

    p = tmp_path / "config.toml"
    p.write_text(
        "[criteria]\n"
        "# keep me\n"
        "target_classes = [\n"
        '  "MAGE", "PALADIN",\n'
        '  "WARRIOR",\n'
        "]\n"
        "require_second = true\n"
    )
    clear_target_classes(p)                 # must not corrupt the file
    tomllib.loads(p.read_text())            # still valid TOML (raises if the guard failed)
    assert "# keep me" in p.read_text()     # comments untouched


def test_load_config_clean_targets_starts_each_launch_with_nothing_selected(tmp_path):
    """`hop app` / `hop dashboard` treat target classes as session-only: every launch opens
    with nothing selected. _load_config_clean_targets wipes the persisted picks on disk AND
    drops them from the in-memory config the UIs build their checkboxes from, while leaving
    require_second (and the file's comments) intact."""
    from hop.cli import _load_config_clean_targets

    p = tmp_path / "config.toml"
    save_criteria(target_classes=(HeroClass.PALADIN, HeroClass.MAGE), require_second=True,
                  mode="casual", path=p)

    cfg = _load_config_clean_targets(p)
    assert cfg.criteria.target_classes == ()          # in-memory config is a clean slate
    assert cfg.criteria.require_second is True         # other picks survive
    assert load_config(p).criteria.target_classes == ()   # and the disk file was cleared too


def test_load_config_clean_targets_is_a_noop_when_nothing_selected(tmp_path):
    """No picks on disk -> nothing to clear; the config loads unchanged."""
    from hop.cli import _load_config_clean_targets

    p = tmp_path / "config.toml"
    save_criteria(target_classes=(), require_second=False, mode="casual", path=p)
    cfg = _load_config_clean_targets(p)
    assert cfg.criteria.target_classes == ()
