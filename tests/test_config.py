from dataclasses import replace

import pytest

from hop.config import clear_target_classes, load_config, save_criteria, validate_config
from hop.hero_classes import HeroClass


def test_defaults_load():
    cfg = load_config()
    assert cfg.device.opponent_corner == "bottom_left"
    assert cfg.motor.tap_dwell_min_s < cfg.motor.tap_dwell_max_s
    assert cfg.timing.gear_menu_cooldown_s == 0.5
    assert cfg.timing.post_concede_start_cooldown_s == 0.5


@pytest.mark.parametrize("field", [
    "gear_menu_cooldown_s",
    "post_concede_start_cooldown_s",
])
def test_opening_cooldowns_reject_values_above_one_second(field):
    """The two known UI-opening pauses are deliberately short and bounded."""
    cfg = load_config()
    with pytest.raises(ValueError):
        validate_config(replace(cfg, timing=replace(cfg.timing, **{field: 1.001})))


@pytest.mark.parametrize("change", [
    {"post_concede_start_cooldown_s": 0},
    {"post_concede_click_interval_s": 0},
    {"post_concede_click_interval_max_s": 0},
    {"post_concede_click_count": 0},
    {"post_concede_click_count": 1.5},
    {"post_concede_click_count": True},
    {"post_concede_click_interval_s": 0.9, "post_concede_click_interval_max_s": 0.75},
    {"post_concede_click_count": 20, "post_concede_burst_max_s": 1.0},
    {"post_concede_click_interval_s": "fast"},
])
def test_invalid_post_concede_timing_is_rejected_before_engine_use(change):
    cfg = load_config()
    with pytest.raises(ValueError):
        validate_config(replace(cfg, timing=replace(cfg.timing, **change)))


def test_burst_cap_must_remain_below_the_sealed_observed_mulligan_floor():
    cfg = load_config()
    floor = cfg.timing.play_to_mulligan_observed_min_s
    with pytest.raises(ValueError):
        validate_config(replace(cfg, timing=replace(cfg.timing, post_concede_burst_max_s=floor)))


def test_post_concede_boundary_due_time_cannot_follow_the_burst():
    """The blind trace must leave room to capture before a fast mulligan expires."""
    cfg = load_config()
    floor = cfg.timing.play_to_mulligan_observed_min_s
    with pytest.raises(ValueError, match="capture/classification margin"):
        validate_config(replace(
            cfg, timing=replace(
                cfg.timing,
                post_concede_queue_cooldown_s=floor - 4.0,
            ),
        ))


@pytest.mark.parametrize("field", [
    "play_to_mulligan_cooldown_s", "mulligan_confirm_cooldown_s",
    "gear_menu_cooldown_s", "concede_resolve_cooldown_s",
    "end_dismiss_cooldown_s", "post_concede_queue_cooldown_s",
    "post_concede_start_cooldown_s",
    "post_concede_click_interval_s",
])
def test_sealed_calibration_floors_reject_tiny_user_values(field):
    cfg = load_config()
    value = getattr(cfg.timing, field)
    with pytest.raises(ValueError, match="sealed shipped calibration"):
        validate_config(replace(cfg, timing=replace(cfg.timing, **{field: value / 2})))


def test_user_cannot_raise_observed_floor_to_expand_burst_cap(tmp_path):
    """Validation reads unmerged shipped evidence, never the user-merged timing table."""
    path = tmp_path / "config.toml"
    path.write_text(
        "[timing]\n"
        "play_to_mulligan_observed_min_s = 100.0\n"
        "post_concede_burst_max_s = 99.0\n"
    )
    with pytest.raises(ValueError):
        load_config(path)


def test_legacy_matchmaking_wait_keys_are_ignored_on_config_load(tmp_path):
    """Old queue/read retry knobs must not revive a timeout in the raw class watch."""
    shipped = load_config()
    path = tmp_path / "legacy.toml"
    path.write_text(
        "[timing]\n"
        "queue_to_mulligan_cooldown_s = 0.01\n"
        "vs_to_mulligan_cooldown_s = 0.01\n"
        "[vision]\n"
        "queue_wait_attempts = 1\n"
        "mulligan_read_attempts = 1\n"
        "mulligan_unreadable_halt_streak = 1\n"
    )

    loaded = load_config(path)

    assert loaded.timing.post_concede_queue_cooldown_s == shipped.timing.post_concede_queue_cooldown_s
    assert not hasattr(loaded.timing, "queue_to_mulligan_cooldown_s")
    assert not hasattr(loaded.timing, "vs_to_mulligan_cooldown_s")
    assert not hasattr(loaded.vision, "queue_wait_attempts")
    assert not hasattr(loaded.vision, "mulligan_read_attempts")
    assert not hasattr(loaded.vision, "mulligan_unreadable_halt_streak")


def test_safe_direction_timing_overrides_remain_allowed():
    cfg = load_config()
    t = cfg.timing
    safer = replace(
        t,
        play_to_mulligan_cooldown_s=t.play_to_mulligan_cooldown_s + 1,
        mulligan_confirm_cooldown_s=t.mulligan_confirm_cooldown_s + 1,
        gear_menu_cooldown_s=1.0,
        concede_resolve_cooldown_s=t.concede_resolve_cooldown_s + 1,
        end_dismiss_cooldown_s=t.end_dismiss_cooldown_s + 1,
        post_concede_queue_cooldown_s=t.post_concede_queue_cooldown_s + 1,
        post_concede_start_cooldown_s=1.0,
        post_concede_click_interval_s=t.post_concede_click_interval_s + 0.05,
        post_concede_click_interval_max_s=t.post_concede_click_interval_max_s,
        post_concede_burst_max_s=t.post_concede_burst_max_s - 1,
    )
    validate_config(replace(cfg, timing=safer))


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
