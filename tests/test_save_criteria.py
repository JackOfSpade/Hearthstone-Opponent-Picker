"""`save_criteria` writes the user's live config file.

It must never corrupt it. The config is mostly *comments* -- every LIVE-VERIFIED
constant carries a provenance note beside it, because "undocumented constants rot"
-- so a tomllib parse/re-emit round-trip is not an option, and an append is how
`hop calibrate` once managed to emit a duplicate `[motor]` header and leave a file
no `hop` command could load.
"""

import tomllib

import pytest

from hop.config import load_config, save_criteria
from hop.hero_classes import HeroClass


def _user_cfg(tmp_path, body: str):
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p


def test_writes_classes_and_require_second(tmp_path):
    p = _user_cfg(tmp_path, '[criteria]\ntarget_classes = []\nrequire_second = false\n')
    save_criteria(target_classes=(HeroClass.MAGE, HeroClass.WARLOCK),
                  require_second=True, path=p)
    parsed = tomllib.loads(p.read_text())
    assert parsed["criteria"]["target_classes"] == ["MAGE", "WARLOCK"]
    assert parsed["criteria"]["require_second"] is True


def test_round_trips_through_load_config(tmp_path):
    p = _user_cfg(tmp_path, '[criteria]\ntarget_classes = []\nrequire_second = false\n')
    save_criteria(target_classes=(HeroClass.PRIEST,), require_second=True, path=p)
    cfg = load_config(p)
    assert cfg.criteria.target_classes == (HeroClass.PRIEST,)
    assert cfg.criteria.require_second is True


def test_empty_target_classes_means_accept_any(tmp_path):
    p = _user_cfg(tmp_path, '[criteria]\ntarget_classes = ["MAGE"]\nrequire_second = true\n')
    save_criteria(target_classes=(), require_second=False, path=p)
    cfg = load_config(p)
    assert cfg.criteria.target_classes == ()
    assert cfg.criteria.accepts(HeroClass.DRUID, we_go_second=False) is True


def test_saving_targets_clears_avoid_classes(tmp_path):
    p = _user_cfg(
        tmp_path,
        '[criteria]\ntarget_classes = ["MAGE"]\navoid_classes = ["PRIEST"]\n'
        'require_second = false\n',
    )
    save_criteria(target_classes=(), require_second=False, path=p)
    cfg = load_config(p)
    assert cfg.criteria.avoid_classes == ()
    assert cfg.criteria.accepts(HeroClass.PRIEST, we_go_second=False) is True


def test_preserves_comments_and_unrelated_tables(tmp_path):
    body = (
        "# hop user config\n"
        "[criteria]\n"
        "target_classes = []\n"
        "require_second = false\n"
        "\n"
        "[contact]\n"
        "# LIVE-VERIFIED 2026-07-07: ABS_MT_PRESSURE 0..255, calibration physical\n"
        'pressure_semantics = "ramp"\n'
        "\n"
        "[motor]\n"
        "report_rate_hz = 183    # LIVE-VERIFIED: median SYN_REPORT interval\n"
    )
    p = _user_cfg(tmp_path, body)
    save_criteria(target_classes=(HeroClass.MAGE,), require_second=True, path=p)
    out = p.read_text()

    assert "# LIVE-VERIFIED 2026-07-07" in out
    assert "# LIVE-VERIFIED: median SYN_REPORT interval" in out
    parsed = tomllib.loads(out)
    assert parsed["motor"]["report_rate_hz"] == 183
    assert parsed["contact"]["pressure_semantics"] == "ramp"


def test_is_idempotent_and_never_duplicates_a_table(tmp_path):
    p = _user_cfg(tmp_path, '[criteria]\ntarget_classes = []\nrequire_second = false\n')
    save_criteria(target_classes=(HeroClass.MAGE,), require_second=True, path=p)
    once = p.read_text()
    save_criteria(target_classes=(HeroClass.MAGE,), require_second=True, path=p)
    twice = p.read_text()
    assert once == twice
    assert twice.count("[criteria]") == 1
    tomllib.loads(twice)   # still valid


def test_creates_the_file_and_table_when_absent(tmp_path):
    p = tmp_path / "nested" / "config.toml"
    save_criteria(target_classes=(HeroClass.ROGUE,), require_second=False, path=p)
    assert load_config(p).criteria.target_classes == (HeroClass.ROGUE,)


def test_risk_profile_is_validated(tmp_path):
    p = _user_cfg(tmp_path, "[criteria]\ntarget_classes = []\n")
    with pytest.raises(ValueError, match="unknown risk profile"):
        save_criteria(target_classes=(), require_second=False,
                      risk_profile="yolo", path=p)


def test_risk_profile_writes_to_its_own_table(tmp_path):
    p = _user_cfg(tmp_path, "[criteria]\ntarget_classes = []\n")
    save_criteria(target_classes=(), require_second=False,
                  mode="ranked", risk_profile="balanced", path=p)
    cfg = load_config(p)
    assert cfg.risk_profile == "balanced"
    assert cfg.criteria.mode == "ranked"
