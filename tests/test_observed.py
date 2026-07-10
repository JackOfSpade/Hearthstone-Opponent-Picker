"""The observed class distribution is day-scoped and persistent.

The dashboard chart must survive an app restart *within the same day* (close at lunch,
reopen, the morning's games are still there) but reset on a new day. The store is the
JSON file; the EngineController accrues each run's counts into it without double-counting
across a Stop/Start.
"""

from hop.observed import ObservedDistribution, default_path
from hop.runner import EngineController


class _FakeStats:
    def __init__(self):
        self.class_distribution: dict[str, int] = {}


class _FakeEngine:
    def __init__(self):
        self.stats = _FakeStats()


# ── the store: date-scoped load / delete-on-new-day ──────────────────────────

def test_load_same_day_restores_counts(tmp_path):
    p = tmp_path / "d.json"
    ObservedDistribution(p, "2026-07-10", {"Mage": 2}).write()
    assert ObservedDistribution.load(p, "2026-07-10").counts == {"Mage": 2}


def test_load_different_day_deletes_and_starts_fresh(tmp_path):
    p = tmp_path / "d.json"
    ObservedDistribution(p, "2026-07-09", {"Mage": 2}).write()   # yesterday
    loaded = ObservedDistribution.load(p, "2026-07-10")
    assert loaded.counts == {}       # a new day starts empty
    assert not p.exists()            # the stale save was deleted, as asked


def test_load_missing_file_starts_fresh(tmp_path):
    assert ObservedDistribution.load(tmp_path / "nope.json", "2026-07-10").counts == {}


def test_load_malformed_file_starts_fresh(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("{ not json")
    assert ObservedDistribution.load(p, "2026-07-10").counts == {}


def test_load_drops_non_finite_counts_without_raising(tmp_path):
    """`load` promises "Never raises". Python's json accepts NaN/Infinity, which then
    reach `int(v)` and blow up (NaN -> ValueError, Infinity -> OverflowError) at app
    startup. A non-finite count must be dropped like any other bad value, not crash the
    launch -- the finite, valid entries beside it survive."""
    p = tmp_path / "d.json"
    p.write_text('{"date": "2026-07-10", "counts": {"Mage": NaN, "Rogue": Infinity, "Druid": 3}}')
    loaded = ObservedDistribution.load(p, "2026-07-10")   # must not raise
    assert loaded.counts == {"Druid": 3}


def test_default_path_sits_beside_the_config(tmp_path):
    cfg = tmp_path / "sub" / "config.toml"
    assert default_path(cfg) == tmp_path / "sub" / "observed_distribution.json"


# ── the controller: accrual without double-counting ──────────────────────────

def test_accrues_engine_counts_and_persists(tmp_path):
    p = tmp_path / "d.json"
    store = ObservedDistribution(p, "2026-07-10")
    c = EngineController(lambda o: _FakeEngine(), observed=store)

    eng = _FakeEngine()
    c._engine = eng
    eng.stats.class_distribution["Mage"] = 1
    c._accrue_observed()
    assert store.counts == {"Mage": 1}

    # another game on the same run adds only the delta, not the whole tally again
    eng.stats.class_distribution["Mage"] = 2
    eng.stats.class_distribution["Warrior"] = 1
    c._accrue_observed()
    assert store.counts == {"Mage": 2, "Warrior": 1}

    # and it is on disk, so a same-day restart continues from here
    assert ObservedDistribution.load(p, "2026-07-10").counts == {"Mage": 2, "Warrior": 1}


def test_does_not_double_count_across_runs(tmp_path):
    store = ObservedDistribution(tmp_path / "d.json", "2026-07-10")
    c = EngineController(lambda o: _FakeEngine(), observed=store)

    e1 = _FakeEngine()
    e1.stats.class_distribution["Mage"] = 3
    c._engine = e1
    c._accrue_observed()
    assert store.counts == {"Mage": 3}

    # a fresh run (a different engine object, counting from zero) must add its games on
    # top of the day total, not re-add what the first run already contributed.
    e2 = _FakeEngine()
    e2.stats.class_distribution["Mage"] = 2
    c._engine = e2
    c._accrue_observed()
    assert store.counts == {"Mage": 5}


def test_day_total_is_shown_while_idle(tmp_path):
    """On app open, before any run, the chart carries the day's restored counts."""
    store = ObservedDistribution(tmp_path / "d.json", "2026-07-10", {"Paladin": 4})
    c = EngineController(lambda o: _FakeEngine(), observed=store)
    assert c._engine is None
    assert c.observed_distribution() == {"Paladin": 4}
    assert c.status()["class_distribution"] == {"Paladin": 4}


def test_flush_persists_the_current_run(tmp_path):
    p = tmp_path / "d.json"
    store = ObservedDistribution(p, "2026-07-10")
    c = EngineController(lambda o: _FakeEngine(), observed=store)
    eng = _FakeEngine()
    eng.stats.class_distribution["Rogue"] = 2
    c._engine = eng
    c.flush_observed()
    assert ObservedDistribution.load(p, "2026-07-10").counts == {"Rogue": 2}


def test_without_a_store_falls_back_to_per_run_counts():
    c = EngineController(lambda o: _FakeEngine())   # observed=None (CLI / tests)
    eng = _FakeEngine()
    eng.stats.class_distribution["Mage"] = 1
    c._engine = eng
    assert c.observed_distribution() == {"Mage": 1}
