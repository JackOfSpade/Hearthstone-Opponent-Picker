"""Retention: capture is for analysis, and analysed pixels are dead weight.

The planners are pure so the policy is testable without a phone, a run, or Pillow.
"""

from pathlib import Path

import pytest

from hop.debuglog import (
    DEFAULT_KEEP_RUNS,
    DEFAULT_MAX_ANOMALY_FRAMES,
    DebugLog,
    plan_frame_pruning,
    plan_run_pruning,
    prune_runs,
)


def _frames(n: int) -> list[Path]:
    """n anomalies, before+after each, deliberately shuffled."""
    out = []
    for i in range(1, n + 1):
        out += [Path(f"anomaly_{i}_after.png"), Path(f"anomaly_{i}_before.png")]
    return out[::-1]


def test_frame_pruning_keeps_the_newest_by_anomaly_index():
    doomed = plan_frame_pruning(_frames(6), keep=4)
    assert len(doomed) == 8
    # anomalies 1..4 die; 5 and 6 - the ones that explain the halt - survive
    survivors = set(_frames(6)) - set(doomed)
    assert {p.name for p in survivors} == {
        "anomaly_5_before.png", "anomaly_5_after.png",
        "anomaly_6_before.png", "anomaly_6_after.png",
    }


def test_frame_pruning_orders_numerically_not_lexicographically():
    """anomaly_10 is newer than anomaly_9; a string sort would delete the wrong one."""
    frames = [Path("anomaly_9_before.png"), Path("anomaly_10_before.png")]
    assert plan_frame_pruning(frames, keep=1) == [Path("anomaly_9_before.png")]


def test_frame_pruning_is_a_noop_below_the_cap():
    assert plan_frame_pruning(_frames(2), keep=8) == []


def test_frame_pruning_keep_zero_deletes_everything():
    assert len(plan_frame_pruning(_frames(3), keep=0)) == 6


def test_frame_pruning_tolerates_unparseable_names():
    """A stray file sorts first (oldest) and is culled before real evidence."""
    frames = [Path("anomaly_2_before.png"), Path("stray.png")]
    assert plan_frame_pruning(frames, keep=1) == [Path("stray.png")]


def test_run_pruning_is_chronological_because_names_are_timestamps():
    dirs = [Path("20260708-024512"), Path("20260707-235959"), Path("20260708-002932")]
    assert plan_run_pruning(dirs, keep=1) == [
        Path("20260707-235959"), Path("20260708-002932"),
    ]


def test_negative_keep_is_a_bug_not_a_silent_wipe():
    with pytest.raises(ValueError):
        plan_frame_pruning([], keep=-1)
    with pytest.raises(ValueError):
        plan_run_pruning([], keep=-1)


def test_prune_runs_removes_oldest_dirs_and_leaves_the_newest(tmp_path):
    for name in ("20260708-000000", "20260708-010000", "20260708-020000"):
        d = tmp_path / name
        d.mkdir()
        (d / "journal.jsonl").write_text("{}\n")
    removed = prune_runs(tmp_path, keep=2)
    assert [p.name for p in removed] == ["20260708-000000"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "20260708-010000", "20260708-020000",
    ]
    # the survivors kept their journals - the thing a halt is diagnosed from
    assert (tmp_path / "20260708-020000" / "journal.jsonl").exists()


def test_prune_runs_ignores_a_missing_root(tmp_path):
    assert prune_runs(tmp_path / "nope", keep=3) == []


def test_prune_runs_ignores_loose_files(tmp_path):
    (tmp_path / "stray.txt").write_text("x")
    (tmp_path / "20260708-000000").mkdir()
    assert prune_runs(tmp_path, keep=0) == [tmp_path / "20260708-000000"]
    assert (tmp_path / "stray.txt").exists()


def test_debuglog_prunes_its_own_frames_on_each_anomaly(tmp_path):
    """Pruning happens per-anomaly, not at exit: a hard halt never reaches an exit."""
    log = DebugLog(tmp_path, max_anomaly_frames=4)
    for i in range(1, 6):
        # stand in for _save_frame, which needs Pillow
        for tag in ("before", "after"):
            (tmp_path / f"anomaly_{i}_{tag}.png").write_bytes(b"x")
        log._n = i
        log._prune_frames()
    pngs = sorted(p.name for p in tmp_path.glob("anomaly_*.png"))
    assert pngs == [
        "anomaly_4_after.png", "anomaly_4_before.png",
        "anomaly_5_after.png", "anomaly_5_before.png",
    ]


def test_debuglog_journal_is_never_pruned(tmp_path):
    log = DebugLog(tmp_path, max_anomaly_frames=0)
    log.record("tap", what="gear")
    (tmp_path / "anomaly_1_before.png").write_bytes(b"x")
    log._n = 1
    log._prune_frames()
    assert not list(tmp_path.glob("*.png"))
    assert "gear" in (tmp_path / "journal.jsonl").read_text()


def test_defaults_are_bounded():
    assert 0 < DEFAULT_MAX_ANOMALY_FRAMES < 100
    assert 0 < DEFAULT_KEEP_RUNS < 100
