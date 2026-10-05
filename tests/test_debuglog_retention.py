"""Retention: capture is for analysis, and analysed pixels are dead weight.

...with exactly one exception. A frame of a screen the classifier could not *name*
cannot be re-taken - nobody knows how to navigate back to a screen nobody identified -
and it is what the anchor that prevents the next halt gets built from. Those are kept,
outside the run dirs, in a folder that is empty when the hunt is healthy.

The planners are pure so the policy is testable without a phone, a run, or Pillow.
"""

from pathlib import Path
import json
from datetime import datetime

import pytest

from hop.debuglog import (
    DEFAULT_KEEP_RUNS,
    DEFAULT_MAX_ANOMALY_FRAMES,
    DEFAULT_MAX_UNKNOWN_FRAMES,
    DebugLog,
    plan_frame_pruning,
    plan_run_pruning,
    plan_unknown_pruning,
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


def test_journal_entry_has_a_local_iso_timestamp_from_its_single_clock_sample(tmp_path):
    class FrozenClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return 1_791_002_909.123456

    clock = FrozenClock()
    log = DebugLog(tmp_path, clock=clock)
    log.record("tap", what="gear")

    entry = json.loads((tmp_path / "journal.jsonl").read_text())
    assert clock.calls == 1
    assert entry["t"] == 1_791_002_909.123456  # retained for existing readers
    parsed = datetime.fromisoformat(entry["ts"])
    assert parsed.tzinfo is not None and parsed.utcoffset() is not None
    assert parsed.timestamp() == pytest.approx(entry["t"], abs=0.000001)


def test_new_journal_timestamp_is_ignored_by_existing_bugreport_parser(tmp_path):
    """`ts` is additive: readers of the old numeric-`t` schema still consume it."""
    from hop.bugreport import summarize_journal

    log = DebugLog(tmp_path, clock=lambda: 0.0)
    log.record("mulligan_read", opponent="Mage", second=False)
    journal = (tmp_path / "journal.jsonl").read_text()

    assert "ts" in json.loads(journal)
    assert "Mage" in summarize_journal(journal)


def test_jsonable_preserves_nested_dicts_in_lists():
    """A journalled list of dicts (e.g. unknown-screen near_misses) must round-trip as
    dicts, not as Python-repr strings -- the summariser reads them as objects."""
    import json

    from hop.debuglog import _jsonable

    out = _jsonable({"near_misses": [{"state": "in_game", "score": 0.53, "thr": 0.72}],
                     "n": 3, "where": "dispatch"})
    assert out["near_misses"][0] == {"state": "in_game", "score": 0.53, "thr": 0.72}
    json.dumps(out)                        # and the whole thing stays JSON-serialisable


def test_jsonable_stringifies_only_genuine_non_serialisable_leaves():
    from hop.debuglog import _jsonable

    class Weird:
        def __repr__(self): return "<weird>"

    out = _jsonable({"items": [Weird(), {"ok": 1}], "obj": Weird()})
    assert out["items"] == ["<weird>", {"ok": 1}]
    assert out["obj"] == "<weird>"


def test_defaults_are_bounded():
    assert 0 < DEFAULT_MAX_ANOMALY_FRAMES < 100
    assert 0 < DEFAULT_KEEP_RUNS < 100
    assert 0 < DEFAULT_MAX_UNKNOWN_FRAMES < 100


# ── the exception: frames of screens nobody named ────────────────────────────

def _unknowns(n: int) -> list[Path]:
    return [Path(f"unknown_20260708-1200{i:02d}-{i:03d}_dispatch.png") for i in range(1, n + 1)]


def test_unknown_pruning_is_chronological_because_names_are_timestamps():
    assert plan_unknown_pruning(_unknowns(4), keep=2) == _unknowns(2)


def test_unknown_pruning_keep_zero_means_never_prune_not_delete_everything():
    """The store exists to be read by a human. `keep <= 0` must not wipe it.

    This is the opposite convention to `plan_frame_pruning`, deliberately: anomaly
    frames are bulk to be bounded, unknown frames are evidence to be preserved.
    """
    assert plan_unknown_pruning(_unknowns(3), keep=0) == []
    assert plan_unknown_pruning(_unknowns(3), keep=-1) == []


def test_unknown_pruning_is_a_noop_below_the_cap():
    assert plan_unknown_pruning(_unknowns(2), keep=30) == []


def test_unknown_dir_is_a_sibling_of_the_run_dirs_not_a_child(tmp_path):
    """`keep_runs` retires run dirs. The one frame worth keeping must not ride along."""
    run_dir = tmp_path / "runs" / "20260708-020000"
    log = DebugLog(run_dir)
    assert log.unknown_dir == tmp_path / "unknowns"
    assert log.unknown_dir not in run_dir.parents

    (tmp_path / "unknowns").mkdir(parents=True)
    (tmp_path / "unknowns" / "unknown_x_dispatch.png").write_bytes(b"x")
    prune_runs(tmp_path / "runs", keep=0)
    assert not run_dir.exists()
    assert (tmp_path / "unknowns" / "unknown_x_dispatch.png").exists()


def test_unknown_screen_journals_even_without_pillow(tmp_path):
    log = DebugLog(tmp_path / "runs" / "r1", clock=lambda: 0.0)
    log.unknown_screen(None, where="clear_end_screens", confidence=0.31)
    journal = (tmp_path / "runs" / "r1" / "journal.jsonl").read_text()
    assert "unknown_screen" in journal
    assert "clear_end_screens" in journal


def test_unknown_screen_writes_frame_and_sidecar(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    import numpy as np

    from hop.perception.image import Frame

    rgb = np.zeros((4, 6, 3), dtype="uint8")
    rgb[..., 0] = 200                       # colour must survive: the vision layer is
    frame = Frame(6, 4, np.zeros((4, 6), dtype="uint8"), rgb=rgb)   # gray, humans aren't

    log = DebugLog(tmp_path / "runs" / "r1", clock=lambda: 0.0,
                   unknown_dir=tmp_path / "unknowns")
    path = log.unknown_screen(frame, where="dispatch", confidence=0.4)
    assert path is not None and path.exists()
    assert path.parent == tmp_path / "unknowns"
    assert "dispatch" in path.name

    from PIL import Image
    assert Image.open(path).mode == "RGB"
    assert Image.open(path).getpixel((0, 0)) == (200, 0, 0)

    sidecar = path.with_suffix(".json")
    assert sidecar.exists() and "dispatch" in sidecar.read_text()


def test_colour_anomaly_preserves_rgb_for_hue_dependent_diagnostics(tmp_path):
    """The green mulligan glow is not recoverable from a normal grayscale anomaly."""
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    import numpy as np
    from PIL import Image

    from hop.perception.image import Frame

    rgb = np.zeros((4, 6, 3), dtype="uint8")
    rgb[..., 1] = 200
    frame = Frame(6, 4, np.zeros((4, 6), dtype="uint8"), rgb=rgb)
    log = DebugLog(tmp_path, clock=lambda: 0.0)

    log.anomaly("mulligan card count unreadable", before=frame, colour=True)

    saved = tmp_path / "anomaly_1_before.png"
    assert saved.exists()
    with Image.open(saved) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (0, 200, 0)
    entry = json.loads((tmp_path / "journal.jsonl").read_text())
    assert entry["kind"] == "anomaly"
    assert entry["detail"]["reason"] == "mulligan card count unreadable"


def test_unknown_store_is_empty_when_nothing_is_unknown(tmp_path):
    """Its non-emptiness IS the signal, so a healthy run must not create it."""
    log = DebugLog(tmp_path / "runs" / "r1", unknown_dir=tmp_path / "unknowns")
    log.record("tap", what="gear")
    log.anomaly("missed tap")
    assert not (tmp_path / "unknowns").exists()
