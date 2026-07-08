"""Silent debug log - Layer 6.

Standard: *"Silent debug log (per-run action journal + before/after screencaps
and sensor snippets on any anomaly) makes every halt reconstructable offline at
~no runtime cost."*

The journal is always on (cheap: append-only JSONL). Frames/sensor snippets are
persisted only on an anomaly, so a green run costs nothing but a few KB of text.

**Retention.** "Preserving evidence" and "don't let the disk grow without bound"
pull against each other, and only one of them is a standard. Frames are the bulky
half (a 2400x1080 grayscale PNG is ~1 MB; a debugging session left 10 MB of them
behind) while the journal that every diagnosis is actually read from is a few KB.
So both are bounded, asymmetrically: keep the newest ``max_anomaly_frames`` frames
in a run and the newest ``keep_runs`` run directories, and never prune a journal
that still has a run dir. Deleting the *oldest* frames is the right end to cut
from - a halt is diagnosed from the anomaly that stopped it, which is the last one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .perception.image import Frame, pil_available

#: A frame pair per anomaly, so this is ~4 anomalies' worth of before/after.
DEFAULT_MAX_ANOMALY_FRAMES = 8
DEFAULT_KEEP_RUNS = 5


def plan_frame_pruning(frames: list[Path], keep: int) -> list[Path]:
    """Which anomaly frames to delete so at most ``keep`` newest survive.

    Pure: takes the paths, returns the doomed subset. Ordering is by the name's
    embedded index, not mtime - two frames of one anomaly are written in the same
    instant and mtime cannot separate them, but ``anomaly_<n>_<tag>.png`` can.
    """
    if keep < 0:
        raise ValueError("keep must be >= 0")

    def sort_key(p: Path) -> tuple[int, str]:
        parts = p.stem.split("_")
        try:
            return (int(parts[1]), p.name)
        except (IndexError, ValueError):
            return (-1, p.name)

    ordered = sorted(frames, key=sort_key)
    return ordered[: max(0, len(ordered) - keep)]


def plan_run_pruning(run_dirs: list[Path], keep: int) -> list[Path]:
    """Which run directories to delete so at most ``keep`` newest survive.

    Run dirs are named ``%Y%m%d-%H%M%S``, so lexicographic order *is* chronological
    order - no stat() call, and no dependence on a filesystem that may not preserve
    mtime across a copy.
    """
    if keep < 0:
        raise ValueError("keep must be >= 0")
    ordered = sorted(run_dirs, key=lambda p: p.name)
    return ordered[: max(0, len(ordered) - keep)]


def prune_runs(root: str | Path, keep: int = DEFAULT_KEEP_RUNS) -> list[Path]:
    """Delete all but the ``keep`` newest run dirs under ``root``. Returns removed."""
    import shutil

    root = Path(root)
    if not root.is_dir():
        return []
    doomed = plan_run_pruning([p for p in root.iterdir() if p.is_dir()], keep)
    for p in doomed:
        shutil.rmtree(p, ignore_errors=True)
    return doomed


@dataclass
class JournalEntry:
    t: float
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


class DebugLog:
    def __init__(self, run_dir: str | Path, clock=time.time,
                 max_anomaly_frames: int = DEFAULT_MAX_ANOMALY_FRAMES):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._journal = self.run_dir / "journal.jsonl"
        self._clock = clock
        self._n = 0
        self.max_anomaly_frames = max_anomaly_frames

    def record(self, kind: str, **detail: Any) -> None:
        entry = JournalEntry(t=self._clock(), kind=kind, detail=_jsonable(detail))
        with self._journal.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def anomaly(
        self,
        reason: str,
        before: Frame | None = None,
        after: Frame | None = None,
        **context: Any,
    ) -> None:
        """Record an anomaly and, if possible, dump before/after frames."""
        self._n += 1
        self.record("anomaly", reason=reason, index=self._n, **context)
        if pil_available():
            for tag, frame in (("before", before), ("after", after)):
                if frame is not None:
                    self._save_frame(frame, f"anomaly_{self._n}_{tag}.png")
            self._prune_frames()

    def _prune_frames(self) -> None:
        """Hold this run to ``max_anomaly_frames`` newest frames.

        Called after each anomaly rather than at exit: a run that halts hard, or is
        killed, must not be the one that leaves a gigabyte behind.
        """
        for p in plan_frame_pruning(list(self.run_dir.glob("anomaly_*.png")),
                                    self.max_anomaly_frames):
            p.unlink(missing_ok=True)

    def _save_frame(self, frame: Frame, name: str) -> None:  # pragma: no cover - needs Pillow
        try:
            from PIL import Image
            import numpy as np
            path = self.run_dir / name
            if isinstance(frame.data, (bytes, bytearray)):
                Image.frombytes("L", (frame.width, frame.height), bytes(frame.data)).save(path)
            else:
                Image.fromarray(np.asarray(frame.data, dtype="uint8"), "L").save(path)
        except Exception:
            pass


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if is_dataclass(v):
            out[k] = asdict(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (str, int, float, bool)) or x is None else str(x) for x in v]
        else:
            out[k] = str(v)
    return out
