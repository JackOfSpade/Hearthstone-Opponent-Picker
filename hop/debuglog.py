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

**One exception: UNKNOWN screens.** Every other frame can be re-captured on
demand - point the phone at the mulligan again and take another screencap. A frame
the classifier could not *name* cannot. It is, by definition, a screen nobody
anticipated (a "Ban Notice" modal; the Collection), it halts the loop, and the only
way to stop the *next* halt is to build an anchor from those exact pixels. So
:meth:`DebugLog.unknown_screen` writes them, in **colour**, to a directory that
lives *outside* the run dirs and is never touched by ``keep_runs``:

    ~/.config/hop/unknowns/

That directory is **empty when the hunt is healthy**. Its non-emptiness is the
signal. Empty it by hand once each frame has become an anchor (``hop capture
--from-file``). ``max_unknown_frames`` is a runaway backstop - a bound on a
pathological loop, not a retention policy - not a licence to discard evidence.
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
#: Runaway backstop for the unknown-screen store, NOT a retention policy. A Halt
#: ends the run, so a healthy-then-surprised hunt writes one frame and stops; only
#: a pathological caller could reach this. Keep it well above any real run.
DEFAULT_MAX_UNKNOWN_FRAMES = 30


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


def plan_unknown_pruning(frames: list[Path], keep: int) -> list[Path]:
    """Which unknown-screen frames to delete so at most ``keep`` newest survive.

    ``keep <= 0`` means *never prune* - the store is meant to be emptied by a human
    who has looked at it. Names are ``unknown_<YYYYmmdd-HHMMSS-ffffff>_<where>.png``,
    so lexicographic order is chronological and no stat() is needed.
    """
    if keep <= 0:
        return []
    ordered = sorted(frames, key=lambda p: p.name)
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
                 max_anomaly_frames: int = DEFAULT_MAX_ANOMALY_FRAMES,
                 unknown_dir: str | Path | None = None,
                 max_unknown_frames: int = DEFAULT_MAX_UNKNOWN_FRAMES):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._journal = self.run_dir / "journal.jsonl"
        self._clock = clock
        self._n = 0
        self._unknowns = 0
        self.max_anomaly_frames = max_anomaly_frames
        # Deliberately a SIBLING of the run dirs, not a child: `keep_runs` retires
        # run dirs, and the one frame we must never lose lives here.
        self.unknown_dir = Path(unknown_dir) if unknown_dir is not None \
            else self.run_dir.parent.parent / "unknowns"
        self.max_unknown_frames = max_unknown_frames

    def record(self, kind: str, /, **detail: Any) -> None:
        # `kind` is positional-only (the `/`) on purpose: `anomaly`/`unknown_screen`
        # forward caller **context into here, and a caller whose context happens to
        # carry a `kind=` key would otherwise raise "got multiple values for argument
        # 'kind'" and crash the recorder mid-anomaly. Positional-only makes that key
        # land in `detail` instead of colliding with this parameter.
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
                    self._save_frame(frame, self.run_dir / f"anomaly_{self._n}_{tag}.png")
            self._prune_frames()

    def unknown_screen(self, frame: Frame | None, *, where: str, **context: Any) -> Path | None:
        """Persist a frame the classifier could not name. Returns the path written.

        This is the *only* capture `hop` keeps on purpose. Written in colour (the
        vision layer is grayscale, but a human diagnosing "what screen is this?"
        needs the hue, and `hop capture --from-file` builds the new anchor from
        this file), to :attr:`unknown_dir`, which ``keep_runs`` never touches.

        ``where`` names the call site - ``dispatch``, ``clear_end_screens``, ... -
        so a folder listing already says which part of the loop got lost.
        """
        self._unknowns += 1
        self.record("unknown_screen", where=where, **context)
        if frame is None or not pil_available():
            return None
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        # microseconds: two unknowns inside one second must not collide, and the
        # name has to sort chronologically for `plan_unknown_pruning`.
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        stamp = f"{stamp}-{self._unknowns:03d}"
        path = self.unknown_dir / f"unknown_{stamp}_{where}.png"
        written = self._save_frame(frame, path, colour=True)
        sidecar = path.with_suffix(".json")
        try:
            sidecar.write_text(json.dumps({"where": where, "t": self._clock(),
                                           **_jsonable(context)}, indent=2) + "\n")
        except OSError:
            pass
        self._prune_unknowns()
        return path if written else None

    def _prune_frames(self) -> None:
        """Hold this run to ``max_anomaly_frames`` newest frames.

        Called after each anomaly rather than at exit: a run that halts hard, or is
        killed, must not be the one that leaves a gigabyte behind.
        """
        for p in plan_frame_pruning(list(self.run_dir.glob("anomaly_*.png")),
                                    self.max_anomaly_frames):
            p.unlink(missing_ok=True)

    def _prune_unknowns(self) -> None:
        """Runaway backstop only - see ``DEFAULT_MAX_UNKNOWN_FRAMES``."""
        for p in plan_unknown_pruning(list(self.unknown_dir.glob("unknown_*.png")),
                                      self.max_unknown_frames):
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)

    def _save_frame(self, frame: Frame, path: Path,
                    colour: bool = False) -> bool:  # pragma: no cover - needs Pillow
        try:
            from PIL import Image
            import numpy as np
            if colour and frame.rgb is not None:
                Image.fromarray(np.asarray(frame.rgb, dtype="uint8"), "RGB").save(path)
            elif isinstance(frame.data, (bytes, bytearray)):
                Image.frombytes("L", (frame.width, frame.height), bytes(frame.data)).save(path)
            else:
                Image.fromarray(np.asarray(frame.data, dtype="uint8"), "L").save(path)
            return True
        except Exception:
            return False


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _json_value(v) for k, v in d.items()}


def _json_value(v: Any) -> Any:
    """Coerce one value to a JSON-serialisable form, preserving nested structure.

    The list branch used to ``str()`` every non-scalar element, which flattened a
    list of dicts - e.g. an unknown screen's ``near_misses`` ``[{"state":..}, ...]`` -
    into a list of Python reprs. The journal then held strings where the bug-report
    summariser expected dicts (``near_misses[0].get("state")``), so a single
    unknown-screen halt made ``hop bugreport`` raise ``AttributeError`` on the exact
    run it most needed to describe. Recurse instead, and only stringify genuine
    non-serialisable leaves.
    """
    if is_dataclass(v) and not isinstance(v, type):
        return _json_value(asdict(v))
    if isinstance(v, dict):
        return {str(k): _json_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_value(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)
