"""Layer 5 - the non-repetition safeguard, plus informational activity counters.

Rev-2 non-repetition obligation: *"Replay ... or repeated auto runs, must not
reuse identical trajectories, inter-event timings, or navigation rhythms."* Every
emitted gesture is fingerprinted; a near-duplicate of a prior one is rejected
*before* emission (the engine resamples, and only a genuinely degenerate generator
halts - fail closed).

A hunt runs until it finds a target, an error halts it, or the user stops it:
there are deliberately **no** volume, per-session, or time ceilings, and no
mandatory breaks. (Those caps used to live here; they were removed by request -
the operator owns pacing and volume.) The run counters kept here are
informational only - the status display and the concede ratio - never gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import CapsConfig
from ..touchstream import Gesture


@dataclass
class Limiter:
    """The non-repetition memory for one persistent identity, plus informational
    activity counters. All counters reset per process; none of them gate a run."""

    caps: CapsConfig

    # informational counters (status display + concede ratio; never gates)
    actions_this_run: int = 0
    commits_this_run: int = 0
    games_this_session: int = 0
    session_seconds: float = 0.0

    _traj: "TrajectoryMemory" = field(default_factory=lambda: TrajectoryMemory())

    # counters (informational only; never gate a run) -------------------------

    def register_action(self, committing: bool) -> None:
        self.actions_this_run += 1
        if committing:
            self.commits_this_run += 1

    def register_game(self) -> None:
        self.games_this_session += 1

    def register_time(self, dt: float) -> None:
        self.session_seconds += max(0.0, dt)

    def committing_ratio(self) -> float:
        """Concedes / total actions this run - surfaced in status so a high concede
        rate (the barcode-shaped pattern) stays visible to the operator."""
        return self.commits_this_run / self.actions_this_run if self.actions_this_run else 0.0

    # non-repetition ----------------------------------------------------------

    def is_near_duplicate(self, gesture: Gesture) -> bool:
        return self._traj.is_near_duplicate(
            trajectory_fingerprint(gesture), self.caps.non_repetition_threshold
        )

    def remember_trajectory(self, gesture: Gesture) -> None:
        self._traj.add(trajectory_fingerprint(gesture))

    @property
    def trajectory_memory_size(self) -> int:
        return len(self._traj.fingerprints)


def trajectory_fingerprint(g: Gesture) -> list[float]:
    """A small, comparable feature vector describing a gesture's shape+timing.

    Chosen so that two *materially different* human-ish gestures land far apart
    while a verbatim replay lands at ~0 distance. Shape terms are normalized by
    the samples' own bounding box so absolute location doesn't dominate; the
    release point is still included so taps on different controls do not poison
    each other's history during long hunts.
    """
    if not g.samples:
        return [0.0] * 8
    xs = [s.x for s in g.samples]
    ys = [s.y for s in g.samples]
    ts = [s.t for s in g.samples]
    dur = max(1e-6, ts[-1] - ts[0])
    # path length
    plen = 0.0
    peak_v = 0.0
    for i in range(1, len(g.samples)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dt = max(1e-6, ts[i] - ts[i - 1])
        seg = math.hypot(dx, dy)
        plen += seg
        peak_v = max(peak_v, seg / dt)
    span_x = max(1.0, max(xs) - min(xs))
    span_y = max(1.0, max(ys) - min(ys))
    return [
        dur,
        float(len(g.samples)),
        plen / max(span_x, span_y),
        peak_v / 1000.0,
        (xs[-1] - xs[0]) / span_x,
        (ys[-1] - ys[0]) / span_y,
        xs[-1],
        ys[-1],
    ]


@dataclass
class TrajectoryMemory:
    """Remembers gesture fingerprints and answers near-duplicate queries."""

    fingerprints: list[list[float]] = field(default_factory=list)
    # Recent history, not whole-session history. The old 500-entry window made an hour-long
    # hunt fail closed after normal repeated use of the same few Hearthstone controls: the
    # memory became dense enough that eight fresh draws could all collide. Thirty-two still
    # blocks immediate replay patterns while letting the motor model keep drawing for long runs.
    keep: int = 32

    def add(self, fp: list[float]) -> None:
        self.fingerprints.append(fp)
        if len(self.fingerprints) > self.keep:
            self.fingerprints.pop(0)

    def is_near_duplicate(self, fp: list[float], threshold: float) -> bool:
        for prev in self.fingerprints:
            if _normalized_distance(fp, prev) < threshold:
                return True
        return False


def _normalized_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 1e9
    # RMS relative difference, robust to the differing scales of the features.
    acc = 0.0
    for x, y in zip(a, b):
        denom = max(1e-6, abs(x) + abs(y))
        acc += ((x - y) / denom) ** 2
    return math.sqrt(acc / len(a))
