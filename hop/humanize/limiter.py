"""Layer 5 - volume/session caps and the non-repetition safeguard.

Standard: *"Per-session and per-day volume caps ... When a cap is hit, stop the
run - don't substitute a different action to keep going; that both looks robotic
and corrupts any labels/data. Checks run before each autonomous action."* Plus
the Rev-2 non-repetition obligation: *"Replay ... or repeated auto runs, must
not reuse identical trajectories, inter-event timings, or navigation rhythms."*

For Hearthstone the *committing action* - the tap a bot over-produces and the
one an anomalous ratio is computed on - is the **concede**. Rapid, repeated
concedes are exactly the "barcode account" pattern Blizzard segregates, so the
concede is capped below total actions and paced with mandatory breaks.

The counting logic is pure and in-memory; :meth:`Limiter.to_dict` /
:func:`Limiter.from_dict` persist day-level counters across runs (the engine
saves them to a small JSON state file - longitudinal ceilings need persistence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random

from ..config import CapsConfig
from ..touchstream import Gesture


class CapReached(Exception):
    """Raised when a cap forbids the next autonomous action. The engine treats
    this as a clean stop, never as a signal to substitute another action."""

    def __init__(self, which: str, message: str):
        super().__init__(message)
        self.which = which


@dataclass
class Limiter:
    """Tracks and enforces the behavioral caps for one persistent identity.

    ``scale`` comes from the risk profile (``cap_scale``): cautious=1.0 keeps
    the conservative starting caps; aggressive loosens them. Session and run
    counters reset per process; day counters persist.
    """

    caps: CapsConfig
    scale: float = 1.0

    # run/session counters
    actions_this_run: int = 0
    commits_this_run: int = 0
    games_this_session: int = 0
    session_seconds: float = 0.0
    games_since_break: int = 0

    # persisted day counters
    actions_today: int = 0
    commits_today: int = 0

    _traj: "TrajectoryMemory" = field(default_factory=lambda: TrajectoryMemory())

    # scaled cap accessors ----------------------------------------------------

    def _cap(self, base: int) -> int:
        return int(round(base * self.scale))

    @property
    def max_actions_per_run(self) -> int:
        return self._cap(self.caps.max_actions_per_run)

    @property
    def max_actions_per_day(self) -> int:
        return self._cap(self.caps.max_actions_per_day)

    @property
    def committing_action_cap(self) -> int:
        return self._cap(self.caps.committing_action_cap)

    @property
    def max_games_per_session(self) -> int:
        return self._cap(self.caps.max_games_per_session)

    # checks ------------------------------------------------------------------

    def check_elapsed_caps(self) -> None:
        """Raise :class:`CapReached` on the caps that time alone can breach.

        Separate from :meth:`check_before_action` because *every* cap used to be
        checked only there - i.e. only inside a tap. A loop that never taps could
        therefore never trip a cap, however long it ran, which is exactly what a
        soft-locked matchmaking queue produces. The hunt loop calls this once per
        iteration, tap or no tap.
        """
        if self.session_seconds >= self.caps.max_session_minutes * 60 * self.scale:
            raise CapReached("session_minutes", "Session time limit reached; stopping.")
        if self.games_this_session >= self.max_games_per_session:
            raise CapReached("games_session", "Per-session game cap reached; stopping.")

    def check_before_action(self, committing: bool) -> None:
        """Raise :class:`CapReached` if the next action would breach a cap.

        Called before every autonomous tap. ``committing`` marks a concede (or
        other conversion-like action) so its dedicated, tighter cap applies.
        """
        self.check_elapsed_caps()
        if self.actions_this_run >= self.max_actions_per_run:
            raise CapReached("actions_run", "Per-run action cap reached; stopping.")
        if self.actions_today >= self.max_actions_per_day:
            raise CapReached("actions_day", "Per-day action cap reached; stopping.")
        if committing and self.commits_this_run >= self.committing_action_cap:
            raise CapReached("committing", "Committing-action (concede) cap reached; stopping.")

    def register_action(self, committing: bool) -> None:
        self.actions_this_run += 1
        self.actions_today += 1
        if committing:
            self.commits_this_run += 1
            self.commits_today += 1

    def register_game(self) -> None:
        self.games_this_session += 1
        self.games_since_break += 1

    def register_time(self, dt: float) -> None:
        self.session_seconds += max(0.0, dt)

    def needs_break(self, rng: Random) -> float | None:
        """If a mandatory break is due, return its length in seconds, else None.

        Humans queue in bursts and stop; a break every N games (jittered length)
        breaks up an otherwise metronomic session. Resets the counter.
        """
        every = max(1, int(round(self.caps.mandatory_break_every_games * self.scale)))
        if self.games_since_break >= every:
            self.games_since_break = 0
            lo = self.caps.break_min_minutes * 60
            hi = self.caps.break_max_minutes * 60
            return lo + rng.random() * (hi - lo)
        return None

    def committing_ratio(self) -> float:
        """Concedes / total actions this run - the key aggregate to keep human."""
        return self.commits_this_run / self.actions_this_run if self.actions_this_run else 0.0

    # non-repetition ----------------------------------------------------------

    def is_near_duplicate(self, gesture: Gesture) -> bool:
        return self._traj.is_near_duplicate(
            trajectory_fingerprint(gesture), self.caps.non_repetition_threshold
        )

    def remember_trajectory(self, gesture: Gesture) -> None:
        self._traj.add(trajectory_fingerprint(gesture))

    # persistence -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "actions_today": self.actions_today,
            "commits_today": self.commits_today,
            "trajectories": self._traj.fingerprints,
        }

    @classmethod
    def from_dict(cls, caps: CapsConfig, scale: float, data: dict) -> "Limiter":
        lim = cls(caps=caps, scale=scale)
        lim.actions_today = int(data.get("actions_today", 0))
        lim.commits_today = int(data.get("commits_today", 0))
        lim._traj = TrajectoryMemory(list(data.get("trajectories", [])))
        return lim


def trajectory_fingerprint(g: Gesture) -> list[float]:
    """A small, comparable feature vector describing a gesture's shape+timing.

    Chosen so that two *materially different* human-ish gestures land far apart
    while a verbatim replay lands at ~0 distance. Endpoints are in screen
    fractions of the samples' own bounding box so absolute location doesn't
    dominate; duration, sample count, path length and peak speed capture the
    motor/timing signature.
    """
    if not g.samples:
        return [0.0] * 6
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
    ]


@dataclass
class TrajectoryMemory:
    """Remembers gesture fingerprints and answers near-duplicate queries."""

    fingerprints: list[list[float]] = field(default_factory=list)
    keep: int = 500

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
