"""Day-scoped persistence for the dashboard's observed class distribution.

The observed distribution is the ladder sample the dashboard charts: which opponent
classes you actually faced. It should *survive an app restart within the same day* --
close the app at lunch, reopen it, and the morning's games are still there -- but reset
on a new day, since yesterday's ladder is not today's.

So it is persisted as a tiny JSON file next to the config::

    {"date": "2026-07-10", "counts": {"Paladin": 3, "Mage": 2}}

On app start we load it only if its ``date`` is today; a stale (different-day) save is
deleted and the day starts fresh. The store is owned by the
:class:`~hop.runner.EngineController`, which outlives individual hunt runs, so the tally
accumulates across every start/stop within a day. The controller accrues each run's
counts into it (see ``EngineController._accrue_observed``); this module is just the
date-scoped file.
"""

from __future__ import annotations

import datetime
import json
import math
from pathlib import Path


def today_iso() -> str:
    """Local calendar date as ``YYYY-MM-DD`` -- the day a distribution is scoped to."""
    return datetime.date.today().isoformat()


def default_path(config_path: str | Path | None = None) -> Path:
    """Where the saved distribution lives: beside the user config, so a custom
    ``--config`` keeps its stats next to it (and the default lands in ``~/.config/hop``).
    """
    if config_path:
        return Path(config_path).parent / "observed_distribution.json"
    return Path.home() / ".config" / "hop" / "observed_distribution.json"


class ObservedDistribution:
    """A day-scoped, persistent tally of opponent classes (display-name -> count)."""

    def __init__(self, path: str | Path, date: str, counts: dict[str, int] | None = None):
        self.path = Path(path)
        self.date = date
        self.counts: dict[str, int] = dict(counts or {})

    @classmethod
    def load(cls, path: str | Path, today: str | None = None) -> "ObservedDistribution":
        """Load the saved tally iff it is for ``today``; otherwise delete it and start
        fresh. A missing or malformed file also starts fresh. Never raises.
        """
        path = Path(path)
        today = today or today_iso()
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return cls(path, today)
        if isinstance(data, dict) and data.get("date") == today:
            counts = data.get("counts")
            if isinstance(counts, dict):
                clean = {str(k): int(v) for k, v in counts.items()
                         if isinstance(v, (int, float)) and math.isfinite(v) and int(v) >= 0}
                return cls(path, today, clean)
            return cls(path, today)
        # a different-day (or shapeless) save: the user asked us to delete it and reset
        try:
            path.unlink()
        except OSError:
            pass
        return cls(path, today)

    def write(self, counts: dict[str, int] | None = None) -> None:
        """Persist ``counts`` (default: our own) as today's distribution. Best-effort;
        a write failure just means this update is not saved (the next one will be).
        """
        payload = self.counts if counts is None else counts
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"date": self.date, "counts": payload}))
        except OSError:
            pass
