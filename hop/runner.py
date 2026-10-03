"""EngineController - run the engine in a background thread with shared status.

The dashboard (and any future controller) drives the engine through this small
interface: :meth:`start`, :meth:`stop`, and :meth:`status`. The engine's hunt loop
is blocking, so it runs on a daemon thread; status is aggregated from the live
``HumanState``, limiter counters and :class:`~hop.engine.RunStats`.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .engine import Engine


class EngineController:
    def __init__(self, engine_factory: Callable[[dict], Engine], alerter=None,
                 observed=None):
        self.engine_factory = engine_factory
        self.alerter = alerter
        #: day-scoped observed-class tally that outlives individual runs, so the dashboard
        #: chart survives an app restart within the same day (see hop.observed). None
        #: disables persistence (tests, and the headless CLI, which owns no store).
        self._observed = observed
        #: the run currently accruing into `_observed`, and how much of its counts we have
        #: already folded in -- so re-reading the same engine adds only the new delta, and
        #: a fresh run (a different engine object) starts its accrual from zero.
        self._accrued_engine: Engine | None = None
        self._accrued: dict[str, int] = {}
        self._observed_lock = threading.Lock()
        self._engine: Engine | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_error: str = ""
        #: full traceback of the last crash, captured so a pasted bug report pinpoints
        #: the failing line instead of just its message (a bare "TypeError: ..." once
        #: cost a grep-hunt to locate). Surfaced in status() -> the report, not the UI.
        self._last_error_tb: str = ""
        self._started_at: float = 0.0
        #: a stop asked for before the engine exists (during construction) must not be
        #: lost -- see start()/stop(). Building an engine (ADB connect + UHID enumerate)
        #: takes a beat, and a stop in that window used to no-op and let the hunt begin.
        self._stop_requested: bool = False
        #: coarse lifecycle phase for the UI, so the seconds spent connecting/enumerating
        #: read as progress ("Connecting to phone…") instead of a blank, stuck-looking
        #: window. Once the loop is running the engine's own finer phase takes over.
        self._phase: str = "idle"

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _current_phase(self) -> str:
        eng = self._engine
        if eng is not None and self.running:
            return getattr(eng, "phase", None) or self._phase
        return self._phase

    def start(self, overrides: dict | None = None) -> bool:
        with self._lock:
            if self.running:
                return False
            # Fold the just-finished run's opponent counts into the day tally before we
            # drop its engine, so its games are not lost when the next run replaces it.
            self._accrue_observed()
            # Drop the previous run's finished engine so a restart takes the light
            # status path ("connecting to phone…") during the slow reconnect/enumerate
            # window instead of reporting the OLD run's stats and a stale phase. Safe:
            # `self.running` is false here (no live thread), so nothing reads it mid-swap.
            self._engine = None
            self._last_error = ""
            self._last_error_tb = ""
            self._stop_requested = False
            self._phase = "connecting to phone…"
            self._started_at = time.time()

            def _run():
                try:
                    # construction is the slow, silent part: adb connect + panel probe +
                    # UHID enumerate. Keep _phase = "connecting…" across it so the UI has
                    # something to show instead of a blank window.
                    eng = self.engine_factory(overrides or {})
                    self._engine = eng
                    # a stop that arrived while we were constructing: honour it now, so
                    # the hunt exits at the first loop check instead of booking a game.
                    if self._stop_requested:
                        eng.request_stop()
                    self._phase = "running"
                    eng.run()
                    self._phase = "stopped"
                except Exception as e:  # surface, don't crash the dashboard
                    import traceback
                    self._last_error = f"{type(e).__name__}: {e}"
                    self._last_error_tb = traceback.format_exc()
                    self._phase = "error"
                    # A crash here is a stop the engine never got to announce: it escaped
                    # run()'s own halt handling, or construction (adb connect / UHID
                    # enumerate) failed before the loop began. Either way the hunt is dead
                    # and the user did not ask for it, so alert audibly -- same reasoning as
                    # a halt (they are not watching the screen). The one exception is a Stop
                    # pressed mid-construction: then the failure is moot and a sound is noise.
                    if self.alerter is not None and not self._stop_requested:
                        self.alerter.halt(self._last_error)

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        # set the flag BEFORE checking _engine so a stop can't slip through the window
        # where the engine is being constructed (see _run): whichever order the two
        # threads interleave, the stop lands.
        self._stop_requested = True
        eng = self._engine
        if eng is not None:
            eng.request_stop()

    # ── observed class distribution (day-scoped, persistent) ──────────────────

    def _accrue_observed(self) -> None:
        """Fold the live run's opponent counts into the day tally and persist.

        The engine only ever *grows* its per-run ``class_distribution`` (one increment per
        game), so we add the delta since we last looked -- keyed on the engine object, so a
        fresh run starts its accrual from zero rather than double-counting the day total.
        Called on each status tick (the app polls ~1 Hz; games are minutes apart, so writes
        are rare) and on quit, so a day's progress survives even an unclean exit.
        """
        if self._observed is None:
            return
        eng = self._engine
        stats = getattr(eng, "stats", None)
        if stats is None:
            return
        with self._observed_lock:
            if eng is not self._accrued_engine:
                self._accrued_engine = eng
                self._accrued = {}
            changed = False
            for name, n in dict(stats.class_distribution).items():
                prev = self._accrued.get(name, 0)
                if n > prev:
                    self._observed.counts[str(name)] = (
                        self._observed.counts.get(str(name), 0) + (n - prev))
                    self._accrued[name] = n
                    changed = True
            if changed:
                self._observed.write()

    def observed_distribution(self) -> dict:
        """The class distribution the dashboard charts: the persisted day tally when a
        store is configured (survives a same-day restart), else the live run's own counts.
        """
        if self._observed is not None:
            return dict(self._observed.counts)
        stats = getattr(self._engine, "stats", None)
        return dict(stats.class_distribution) if stats is not None else {}

    def flush_observed(self) -> None:
        """Persist the current run's counts on the way out (clean quit / Ctrl-C), so the
        day's tally is not lost when the app closes mid-run."""
        self._accrue_observed()

    def status(self) -> dict:
        self._accrue_observed()   # keep the day tally current + persisted on each poll
        eng = self._engine
        base = {
            "running": self.running,
            "phase": self._current_phase(),
            "last_error": self._last_error,
            "last_error_traceback": self._last_error_tb,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
            # The dashboard's original distribution is deliberately day-scoped: it shows even
            # before the first run / while idle, so the chart carries yesterday-cleared,
            # same-day-restored counts on app open.  Keep that API intact, but name its scope so
            # report consumers never confuse a prior run's target with this run's opponent.
            "class_distribution": self.observed_distribution(),
            "class_distribution_scope": "day" if self._observed is not None else "run",
        }
        if eng is None:
            return base
        s, st, lim = eng.stats, eng.state, eng.limiter
        crit = eng.cfg.criteria
        # A persistent UHID writer can die after an ADB reconnect.  Keep its bounded
        # in-process status alongside the controller's caught error so a bug report
        # still has the child's exit/stderr snapshot even when the journal is disabled
        # or the crash escaped Engine.run.  Optional for old/fake engines.
        transport_status = {}
        transport_probe = getattr(eng, "transport_status", None)
        if callable(transport_probe):
            try:
                candidate = transport_probe()
                if isinstance(candidate, dict):
                    transport_status = candidate
            except Exception:
                pass
        base.update({
            # the criteria THIS run is actually using (config + dashboard/menu overrides,
            # merged at construction). Surfaced so a bug report shows what the run hunted
            # for -- the config file alone can be stale when the dashboard overrides it.
            "criteria": {
                "target_classes": [c.name for c in crit.target_classes],
                "require_second": crit.require_second,
                "mode": crit.mode,
            },
            "games": s.games,
            "concedes": s.concedes,
            "target_found": s.target_found,
            "concedes_until_target": s.concedes_until_target,
            "last_opponent": s.last_opponent,
            "stop_reason": s.stop_reason,
            # Keep the observed total for the dashboard, but expose the engine's own current
            # run separately for diagnostics.  The two legitimately differ after a restart:
            # a day's Priest target plus this run's rejected Hunter must never look like one
            # run that stopped on Hunter.  Additive field: existing dashboard clients continue
            # reading `class_distribution` unchanged.
            "class_distribution": self.observed_distribution(),
            "run_class_distribution": dict(s.class_distribution),
            "going_first": s.going_first,
            "going_second": s.going_second,
            # Non-fatal, but a rising count means our touch profile is drifting from
            # what Hearthstone accepts. Surface it rather than let it stay silent.
            "ignored_card_taps": s.ignored_card_taps,
            "human_state": {
                "attention": round(st.attention, 2),
                "confidence": round(st.confidence, 2),
                "fatigue": round(st.fatigue, 2),
                "familiarity": round(st.familiarity, 2),
                "actions": st.actions_taken,
            },
            # activity counters (informational; there are no volume caps to hit)
            "budget": {
                "actions_run": lim.actions_this_run,
                "concedes_run": lim.commits_this_run,
                "games_session": lim.games_this_session,
                "committing_ratio": round(lim.committing_ratio(), 2),
                "session_minutes": round(lim.session_seconds / 60, 1),
            },
        })
        if transport_status:
            base["transport_status"] = transport_status
        return base
