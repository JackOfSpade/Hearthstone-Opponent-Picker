"""EngineController - run the engine in a background thread with shared status.

The dashboard (and any future controller) drives the engine through this small
interface: :meth:`start`, :meth:`stop`, :meth:`status`, and
:meth:`latest_png`. The engine's hunt loop is blocking, so it runs on a daemon
thread; status is aggregated from the live ``HumanState``, limiter counters and
:class:`~hop.engine.RunStats`.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .engine import Engine


class EngineController:
    def __init__(self, engine_factory: Callable[[dict], Engine], alerter=None):
        self.engine_factory = engine_factory
        self.alerter = alerter
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

    def latest_png(self) -> bytes | None:
        eng = self._engine
        if eng is None:
            return None
        try:
            return eng.adb.screencap_png()
        except Exception:
            return None

    def status(self) -> dict:
        eng = self._engine
        base = {
            "running": self.running,
            "phase": self._current_phase(),
            "last_error": self._last_error,
            "last_error_traceback": self._last_error_tb,
            "uptime_s": round(time.time() - self._started_at, 1) if self._started_at else 0,
        }
        if eng is None:
            return base
        s, st, lim = eng.stats, eng.state, eng.limiter
        crit = eng.cfg.criteria
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
            # the observed class distribution + coin split, for the dashboard charts
            "class_distribution": dict(s.class_distribution),
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
        return base
