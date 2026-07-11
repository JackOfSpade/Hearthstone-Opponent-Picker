"""EngineController: the stop must survive the engine-construction window.

Building an engine (ADB connect + UHID enumerate) takes a beat. A stop pressed in that
window used to no-op -- the controller's ``_engine`` was still ``None`` -- and the hunt
then began anyway, which read to the user as "Stop takes forever". These tests pin the
race closed in both interleavings.
"""

import threading

from hop.runner import EngineController


class _FakeEngine:
    def __init__(self):
        self.stops = 0

    def request_stop(self):
        self.stops += 1

    def run(self):
        return None


class _SpyAlerter:
    """Records the audible-stop alerts the controller fires, so a test can pin that a caught
    crash rings and a user stop does not."""

    def __init__(self):
        self.halts: list[str] = []

    def halt(self, message: str) -> None:
        self.halts.append(message)


def test_stop_during_construction_is_applied_after_the_engine_exists():
    started = threading.Event()
    release = threading.Event()
    engine = _FakeEngine()

    def factory(overrides):
        started.set()
        release.wait(2)        # hold the engine mid-construction
        return engine

    c = EngineController(factory)
    assert c.start() is True
    assert started.wait(2)     # factory is running; controller._engine is still None
    c.stop()                   # stop BEFORE the engine object exists
    release.set()              # let construction finish
    c._thread.join(2)

    assert engine.stops == 1   # the deferred stop was applied, not dropped


def test_status_shows_the_day_distribution_while_idle(tmp_path):
    """An idle controller (app just opened, no Search yet) must still report the day-scoped
    observed distribution -- that is what lets the dashboard chart show the day's games on
    open instead of only after Start. The per-run stats ('games', ...) stay absent while
    idle, which is exactly why the frontend must draw the chart OUTSIDE its games guard."""
    from hop.observed import ObservedDistribution

    store = ObservedDistribution(tmp_path / "obs.json", date="2026-07-10",
                                 counts={"Mage": 3, "Paladin": 2})
    c = EngineController(lambda o: _FakeEngine(), observed=store)
    st = c.status()                                   # never started
    assert st["running"] is False
    assert st["class_distribution"] == {"Mage": 3, "Paladin": 2}
    assert "games" not in st                          # per-run stats absent while idle


def test_stop_after_the_engine_exists_calls_request_stop():
    engine = _FakeEngine()
    c = EngineController(lambda overrides: engine)
    c.start()
    c._thread.join(2)          # run() returns immediately
    c.stop()
    assert engine.stops == 1


def test_status_shows_connecting_phase_during_construction():
    in_factory = threading.Event()
    release = threading.Event()

    class SlowEngine(_FakeEngine):
        phase = "in queue — waiting for a match"

    def factory(overrides):
        in_factory.set()
        release.wait(2)            # stay in construction so we can observe the phase
        return SlowEngine()

    c = EngineController(factory)
    c.start()
    assert in_factory.wait(2)
    # engine is None during construction, so status() takes its light path
    assert c.status()["phase"].startswith("connecting")
    release.set()
    c._thread.join(2)
    # after run() returns, the controller phase is "stopped" (checked directly to avoid
    # status()'s full engine read, which the minimal fake doesn't implement)
    assert c._current_phase() == "stopped"


def test_start_clears_a_stale_stop_request():
    """A stop from a previous run must not pre-stop the next start."""
    engines = [_FakeEngine(), _FakeEngine()]
    it = iter(engines)
    c = EngineController(lambda overrides: next(it))
    c.start()
    c._thread.join(2)
    c.stop()                   # stop the (finished) first engine
    assert engines[0].stops == 1

    c.start()                  # a fresh start must reset _stop_requested
    c._thread.join(2)
    assert engines[1].stops == 0


def test_restart_drops_the_previous_engine_during_reconnect():
    """A restart (Stop then Start) must show 'connecting…' with no stats, not the
    finished run's engine. _engine was only ever set (never cleared), so during the
    slow second construction a /status poll read the OLD engine as if it were live.
    """
    in_factory = threading.Event()
    release = threading.Event()
    engines = [_FakeEngine(), _FakeEngine()]
    it = iter(engines)

    def factory(overrides):
        e = next(it)
        if e is engines[1]:
            in_factory.set()
            release.wait(2)        # hold the SECOND construction so we can observe status
        return e

    c = EngineController(factory)
    c.start()
    c._thread.join(2)              # first run finishes; _engine is engines[0]
    assert c._engine is engines[0]

    c.start()                      # restart
    assert in_factory.wait(2)
    assert c._engine is None       # the finished engine was dropped, not reported as live
    assert c.status()["phase"].startswith("connecting")   # light path, no stale stats
    release.set()
    c._thread.join(2)


def test_status_captures_a_traceback_when_construction_crashes():
    """A crash must leave a full traceback in status so a bug report pinpoints the line."""
    def boom(overrides):
        raise TypeError("record() got multiple values for argument 'kind'")

    c = EngineController(boom)
    c.start()
    c._thread.join(2)
    s = c.status()
    assert s["last_error"].startswith("TypeError:")
    assert "Traceback (most recent call last)" in s["last_error_traceback"]
    assert "boom" in s["last_error_traceback"]        # the failing frame is named


def test_construction_crash_alerts_the_user():
    """A crash that stops the hunt without the user asking must ring: the engine never got to
    announce it (it failed before or outside run()'s own halt handling), so the controller
    does -- otherwise the hunt dies silently and the user, not watching, waits on nothing."""
    alerter = _SpyAlerter()

    def boom(overrides):
        raise RuntimeError("adb connect failed")

    c = EngineController(boom, alerter=alerter)
    c.start()
    c._thread.join(2)
    assert alerter.halts == ["RuntimeError: adb connect failed"]


def test_a_crash_after_a_user_stop_stays_silent():
    """If the user pressed Stop mid-construction, a construction failure is moot -- the user
    already chose to stop, so no alert. A manual stop is never announced with a sound."""
    alerter = _SpyAlerter()
    started = threading.Event()
    release = threading.Event()

    def factory(overrides):
        started.set()
        release.wait(2)                 # hold construction so the stop lands first
        raise RuntimeError("adb connect failed")

    c = EngineController(factory, alerter=alerter)
    c.start()
    assert started.wait(2)
    c.stop()                            # user asks to stop while we are still constructing
    release.set()
    c._thread.join(2)
    assert alerter.halts == []          # the moot crash rang nothing


def test_a_clean_run_never_alerts():
    """The controller alerts only on a crash it had to catch. A run that returns normally --
    the engine's own handled halts included -- must not ring again over the top of it."""
    alerter = _SpyAlerter()
    c = EngineController(lambda overrides: _FakeEngine(), alerter=alerter)
    c.start()
    c._thread.join(2)
    assert alerter.halts == []


def test_start_clears_a_stale_traceback():
    """A fresh start must not carry the previous crash's traceback."""
    calls = [lambda: (_ for _ in ()).throw(RuntimeError("first boom")), None]

    def factory(overrides):
        f = calls.pop(0)
        if f is not None:
            f()
        return _FakeEngine()

    c = EngineController(factory)
    c.start(); c._thread.join(2)
    assert c._last_error_tb                              # first run crashed
    # second start succeeds; read the field directly (status()'s full engine read is not
    # implemented by the minimal fake -- same reason as the phase test above)
    c.start(); c._thread.join(2)
    assert c._last_error_tb == ""                        # second run is clean
