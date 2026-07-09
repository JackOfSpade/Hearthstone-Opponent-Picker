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
