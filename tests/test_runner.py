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
