"""Shared test fixtures and fakes.

The pure humanization layers (L3/L4/L5) need no fakes - they take an injectable
RNG. The hardware-facing layers are exercised through these fakes so the engine
and transport can be tested without a phone.
"""

from __future__ import annotations

from random import Random

import pytest

from hop.config import load_config
from hop.geometry import PanelGeometry
from hop.perception.image import Frame


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    """Keep default load_config() calls from reading the developer's real config."""
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def panel():
    return PanelGeometry(width_px=2400, height_px=1080, dpi=400.0)


@pytest.fixture
def rng():
    return Random(1234)


def gray_frame(w: int, h: int, value: int = 30) -> Frame:
    return Frame.from_gray_bytes(w, h, bytes([value]) * (w * h))


class FakeAdb:
    """Minimal Adb surface: connection is always up; shell returns canned text."""

    def __init__(self):
        self.shell_calls: list[str] = []
        self.emitted = []

    def ensure_connected(self):
        pass

    def is_connected(self):
        return True

    def connect(self, retries=4):
        pass

    def shell(self, cmd):
        self.shell_calls.append(cmd)
        return "yes"

    def screencap_png(self):
        return b"\x89PNG" + b"\x00" * 200  # decoded via FakeCapturer, not Pillow

    def stay_awake(self, on=True):
        pass


class FakeBackend:
    """Records emitted gestures; never touches hardware."""

    fidelity = "full"

    def __init__(self):
        self.opened = False
        self.closed = False
        self.gestures = []

    def open(self, panel):
        self.opened = True

    def emit(self, gesture):
        self.gestures.append(gesture)

    def close(self):
        self.closed = True


class ScriptedCapturer:
    """Returns frames from a script; repeats the last one when exhausted."""

    def __init__(self, frames: list[Frame]):
        self.frames = list(frames)
        self.i = 0

    def capture(self) -> Frame:
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return f


class FakeClassifier:
    """Duck-types ScreenClassifier: returns scripted states."""

    def __init__(self, states):
        from hop.perception.screens import Classification
        self._states = list(states)
        self._i = 0
        self._Classification = Classification

    @property
    def has_templates(self):
        return True

    def classify(self, frame):
        st = self._states[min(self._i, len(self._states) - 1)]
        self._i += 1
        return self._Classification(st, 0.95)
