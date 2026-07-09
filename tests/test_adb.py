"""adb binary resolution and fail-fast behaviour.

A .app launched from Finder gets a minimal PATH that excludes Homebrew and the Android
SDK, so bare "adb" isn't found and Start Search died with "No such file: 'adb'" only
after ~30 s of pointless connect retries. These pin the resolution and the fast failure.
"""

import time

import pytest

from hop.adb import Adb, AdbError, resolve_adb


def test_resolve_adb_prefers_path(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/custom/adb" if name == "adb" else None)
    assert resolve_adb() == "/custom/adb"


def test_resolve_adb_falls_back_to_known_location(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("os.path.isfile", lambda p: p == "/opt/homebrew/bin/adb")
    monkeypatch.setattr("os.access", lambda p, mode: True)
    assert resolve_adb() == "/opt/homebrew/bin/adb"


def test_resolve_adb_returns_bare_name_when_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    monkeypatch.setattr("os.path.isfile", lambda p: False)
    assert resolve_adb() == "adb"        # lets the call fail with a clear message


def test_connect_fails_fast_and_clearly_when_adb_missing():
    a = Adb("1.2.3.4:5555", adb_bin="/nonexistent/adb", timeout=1.0)
    t0 = time.time()
    with pytest.raises(AdbError) as ei:
        a.connect(retries=4)             # would be 2+4+8+16 s if it retried
    assert "adb not found" in str(ei.value)
    assert time.time() - t0 < 1.0        # no backoff burned on a hopeless retry


def test_shell_reports_missing_adb_clearly():
    a = Adb("1.2.3.4:5555", adb_bin="/nonexistent/adb")
    with pytest.raises(AdbError) as ei:
        a.shell("echo hi")
    assert "adb not found" in str(ei.value)
