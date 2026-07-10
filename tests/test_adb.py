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


def test_screencap_timeout_becomes_a_typed_adberror(monkeypatch):
    """Regression: a wireless-ADB screencap that HANGS to the timeout used to raise a raw
    subprocess.TimeoutExpired, which escaped every handler in Engine.run and killed the whole
    hunt. It must convert to AdbError at the boundary (like the missing-binary case), so the
    engine's capture-retry/clean-halt machinery can see it as an adb failure."""
    import subprocess

    a = Adb("1.2.3.4:5555", timeout=0.5)

    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="adb exec-out screencap -p", timeout=0.5)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(AdbError) as ei:
        a.screencap_png()
    msg = str(ei.value).lower()
    assert "timed out" in msg and "unreachable" in msg    # names the wireless-drop cause


def test_spawn_time_oserror_becomes_a_typed_adberror(monkeypatch):
    """The boundary promise is that NO raw subprocess/OS exception escapes _run. Besides
    TimeoutExpired, subprocess.run can raise an OSError at spawn (ENOMEM/EAGAIN under memory
    pressure -- a Mac thrashing on wake -- or PermissionError on a non-executable adb). That
    must also convert to AdbError so the engine's retry/clean-halt machinery sees it."""
    import subprocess

    a = Adb("1.2.3.4:5555", timeout=0.5)

    def boom(*args, **kwargs):
        raise OSError(35, "Resource temporarily unavailable")   # EAGAIN from posix_spawn

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(AdbError) as ei:
        a.screencap_png()
    assert "failed to run" in str(ei.value).lower()


def test_reconnect_disconnects_before_connecting(monkeypatch):
    """A dropped Wi-Fi link often lingers as a zombie 'device' whose exec-out still hangs, so a
    plain re-connect reuses the dead socket. reconnect() must disconnect FIRST, then connect."""
    import subprocess

    a = Adb("1.2.3.4:5555", timeout=0.5)
    verbs = []

    class _Proc:
        stdout = b"connected to 1.2.3.4:5555\n"
        stderr = b""
        returncode = 0

    def record(args, **kwargs):
        verbs.append(args[1] if len(args) > 1 else args[0])   # connect / disconnect
        return _Proc()

    monkeypatch.setattr(subprocess, "run", record)
    a.reconnect()
    assert "disconnect" in verbs and "connect" in verbs
    assert verbs.index("disconnect") < verbs.index("connect")   # zombie dropped before re-handshake
