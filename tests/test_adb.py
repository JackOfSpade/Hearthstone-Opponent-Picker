"""adb binary resolution and fail-fast behaviour.

A .app launched from Finder gets a minimal PATH that excludes Homebrew and the Android
SDK, so bare "adb" isn't found and Start Search died with "No such file: 'adb'" only
after ~30 s of pointless connect retries. These pin the resolution and the fast failure.
"""

import time

import pytest

from hop.adb import Adb, AdbError, parse_usb_serials, resolve_adb


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


def test_connect_names_a_refused_connection_as_a_dead_listener(monkeypatch):
    """A REFUSED connect (the phone is on the network but adbd isn't listening on this port)
    is a different failure from a timeout, and no retry backoff fixes it -- retrying a closed
    port can't make it listen. The raised error must say so instead of just echoing adb's raw
    stdout, so the live status / bug report point straight at `adb tcpip 5555`, not a network
    guess (see hop/bugreport.py's _diagnose_last_error for the fuller report-side diagnosis).

    Regression: a user hit this live, re-enabled Android's "Wireless debugging" toggle (a
    SEPARATE feature that opens its own random port), and was still refused -- because the
    first version of this message named that toggle as an alternative fix, which is wrong.
    The message must point ONLY at `adb tcpip 5555` and must not repeat that equivalence."""
    import subprocess

    a = Adb("192.168.99.139:5555", timeout=1.0)

    class _Proc:
        stdout = b"failed to connect to '192.168.99.139:5555': Connection refused\n"
        stderr = b""

    monkeypatch.setattr(subprocess, "run", lambda *a_, **k: _Proc())
    with pytest.raises(AdbError) as ei:
        a.connect(retries=0)          # no retries needed: assert the message, not the backoff
    msg = str(ei.value)
    assert "Connection refused" in msg
    assert "adb tcpip 5555" in msg
    assert "does not restore 5555" in msg.replace("NOT", "not")
    assert "(or re-enable Wireless" not in msg   # the old, misleading equivalence


def test_parse_usb_serials_filters_wireless_and_unauthorized_entries():
    """Shared by the USB rescue below and hop.bugreport's device probe -- a plain text parse
    with no adb binary involved, so it's worth pinning on its own."""
    out = (
        "List of devices attached\n"
        "R58N70ABCDE            device usb:1-1 product:panther\n"
        "192.168.99.139:5555    offline transport_id:2\n"
        "ZY223JQZ8G              unauthorized usb:1-2\n"
    )
    assert parse_usb_serials(out) == ["R58N70ABCDE"]


def test_connect_auto_rescues_via_usb_when_refused(monkeypatch):
    """The behaviour a user asked for directly after hitting this live: if the SAME phone is
    already plugged in over USB when the wireless connect is refused, hop should just fix it
    -- `adb tcpip <port>` on the USB serial, then retry -- instead of only printing
    instructions for a fix the phone was sitting right there to receive."""
    import subprocess

    monkeypatch.setattr(time, "sleep", lambda *_a: None)   # skip the real settle/backoff waits
    a = Adb("192.168.99.139:5555", timeout=1.0)
    calls = []

    class _Proc:
        def __init__(self, stdout=b"", returncode=0):
            self.stdout = stdout
            self.stderr = b""
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "connect":
            rescued = any("tcpip" in c for c in calls[:-1])
            return _Proc(stdout=(b"connected to 192.168.99.139:5555\n" if rescued else
                                 b"failed to connect to '192.168.99.139:5555': Connection refused\n"))
        if args[1] == "devices":
            return _Proc(stdout=b"List of devices attached\nR58N70ABCDE   device usb:1-1\n")
        if "tcpip" in args:
            return _Proc(returncode=0)
        raise AssertionError(f"unexpected adb call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    a.connect()   # must NOT raise
    tcpip_calls = [c for c in calls if "tcpip" in c]
    assert len(tcpip_calls) == 1                                  # rescue ran exactly once
    assert "R58N70ABCDE" in tcpip_calls[0] and "5555" in tcpip_calls[0]


def test_connect_falls_back_to_manual_fix_with_no_usb_device_present(monkeypatch):
    """No USB device present means there's nothing to safely rescue from -- fall through to
    the existing manual-fix error instead of guessing."""
    import subprocess

    monkeypatch.setattr(time, "sleep", lambda *_a: None)
    a = Adb("192.168.99.139:5555", timeout=1.0)

    class _Proc:
        def __init__(self, stdout=b""):
            self.stdout = stdout
            self.stderr = b""

    def fake_run(args, **kwargs):
        if args[1] == "connect":
            return _Proc(stdout=b"failed to connect to '192.168.99.139:5555': Connection refused\n")
        if args[1] == "devices":
            return _Proc(stdout=b"List of devices attached\n")   # nothing plugged in
        raise AssertionError(f"unexpected adb call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AdbError) as ei:
        a.connect(retries=0)
    msg = str(ei.value)
    assert "Connection refused" in msg
    assert "adb tcpip 5555" in msg
    assert "hop will re-pin it automatically" in msg


def test_connect_names_a_rescue_that_ran_but_still_failed(monkeypatch):
    """A rarer case: the USB rescue itself succeeded (`adb tcpip` returned 0) but the phone
    still refuses the wireless connect afterward -- a different problem (bad address, Wi-Fi
    actually down on the phone), so the message must say the rescue WAS applied, not repeat
    the generic dead-listener advice as if nothing had been tried."""
    import subprocess

    monkeypatch.setattr(time, "sleep", lambda *_a: None)
    a = Adb("192.168.99.139:5555", timeout=1.0)

    class _Proc:
        def __init__(self, stdout=b"", returncode=0):
            self.stdout = stdout
            self.stderr = b""
            self.returncode = returncode

    def fake_run(args, **kwargs):
        if args[1] == "connect":
            return _Proc(stdout=b"failed to connect to '192.168.99.139:5555': Connection refused\n")
        if args[1] == "devices":
            return _Proc(stdout=b"List of devices attached\nR58N70ABCDE   device usb:1-1\n")
        if "tcpip" in args:
            return _Proc(returncode=0)
        raise AssertionError(f"unexpected adb call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(AdbError) as ei:
        a.connect()   # default retries=4 -- gives the post-rescue retry a chance to fail too
    msg = str(ei.value)
    assert "USB rescue re-pinned the wireless listener" in msg
    assert "still failed afterward" in msg


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
