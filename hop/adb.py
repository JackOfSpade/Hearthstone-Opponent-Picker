"""Wireless-ADB connection management, screencap, and device probing.

This is the one place that shells out to the ``adb`` binary. Everything the
higher layers need from the phone goes through :class:`Adb`:

* connect / keepalive / auto-reconnect (Wi-Fi ADB drops on screen lock and long
  idle - the standard's Layer 0/portability notes),
* ``exec-out screencap`` frames for perception (Layer 2),
* a persistent shell + FIFO writer for the UHID transport (Layer 1),
* panel geometry and input-device metadata for calibration (``hop doctor``).

The class is deliberately thin and side-effecting; the pure layers never import
it. Tests use a ``FakeAdb`` with the same surface (see tests/).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from .geometry import PanelGeometry


class AdbError(RuntimeError):
    pass


#: Where `adb` commonly lives when it isn't on PATH. A .app launched from Finder gets a
#: minimal PATH that excludes Homebrew and the Android SDK, so bare "adb" isn't found;
#: we resolve an absolute path as a fallback (the launcher also augments PATH).
_ADB_FALLBACK_PATHS = (
    "/opt/homebrew/bin/adb",
    "/usr/local/bin/adb",
    os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
    os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
)


def resolve_adb() -> str:
    """Absolute path to `adb`, or the bare name if nothing is found.

    Prefers PATH (respects a user override), then the common install locations. Returning
    "adb" when truly absent lets the call fail with a clear "adb not found" message.
    """
    found = shutil.which("adb")
    if found:
        return found
    for cand in _ADB_FALLBACK_PATHS:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return "adb"


class Adb:
    def __init__(self, address: str, adb_bin: str | None = None, timeout: float = 20.0):
        self.address = address
        self.adb_bin = adb_bin or resolve_adb()
        self.timeout = timeout

    # ── raw invocation ──────────────────────────────────────────────────────

    def _base(self) -> list[str]:
        return [self.adb_bin, "-s", self.address] if self.address else [self.adb_bin]

    def _missing_adb_msg(self) -> str:
        return (f"adb not found (tried {self.adb_bin!r}). Install Android platform-tools "
                "(brew install --cask android-platform-tools) or add adb to your PATH.")

    def _run(self, args: list[str], binary: bool = False) -> bytes:
        try:
            proc = subprocess.run(
                self._base() + args,
                capture_output=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise AdbError(self._missing_adb_msg())
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace")
            raise AdbError(f"adb {' '.join(args)} failed: {err.strip()}")
        return proc.stdout if binary else proc.stdout

    def shell(self, cmd: str) -> str:
        return self._run(["shell", cmd]).decode(errors="replace")

    def exec_out(self, cmd: str) -> bytes:
        return self._run(["exec-out", cmd], binary=True)

    def popen_shell(self, cmd: str) -> subprocess.Popen:
        """Long-lived shell process with binary stdin/stdout (for the UHID FIFO
        writer and the ``hid`` reader). Caller manages its lifecycle."""
        return subprocess.Popen(
            self._base() + ["shell", cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    # ── connection lifecycle ────────────────────────────────────────────────

    def connect(self, retries: int = 4) -> None:
        """`adb connect <address>` with exponential backoff on failure."""
        delay = 2.0
        last = ""
        for attempt in range(retries + 1):
            try:
                out = subprocess.run(
                    [self.adb_bin, "connect", self.address],
                    capture_output=True, timeout=self.timeout,
                ).stdout.decode(errors="replace")
                if "connected" in out or "already" in out:
                    return
                last = out
            except FileNotFoundError:
                # a missing binary can never succeed - fail immediately instead of
                # burning the whole 2+4+8+16 s backoff on a hopeless retry loop.
                raise AdbError(self._missing_adb_msg())
            except Exception as e:  # pragma: no cover - network
                last = str(e)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        raise AdbError(f"could not connect to {self.address}: {last.strip()}")

    def is_connected(self) -> bool:
        try:
            state = self._run(["get-state"]).decode(errors="replace").strip()
            return state == "device"
        except Exception:
            return False

    def ensure_connected(self) -> None:
        """Reconnect if the Wi-Fi ADB link dropped (screen lock / idle)."""
        if not self.is_connected():
            self.connect()

    # ── device probing ──────────────────────────────────────────────────────

    def measure_panel(self) -> PanelGeometry:
        """Read live width/height and effective dpi from the device.

        Motor synthesis needs physical density (mm<->px). We use ``wm size`` and
        ``wm density``; the density is the reported dpi (LIVE-VERIFY against the
        true physical dpi from ``dumpsys display`` if you want mm-exact tremor).
        """
        size = self.shell("wm size")           # "Physical size: 2400x1080"
        density = self.shell("wm density")     # "Physical density: 400"
        w, h = _parse_size(size)
        dpi = _parse_density(density)
        return PanelGeometry(width_px=w, height_px=h, dpi=dpi)

    def get_rotation(self) -> int:
        """Current display rotation (0,1,2,3 == 0/90/180/270) of the internal
        display, as the input system sees it - this is the rotation the
        framework applies to our virtual touchscreen's raw coordinates.

        Primary source is the INTERNAL viewport in ``dumpsys input`` (the input
        system's own source of truth), which on modern Android prints
        ``Viewport INTERNAL: ... orientation=3``. Falls back to older viewport
        formatting and then to ``dumpsys window``'s ``mDisplayRotation``.
        Defaults to 0 if nothing parses.
        """
        import re
        try:
            out = self.shell("dumpsys input")
            for pat in (r"Viewport INTERNAL:[^\n]*?orientation=(\d)",
                        r"DisplayViewport\{type=INTERNAL.*?orientation=(\d)"):
                m = re.search(pat, out, re.DOTALL)
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        try:
            win = self.shell("dumpsys window")
            m = re.search(r"mDisplayRotation=ROTATION_(\d+)", win) or \
                re.search(r"\bmRotation=ROTATION_(\d+)", win)
            if m:
                return {0: 0, 90: 1, 180: 2, 270: 3}.get(int(m.group(1)), 0)
        except Exception:
            pass
        return 0

    def screencap_png(self) -> bytes:
        """Raw PNG bytes of the current screen (works while backgrounded)."""
        return self.exec_out("screencap -p")

    def input_devices_dump(self) -> str:
        """`dumpsys input` - used by `hop doctor` to read the real panel's
        source/vendor/product for panel-matched UHID identity."""
        return self.shell("dumpsys input")

    def has_hid_tool(self) -> bool:
        return "yes" in self.shell("[ -e /system/bin/hid ] && echo yes || echo no")

    def stay_awake(self, on: bool = True) -> None:
        """Keep the screen on while plugged in (prevents Doze mid-session)."""
        self.shell(f"svc power stayon {'true' if on else 'false'}")


def _parse_size(text: str) -> tuple[int, int]:
    for prefix in ("Override size", "Physical size"):
        for line in text.splitlines():
            if prefix in line:
                for token in line.split():
                    if "x" in token and token.replace("x", "").isdigit():
                        w, h = token.split("x")
                        return int(w), int(h)
    raise AdbError(f"could not parse `wm size`: {text!r}")


def _parse_density(text: str) -> float:
    for token in text.split():
        if token.strip().isdigit():
            return float(token.strip())
    return 400.0  # LIVE-VERIFY fallback
