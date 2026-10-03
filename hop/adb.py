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

import ipaddress
import os
import re
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


def parse_usb_serials(devices_output: str) -> list[str]:
    """Serials of USB-attached, authorized ("device"-state) phones from `adb devices` output.

    `adb devices` lists BOTH USB and wireless endpoints; a wireless one's id is the ip:port
    we're already dialing (and, when this matters, failing to reach), so filtering out any
    id containing ':' leaves only real USB serials. "device" (not "offline"/"unauthorized")
    is the only state that means actually usable right now. Shared by :meth:`Adb.connect`'s
    USB rescue and hop.bugreport's device probe, so the parsing rule can't drift between the
    two -- a plain pure-text parse, so it needs no adb binary and can't fail.
    """
    serials = []
    for line in devices_output.splitlines():
        tok = line.split()
        if len(tok) >= 2 and tok[1] == "device" and ":" not in tok[0]:
            serials.append(tok[0])
    return serials


def parse_wlan_ipv4(ip_output: str) -> str | None:
    """The one usable IPv4 address reported for Android's ``wlan0``, if unambiguous.

    A USB-attached phone is the only trustworthy source when a configured wireless endpoint
    says ``Network is unreachable``: DHCP may have assigned the phone a new address.  Android's
    ``ip -f inet addr show dev wlan0`` output is deliberately parsed as text so this stays a
    tiny, testable probe rather than importing platform-specific networking machinery.

    Do not guess when multiple addresses are present.  Likewise reject addresses which cannot
    be a normal LAN endpoint (loopback, link-local, unspecified, multicast, reserved).  In
    either case the caller can still use the known-good USB transport for this session.
    """
    found: set[str] = set()
    for raw in re.findall(r"\binet\s+(\d{1,3}(?:\.\d{1,3}){3})/\d{1,2}\b", ip_output or ""):
        try:
            addr = ipaddress.IPv4Address(raw)
        except ipaddress.AddressValueError:
            continue
        if (addr.is_unspecified or addr.is_loopback or addr.is_link_local
                or addr.is_multicast or addr.is_reserved):
            continue
        found.add(str(addr))
    return next(iter(found)) if len(found) == 1 else None


def _proc_output(proc) -> str:
    """Decoded stdout and stderr from a subprocess result, without assuming both exist.

    ``adb connect`` varies by platform/version about which stream carries a failure.  Looking
    at both is important here: a route failure written to stderr must take the same recovery
    path as the stdout form captured in the live report.
    """
    parts = []
    for stream in (getattr(proc, "stdout", b""), getattr(proc, "stderr", b"")):
        if not stream:
            continue
        if isinstance(stream, (bytes, bytearray)):
            stream = bytes(stream).decode(errors="replace")
        parts.append(str(stream))
    return "\n".join(parts)


def _is_network_unreachable(message: str) -> bool:
    """Whether ADB reported a local route failure, rather than a dead remote listener."""
    m = (message or "").lower()
    return any(needle in m for needle in (
        "network is unreachable", "network unreachable", "no route to host", "host is unreachable",
    ))


class Adb:
    def __init__(self, address: str, adb_bin: str | None = None, timeout: float = 20.0):
        # ``address`` remains the public/backward-compatible configured endpoint.  A live
        # session may discover a newer Wi-Fi address, or temporarily use USB; keep those facts
        # separate so a transient recovery never rewrites the user's config or makes a later
        # reconnect fall back to the stale configured IP.
        self.address = address
        self.configured_address = address
        self._wireless_address = address
        self._active_address = address
        self._usb_serial: str | None = None
        # Monotonic epoch for *long-lived* adb shell users.  A regular adb command
        # is stateless, but `adb disconnect` (and a newly selected transport) can
        # sever a resident `adb shell hid -` stream while later one-shot commands
        # work again.  Consumers snapshot this and rebuild their stream before the
        # next logical operation rather than discovering the dead pipe mid-write.
        self._connection_generation = 0
        self.adb_bin = adb_bin or resolve_adb()
        self.timeout = timeout

    @property
    def active_address(self) -> str:
        """ADB serial/endpoint used for shell, screencap, and input right now."""
        return self._active_address

    @property
    def wireless_address(self) -> str:
        """Wireless endpoint currently known to work in this session (config initially)."""
        return self._wireless_address

    @property
    def using_usb(self) -> bool:
        """Whether the active session transport is a directly attached USB serial."""
        return self._usb_serial is not None

    @property
    def connection_generation(self) -> int:
        """Epoch of the active ADB transport for persistent-shell consumers.

        This advances not only after a verified new connection, but also before a
        connection attempt or wireless ``adb disconnect``.  The latter two are
        deliberately conservative: a reconnect can fail while a later one-shot
        command recovers, but the old persistent shell is no longer safe to use.
        """
        return self._connection_generation

    # ── raw invocation ──────────────────────────────────────────────────────

    def _base(self) -> list[str]:
        return [self.adb_bin, "-s", self._active_address] if self._active_address else [self.adb_bin]

    def _missing_adb_msg(self) -> str:
        return (f"adb not found (tried {self.adb_bin!r}). Install Android platform-tools "
                "(brew install --cask android-platform-tools) or add adb to your PATH.")

    def _run(self, args: list[str]) -> bytes:
        try:
            proc = subprocess.run(
                self._base() + args,
                capture_output=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            raise AdbError(self._missing_adb_msg())
        except subprocess.TimeoutExpired:
            # A wireless-ADB link that dropped (screen lock, Doze, a Wi-Fi power-save blip, or
            # the Mac sleeping) does NOT error fast -- `exec-out screencap` HANGS on the dead
            # socket until this timeout. Convert it to our own typed error at the boundary, the
            # same way the missing-binary case is: this is the one place that shells out, and a
            # raw subprocess exception must never escape it. A single stalled frame used to
            # propagate a bare TimeoutExpired up through every handler in Engine.run and kill
            # the whole hunt (see hop.engine._capture_resilient, which now retries these).
            raise AdbError(
                f"adb {' '.join(args)} timed out after {self.timeout:.0f}s -- device "
                "unreachable (Wi-Fi ADB drops on screen lock / Doze / Mac sleep)")
        except (OSError, subprocess.SubprocessError) as e:
            # Any OTHER spawn-time failure: an OSError from posix_spawn/fork under memory pressure
            # (ENOMEM / EAGAIN "Resource temporarily unavailable" -- e.g. a Mac thrashing as it
            # wakes from the very sleep that drops the link), a present-but-non-executable adb_bin
            # (PermissionError), etc. FileNotFoundError and TimeoutExpired are handled above for
            # their specific messages (both are subclasses caught by the earlier clauses); this
            # backstop makes good on this class's promise that NO raw subprocess/OS exception
            # escapes the one place that shells out -- so every such failure reaches the engine as
            # an AdbError it can retry/reconnect/halt-cleanly on, not a bare crash.
            raise AdbError(f"adb {' '.join(args)} failed to run: {e}")
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace")
            raise AdbError(f"adb {' '.join(args)} failed: {err.strip()}")
        return proc.stdout

    def shell(self, cmd: str) -> str:
        return self._run(["shell", cmd]).decode(errors="replace")

    def exec_out(self, cmd: str) -> bytes:
        return self._run(["exec-out", cmd])

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
        """Connect the configured/current wireless endpoint, with bounded USB recovery.

        A REFUSED connection (the phone answered and closed the socket) means adbd simply
        isn't listening on this port -- retrying with backoff can't fix that, only running
        `adb tcpip <port>` on the phone can (see :func:`parse_usb_serials`). So on the FIRST
        refusal this looks for that same phone attached over USB right now and, if found
        alone, runs the fix itself and retries immediately -- turning "plug it in and run
        this command" into something that just works when the phone happens to already be
        plugged in. Falls through to the backoff loop (unchanged) for any other failure, and
        to a manual-fix error message if no single USB device was there to rescue from.

        ``Network is unreachable`` / ``No route to host`` is materially different: re-pinning
        adbd cannot create a route on the Mac, and retry backoff only makes the user wait.  If
        exactly one authorized USB phone is present, first ask it for its current ``wlan0``
        IPv4 address (the configured address may be stale after DHCP), try that address, and
        re-pin only if that *new* reachable address explicitly refuses.  If wireless recovery
        cannot establish a target, keep the verified USB serial as this session's active ADB
        transport.  The configured address is never changed on disk or in ``self.address``.
        """
        # A caller may invoke connect() again while the USB fallback is healthy.  It is already
        # connected; do not turn a working wired session back into a stale Wi-Fi retry.
        if self.using_usb:
            if self.is_connected():
                return
            # The cable went away (or authorization changed). Resume normal wireless recovery.
            self._usb_serial = None
            self._active_address = self._wireless_address

        # This call is only reached when we are about to establish a fresh ADB
        # transport (initial setup is harmless; a later call may follow a failed
        # health check).  Even if it fails, do not let a later recovered one-shot
        # command make an old resident shell look valid.
        self._connection_generation += 1

        delay = 2.0
        last = ""
        rescue_tried = False
        rescue_ok = False
        for attempt in range(retries + 1):
            connected, last = self._connect_once(self._wireless_address)
            if connected:
                self._use_wireless(self._wireless_address)
                return

            if _is_network_unreachable(last):
                # This is a deterministic local routing failure, not a remote listener that
                # might wake up during 2+4+8+16 seconds of retries.  Make one bounded USB
                # recovery pass, then either use its verified transport or fail clearly now.
                if self._recover_network_unreachable_via_usb():
                    return
                break

            if "refused" in last.lower() and not rescue_tried:
                # Only ONE attempt at this, win or lose: repeating it on every subsequent
                # retry would just re-run `adb devices`/`adb tcpip` for no new information.
                rescue_tried = True
                rescue_ok = self._rescue_via_usb()
                if rescue_ok:
                    continue   # retry now -- no backoff burned on a fix we just made
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        reason = last.strip()
        if rescue_ok:
            # We DID find the fix and applied it (adb tcpip succeeded on the USB-attached
            # phone), but the wireless connect still failed afterward -- a different,
            # rarer problem (wrong IP in device.adb_address, phone's Wi-Fi actually down),
            # not the common dead-listener case, so don't repeat that advice here.
            hint = (" -- a USB rescue re-pinned the wireless listener (`adb tcpip` succeeded "
                    "on the USB-attached phone) but the wireless connect still failed "
                    "afterward; check the phone's Wi-Fi and device.adb_address")
        elif "refused" in reason.lower():
            # A REFUSED connection (not a timeout) means the phone answered and closed the
            # socket -- adbd simply isn't listening on THIS port right now, which is NOT a
            # network/routing problem (a dead link times out; it does not refuse). A fixed
            # port (like 5555 here) is pinned by running `adb tcpip 5555` over USB, and a
            # reboot or a USB replug drops it until that's re-run. Android's own "Wireless
            # debugging" toggle (Settings > Developer options) is a SEPARATE, unrelated
            # feature: it opens its own listener on a random port shown on that screen, NOT
            # 5555 -- so re-enabling it does NOT fix a refusal on a fixed port and is a dead
            # end some users otherwise try first (confirmed live: a user re-enabled it and
            # still got refused on 5555). The USB rescue above already covers the case where
            # the phone is plugged in; reaching this message means it either wasn't plugged
            # in, or more than one USB device was attached (too ambiguous to guess which).
            hint = (" -- the phone's wireless-ADB listener is down, not a network drop: this "
                    "address is pinned to port 5555 by `adb tcpip 5555` (run once over USB) "
                    "-- a DIFFERENT mechanism from Android's 'Wireless debugging' toggle, "
                    "which opens its own listener on a random port shown on that screen and "
                    "does NOT restore 5555. Plug in ONLY the target phone over USB and retry "
                    "(hop will re-pin it automatically), or run `adb tcpip 5555` yourself.")
        elif _is_network_unreachable(reason):
            hint = (" -- this is a host routing/stale Wi-Fi-address failure, not a dead ADB "
                    "listener: `adb tcpip` cannot create a route to the phone. Join the "
                    "phone's Wi-Fi or update device.adb_address; with exactly one authorized "
                    "USB phone attached, hop will use USB for this session automatically.")
        else:
            hint = ""
        raise AdbError(f"could not connect to {self._wireless_address}: {reason}{hint}")

    def _connect_once(self, endpoint: str) -> tuple[bool, str]:
        """Run one ``adb connect`` and return ``(connected, diagnostic_output)``.

        ``adb`` has put failures on both stdout and stderr across versions.  Normalizing the
        streams here prevents the recovery decision from depending on that implementation
        detail.  A missing binary remains an immediate typed error, as it was before.
        """
        try:
            proc = subprocess.run([self.adb_bin, "connect", endpoint], capture_output=True,
                                  timeout=self.timeout)
        except FileNotFoundError:
            raise AdbError(self._missing_adb_msg())
        except Exception as e:  # pragma: no cover - network/spawn
            return False, str(e)
        out = _proc_output(proc)
        low = out.lower()
        if "connected to" in low or "already connected" in low:
            return True, out
        if not out.strip():
            code = getattr(proc, "returncode", None)
            out = f"adb connect exited with status {code}" if code is not None else "adb connect failed"
        return False, out

    def _single_usb_serial(self) -> str | None:
        """The sole authorized USB serial, or ``None`` when selection would be unsafe."""
        try:
            proc = subprocess.run([self.adb_bin, "devices"], capture_output=True,
                                  timeout=self.timeout)
        except Exception:
            return None
        if getattr(proc, "returncode", 0) != 0:
            return None
        serials = parse_usb_serials(_proc_output(proc))
        return serials[0] if len(serials) == 1 else None

    def _wireless_port(self) -> str:
        endpoint = self._wireless_address or self.configured_address
        return endpoint.rsplit(":", 1)[-1] if ":" in endpoint else "5555"

    def _discover_wlan_ipv4(self, usb_serial: str) -> str | None:
        """Read the currently assigned Wi-Fi IPv4 from a known USB-connected phone."""
        try:
            # ``-f inet`` and ``dev`` are supported by Android's toybox `ip`; unlike `-4`,
            # they work on the Pixel's stock shell too.  The interface is deliberately pinned
            # to wlan0 so a cellular/VPN address cannot be mistaken for the LAN endpoint.
            proc = subprocess.run(
                [self.adb_bin, "-s", usb_serial, "shell", "ip -f inet addr show dev wlan0"],
                capture_output=True, timeout=self.timeout,
            )
        except Exception:
            return None
        if getattr(proc, "returncode", 0) != 0:
            return None
        return parse_wlan_ipv4(_proc_output(proc))

    def _usb_is_connected(self, usb_serial: str) -> bool:
        """Re-check a USB transport before making it active after a recovery attempt."""
        try:
            proc = subprocess.run([self.adb_bin, "-s", usb_serial, "get-state"],
                                  capture_output=True, timeout=self.timeout)
        except Exception:
            return False
        return (getattr(proc, "returncode", 0) == 0
                and _proc_output(proc).strip() == "device")

    def _use_usb_if_connected(self, usb_serial: str) -> bool:
        if not self._usb_is_connected(usb_serial):
            return False
        self._active_address = usb_serial
        self._usb_serial = usb_serial
        self._connection_generation += 1
        return True

    def _use_wireless(self, endpoint: str) -> None:
        """Make a verified wireless endpoint the active session transport."""
        self._wireless_address = endpoint
        self._active_address = endpoint
        self._usb_serial = None
        self._connection_generation += 1

    def _start_tcpip_via_usb(self, usb_serial: str, port: str) -> bool:
        """Best-effort ``adb tcpip`` for an explicitly reachable-but-refused endpoint."""
        try:
            proc = subprocess.run([self.adb_bin, "-s", usb_serial, "tcpip", port],
                                  capture_output=True, timeout=self.timeout)
        except Exception:
            return False
        if getattr(proc, "returncode", 0) != 0:
            return False
        time.sleep(1.5)   # adbd restarts in TCP mode; give it a beat before the retry connects
        return True

    def _recover_network_unreachable_via_usb(self) -> bool:
        """Recover one explicit local-route failure through a sole attached USB phone.

        The ordering is intentional.  A changed DHCP address may already have adbd listening,
        so try the phone's live WLAN address first.  Only a *refused* new address proves a
        listener restart is useful; a second no-route/timeout would make ``adb tcpip`` harmful
        by disrupting the USB fallback without repairing the Mac's route.
        """
        usb_serial = self._single_usb_serial()
        if usb_serial is None:
            return False

        current_ip = self._discover_wlan_ipv4(usb_serial)
        if current_ip:
            candidate = f"{current_ip}:{self._wireless_port()}"
            # If it is the same stale/routeless endpoint we already tried, do not restart adbd
            # pointlessly.  Keep USB usable instead.
            if candidate != self._wireless_address:
                connected, out = self._connect_once(candidate)
                if connected:
                    self._use_wireless(candidate)
                    return True
                if "refused" in out.lower() and self._start_tcpip_via_usb(
                        usb_serial, self._wireless_port()):
                    connected, _ = self._connect_once(candidate)
                    if connected:
                        self._use_wireless(candidate)
                        return True

        # The initial `adb devices` result was authorized, but an address probe / tcpip can race
        # with a cable removal or an adbd restart.  Verify it again before targeting the session
        # at this serial; never blindly issue `adb disconnect <USB serial>` later.
        return self._use_usb_if_connected(usb_serial)

    def _rescue_via_usb(self) -> bool:
        """If this phone is reachable over USB right now, run `adb tcpip <port>` on it to
        re-pin the wireless listener that a REFUSED :meth:`connect` needs restarted.

        Only acts when exactly one authorized ("device"-state) USB serial is attached: zero
        means nothing to rescue from, and more than one makes guessing which is this phone a
        real risk (tcpip-ing the wrong device). Best-effort and never raises -- any failure
        (adb missing, no USB device, `tcpip` itself failing) returns False so :meth:`connect`
        falls through to its normal error, not a broken rescue standing in for it.
        """
        usb_serial = self._single_usb_serial()
        if usb_serial is None:
            return False
        return self._start_tcpip_via_usb(usb_serial, self._wireless_port())

    def is_connected(self) -> bool:
        try:
            state = self._run(["get-state"]).decode(errors="replace").strip()
            return state == "device"
        except Exception:
            return False

    def ensure_connected(self) -> None:
        """Reconnect if the active Wi-Fi or USB ADB transport dropped."""
        if not self.is_connected():
            self.connect()

    def disconnect(self) -> None:
        """Drop the cached wireless endpoint. A Wi-Fi link that dropped often lingers as a
        zombie 'device' whose ``exec-out`` still HANGS on the dead socket; a plain reconnect
        reuses that zombie, so we disconnect first to force a fresh handshake. Best-effort and
        never raises -- it is only ever a prelude to :meth:`reconnect`."""
        if self.using_usb:
            # `adb disconnect` is a TCP endpoint operation.  Sending it a USB serial is both
            # meaningless and risks throwing away the only transport that survived the outage.
            return
        # A successful `adb disconnect` definitely tears down resident shell
        # streams.  Advance before attempting it so a spawn/timeout ambiguity is
        # still fail-safe: a later successful one-shot command must not make a
        # stale writer look trustworthy.
        self._connection_generation += 1
        try:
            subprocess.run([self.adb_bin, "disconnect", self._active_address],
                           capture_output=True, timeout=self.timeout)
        except Exception:  # pragma: no cover - network/binary; reconnect() reports real failure
            pass

    def reconnect(self) -> None:
        """Force a fresh wireless link after a drop, or retain a healthy USB fallback.

        A directly attached serial has no cached TCP socket to disconnect.  If it still answers
        ``get-state``, leave it alone and let the caller retry the failed frame; if it is gone,
        restore normal wireless recovery.  Wireless mode retains the existing disconnect-first
        zombie-socket protection and uses only a single bounded connect attempt.
        """
        if self.using_usb:
            if self.is_connected():
                return
            self._usb_serial = None
            self._active_address = self._wireless_address
            self.connect(retries=1)
            return
        self.disconnect()
        self.connect(retries=1)

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
