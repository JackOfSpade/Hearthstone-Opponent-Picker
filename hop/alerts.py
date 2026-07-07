"""Alerts: Mac notification + looping sound + optional phone push.

When a target matchup is found the engine stops touching the game and fires
these. Two destinations:

* **Mac** (where you're sitting): a native notification (``osascript``), a
  looping sound (``afplay``) that keeps going until acknowledged, and an
  optional spoken announcement (``say``).
* **Phone** (since the point is to walk away): a push via ntfy.sh - one HTTP
  POST, no account, plays a loud custom sound on the phone. Never interacts with
  Hearthstone. Uses only the standard library (urllib), so no extra deps.

All of it degrades gracefully off macOS / without network to plain prints, so
the tool never crashes because an alert channel is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import urllib.request

from .config import AlertConfig


class Alerter:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._alarm_stop: threading.Event | None = None
        self._alarm_thread: threading.Thread | None = None

    # ── public API used by the engine ────────────────────────────────────────

    def target_found(self, class_name: str, we_go_second: bool) -> None:
        second = "going 2nd" if we_go_second else "going 1st"
        title = "Hearthstone target found"
        body = f"{class_name} ({second}) - your turn!"
        self._mac_notify(title, body)
        if self.cfg.mac_speak:
            self._mac_say(f"{class_name} found")
        if self.cfg.mac_sound:
            self.start_alarm()
        self._ntfy(title, body, priority="urgent", tags="tada")
        print(f"*** {title}: {body} ***")

    def info(self, message: str) -> None:
        self._mac_notify("hop", message)
        print(f"[hop] {message}")

    def halt(self, message: str) -> None:
        self._mac_notify("hop HALTED", message)
        self._ntfy("hop HALTED", message, priority="high", tags="warning")
        print(f"[hop][HALT] {message}")

    # ── looping alarm (until acknowledged) ───────────────────────────────────

    def start_alarm(self) -> None:
        """Loop the alert sound in a background thread until :meth:`stop_alarm`."""
        if self._alarm_thread and self._alarm_thread.is_alive():
            return
        self._alarm_stop = threading.Event()

        def loop():
            path = f"/System/Library/Sounds/{self.cfg.sound_name}.aiff"
            afplay = shutil.which("afplay")
            while self._alarm_stop and not self._alarm_stop.is_set():
                if afplay:
                    try:
                        subprocess.run([afplay, path], timeout=10)
                    except Exception:
                        self._alarm_stop.wait(1.0)
                else:  # non-mac: terminal bell + wait
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    self._alarm_stop.wait(1.5)

        self._alarm_thread = threading.Thread(target=loop, daemon=True)
        self._alarm_thread.start()

    def stop_alarm(self) -> None:
        if self._alarm_stop:
            self._alarm_stop.set()
        self._alarm_thread = None

    # ── platform helpers ─────────────────────────────────────────────────────

    def _mac_notify(self, title: str, body: str) -> None:
        if sys.platform != "darwin":
            return
        osa = shutil.which("osascript")
        if not osa:
            return
        script = f'display notification {_q(body)} with title {_q(title)}'
        try:
            subprocess.run([osa, "-e", script], timeout=5)
        except Exception:
            pass

    def _mac_say(self, text: str) -> None:
        if sys.platform != "darwin":
            return
        say = shutil.which("say")
        if not say:
            return
        try:
            subprocess.Popen([say, text])
        except Exception:
            pass

    def _ntfy(self, title: str, body: str, priority: str = "default", tags: str = "") -> None:
        if not self.cfg.ntfy_topic:
            return
        url = f"{self.cfg.ntfy_server.rstrip('/')}/{self.cfg.ntfy_topic}"
        req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
        req.add_header("Title", title)
        req.add_header("Priority", priority)
        if tags:
            req.add_header("Tags", tags)
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # phone push is best-effort; the Mac alert is the primary


def _q(s: str) -> str:
    """Quote a string for AppleScript."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
