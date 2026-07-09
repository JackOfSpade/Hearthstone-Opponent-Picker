"""Alerts: a single Mac sound on target; Mac notifications for halts/info.

When a target matchup is found the engine stops touching the game and alerts. Per the
user's setup (sitting at the Mac, earbuds on, watching YouTube) the target alert is
deliberately minimal: **play the configured Mac sound exactly once** - no looping alarm,
no acknowledge/silence step, no phone push, no spoken announcement. One sound they'll
hear over their audio, and the dashboard/menu-bar show which class.

Halts and info still post a silent Mac notification (and a halt also pings the phone),
since those are error conditions worth surfacing even if the window isn't focused.

All of it degrades gracefully off macOS / without network to plain prints, so the tool
never crashes because an alert channel is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request

from .config import AlertConfig


class Alerter:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg

    # ── public API used by the engine ────────────────────────────────────────

    def target_found(self, class_name: str, we_go_second: bool) -> None:
        """The whole target alert: one Mac sound. Nothing else, by design."""
        second = "going 2nd" if we_go_second else "going 1st"
        if self.cfg.mac_sound:
            self._mac_play_sound_once()
        print(f"*** target found: {class_name} ({second}) - your turn! ***")

    def info(self, message: str) -> None:
        self._mac_notify("hop", message)
        print(f"[hop] {message}")

    def halt(self, message: str) -> None:
        self._mac_notify("hop HALTED", message)
        self._ntfy("hop HALTED", message, priority="high", tags="warning")
        print(f"[hop][HALT] {message}")

    # ── the single target sound ──────────────────────────────────────────────

    def _mac_play_sound_once(self) -> None:
        """Play the configured system sound exactly once, non-blocking.

        ``Popen`` (not ``run``) so the one ~1s playback never blocks the caller; off
        macOS it falls back to a single terminal bell. Best-effort: a missing player
        just prints (the caller already logged the find).
        """
        if sys.platform != "darwin":
            sys.stdout.write("\a")
            sys.stdout.flush()
            return
        afplay = shutil.which("afplay")
        if not afplay:
            return
        try:
            subprocess.Popen([afplay, f"/System/Library/Sounds/{self.cfg.sound_name}.aiff"])
        except Exception:
            pass

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
