"""The target alert is a single Mac sound -- nothing else, by design.

The user sits at the Mac with earbuds in, so on a target hop plays the configured system
sound exactly once. No looping alarm, no acknowledge/silence step, no phone push, no
spoken announcement. These pin that so the alert can't silently regrow a channel.
"""

from hop.alerts import Alerter
from hop.config import AlertConfig


def _cfg(**over) -> AlertConfig:
    base = dict(mac_notification=True, mac_sound=True, mac_speak=True,
                sound_name="Glass", ntfy_topic="secret-topic", ntfy_server="https://ntfy.sh")
    base.update(over)
    return AlertConfig(**base)


def _spy(alerter):
    """Replace every outward channel with a call-recorder."""
    calls: list[str] = []
    alerter._mac_play_sound_once = lambda: calls.append("sound")
    alerter._mac_notify = lambda *a, **k: calls.append("notify")
    alerter._mac_say = lambda *a, **k: calls.append("say")
    alerter._ntfy = lambda *a, **k: calls.append("ntfy")
    return calls


def test_target_found_plays_the_sound_once_and_nothing_else():
    a = Alerter(_cfg(mac_speak=True))          # even with speak on in config...
    calls = _spy(a)
    a.target_found("Mage", we_go_second=True)
    assert calls == ["sound"]                  # ...only the sound fires


def test_target_found_is_silent_when_mac_sound_is_off():
    a = Alerter(_cfg(mac_sound=False))
    calls = _spy(a)
    a.target_found("Priest", we_go_second=False)
    assert calls == []                          # no sound, and still no other channel


def test_no_looping_alarm_api_remains():
    """The looping alarm + silence step are gone; nothing may depend on them."""
    a = Alerter(_cfg())
    for gone in ("start_alarm", "stop_alarm", "is_alarming"):
        assert not hasattr(a, gone), f"{gone} should have been removed"


def test_halt_still_notifies_and_pings_the_phone():
    """Halts are error conditions, not the target alert, so they keep their channels."""
    a = Alerter(_cfg())
    calls = _spy(a)
    a.halt("stuck on an unknown screen")
    assert "notify" in calls and "ntfy" in calls


def test_halt_plays_the_sound_so_a_dead_hunt_is_heard():
    """An unexpected stop is not user-requested, so it is audible like a target find -- the
    user is not watching the screen. The sound leads; the banner + phone push still follow."""
    a = Alerter(_cfg())
    calls = _spy(a)
    a.halt("stuck on an unknown screen")
    assert calls[0] == "sound"                  # audible, and first
    assert "notify" in calls and "ntfy" in calls


def test_halt_sound_obeys_the_mac_sound_flag():
    """mac_sound gates the halt sound exactly as it gates the target sound; the banner and
    phone push are independent of it and still fire."""
    a = Alerter(_cfg(mac_sound=False))
    calls = _spy(a)
    a.halt("stuck on an unknown screen")
    assert "sound" not in calls                 # flag off -> no sound
    assert "notify" in calls and "ntfy" in calls


def test_mac_notification_flag_actually_suppresses_the_banner(monkeypatch):
    """Regression: mac_notification=false was parsed but never consulted, so banners
    still fired. The flag must gate the osascript call, like mac_sound gates afplay."""
    import hop.alerts as alerts_mod

    calls: list = []
    # force the darwin path with a resolvable osascript so ONLY the flag decides
    monkeypatch.setattr(alerts_mod.sys, "platform", "darwin")
    monkeypatch.setattr(alerts_mod.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(alerts_mod.subprocess, "run", lambda *a, **k: calls.append(a))

    Alerter(_cfg(mac_notification=False))._mac_notify("t", "b")
    assert calls == []                     # flag off -> no banner

    Alerter(_cfg(mac_notification=True))._mac_notify("t", "b")
    assert len(calls) == 1                 # flag on -> banner attempted
