"""The Mac control panel.

Two things are worth pinning without a GUI session:

* the pyobjc delegate class **constructs at import**. Every method of an NSObject
  subclass becomes an Objective-C selector, so a plain Python helper with the wrong
  arity raises ``BadPrototypeError`` at class-creation time. A broad
  ``except Exception`` around the class body once swallowed exactly that and made
  the app quietly claim pyobjc was missing.
* ``_AppState`` is pure Python and holds all the logic the menu items call, so the
  criteria round-trip and the risk meter are testable off-GUI.
"""

import tomllib

import pytest

from hop.config import load_config
from hop.hero_classes import HeroClass


def test_delegate_class_constructs_when_pyobjc_is_present():
    """A selector-prototype error must fail loudly, not look like 'pyobjc missing'."""
    pytest.importorskip("AppKit")
    import hop.macapp as macapp

    assert macapp._HopMenuDelegate is not None, (
        "the pyobjc delegate failed to build; the import guard is hiding a real error"
    )
    for selector in ("start_", "stop_", "toggle_", "toggleClass_",
                     "toggleSecond_", "dashboard_", "quit_", "refresh_"):
        assert hasattr(macapp._HopMenuDelegate, selector)
    # closing the window (red X) must quit the whole app, not leave it headless
    assert hasattr(macapp._HopMenuDelegate, "applicationShouldTerminateAfterLastWindowClosed_")


def test_missing_pyobjc_raises_a_helpful_error(monkeypatch):
    import builtins

    import hop.macapp as macapp

    real_import = builtins.__import__

    def no_appkit(name, *a, **k):
        if name in ("AppKit", "objc"):
            raise ImportError("no pyobjc")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_appkit)
    with pytest.raises(macapp.MacAppUnavailable, match=r"hop\[mac\]"):
        macapp._require_appkit()


def _state(tmp_path, body='[criteria]\ntarget_classes = []\nrequire_second = false\n'):
    import hop.macapp as macapp

    p = tmp_path / "config.toml"
    p.write_text(body)

    class FakeController:
        def __init__(self):
            self._status = {"running": False}

        def status(self):
            return self._status

    return macapp._AppState(cfg=load_config(p), controller=FakeController(),
                            config_path=p, AppKit=None), p


def test_toggling_a_class_persists_to_the_config(tmp_path):
    s, p = _state(tmp_path)
    s.toggle_class(HeroClass.MAGE)
    s.toggle_class(HeroClass.WARLOCK)
    assert set(s.targets) == {HeroClass.MAGE, HeroClass.WARLOCK}
    assert set(tomllib.loads(p.read_text())["criteria"]["target_classes"]) == {"MAGE", "WARLOCK"}

    s.toggle_class(HeroClass.MAGE)          # toggling off removes it
    assert s.targets == (HeroClass.WARLOCK,)
    assert load_config(p).criteria.target_classes == (HeroClass.WARLOCK,)


def test_toggling_require_second_persists(tmp_path):
    s, p = _state(tmp_path)
    assert s.cfg.criteria.require_second is False
    s.toggle_require_second()
    assert load_config(p).criteria.require_second is True


def test_menu_bar_title_reflects_engine_state(tmp_path):
    import hop.macapp as macapp

    s, _ = _state(tmp_path)
    s.controller._status = {"running": False}
    assert s.title() == macapp.MENU_BAR_IDLE
    s.controller._status = {"running": True}
    assert s.title() == macapp.MENU_BAR_HUNTING
    s.controller._status = {"running": True, "target_found": True}
    assert s.title() == macapp.MENU_BAR_TARGET
    s.controller._status = {"running": False, "stop_reason": "halt:unknown screen"}
    assert s.title() == macapp.MENU_BAR_HALTED


def test_app_is_branded_with_the_project_name_not_hop():
    import hop.macapp as macapp

    assert macapp.APP_NAME == "Hearthstone Opponent Picker"
    for label in (macapp.MENU_BAR_IDLE, macapp.MENU_BAR_HUNTING,
                  macapp.MENU_BAR_TARGET, macapp.MENU_BAR_HALTED):
        assert label.startswith(macapp.APP_NAME)
        assert "hop" not in label.lower()


def test_risk_meter_flags_barcode_shaped_criteria(tmp_path):
    """One class + require-second is ~95% concedes: the shape Blizzard tracks."""
    s, _ = _state(tmp_path)
    assert "moderate" in s.risk_text()          # accept any class -> never concede
    s.toggle_class(HeroClass.MAGE)
    s.toggle_require_second()
    text = s.risk_text()
    assert "HIGH" in text and "barcode" in text
    assert "95%" in text
