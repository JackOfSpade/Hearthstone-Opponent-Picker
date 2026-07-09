"""The Mac control panel: a menu-bar item plus a native stats window.

On launch it puts a status item in the menu bar (glanceable state, a loud target
alert) AND opens a **native window** - a WKWebView hosting the local dashboard, so the
settings/stats hub is a real macOS window, not a browser tab and not Electron. The
window renders in-process (the dashboard server is a daemon thread in this same Python
process), so there is no second runtime and no IPC seam.

Why pyobjc and not Swift/Electron: the engine is Python. A Swift or Electron shell
would add a build step, a signing story, and an IPC boundary between the UI and the
Python engine, in exchange for nothing a native WKWebView window doesn't already give.
The menu items call :class:`~hop.runner.EngineController` directly, and the window shows
the same in-process dashboard.

Threading: the engine's hunt loop is blocking, so ``EngineController`` already runs
it on a daemon thread. AppKit owns the main thread and drives everything here from
a repeating timer, so no engine call ever blocks the UI.

    hop app          # menu bar
    hop dashboard    # the web panel

Requires ``pip install 'hop[mac]'`` (pyobjc). Everything degrades to the CLI and the
dashboard without it.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

from .config import Config, save_criteria
from .hero_classes import DISPLAY_NAMES, HeroClass

MENU_BAR_IDLE = "hop"
MENU_BAR_HUNTING = "hop ▶"
MENU_BAR_TARGET = "hop ●"
MENU_BAR_HALTED = "hop ⚠"
DASHBOARD_PORT = 8765


class MacAppUnavailable(RuntimeError):
    """pyobjc is not installed, so there is no menu bar to attach to."""


def _require_appkit():
    try:
        import AppKit  # noqa: F401
        import objc  # noqa: F401
    except Exception as e:  # pragma: no cover - import guard
        raise MacAppUnavailable(
            "the control panel needs pyobjc: pip install 'hop[mac]'\n"
            "(the CLI and `hop dashboard` work without it)"
        ) from e
    return __import__("AppKit"), __import__("objc")


def _brand_process(app_name: str = "Hearthstone Opponent Picker") -> None:
    """Make the menu bar and Dock read our name, not "Python".

    A framework-Python GUI app re-execs through ``Python.app``, whose bundle name is
    "Python"; AppKit reads ``CFBundleName`` from the main bundle's info dictionary for
    the app menu (and it seeds the Dock tile's label). Mutating that in-memory dict
    before ``NSApplication`` is built relabels it -- the same trick rumps/matplotlib
    use. Best-effort: a failure just leaves the default name.
    """
    try:
        from Foundation import NSBundle
        b = NSBundle.mainBundle()
        info = b.localizedInfoDictionary() or b.infoDictionary()
        if info is not None:
            info["CFBundleName"] = app_name
    except Exception:
        pass


def _apply_dock_icon(app, AppKit) -> None:
    """Set the Dock icon to the .app's AppIcon.icns (path passed via ``HOP_APP_ICON``).

    The re-exec'd ``Python.app`` owns the Dock tile, so its default is the generic
    Python rocket; ``setApplicationIconImage_`` overrides it at runtime with ours. The
    launcher exports ``HOP_APP_ICON`` because this Python process can't otherwise locate
    the surrounding .app bundle. Best-effort; a missing file just leaves the default.
    """
    import os
    path = os.environ.get("HOP_APP_ICON", "")
    if not path or not os.path.exists(path):
        return
    try:
        img = AppKit.NSImage.alloc().initWithContentsOfFile_(path)
        if img is not None:
            app.setApplicationIconImage_(img)
    except Exception:
        pass


def run_menubar(cfg: Config, engine_factory, alerter=None,
                config_path: str | Path | None = None) -> int:
    """Run the menu-bar control panel. Blocks until the user quits."""
    AppKit, objc = _require_appkit()
    from .runner import EngineController

    _brand_process()   # relabel "Python" -> our name BEFORE NSApplication reads it
    controller = EngineController(engine_factory, alerter=alerter)
    app = AppKit.NSApplication.sharedApplication()
    # Regular app: a Dock icon and a Cmd-Tab entry, PLUS the menu-bar item. The Dock
    # icon lets the hub window be reopened after it's closed (see the reopen handler);
    # the menu bar keeps the glanceable state and the loud target alert.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    _apply_dock_icon(app, AppKit)   # our AppIcon.icns over the generic Python rocket

    delegate = _HopMenuDelegate.alloc().initWithState_(
        _AppState(cfg=cfg, controller=controller, config_path=config_path, AppKit=AppKit)
    )
    app.setDelegate_(delegate)
    delegate.build()
    # Open the stats/settings window on launch so a double-click gives a visible hub -
    # a bare menu-bar item alone reads as "nothing happened". The menu bar stays too.
    try:
        _open_dashboard_window(delegate._state)
    except Exception:
        pass   # a webview failure must not stop the menu bar from running
    app.run()
    return 0


class _AppState:
    """Plain-Python state, kept out of the Objective-C subclass."""

    def __init__(self, cfg: Config, controller, config_path, AppKit):
        self.cfg = cfg
        self.controller = controller
        self.config_path = config_path
        self.AppKit = AppKit
        self.status_item = None
        self.menu = None
        self.class_items: dict[HeroClass, object] = {}
        self.require_second_item = None
        self.status_line = None
        self.risk_line = None
        self.dashboard_thread: threading.Thread | None = None
        #: the native WKWebView window (kept referenced so it isn't collected)
        self.dash_window = None
        self.dash_webview = None

    # ── criteria ──────────────────────────────────────────────────────────────

    @property
    def targets(self) -> tuple[HeroClass, ...]:
        return self.cfg.criteria.target_classes

    def toggle_class(self, hero: HeroClass) -> None:
        current = list(self.targets)
        if hero in current:
            current.remove(hero)
        else:
            current.append(hero)
        self._commit(tuple(current), self.cfg.criteria.require_second)

    def toggle_require_second(self) -> None:
        self._commit(self.targets, not self.cfg.criteria.require_second)

    def _commit(self, targets: tuple[HeroClass, ...], require_second: bool) -> None:
        self.cfg = replace(self.cfg, criteria=replace(
            self.cfg.criteria, target_classes=targets, require_second=require_second))
        save_criteria(target_classes=targets, require_second=require_second,
                      path=self.config_path)

    # ── derived text ──────────────────────────────────────────────────────────

    def title(self) -> str:
        st = self.controller.status()
        if st.get("target_found"):
            return MENU_BAR_TARGET
        if st.get("last_error") or (st.get("stop_reason") or "").startswith("halt"):
            return MENU_BAR_HALTED
        return MENU_BAR_HUNTING if st.get("running") else MENU_BAR_IDLE

    def status_text(self) -> str:
        st = self.controller.status()
        if st.get("target_found"):
            return f"TARGET: {st.get('last_opponent', '?')} — your turn"
        err = st.get("last_error")
        if err:
            return f"Halted: {err[:48]}"
        if not st.get("running"):
            reason = st.get("stop_reason")
            return f"Idle ({reason})" if reason else "Idle"
        b = st.get("budget") or {}
        text = (f"Hunting · {st.get('games', 0)} games · "
                f"{b.get('concedes_run', 0)}/{b.get('concedes_cap', '?')} concedes")
        ignored = st.get("ignored_card_taps") or 0
        if ignored:
            text += f" · {ignored} card taps ignored"
        return text

    def risk_text(self) -> str:
        crit = self.cfg.criteria
        pass_rate = crit.pass_rate_estimate()
        concede = 1.0 - pass_rate
        if concede >= 0.9:
            label = "HIGH — barcode-shaped"
        elif concede >= 0.7:
            label = "elevated"
        else:
            label = "moderate"
        return f"Concede rate ≈{concede * 100:.0f}%  ·  risk: {label}"


# pyobjc turns EVERY method of an NSObject subclass into an Objective-C selector,
# so a plain Python helper like `_disabled(self, menu, title)` raises BadPrototypeError
# at class-creation time. Helpers must be marked `@objc.python_method`. Only the
# import is guarded here -- a broad `except Exception` around the class body once hid
# exactly that error and made the app pretend pyobjc was missing.
try:  # pragma: no cover - macOS + pyobjc only
    import AppKit as _AppKit
    import objc as _objc
except ImportError:  # pragma: no cover
    _HopMenuDelegate = None  # type: ignore[assignment]
else:  # pragma: no cover - needs a Mac GUI session to exercise

    class _HopMenuDelegate(_AppKit.NSObject):
        """The Objective-C side. Holds only a pointer to :class:`_AppState`."""

        def initWithState_(self, state):
            self = _objc.super(_HopMenuDelegate, self).init()
            if self is None:
                return None
            self._state = state
            return self

        # ── construction ──────────────────────────────────────────────────────

        @_objc.python_method
        def build(self):
            s = self._state
            AppKit = s.AppKit
            bar = AppKit.NSStatusBar.systemStatusBar()
            s.status_item = bar.statusItemWithLength_(AppKit.NSVariableStatusItemLength)
            s.status_item.button().setTitle_(MENU_BAR_IDLE)

            menu = AppKit.NSMenu.alloc().init()
            menu.setAutoenablesItems_(False)

            s.status_line = self._disabled(menu, "Idle")
            s.risk_line = self._disabled(menu, s.risk_text())
            menu.addItem_(AppKit.NSMenuItem.separatorItem())

            self._action(menu, "Start Search", "start_", key="s")
            self._action(menu, "Stop", "stop_", key=".")
            self._action(menu, "Silence alarm", "ack_", key="a")
            menu.addItem_(AppKit.NSMenuItem.separatorItem())

            # target classes: a checkable submenu; empty selection = accept any class
            classes_item = AppKit.NSMenuItem.alloc().init()
            classes_item.setTitle_("Target classes")
            submenu = AppKit.NSMenu.alloc().init()
            submenu.setAutoenablesItems_(False)
            for hero in HeroClass:
                item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                    DISPLAY_NAMES[hero], b"toggleClass:", "")
                item.setTarget_(self)
                item.setRepresentedObject_(hero.name)
                item.setState_(1 if hero in s.targets else 0)
                submenu.addItem_(item)
                s.class_items[hero] = item
            classes_item.setSubmenu_(submenu)
            menu.addItem_(classes_item)

            s.require_second_item = self._action(menu, "Only when going 2nd", "toggleSecond_")
            s.require_second_item.setState_(1 if s.cfg.criteria.require_second else 0)

            menu.addItem_(AppKit.NSMenuItem.separatorItem())
            self._action(menu, "Open dashboard\u2026", "dashboard_")
            self._action(menu, "Quit hop", "quit_", key="q")

            s.status_item.setMenu_(menu)
            s.menu = menu

            self._install_main_menu()

            AppKit.NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1.0, self, b"refresh:", None, True)

        @_objc.python_method
        def _install_main_menu(self):
            """A minimal main menu so the Dock app behaves like one: an app menu with a
            working Cmd-Q (routed through our clean stop) and "Open hop Window", plus a
            standard Edit menu so Cmd-C/V/X/A work in the WKWebView. Accessory apps don't
            need this; a Regular (Dock) app looks broken without it (no Cmd-Q, no app
            menu, and no working copy/paste in text fields).
            """
            AppKit = self._state.AppKit
            app = AppKit.NSApplication.sharedApplication()
            main = AppKit.NSMenu.alloc().init()

            app_item = AppKit.NSMenuItem.alloc().init()
            main.addItem_(app_item)
            app_menu = AppKit.NSMenu.alloc().init()
            show = app_menu.addItemWithTitle_action_keyEquivalent_(
                "Open hop Window", b"dashboard:", "0")
            show.setTarget_(self)
            app_menu.addItem_(AppKit.NSMenuItem.separatorItem())
            quit_item = app_menu.addItemWithTitle_action_keyEquivalent_(
                "Quit Hearthstone Opponent Picker", b"quit:", "q")
            quit_item.setTarget_(self)
            app_item.setSubmenu_(app_menu)

            # Edit menu. Its items keep the default nil target so Cmd-C/V/X/A dispatch
            # down the responder chain to the focused WKWebView / text field. Without
            # these menu items nothing maps those keys to copy:/paste:/cut:/selectAll:,
            # which is why keyboard editing did nothing in the window.
            edit_item = AppKit.NSMenuItem.alloc().init()
            edit_item.setTitle_("Edit")
            main.addItem_(edit_item)
            edit_menu = AppKit.NSMenu.alloc().initWithTitle_("Edit")
            edit_menu.addItemWithTitle_action_keyEquivalent_("Undo", b"undo:", "z")
            redo = edit_menu.addItemWithTitle_action_keyEquivalent_("Redo", b"redo:", "z")
            redo.setKeyEquivalentModifierMask_(
                AppKit.NSEventModifierFlagCommand | AppKit.NSEventModifierFlagShift)
            edit_menu.addItem_(AppKit.NSMenuItem.separatorItem())
            edit_menu.addItemWithTitle_action_keyEquivalent_("Cut", b"cut:", "x")
            edit_menu.addItemWithTitle_action_keyEquivalent_("Copy", b"copy:", "c")
            edit_menu.addItemWithTitle_action_keyEquivalent_("Paste", b"paste:", "v")
            edit_menu.addItemWithTitle_action_keyEquivalent_("Select All", b"selectAll:", "a")
            edit_item.setSubmenu_(edit_menu)

            app.setMainMenu_(main)

        @_objc.python_method
        def _disabled(self, menu, title):
            AppKit = self._state.AppKit
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, None, "")
            item.setEnabled_(False)
            menu.addItem_(item)
            return item

        @_objc.python_method
        def _action(self, menu, title, method_name, key=""):
            """Wire a menu item to one of our selectors (``start_`` -> ``start:``)."""
            AppKit = self._state.AppKit
            selector = (method_name[:-1] + ":").encode()
            item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, selector, key)
            item.setTarget_(self)
            menu.addItem_(item)
            return item

        # ── actions (selectors: one argument, the sender) ─────────────────────

        def start_(self, sender):
            crit = self._state.cfg.criteria
            self._state.controller.start({
                "target_classes": [c.name for c in crit.target_classes],
                "require_second": crit.require_second,
            })

        def stop_(self, sender):
            self._state.controller.stop()

        def ack_(self, sender):
            self._state.controller.ack_alarm()

        def toggleClass_(self, sender):
            s = self._state
            hero = HeroClass[sender.representedObject()]
            s.toggle_class(hero)
            sender.setState_(1 if hero in s.targets else 0)
            s.risk_line.setTitle_(s.risk_text())

        def toggleSecond_(self, sender):
            s = self._state
            s.toggle_require_second()
            sender.setState_(1 if s.cfg.criteria.require_second else 0)
            s.risk_line.setTitle_(s.risk_text())

        def dashboard_(self, sender):
            _open_dashboard_window(self._state)

        def quit_(self, sender):
            self._state.controller.stop()
            self._state.AppKit.NSApplication.sharedApplication().terminate_(self)

        # ── the 1 Hz refresh ──────────────────────────────────────────────────

        def refresh_(self, timer):
            s = self._state
            s.status_item.button().setTitle_(s.title())
            s.status_line.setTitle_(s.status_text())

        # ── Dock behaviour ────────────────────────────────────────────────────
        def applicationShouldHandleReopen_hasVisibleWindows_(self, app, has_windows):
            """Clicking the Dock icon with no window open reopens the hub window.

            Closing the window only hides it (releasedWhenClosed=False), so this is
            how you get it back from the Dock instead of quitting and relaunching.
            """
            if not has_windows:
                _open_dashboard_window(self._state)
            return True


def _ensure_dashboard(state: _AppState) -> threading.Thread:
    """Start the stdlib dashboard on a daemon thread, once.

    Note ``webui.serve()`` already blocks in ``serve_forever``; we construct the
    server directly so the thread owns the loop.
    """
    if state.dashboard_thread and state.dashboard_thread.is_alive():
        return state.dashboard_thread
    from .webui import DashboardServer

    def _serve():
        DashboardServer(state.controller, state.cfg, port=DASHBOARD_PORT).serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    return t


def _open_dashboard_window(state: _AppState) -> None:
    """Show the dashboard in a NATIVE window (a WKWebView), not a browser.

    The dashboard is already served in-process by :func:`_ensure_dashboard`; this just
    renders that local page in a real macOS window. It is *not* Electron and *not*
    Chrome - it's the system WebKit view, hosted inside the same Python process, so the
    settings/stats hub is a native window with zero extra runtime.

    Re-opening reuses the existing window (closing only hides it). We activate the app so
    the window comes to the front whether it was launched, reopened from the Dock, or
    summoned from the menu bar.
    """
    AppKit = state.AppKit
    _ensure_dashboard(state)

    if state.dash_window is not None:
        state.dash_window.makeKeyAndOrderFront_(None)
        AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        return

    import WebKit
    from Foundation import NSURL, NSURLRequest, NSMakeRect

    rect = NSMakeRect(0, 0, 920, 700)
    mask = (AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable
            | AppKit.NSWindowStyleMaskResizable | AppKit.NSWindowStyleMaskMiniaturizable)
    win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        rect, mask, AppKit.NSBackingStoreBuffered, False)
    win.setTitle_("hop")
    win.setReleasedWhenClosed_(False)   # we hold the ref; closing hides, not frees
    win.setMinSize_(AppKit.NSMakeSize(560, 480))

    config = WebKit.WKWebViewConfiguration.alloc().init()
    webview = WebKit.WKWebView.alloc().initWithFrame_configuration_(rect, config)
    win.setContentView_(webview)
    url = NSURL.URLWithString_(f"http://127.0.0.1:{DASHBOARD_PORT}/")
    webview.loadRequest_(NSURLRequest.requestWithURL_(url))

    win.center()
    win.makeKeyAndOrderFront_(None)
    AppKit.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
    state.dash_window = win
    state.dash_webview = webview
