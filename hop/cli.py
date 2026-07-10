"""hop command-line interface.

Commands:
* ``hop doctor``            - end-to-end preflight: adb, screencap, UHID tool,
  panel geometry + input-device identity (for panel-matching), OCR availability,
  template pack, permissions. Run this first.
* ``hop connect``          - connect wireless ADB and print device/panel info.
* ``hop capture``          - grab a labeled screen into the template pack (build
  the anchors the classifier needs).
* ``hop calibrate``        - measure panel + report rate and write them to your
  user config (the §10 calibration protocol's on-device steps).
* ``hop test-click``       - emit one humanized tap at a screen fraction, to
  confirm the transport actually delivers touches.
* ``hop run``              - run the hunt loop headless (Ctrl-C or a hotkey stops).
* ``hop dashboard``        - run the local web dashboard and the hunt from there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path
from random import Random

from . import __version__
from .adb import Adb, AdbError
from .alerts import Alerter
from .config import Config, clear_target_classes, load_config
from .debuglog import DebugLog, prune_runs
from .engine import Engine
from .geometry import PanelGeometry
from .hearthstone import GameLayout
from .hero_classes import parse_class
from .orientation import display_size
from .perception.ocr import ClassReader, tesseract_available
from .perception.screens import ScreenState, load_template_pack
from .transport import make_backend


def _default_pack_dir() -> Path:
    return Path.home() / ".config" / "hop" / "templates"


def _runs_root() -> Path:
    return Path.home() / ".config" / "hop" / "runs"


def _unknowns_dir() -> Path:
    """Where frames of screens the classifier could not name are kept.

    A *sibling* of ``runs/``, not a child, so ``keep_runs`` pruning can never retire
    the one capture worth keeping. Empty when the hunt is healthy.
    """
    return Path.home() / ".config" / "hop" / "unknowns"


def _default_run_dir(keep_runs: int | None = None) -> Path:
    """A fresh run dir, after retiring the oldest runs past ``keep_runs``.

    Pruning on the way *in* rather than on the way out: a run that halts hard or is
    killed never reaches its own cleanup, and those are exactly the runs that dump
    frames. ``None`` skips pruning (callers without a Config).
    """
    if keep_runs is not None:
        removed = prune_runs(_runs_root(), keep_runs)
        if removed:
            print(f"pruned {len(removed)} old run dir(s)", flush=True)
    return _runs_root() / time.strftime("%Y%m%d-%H%M%S")


# ── engine wiring shared by `run` and `dashboard` ───────────────────────────

def build_engine(cfg: Config, *, pack_dir: Path, debug_dir: Path | None = None,
                 alerter: Alerter | None = None, seed: int | None = None):
    """Connect, open the transport, load perception, and construct the Engine.

    Returns an engine-factory-compatible callable's product: a fully set-up
    Engine ready to ``run()``. Raises AdbError/RuntimeError on setup failure.
    """
    adb = Adb(cfg.device.adb_address)
    adb.connect()
    adb.ensure_connected()
    try:
        adb.stay_awake(True)
    except Exception:
        pass
    # measure_panel() is the NATIVE (portrait) panel - the UHID descriptor space.
    # Hearthstone runs landscape, so perception + the engine work in the rotated
    # DISPLAY geometry; the transport rotates display->native at emit.
    native = adb.measure_panel()
    display = _display_geometry(adb, native)

    backend = make_backend(cfg.device.touch_backend, adb, cfg.uhid)
    backend.open(native)   # descriptor in native space; backend rotates samples

    classifier = load_template_pack(pack_dir)
    reader = ClassReader(cfg.vision.ocr_max_edit_distance)
    debug = DebugLog(debug_dir, max_anomaly_frames=cfg.debug.max_anomaly_frames,
                     unknown_dir=_unknowns_dir(),
                     max_unknown_frames=cfg.debug.max_unknown_frames) if debug_dir else None

    return Engine(
        cfg, adb, backend, display, classifier, reader,
        alerts=alerter, debug=debug,
        rng=Random(seed) if seed is not None else Random(),
        layout=GameLayout(),
    )


def _display_geometry(adb, native: PanelGeometry) -> PanelGeometry:
    """The current on-screen (possibly rotated) geometry perception works in."""
    rotation = adb.get_rotation()
    dw, dh = display_size(native.width_px, native.height_px, rotation)
    return PanelGeometry(width_px=dw, height_px=dh, dpi=native.dpi)


def _apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Return a copy of cfg with criteria overrides from the dashboard applied."""
    from dataclasses import replace
    crit = cfg.criteria
    tgt = overrides.get("target_classes")
    if tgt is not None:
        crit = replace(crit, target_classes=tuple(parse_class(c) for c in tgt))
    if "require_second" in overrides:
        crit = replace(crit, require_second=bool(overrides["require_second"]))
    return replace(cfg, criteria=crit)


# ── commands ────────────────────────────────────────────────────────────────

def cmd_doctor(args) -> int:
    cfg = load_config(args.config)
    pack_dir = Path(args.templates or _default_pack_dir())
    print(f"hop {__version__} doctor\n" + "=" * 40)
    ok = True

    def check(label, cond, hint=""):
        nonlocal ok
        mark = "OK " if cond else "!! "
        print(f"[{mark}] {label}")
        if not cond and hint:
            print(f"        -> {hint}")
        ok = ok and cond

    addr = cfg.device.adb_address
    check(f"adb_address configured ({addr or 'MISSING'})", bool(addr),
          "set device.adb_address in your config (host:port from Wireless debugging)")
    adb = Adb(addr) if addr else None
    connected = False
    if adb:
        try:
            adb.connect()
            connected = adb.is_connected()
        except AdbError as e:
            print(f"        adb connect error: {e}")
    check("device connected", connected, "enable Wireless debugging and pair; check same LAN")

    if connected:
        try:
            panel = adb.measure_panel()
            print(f"        panel: {panel.width_px}x{panel.height_px} @ {panel.dpi:.0f} dpi "
                  f"({panel.px_per_mm:.1f} px/mm)")
        except Exception as e:
            check("panel measurable", False, str(e))
        try:
            png = adb.screencap_png()
            check(f"screencap works ({len(png)} bytes)", len(png) > 100)
        except Exception as e:
            check("screencap works", False, str(e))
        try:
            has_hid = adb.has_hid_tool()
            check("/system/bin/hid present (UHID transport)", has_hid,
                  "falls back to degraded `adb input`; set touch_backend=adb to silence")
        except Exception:
            check("/system/bin/hid present (UHID transport)", False)
        try:
            dump = adb.input_devices_dump()
            ident = parse_touch_identity(dump)
            if ident:
                print(f"        real panel identity: name={ident['name']!r} "
                      f"vendor=0x{ident['vendor']:04x} product=0x{ident['product']:04x} "
                      f"(bus=0x{ident['bus']:04x})")
                print(f"        -> clone into [uhid]: device_name = \"{ident['name']}\", "
                      f"vendor_id = 0x{ident['vendor']:04x}, product_id = 0x{ident['product']:04x}")
            else:
                print("        (could not parse a touchscreen identity from `dumpsys input`; "
                      "set [uhid] manually - see CALIBRATION.md §4)")
        except Exception:
            pass

    check("OCR engine (pytesseract) available", tesseract_available(),
          "pip install 'hop[vision]' OR provide label templates via `hop capture`")
    classifier = load_template_pack(pack_dir)
    check(f"screen template pack present ({pack_dir})", classifier.has_templates,
          "run `hop capture` for each screen (mulligan, victory, defeat, menu, ...)")

    print("-" * 40)
    print(f"posture={cfg.device.posture}  touch_backend={cfg.device.touch_backend}")
    print(f"criteria pass-rate estimate: {cfg.criteria.pass_rate_estimate()*100:.0f}%  "
          f"(concede-rate {100-cfg.criteria.pass_rate_estimate()*100:.0f}%)")
    print("READY" if ok else "NOT READY - resolve the !! items above")
    return 0 if ok else 1


def _device_summary(cfg: Config, pack_dir: Path) -> str:
    """A compact, best-effort device/anchor summary string for a bug report.

    Best-effort: each probe is guarded so an offline phone yields "(not connected)"
    rather than an exception. Returns Markdown lines.
    """
    out: list[str] = [f"- hop {__version__}",
                      f"- adb_address: `{cfg.device.adb_address or '(unset)'}`",
                      f"- touch_backend: {cfg.device.touch_backend}  posture: {cfg.device.posture}",
                      f"- OCR (pytesseract): {'available' if tesseract_available() else 'MISSING'}"]
    clf = load_template_pack(pack_dir)
    out.append(f"- anchors: {len(clf.anchors)} in {pack_dir}")
    addr = cfg.device.adb_address
    if addr:
        try:
            adb = Adb(addr)
            adb.connect()
            if adb.is_connected():
                panel = adb.measure_panel()
                out.append(f"- device: CONNECTED, panel {panel.width_px}x{panel.height_px} "
                           f"@ {panel.dpi:.0f} dpi, UHID tool: {'yes' if adb.has_hid_tool() else 'no'}")
                try:
                    c = load_template_pack(pack_dir).classify(_capture_frame(adb))
                    out.append(f"- current screen: {c.state.value} (confidence {c.confidence:.2f})")
                except Exception:
                    pass
            else:
                out.append("- device: NOT connected")
        except Exception as e:
            out.append(f"- device: probe failed ({e})")
    return "\n".join(out)


def _capture_frame(adb):
    from .perception.capture import Capturer
    return Capturer(adb).capture()


def cmd_bugreport(args) -> int:
    from . import bugreport as br

    cfg = load_config(args.config)
    pack_dir = Path(args.templates or _default_pack_dir())
    description = args.description
    if description is None:
        # Interactive: read a description from stdin so `hop bugreport` alone works.
        try:
            print("Describe what went wrong (end with Ctrl-D):", file=sys.stderr)
            description = sys.stdin.read()
        except KeyboardInterrupt:
            return 1

    paths = br.ReportPaths(
        config=Path(args.config) if args.config else (Path.home() / ".config" / "hop" / "config.toml"),
        runs_root=_runs_root(),
        unknowns_dir=_unknowns_dir(),
        app_log=Path.home() / "Library" / "Logs" / "hop.log",
        templates=pack_dir,
    )
    probe = None if args.no_device else (lambda: _device_summary(cfg, pack_dir))
    markdown = br.collect(description, paths, version=__version__, device_probe=probe)

    if args.out:                       # explicit file opt-in
        path = br.write_report(markdown, Path(args.out))
        print(f"bug report written -> {path}")
        print("Paste this file into Claude Code; it will self-improve the harness.")
    elif br.copy_to_clipboard(markdown):
        print("bug report copied to clipboard — paste it into Claude Code; it will self-improve the harness.")
    else:
        # no clipboard (e.g. over SSH): print it so the report is never lost
        print(markdown)
        print("(could not reach the clipboard; report printed above — copy it manually)", file=sys.stderr)
    return 0


def cmd_connect(args) -> int:
    cfg = load_config(args.config)
    addr = args.address or cfg.device.adb_address
    if not addr:
        print("no address; pass --address host:port or set device.adb_address", file=sys.stderr)
        return 2
    adb = Adb(addr)
    adb.connect()
    if not adb.is_connected():
        print(f"failed to connect to {addr}", file=sys.stderr)
        return 1
    panel = adb.measure_panel()
    print(f"connected: {addr}")
    print(f"panel: {panel.width_px}x{panel.height_px} @ {panel.dpi:.0f} dpi")
    print(f"UHID tool: {'yes' if adb.has_hid_tool() else 'no (will use adb input)'}")
    return 0


def _expand_box(box: list[float], margin: float = 0.35) -> list[float]:
    """Grow a glyph box into a search region by ``margin`` of its size, clamped."""
    xf, yf, wf, hf = box
    mx, my = wf * margin, hf * margin
    x0, y0 = max(0.0, xf - mx), max(0.0, yf - my)
    x1, y1 = min(1.0, xf + wf + mx), min(1.0, yf + hf + my)
    return [x0, y0, x1 - x0, y1 - y0]


def cmd_capture(args) -> int:
    cfg = load_config(args.config)
    pack_dir = Path(args.templates or _default_pack_dir())
    pack_dir.mkdir(parents=True, exist_ok=True)
    if args.from_file:
        # Rebuild an anchor offline from a saved frame (e.g. a <state>_full.png).
        png = Path(args.from_file).read_bytes()
    else:
        adb = Adb(cfg.device.adb_address)
        adb.connect()
        png = adb.screencap_png()

    state = args.state
    valid = [s.value for s in ScreenState if s != ScreenState.UNKNOWN]
    if state not in valid:
        print(f"--state must be one of: {', '.join(valid)}", file=sys.stderr)
        return 2

    # One screen state can wear several faces. Hearthstone's reward popup is a scroll
    # whose header ("Level 12 Reward!") and gold count change every time, and its ranked
    # medal screen is different again - but all dismiss identically, so they are one
    # STATE with several anchors. `--variant` names the face; classify() already accepts
    # any number of anchors per state and takes the best. Without it, behaviour is
    # exactly as before (one anchor per state, replaced on recapture).
    variant = getattr(args, "variant", None)
    suffix = f"_{variant}" if variant else ""

    # Always keep the full frame for reference/recalibration.
    (pack_dir / f"{state}{suffix}_full.png").write_bytes(png)

    # The template image MUST be smaller than its search region: best_match()
    # slides the template inside the region and bails when tw>rw. So crop the
    # anchor glyph out of the frame rather than storing the whole screen.
    img_name = f"{state}{suffix}.png"
    if args.glyph:
        glyph = [float(v) for v in args.glyph.split(",")]
        try:
            import io
            from PIL import Image
        except Exception:
            print("--glyph needs Pillow: pip install 'hop[vision]'", file=sys.stderr)
            return 2
        im = Image.open(io.BytesIO(png))
        gx, gy = int(glyph[0] * im.width), int(glyph[1] * im.height)
        gw, gh = int(glyph[2] * im.width), int(glyph[3] * im.height)
        im.crop((gx, gy, gx + gw, gy + gh)).save(pack_dir / img_name)
        region = [float(x) for x in args.region.split(",")] if args.region else _expand_box(glyph)
    else:
        (pack_dir / img_name).write_bytes(png)
        region = [float(x) for x in args.region.split(",")] if args.region else [0.0, 0.0, 1.0, 1.0]
        print("WARNING: no --glyph given, so the whole screen is the template. It can only "
              "match with a full-frame region (brittle). Prefer --glyph xf,yf,wf,hf.")

    meta_path = pack_dir / "screens.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"anchors": []}
    # Replace only the SAME (state, variant). A pre-existing entry has no "variant"
    # key -> None, so a plain `--state rewards` still replaces the old single anchor,
    # while `--state rewards --variant banner` coexists with it.
    meta["anchors"] = [a for a in meta["anchors"]
                       if not (a["state"] == state and a.get("variant") == variant)]
    entry = {
        "state": state, "image": img_name, "region": region,
        "threshold": args.threshold, "priority": args.priority,
    }
    if variant:
        entry["variant"] = variant
    meta["anchors"].append(entry)
    meta_path.write_text(json.dumps(meta, indent=2))
    label = f"{state}/{variant}" if variant else state
    print(f"saved {label} anchor -> {pack_dir/img_name} (search region {[round(v,4) for v in region]})")
    return 0


def cmd_calibrate(args) -> int:
    from .tomledit import upsert_toml_scalar

    cfg = load_config(args.config)
    adb = Adb(cfg.device.adb_address)
    adb.connect()
    panel = adb.measure_panel()
    rate = _measure_report_rate(adb, args.swipe_seconds) if args.report_rate else None
    if args.report_rate and rate is None:
        print("could not measure a report rate - did you swipe during the window?",
              file=sys.stderr)
        return 1

    user_path = Path(args.config) if args.config else (Path.home() / ".config" / "hop" / "config.toml")
    user_path.parent.mkdir(parents=True, exist_ok=True)
    text = user_path.read_text() if user_path.exists() else ""

    date = time.strftime("%Y-%m-%d")
    model = _device_model(adb)
    if rate:
        text = upsert_toml_scalar(
            text, "motor", "report_rate_hz", str(int(round(rate))),
            comment=f"LIVE-VERIFIED {date} ({model}): median SYN_REPORT interval",
        )
    text = upsert_toml_scalar(text, "calibration", "device", f'"{model}"')
    text = upsert_toml_scalar(text, "calibration", "date", f'"{date}"')
    user_path.write_text(text)

    print(f"calibration written to {user_path}")
    print(f"panel: {panel.width_px}x{panel.height_px} @ {panel.dpi:.0f} dpi"
          + (f", report_rate = {rate:.1f} Hz -> {int(round(rate))}" if rate else ""))
    print("Complete the remaining §10 steps (getevent pressure/dwell, InputDevice "
          "vid/pid via `hop doctor`) and record provenance.")
    return 0


def cmd_test_click(args) -> int:
    cfg = load_config(args.config)
    adb = Adb(cfg.device.adb_address)
    adb.connect()
    native = adb.measure_panel()
    display = _display_geometry(adb, native)
    backend = make_backend(cfg.device.touch_backend, adb, cfg.uhid)
    backend.open(native)   # descriptor native; backend rotates display->native
    try:
        from .humanize.contact import ContactModel
        from .humanize.motor import synth_tap
        from .humanize.state import HumanState
        xf, yf = (float(v) for v in args.at.split(","))
        # target is a DISPLAY-space fraction (what you see on the landscape screen)
        target = (xf * display.width_px, yf * display.height_px)
        g = synth_tap(Random(), target, 0.03 * display.width_px, display, cfg.motor,
                      ContactModel(cfg.contact), HumanState())
        print(f"emitting tap at display ({target[0]:.0f},{target[1]:.0f}) "
              f"[{display.width_px}x{display.height_px} rot={adb.get_rotation()}] via "
              f"{type(backend).__name__} (fidelity={backend.fidelity})")
        backend.emit(g)
        # Let the release dispatch before we tear the device down, otherwise a
        # single test tap can be canceled (the engine keeps the device open, so
        # this only matters for the one-shot test-click).
        time.sleep(0.4)
        print("done - watch the phone; if nothing happened, check UHID/SELinux or use touch_backend=adb")
    finally:
        backend.close()
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    if args.classes:
        from dataclasses import replace
        cfg = replace(cfg, criteria=replace(cfg.criteria,
                      target_classes=tuple(parse_class(c) for c in args.classes)))
    alerter = Alerter(cfg.alerts)
    pack_dir = Path(args.templates or _default_pack_dir())
    debug_dir = _default_run_dir(cfg.debug.keep_runs)
    print(f"run: criteria pass-rate ~{cfg.criteria.pass_rate_estimate()*100:.0f}%, "
          f"logs -> {debug_dir}")
    engine = build_engine(cfg, pack_dir=pack_dir, debug_dir=debug_dir, alerter=alerter,
                          seed=args.seed)
    _install_hotkeys(engine)
    try:
        stats = engine.run(max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        engine.request_stop()
        stats = engine.stats
    print(f"stopped: reason={stats.stop_reason or 'user'} games={stats.games} "
          f"concedes={stats.concedes} target_found={stats.target_found}")
    return 0


def _load_config_clean_targets(config_path) -> Config:
    """Load the config for an app launch with *nothing selected*.

    Target classes are session-only (see :func:`hop.config.clear_target_classes`): the
    app starts each launch with an empty selection and you re-pick. We wipe the persisted
    picks on disk AND drop them from the in-memory config the UIs build their checkboxes
    from, so neither surface shows a stale class carried over from last time.
    """
    from dataclasses import replace

    cfg = load_config(config_path)
    if cfg.criteria.target_classes:
        clear_target_classes(config_path)
        cfg = replace(cfg, criteria=replace(cfg.criteria, target_classes=()))
    return cfg


def cmd_app(args) -> int:
    """The Mac control panel: a menu-bar item that drives the hunt loop."""
    from .macapp import MacAppUnavailable, run_menubar

    cfg = _load_config_clean_targets(args.config)
    alerter = Alerter(cfg.alerts)
    pack_dir = Path(args.templates or _default_pack_dir())

    def factory(overrides: dict) -> Engine:
        c = _apply_overrides(cfg, overrides)
        return build_engine(c, pack_dir=pack_dir, debug_dir=_default_run_dir(c.debug.keep_runs),
                            alerter=alerter, seed=args.seed)

    try:
        return run_menubar(cfg, factory, alerter=alerter, config_path=args.config)
    except MacAppUnavailable as e:
        print(e, file=sys.stderr)
        return 2


def cmd_dashboard(args) -> int:
    cfg = _load_config_clean_targets(args.config)
    alerter = Alerter(cfg.alerts)
    pack_dir = Path(args.templates or _default_pack_dir())
    from .observed import ObservedDistribution, default_path
    from .runner import EngineController
    from .webui import DashboardServer

    def factory(overrides: dict) -> Engine:
        c = _apply_overrides(cfg, overrides)
        return build_engine(c, pack_dir=pack_dir, debug_dir=_default_run_dir(c.debug.keep_runs),
                            alerter=alerter, seed=args.seed)

    # day-scoped observed-class tally: continues within a day, resets on a new one.
    observed = ObservedDistribution.load(default_path(args.config))
    controller = EngineController(factory, alerter=alerter, observed=observed)
    server = DashboardServer(controller, cfg, host=args.host, port=args.port,
                             config_path=args.config)
    url = f"http://{args.host}:{args.port}/"
    print(f"dashboard: {url}  (Ctrl-C to quit)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        controller.stop()
        controller.flush_observed()   # save the day's tally before we exit
        server.shutdown()
    return 0


# ── helpers ─────────────────────────────────────────────────────────────────

def _install_hotkeys(engine: Engine) -> None:
    """Global stop/panic hotkeys via pynput, if available (optional dep)."""
    try:
        from pynput import keyboard
    except Exception:
        print("(pynput not installed; use Ctrl-C to stop. `pip install hop[hotkeys]` for F12 panic)")
        return

    def on_press(key):
        if key == keyboard.Key.f12:
            print("\n[panic] stop requested")
            engine.request_stop()

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()
    print("(hotkey: F12 = panic stop)")


def _measure_report_rate(adb, swipe_seconds: int = 20) -> float | None:
    """Measure the panel's touch report rate; needs a human swipe in the window.

    Captures from the touchscreen node **only** (a phone's fingerprint reader and
    haptics also emit input events) and hands the raw text to the pure estimator
    in :mod:`hop.calibrate`, which counts ``SYN_REPORT`` frames rather than
    position axes. See that module for why the distinction matters.
    """
    from .calibrate import estimate_report_rate_hz, parse_touch_event_node

    node = parse_touch_event_node(adb.shell("getevent -pl 2>/dev/null"))
    if not node:
        print("could not identify the touchscreen input node", file=sys.stderr)
        return None
    print(f"sampling {node} for {swipe_seconds}s - SWIPE ON THE PHONE NOW "
          "(a few natural drags)...", flush=True)
    prev_timeout, adb.timeout = adb.timeout, swipe_seconds + 15
    try:
        # `timeout` exits 124 when it fires, which is the *normal* end of this
        # capture - but adb propagates that exit code and Adb._run raises on any
        # non-zero status, which would discard a perfectly good sample. Swallow
        # the status on-device so the events still come back on stdout.
        out = adb.shell(f"timeout {swipe_seconds} getevent -lt {node} || true")
    except AdbError as e:
        print(f"getevent capture failed: {e}", file=sys.stderr)
        return None
    finally:
        adb.timeout = prev_timeout

    frames = sum(1 for line in out.splitlines() if "SYN_REPORT" in line)
    print(f"captured {frames} touch report frames", flush=True)
    return estimate_report_rate_hz(out)


def _device_model(adb) -> str:
    try:
        return adb.shell("getprop ro.product.model").strip() or "unknown"
    except Exception:
        return "unknown"


def parse_touch_identity(dump: str) -> dict | None:
    """Parse ``dumpsys input`` for the touchscreen's InputDevice identity.

    Used to panel-match the virtual UHID digitizer (CALIBRATION.md §4). Modern
    Android (11+) prints, under each device block::

        2: goodix_ts0
          Classes: KEYBOARD | TOUCH | TOUCH_MT
          ...
          Identifier: bus=0x0001, vendor=0x27c6, product=0x0100, version=0x0100, ...

    so we walk blocks (headed by ``<n>: <name>``), remember the block's name and
    Classes, and read the ``Identifier:`` line only for blocks whose Classes
    include TOUCH. Among touch devices we prefer a real multitouch panel
    (TOUCH_MT, non-zero vendor, not a fingerprint sensor). Returns a dict with
    ``name`` and int ``vendor``/``product``/``version``/``bus`` (from the
    ``0x``-prefixed fields), or ``None`` if no touchscreen identity is found.

    Falls back to the legacy single-line ``Vendor: 0x.. Product: 0x..`` format
    if present, so it keeps working on older Android too.
    """
    import re

    header = re.compile(r"^\s+(\d+):\s+(\S.*)$")
    candidates: list[dict] = []
    name = None
    classes = ""
    for line in dump.splitlines():
        m = header.match(line)
        if m:
            name, classes = m.group(2).strip(), ""
            continue
        stripped = line.strip()
        if stripped.startswith("Classes:"):
            classes = stripped
        elif stripped.startswith("Identifier:") and "TOUCH" in classes:
            fields: dict[str, int] = {}
            for part in stripped[len("Identifier:"):].split(","):
                if "=" in part:
                    k, _, v = part.partition("=")
                    v = v.strip()
                    if v.startswith("0x"):
                        try:
                            fields[k.strip()] = int(v, 16)
                        except ValueError:
                            pass
            if {"vendor", "product"} <= fields.keys():
                candidates.append({
                    "name": name or "touchscreen",
                    "vendor": fields.get("vendor", 0),
                    "product": fields.get("product", 0),
                    "version": fields.get("version", 0),
                    "bus": fields.get("bus", 0),
                    "multitouch": "TOUCH_MT" in classes,
                })

    if not candidates:
        # legacy fallback: `... Vendor: 0xNNNN Product: 0xNNNN ...` on one line
        pat = re.compile(r"Vendor:\s*(0x[0-9a-fA-F]+).*Product:\s*(0x[0-9a-fA-F]+)")
        for line in dump.splitlines():
            m = pat.search(line)
            if m:
                return {"name": "touchscreen", "vendor": int(m.group(1), 16),
                        "product": int(m.group(2), 16), "version": 0, "bus": 0,
                        "multitouch": False}
        return None

    def rank(c: dict) -> tuple:
        return (
            c["multitouch"],
            c["vendor"] != 0,
            "fingerprint" not in c["name"].lower(),
        )

    return max(candidates, key=rank)


# ── argparse ────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hop", description="Hearthstone Opponent Picker (Android/wireless-ADB)")
    p.add_argument("--version", action="version", version=f"hop {__version__}")
    p.add_argument("--config", help="path to a user config.toml (defaults to ~/.config/hop/config.toml)")
    p.add_argument("--templates", help="template pack directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor", help="preflight checks"); sp.set_defaults(func=cmd_doctor)
    sp = sub.add_parser("connect", help="connect wireless ADB")
    sp.add_argument("--address", help="host:port"); sp.set_defaults(func=cmd_connect)

    sp = sub.add_parser("capture", help="save a labeled screen into the template pack")
    sp.add_argument("--state", required=True, help="screen state name (mulligan, victory, ...)")
    sp.add_argument("--variant", help="name a second visual face of the same state (e.g. "
                    "'banner' for the reward-scroll popup vs the ranked medal). Coexists "
                    "with other variants; omit to keep one anchor per state.")
    sp.add_argument("--glyph", help="crop box of the anchor glyph 'xf,yf,wf,hf' (strongly recommended)")
    sp.add_argument("--region", help="anchor search region 'xf,yf,wf,hf' (default: glyph box + margin)")
    sp.add_argument("--from-file", dest="from_file",
                    help="build the anchor from a saved PNG instead of a live screencap")
    sp.add_argument("--threshold", type=float, default=0.72)
    sp.add_argument("--priority", type=int, default=0,
                    help="higher wins when several anchors match (modal dialogs > screens)")
    sp.set_defaults(func=cmd_capture)

    sp = sub.add_parser("calibrate", help="measure panel/report-rate into user config")
    sp.add_argument("--report-rate", action="store_true", help="measure touch report rate (swipe during window)")
    sp.add_argument("--swipe-seconds", type=int, default=20,
                    help="length of the sample window during which you swipe (default 20)")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("test-click", help="emit one humanized tap to test the transport")
    sp.add_argument("--at", default="0.5,0.5", help="screen fraction 'xf,yf'")
    sp.set_defaults(func=cmd_test_click)

    sp = sub.add_parser("run", help="run the hunt loop (headless)")
    sp.add_argument("--classes", nargs="*", help="target classes (overrides config)")
    sp.add_argument("--max-iterations", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None, help="deterministic RNG seed (testing)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("app", help="Mac menu-bar control panel (criteria, start/stop, alerts)")
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=cmd_app)

    sp = sub.add_parser("dashboard", help="run the local web dashboard")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("bugreport", help="copy a self-improving bug report (logs + state) to the clipboard")
    sp.add_argument("--description", "--desc", dest="description",
                    help="what went wrong (omit to type it interactively)")
    sp.add_argument("--out", help="write to this directory instead of copying to the clipboard")
    sp.add_argument("--no-device", action="store_true",
                    help="skip the live device probe (faster; use when the phone is offline)")
    sp.set_defaults(func=cmd_bugreport)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except AdbError as e:
        print(f"adb error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
