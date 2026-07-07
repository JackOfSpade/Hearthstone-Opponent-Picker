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
from .config import Config, load_config, profile_multipliers
from .debuglog import DebugLog
from .engine import Engine
from .hearthstone import GameLayout
from .hero_classes import parse_class
from .perception.ocr import ClassReader, tesseract_available
from .perception.screens import ScreenState, load_template_pack
from .transport import make_backend


def _default_pack_dir() -> Path:
    return Path.home() / ".config" / "hop" / "templates"


def _default_run_dir() -> Path:
    return Path.home() / ".config" / "hop" / "runs" / time.strftime("%Y%m%d-%H%M%S")


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
    panel = adb.measure_panel()

    backend = make_backend(cfg.device.touch_backend, adb, cfg.uhid)
    backend.open(panel)

    classifier = load_template_pack(pack_dir)
    reader = ClassReader(cfg.vision.ocr_max_edit_distance)
    debug = DebugLog(debug_dir) if debug_dir else None

    return Engine(
        cfg, adb, backend, panel, classifier, reader,
        alerts=alerter, debug=debug,
        rng=Random(seed) if seed is not None else Random(),
        layout=GameLayout(),
    )


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
            hint = _extract_panel_identity(dump)
            if hint:
                print(f"        real panel identity (clone into [uhid]): {hint}")
        except Exception:
            pass

    check("OCR engine (pytesseract) available", tesseract_available(),
          "pip install 'hop[vision]' OR provide label templates via `hop capture`")
    classifier = load_template_pack(pack_dir)
    check(f"screen template pack present ({pack_dir})", classifier.has_templates,
          "run `hop capture` for each screen (mulligan, victory, defeat, menu, ...)")

    print("-" * 40)
    print(f"posture={cfg.device.posture}  touch_backend={cfg.device.touch_backend}  "
          f"risk={cfg.risk_profile}")
    print(f"criteria pass-rate estimate: {cfg.criteria.pass_rate_estimate()*100:.0f}%  "
          f"(concede-rate {100-cfg.criteria.pass_rate_estimate()*100:.0f}%)")
    print("READY" if ok else "NOT READY - resolve the !! items above")
    return 0 if ok else 1


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


def cmd_capture(args) -> int:
    cfg = load_config(args.config)
    pack_dir = Path(args.templates or _default_pack_dir())
    pack_dir.mkdir(parents=True, exist_ok=True)
    adb = Adb(cfg.device.adb_address)
    adb.connect()
    png = adb.screencap_png()

    state = args.state
    valid = [s.value for s in ScreenState if s != ScreenState.UNKNOWN]
    if state not in valid:
        print(f"--state must be one of: {', '.join(valid)}", file=sys.stderr)
        return 2
    img_name = f"{state}.png"
    (pack_dir / img_name).write_bytes(png)

    meta_path = pack_dir / "screens.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"anchors": []}
    meta["anchors"] = [a for a in meta["anchors"] if a["state"] != state]
    region = [float(x) for x in args.region.split(",")] if args.region else [0.3, 0.02, 0.4, 0.12]
    meta["anchors"].append({
        "state": state, "image": img_name, "region": region,
        "threshold": args.threshold,
    })
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved {state} anchor -> {pack_dir/img_name} (region {region})")
    print("NOTE: crop the saved PNG to just the anchor glyph for a tight template, "
          "or leave full-screen and rely on the region filter.")
    return 0


def cmd_calibrate(args) -> int:
    cfg = load_config(args.config)
    adb = Adb(cfg.device.adb_address)
    adb.connect()
    panel = adb.measure_panel()
    rate = _measure_report_rate(adb) if args.report_rate else None

    user_path = Path(args.config) if args.config else (Path.home() / ".config" / "hop" / "config.toml")
    user_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# written by `hop calibrate`",
        "[motor]",
        f"# panel {panel.width_px}x{panel.height_px} @ {panel.dpi:.0f} dpi",
    ]
    if rate:
        lines.append(f"report_rate_hz = {rate}")
    lines += [
        "[calibration]",
        f'device = "{_device_model(adb)}"',
        f'date = "{time.strftime("%Y-%m-%d")}"',
    ]
    existing = user_path.read_text() if user_path.exists() else ""
    user_path.write_text(existing + "\n" + "\n".join(lines) + "\n")
    print(f"calibration appended to {user_path}")
    print(f"panel: {panel.width_px}x{panel.height_px} @ {panel.dpi:.0f} dpi"
          + (f", report_rate ~= {rate} Hz" if rate else ""))
    print("Complete the remaining §10 steps (getevent pressure/dwell, InputDevice "
          "vid/pid via `hop doctor`) and record provenance.")
    return 0


def cmd_test_click(args) -> int:
    cfg = load_config(args.config)
    adb = Adb(cfg.device.adb_address)
    adb.connect()
    panel = adb.measure_panel()
    backend = make_backend(cfg.device.touch_backend, adb, cfg.uhid)
    backend.open(panel)
    try:
        from .humanize.contact import ContactModel
        from .humanize.motor import synth_tap
        from .humanize.state import HumanState
        xf, yf = (float(v) for v in args.at.split(","))
        target = (xf * panel.width_px, yf * panel.height_px)
        g = synth_tap(Random(), target, 0.03 * panel.width_px, panel, cfg.motor,
                      ContactModel(cfg.contact), HumanState())
        print(f"emitting tap at ({target[0]:.0f},{target[1]:.0f}) via "
              f"{type(backend).__name__} (fidelity={backend.fidelity})")
        backend.emit(g)
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
    debug_dir = _default_run_dir()
    print(f"run: criteria pass-rate ~{cfg.criteria.pass_rate_estimate()*100:.0f}%, "
          f"risk={cfg.risk_profile}, logs -> {debug_dir}")
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


def cmd_dashboard(args) -> int:
    cfg = load_config(args.config)
    alerter = Alerter(cfg.alerts)
    pack_dir = Path(args.templates or _default_pack_dir())
    from .runner import EngineController
    from .webui import DashboardServer

    def factory(overrides: dict) -> Engine:
        c = _apply_overrides(cfg, overrides)
        return build_engine(c, pack_dir=pack_dir, debug_dir=_default_run_dir(),
                            alerter=alerter, seed=args.seed)

    controller = EngineController(factory, alerter=alerter)
    server = DashboardServer(controller, cfg, host=args.host, port=args.port)
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


def _measure_report_rate(adb) -> int | None:
    """Best-effort touch report-rate measurement via getevent timing.

    Requires the human to swipe during the sample window. Returns Hz or None.
    """
    try:
        out = adb.shell("getevent -lt -c 200 2>/dev/null | head -200")
        times = []
        for line in out.splitlines():
            if "ABS_MT_POSITION" in line and "[" in line:
                try:
                    times.append(float(line.split("[", 1)[1].split("]", 1)[0]))
                except Exception:
                    pass
        if len(times) > 10:
            span = times[-1] - times[0]
            if span > 0:
                return int(round((len(times) - 1) / span))
    except Exception:
        pass
    return None


def _device_model(adb) -> str:
    try:
        return adb.shell("getprop ro.product.model").strip() or "unknown"
    except Exception:
        return "unknown"


def _extract_panel_identity(dump: str) -> str:
    """Pull a touchscreen's Vendor/Product from `dumpsys input` for uhid cloning."""
    ln = [l for l in dump.splitlines() if "Vendor" in l and "Product" in l]
    return ln[0].strip() if ln else ""


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
    sp.add_argument("--region", help="anchor search region 'xf,yf,wf,hf'")
    sp.add_argument("--threshold", type=float, default=0.72)
    sp.set_defaults(func=cmd_capture)

    sp = sub.add_parser("calibrate", help="measure panel/report-rate into user config")
    sp.add_argument("--report-rate", action="store_true", help="measure touch report rate (swipe during window)")
    sp.set_defaults(func=cmd_calibrate)

    sp = sub.add_parser("test-click", help="emit one humanized tap to test the transport")
    sp.add_argument("--at", default="0.5,0.5", help="screen fraction 'xf,yf'")
    sp.set_defaults(func=cmd_test_click)

    sp = sub.add_parser("run", help="run the hunt loop (headless)")
    sp.add_argument("--classes", nargs="*", help="target classes (overrides config)")
    sp.add_argument("--max-iterations", type=int, default=None)
    sp.add_argument("--seed", type=int, default=None, help="deterministic RNG seed (testing)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("dashboard", help="run the local web dashboard")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--no-browser", action="store_true")
    sp.add_argument("--seed", type=int, default=None)
    sp.set_defaults(func=cmd_dashboard)
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
