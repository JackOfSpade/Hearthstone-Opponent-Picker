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
from .geometry import PanelGeometry
from .hearthstone import GameLayout
from .hero_classes import parse_class
from .orientation import display_size
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
    # measure_panel() is the NATIVE (portrait) panel - the UHID descriptor space.
    # Hearthstone runs landscape, so perception + the engine work in the rotated
    # DISPLAY geometry; the transport rotates display->native at emit.
    native = adb.measure_panel()
    display = _display_geometry(adb, native)

    backend = make_backend(cfg.device.touch_backend, adb, cfg.uhid)
    backend.open(native)   # descriptor in native space; backend rotates samples

    classifier = load_template_pack(pack_dir)
    reader = ClassReader(cfg.vision.ocr_max_edit_distance)
    debug = DebugLog(debug_dir) if debug_dir else None

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
