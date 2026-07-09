# hop — Hearthstone Opponent Picker (v2)

Filter who you face on ladder. `hop` watches Hearthstone's **mulligan screen on
a physical Android phone** over wireless ADB, reads the opponent's **class** and
whether **you go second**, and then either:

- **alerts you** (Mac notification + looping sound + optional phone push) when
  your target matchup appears — and stops touching the game so you can play it, or
- **concedes** a rejected matchup the way an impatient human would (never an
  instant concede) and re-queues.

It reproduces the two features of the original AHK scripts — *only face chosen
classes* and *only play when going second* — rebuilt from scratch to be robust,
resolution-independent, and, crucially, **paced so it doesn't look like a
barcode account**.

> **This is the desktop-free, Android rebuild.** The old `Archive/` AHK scripts
> targeted desktop Hearthstone via fragile OCR + fixed pixel clicks. This
> version drives a phone from your Mac and reads the game's own on-screen class
> text instead of OCR-guessing hero portraits.

---

## How it works (and why it's robust)

| Signal | How it's read | Why it's reliable |
|---|---|---|
| **Opponent class** | OCR the bottom-left class label at mulligan, snap to the 11 known class words by edit distance | The game prints the literal class word — immune to hero skins, multi-class skins, and golden-hero animation (those only change the portrait). Snapping to 11 fixed words makes OCR errors self-correcting. |
| **Going second** | Count mulligan cards: **3 = first, 4 = second** | The Coin isn't in the mulligan hand and coin skins vary, so we never look for it. Cards are counted as the **interiors between their green keep-glows** — spans that are both card-wide (0.147 W) and card-bright (the background gap between cards is 2.7× narrower and 3× darker). The glow is UI chrome, so **card artwork cannot fake it** — unlike the mana gems (a gem's own digit splits it, and a card's blue sky is the same hue: `b-g=21` vs `22`) or a brightness profile (dark art hides a card entirely). Both were tried; both returned **3 on a real 4-card hand**; both had passing *synthetic* tests. Counting the glow *strips* fails too — 3 spread-out cards make 6 strips while 4 packed cards make 5. Anything that doesn't cohere returns **unusable** and the engine fails closed rather than guess at the signal that decides whether to concede. |
| **Which screen we're on** | Template-match anchor glyphs (the "Starting Hand" banner, victory/defeat marks, the gear menu) | Closed-loop navigation: an unrecognized screen **halts and alerts** instead of blind-tapping. Hearthstone animates between screens, so a frame that matches nothing is re-looked at (never tapped) a bounded number of times before failing closed. Overlays carry a higher anchor `priority` so they beat the screen they occlude — the concede menu is drawn *over* the board, so without that the loop would wait instead of conceding. |

**No cloud API, no per-run cost, no ML training.** OCR runs locally
(Tesseract); everything else is pixel math and ADB. The only Claude usage was
building it.

## The humanization layer

`hop` is built to the **Android App Automation Humanization Standard (Rev 2)**
(see the sibling repo `Android-App-Automation-Humanization`). Every tap is a
projection of a latent *HumanState* rather than independent randomness:

- **Motor (L3):** FFitts target acquisition, minimum-jerk primary stroke + corrective submovements, correlated tremor, beta-ramp pressure with co-evolving contact size/orientation, and a lognormal tap dwell — delivered per-sample.
- **Transport (L1):** a **persistent virtual HID digitizer** via `/system/bin/hid` over `/dev/uhid` (carries real pressure/size/geometry through the genuine kernel input pipeline; registered **once** per session, never per gesture). It clones the panel's **name, vid/pid *and its contact-channel axis ranges*** — Android loads a touch device's calibration by name and applies it to our reports, so declaring a `TOUCH_MAJOR` ceiling of 255 against the panel's real 2399 sizes the contact ~6× too big. Hearthstone silently ignored every mulligan-card tap until that was fixed. Falls back to `adb input` with an explicit fidelity-drop record.
- **Orientation:** Hearthstone runs **landscape** while the phone's panel is native **portrait**, so perception and the engine work in *display* space while the digitizer reports *native panel* pixels. `hop.orientation` maps between them from the live rotation, handling **both** landscape orientations (a phone can sit either way up).
- **Timing (L4):** stateful, per-decision think time (a committing action reacts faster than a rejecting one), modulated by fatigue/familiarity/urgency/confidence. No bare `sleep(constant)` on any game-facing action.
- **Sensorimotor (L4):** predicts the coherent inertial side effect of each touch; a **desk-mounted** run declares its posture and does not claim handheld IMU realism (the honest option for a bench phone driven over ADB, per the standard).
- **Non-repetition (L5):** every gesture is fingerprinted and a near-duplicate of a prior trajectory is resampled *before* it is emitted, so replays and repeated runs never reuse an identical path. (The per-run/-session/-day volume caps and mandatory breaks that used to live here were removed by request — a hunt runs until it finds a target, an error halts it, or you stop it; pacing and volume are yours to manage.)
- **Verify + fail-closed (L6):** after every tap the screen must change *and* cohere; otherwise one evidence-based correction, then a clean halt with a debug snapshot. Verification is **scoped to what the tap aimed at** where that's the honest question — marking one mulligan card moves the whole frame by 4.8 (under the 9.0 threshold) and the card's own rectangle by 24.2. And fail-closed means *unknown state*: a card the game declines to toggle leaves us squarely on the mulligan, so the loop retries a bounded number of times and keeps the card rather than halting.

## Anti-barcode design

Blizzard demonstrably tracks rapid-concede patterns server-side (the Arena
"barcode pool"). `hop`'s defense is behavioral, not technical:

1. **Never insta-concede.** A rejected game performs the mulligan, enters the
   game, plays a beat or two, and concedes at a *randomly chosen* point. Detect
   time is decoupled from concede time.
2. **You own pacing.** There are deliberately no volume, session, or time caps
   and no forced breaks — a hunt runs until a target, an error, or you stop it.
3. **Randomized requeue.** Humans don't hit Play 400 ms after the defeat banner.
4. **A risk meter** in the dashboard: it estimates your concede rate from your
   criteria and flags barcode-shaped setups (one class + require-second ≈ 95%
   concedes). Widening criteria is both safer *and* faster to hit a target.

All of the above is tuned for a single "normal user" — one persistent, unhurried
identity, not a set of selectable aggression levels.
**Honest limit:** this shapes your pattern toward "impatient human"; it cannot
make a high concede rate statistically invisible. With the caps removed, volume
discipline is the real protection — and it is now yours to keep.

---

## Setup

### 1. Phone (one-time)
- Enable **Wireless debugging** (Android 11+: Developer Options → Wireless
  debugging → pair with a code). Older Android: one USB session, then
  `adb tcpip 5555`.
- Keep it on the **same Wi-Fi/LAN** as your Mac, plugged into a charger.
- Install the **ntfy** app (free, no account) and subscribe to a topic if you
  want loud phone alerts.

### 2. Mac
```sh
brew install android-platform-tools            # adb
brew install tesseract                         # OCR engine (or use zero-ML templates)
pip install -e '.[all]'                        # hop + vision + web + hotkeys + menu-bar app
```

### 3. Configure
Copy `config.example.toml` to `~/.config/hop/config.toml` and set at least
`device.adb_address` and your `[criteria]`.

### 4. Preflight + calibrate
```sh
hop connect --address 192.168.1.50:5555        # verify the link
hop doctor                                     # end-to-end checks + panel identity
hop capture --state mulligan                   # grab anchor templates (repeat per screen)
hop calibrate --report-rate                    # measure panel + touch rate -> user config
hop test-click --at 0.5,0.9                    # confirm the transport delivers a tap
```
See **[CALIBRATION.md](CALIBRATION.md)** for the full §10 protocol — this is a
precondition for trusting auto mode.

### 5. Run
Open Hearthstone, choose **Play**, and pick your deck so the big **Play** button
is on screen. That deck-detail screen is the hunt loop's home state — it's what
Hearthstone returns to after every game, so it's where `hop` queues and requeues
from. (Starting from the main menu halts with guidance instead of blind-tapping.)

```sh
hop app                                        # Mac menu-bar control panel
hop run --classes mage warlock                 # headless; Ctrl-C / F12 to stop
hop dashboard                                  # or drive it from the web UI
```

### The Mac control panel

`hop app` puts a status item in the menu bar — because the point of `hop` is that
you *aren't* watching it:

```
hop ▶                     ← idle "hop", hunting "hop ▶", target "hop ●", halted "hop ⚠"
├─ Hunting · 3 games · 2/15 concedes
├─ Concede rate ≈91%  ·  risk: HIGH — barcode-shaped
├─ Start hunting / Stop / Silence alarm
├─ Target classes  ▸  ✓ Mage   ✓ Warlock   Druid  …     (empty = any class)
├─ ☐ Only when going 2nd
└─ Open dashboard…                     ← live screen, stats, risk meter
```

Criteria you set here are **persisted** to `~/.config/hop/config.toml` (comments and
provenance notes intact), so they survive a restart. It drives the engine in-process
— no second process, no HTTP hop — and the web dashboard is one click away for the
rich panel. Needs `pip install 'hop[mac]'` (pyobjc); everything else works without it.

---

## Commands

| Command | Purpose |
|---|---|
| `hop doctor` | Preflight: adb, screencap, UHID tool, panel identity, OCR, templates. |
| `hop connect` | Connect wireless ADB, print device/panel info. |
| `hop capture --state <s> --glyph <box>` | Save a labeled anchor into the template pack. **Always pass `--glyph`** — the template must be smaller than its search region or it can never match. `--priority` makes modal dialogs win; `--from-file` rebuilds an anchor offline. |
| `hop calibrate` | Measure panel + report rate into your user config (§10). |
| `hop test-click --at xf,yf` | Emit one humanized tap to verify the transport. |
| `hop run` | Run the hunt loop headless. |
| `hop app` | Mac menu-bar control panel: criteria, start/stop, live status, alarm. |
| `hop dashboard` | Local web dashboard (criteria, live view, stats, risk meter). |

## Configuration

All settings live in TOML. The shipped `hop/config.default.toml` documents every
key, including the §8 humanization priors (with `LIVE-VERIFY` markers on values
you must measure on your phone). Your `~/.config/hop/config.toml` overrides only
what you change. The user-facing knobs:

```toml
[criteria]
target_classes = ["MAGE", "WARLOCK"]  # empty = any class
require_second = false                # keep only when we draw 4 mulligan cards
mode = "casual"                       # avoid cratering ranked MMR while hunting

[device]
adb_address = "192.168.1.50:5555"
touch_backend = "auto"                # auto | uhid | adb
posture = "desk_mounted"              # handheld | desk_mounted

[alerts]
mac_sound = true
ntfy_topic = "my-hop-alerts"          # optional phone push
```

---

## Repo layout

```
hop/
  humanize/     L3-L5: motor, contact, scroll, timing, sensorimotor, state, journey, limiter
  transport/    L1: uhid (persistent HID digitizer), adb_input (fallback), hid_descriptor
  perception/   L2: image, capture, templates, diffing, ocr, screens
  verify.py     L6: verification + coherence gates + fail-closed
  debuglog.py   L6: silent per-run journal + anomaly snapshots
  orientation.py  display<->native panel rotation (HS is landscape, panel is portrait)
  hearthstone.py  HS screens, class-label region, mulligan card counting, coordinates
  engine.py     the hunt-loop state machine
  alerts.py     Mac notification/sound/say + ntfy push
  webui/        stdlib web dashboard
  macapp.py     Mac menu-bar control panel (pyobjc NSStatusItem)
  calibrate.py  pure §10 parsers (getevent -> report rate)
  tomledit.py   comment-preserving TOML upsert (config writes)
  cli.py        connect / doctor / capture / calibrate / test-click / run / app / dashboard
tests/          pytest suite for the pure layers + engine integration
Archive/        the original desktop AHK scripts (reference only)
```

## Risks & Terms of Use

Automating the Hearthstone client **violates Blizzard's Terms of Use** — the
same exposure as the original AHK setup. `hop` reads pixels and taps the UI; it
does **not** read game memory, inspect packets, or root the phone. The dominant
risk is behavioral (your concede-rate pattern), reduced but not eliminated by
the anti-barcode layer. Use an account you can afford to lose, and keep the
volume human. This project is provided for research and personal use; you assume
the risk.
