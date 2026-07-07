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
| **Going second** | Count mulligan cards: **3 = first, 4 = second** | The Coin isn't in the mulligan hand and coin skins vary, so we never look for it. Card counting is pure pixel math (mana-gem count / brightness peak-count). |
| **Which screen we're on** | Template-match anchor glyphs (the "Starting Hand" banner, victory/defeat marks, the gear menu) | Closed-loop navigation: an unrecognized screen **halts and alerts** instead of blind-tapping. |

**No cloud API, no per-run cost, no ML training.** OCR runs locally
(Tesseract); everything else is pixel math and ADB. The only Claude usage was
building it.

## The humanization layer

`hop` is built to the **Android App Automation Humanization Standard (Rev 2)**
(see the sibling repo `Android-App-Automation-Humanization`). Every tap is a
projection of a latent *HumanState* rather than independent randomness:

- **Motor (L3):** FFitts target acquisition, minimum-jerk primary stroke + corrective submovements, correlated tremor, beta-ramp pressure with co-evolving contact size/orientation, and a lognormal tap dwell — delivered per-sample.
- **Transport (L1):** a **persistent virtual HID digitizer** via `/system/bin/hid` over `/dev/uhid` (carries real pressure/size/geometry through the genuine kernel input pipeline; registered **once** per session, never per gesture). Falls back to `adb input` with an explicit fidelity-drop record.
- **Timing (L4):** stateful, per-decision think time (a committing action reacts faster than a rejecting one), modulated by fatigue/familiarity/urgency/confidence. No bare `sleep(constant)` on any game-facing action.
- **Sensorimotor (L4):** predicts the coherent inertial side effect of each touch; a **desk-mounted** run declares its posture and does not claim handheld IMU realism (the honest option for a bench phone driven over ADB, per the standard).
- **Behavioral caps (L5):** per-run/-session/-day volume caps, a dedicated cap on the **committing action** (the concede — the barcode signal), mandatory jittered breaks, and a non-repetition check so trajectories never replay.
- **Verify + fail-closed (L6):** after every tap the screen must change *and* cohere; otherwise one evidence-based correction, then a clean halt with a debug snapshot.

## Anti-barcode design

Blizzard demonstrably tracks rapid-concede patterns server-side (the Arena
"barcode pool"). `hop`'s defense is behavioral, not technical:

1. **Never insta-concede.** A rejected game performs the mulligan, enters the
   game, plays a beat or two, and concedes at a *randomly chosen* point. Detect
   time is decoupled from concede time.
2. **Session shaping.** Concede caps, mandatory breaks, session ceilings, no
   24/7 running.
3. **Randomized requeue.** Humans don't hit Play 400 ms after the defeat banner.
4. **A risk meter** in the dashboard: it estimates your concede rate from your
   criteria and flags barcode-shaped setups (one class + require-second ≈ 95%
   concedes). Widening criteria is both safer *and* faster to hit a target.

The `cautious | balanced | aggressive` risk profile scales all of the above.
**Honest limit:** this shapes your pattern toward "impatient human"; it cannot
make a high concede rate statistically invisible. Volume discipline is the real
protection.

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
pip install -e '.[all]'                        # hop + vision + web + hotkeys
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
```sh
hop run --classes mage warlock                 # headless; Ctrl-C / F12 to stop
hop dashboard                                  # or drive it from the web UI
```

---

## Commands

| Command | Purpose |
|---|---|
| `hop doctor` | Preflight: adb, screencap, UHID tool, panel identity, OCR, templates. |
| `hop connect` | Connect wireless ADB, print device/panel info. |
| `hop capture --state <screen>` | Save a labeled anchor into the template pack. |
| `hop calibrate` | Measure panel + report rate into your user config (§10). |
| `hop test-click --at xf,yf` | Emit one humanized tap to verify the transport. |
| `hop run` | Run the hunt loop headless. |
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

[risk]
profile = "cautious"                  # cautious | balanced | aggressive

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
  hearthstone.py  HS screens, class-label region, mulligan card counting, coordinates
  engine.py     the hunt-loop state machine
  alerts.py     Mac notification/sound/say + ntfy push
  webui/        stdlib web dashboard
  cli.py        connect / doctor / capture / calibrate / test-click / run / dashboard
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
