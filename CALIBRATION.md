# Calibration (§10 protocol)

The humanization models are general; the *constants* are device-specific. The
values in `hop/config.default.toml` are **reference priors measured on one phone
with one human** — several vary 1–10× across devices, tasks, and users.
Completing this protocol for your actual phone is a **precondition for trusting
auto mode**, not an appendix.

Each step notes which `hop` command automates it and which config keys it fills.

## 1. Capture real touch input
With a human playing a few games, record `getevent -lt` for representative taps
and swipes. Extract:
- report rate (Hz) **and batching structure** (`hop calibrate --report-rate`
  estimates the rate; batching needs manual inspection) → `motor.report_rate_hz`
- samples per gesture, raw pressure min/mean/max (`ABS_MT_PRESSURE`) →
  `contact.pressure_floor`, `contact.pressure_peak`, and crucially the panel's
  **pressure semantics** (`ramp` vs `binary` vs `amplitude`) → `contact.pressure_semantics`
- contact major/minor range + orientation (`ABS_MT_TOUCH_MAJOR/MINOR`) →
  `contact.contact_major_peak`, `contact.contact_minor_ratio`
- tap dwell and reaction times → `motor.tap_dwell_median_s`, `[timing]`

```sh
adb shell getevent -lt /dev/input/eventN     # find N with: adb shell getevent -i
```

## 2. Capture the coupled sensors
Only relevant if you run **handheld** posture. Simultaneously log accel/gyro
while tapping/swiping as the automation will run; extract per-tap impulse and
per-swipe torque → `[sensor]`. On **desk_mounted** posture (the default for a
bench phone over ADB) you skip this and the tool declares flat IMU is expected —
it does **not** claim handheld realism.

## 3. Fit the profiles
Map the measurements from step 1 into `[motor]`, `[contact]`, `[timing]`.
Reaction-time distributions are skewed and per-decision — fit `commit` vs
`reject` think time separately.

## 4. Profile the input device (panel-matched UHID identity)
```sh
hop doctor        # prints the real panel's Vendor/Product line from dumpsys input
```
Clone your panel's `name` / `vendor_id` / `product_id` into `[uhid]` so the
virtual digitizer's metadata matches the built-in panel. A `0x0000` vid/pid with
a generic name is a giveaway.

## 5. Set vision thresholds
Measure frame deltas for real transitions vs noise; set
`vision.state_change_threshold` between them. The default (9.0 mean grayscale
delta over a 24×24 signature) is a starting point.

## 6. Collect templates
**Always pass `--glyph`.** `best_match()` slides the template inside its search
region and gives up when the template is larger, so a full-screen template can
never match. `--glyph xf,yf,wf,hf` crops the anchor out of the frame; the search
region defaults to that box plus a margin.

```sh
hop capture --state play_screen  --glyph 0.6925,0.8167,0.0725,0.0611
hop capture --state queue        --glyph 0.4375,0.0685,0.1525,0.0370
hop capture --state mulligan     --glyph 0.4200,0.1000,0.1600,0.0560
hop capture --state defeat       --glyph 0.4450,0.6170,0.1825,0.0889
hop capture --state victory      --glyph 0.4350,0.5950,0.1650,0.0850
hop capture --state rewards      --glyph 0.4500,0.7500,0.1025,0.0500
hop capture --state deck_select  --glyph 0.4400,0.9333,0.1125,0.0426
hop capture --state in_game      --glyph 0.7642,0.5352,0.0650,0.0481
hop capture --state quest_popup      --glyph 0.4117,0.1000,0.1604,0.1157 --priority 10
hop capture --state concede_menu     --glyph 0.4475,0.0444,0.1075,0.0352 --priority 10
hop capture --state error_dialog     --glyph 0.4350,0.3111,0.1250,0.0444 --priority 10
hop capture --state reconnect_dialog --glyph 0.4200,0.1000,0.1600,0.0560 --priority 10
hop capture --state reconnecting     --glyph 0.4088,0.6500,0.1821,0.0620 --priority 20
```
(Those glyph boxes are measured on a 2400x1080 landscape frame.) `--from-file
<png>` rebuilds an anchor offline from a saved `<state>_full.png`.

**Priority is not cosmetic — it encodes what is drawn over what.** An overlay does
not hide the screen beneath it, so both anchors clear their thresholds on the same
frame and the higher priority must win:

* `concede_menu` is drawn over the board, and the `in_game` anchor still matches
  through it. At equal priority `in_game` wins on score, the engine *waits* instead
  of tapping Concede, and **the loop never concedes**.
* `reconnecting` must outrank `reconnect_dialog`: mid-reconnect the title banner is
  unchanged, so the dialog anchor still matches — but the buttons are gone and a tap
  would hit dead space.

Two anchors need care beyond a glyph box:

* **`in_game`** must be read from the End Turn button's *housing*, not its plate.
  The plate reads `END TURN` on your turn and `ENEMY TURN` on theirs — and since a
  going-second game begins on the opponent's turn, an anchor on the text fails
  exactly when it is first needed. Measured pairwise NCC across both turn states:
  housing **0.86 min**, whole button **0.60 min** (below the 0.72 accept level).
* **`reconnecting`** is the same dialog as `reconnect_dialog` with a different body,
  so anchor it on the body text, not the title.

Then verify the coordinates in `hop.hearthstone.GameLayout` against your
resolution. Note that hit radii should come from each control's **smaller**
half-dimension: the FFitts endpoint spread is isotropic and truncated to
`0.9 * radius`, so an over-large radius throws taps off short, wide buttons.

## 6b. Orientation (landscape vs the native panel)
Hearthstone runs landscape while the panel is native portrait. Perception and the
engine work in *display* space; the UHID digitizer reports *native panel* pixels,
and the framework rotates them. `hop.orientation` inverts that using the live
rotation from `adb.get_rotation()`, so no calibration is needed — but if taps land
somewhere rotated/mirrored, that transform is the first place to look.

## 7. Validate against detectors, not intuition
Run generated taps/swipes through a local multimodal check (touch + IMU
features) and confirm they fall inside human envelopes **jointly**, not just
per-channel. (Left as a manual/offline step; the `tests/` suite verifies the
generators are self-consistent and reproducible.)

## 8. Record provenance
`hop calibrate` writes `[calibration] device/date` into your user config. Add
the tester and method next to any constant you change. **Undocumented constants
rot.**

---

## Which values are safe to ship vs. LIVE-VERIFY

Anything marked `LIVE-VERIFY` in `config.default.toml` — notably
`motor.report_rate_hz` (modern panels run 240/720/1200/2000 Hz, not 180),
`contact.pressure_semantics`, the `[uhid]` panel identity, and the
`hop.hearthstone.GameLayout` fractions — should be confirmed on your device
before auto mode. The pure motor/timing math is device-independent and needs no
recalibration.

### Shortcut: what you can read off the device without a finger

Two of these don't need a human swipe. `getevent -i` and `dumpsys input` report
the panel's own axes and calibration:

```sh
adb shell getevent -i | sed -n '/goodix/,/^  add/p'   # ABS ranges
adb shell dumpsys input | grep -i 'touch\..*calibration'
```

* `ABS_MT_PRESSURE  min 0 max 255` + `touch.pressure.calibration: physical`
  → the panel reports **graded** pressure, so `contact.pressure_semantics = "ramp"`.
  A panel that only ever reports 0/1 would want `"binary"` — a faithful flat
  signal beats a fake ramp.
* `ABS_MT_TOUCH_MAJOR/MINOR` maxima and `touch.size.calibration` tell you what the
  contact-size channel means.
* `hop doctor` prints the panel's `name`/`vendor`/`product` ready to paste into
  `[uhid]`.

`motor.report_rate_hz` genuinely cannot be read this way — no sysfs node exposes
it and only a real finger makes the panel emit reports. Measure it:

```sh
hop calibrate --report-rate     # then SWIPE on the phone during the sample window
```

Reference measurement (Pixel 7a, `goodix_ts0`, 2026-07-07): pressure `ramp`
(0..255, physical), `TOUCH_MAJOR` max 2399, `TOUCH_MINOR` max 1079,
`ORIENTATION` -4096..4096 (interpolated), 10 contact slots, size cal `GEOMETRIC`.

---

## What live bring-up actually changed (Pixel 7a, 2026-07-08)

Everything below was wrong until a real phone said so. Recorded because the next
device will need the same treatment, and because the *method* transfers even where
the numbers don't.

| Constant / assumption | Shipped | Measured | How it failed |
|---|---|---|---|
| `motor.report_rate_hz` | 180 (placeholder) | **183** | The measuring code itself was broken three ways; see `hop/calibrate.py`. |
| Mulligan card count | mana gems / brightness | **card interiors between keep-glows** | Both returned 3 on a real 4-card hand, inverting `we_go_second`. |
| Mulligan card centres | evenly spaced 0.20…0.80 | **measured from the frame** | ~100px off; FFitts spread pushed taps off the card. |
| Gear-tap change kind | `bottom_sheet` | **`full_transition`** | Halted every concede. |
| Card-replace tap | whole-frame verify | **region-scoped verify** | Whole-frame delta 4.80 < threshold 9.0 → "missed tap". |
| `concede_menu` priority | 0 | **10** | The menu draws over the board, so `in_game` won and the loop never conceded. |
| `in_game` anchor | End Turn *plate* | End Turn **housing** | Plate reads "ENEMY TURN" on their turn — i.e. exactly when a going-second game starts. |

Two hazards worth carrying to any future device:

* **A card marked for replacement loses its keep-glow.** On a 4-card hand that
  leaves three interiors: a plausible reading that silently flips `we_go_second`.
  Guarded by contiguity + hand-centre symmetry (`hearthstone.hand_is_coherent`).
* **Synthetic fixtures certify nothing here.** Three separate card counters passed
  full unit-test suites built on drawn gems and bars. Every one was wrong on the
  first real frame. `tests/data/frames/` holds real captures for this reason.
