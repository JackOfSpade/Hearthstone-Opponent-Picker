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
```sh
hop capture --state mulligan --region 0.30,0.02,0.40,0.12
hop capture --state victory
hop capture --state defeat
hop capture --state menu
hop capture --state concede_menu
# ... one per screen the engine navigates
```
Crop each saved PNG down to just the anchor glyph for a tight template (or leave
it full-screen and rely on the region filter). Also verify the Hearthstone-
specific coordinates in `hop.hearthstone.GameLayout` against your resolution —
especially `opponent_class_region` (bottom-left) and the mulligan card row.

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
