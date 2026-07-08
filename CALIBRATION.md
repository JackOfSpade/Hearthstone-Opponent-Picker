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
hop capture --state collection       --glyph 0.7542,0.0148,0.1063,0.0444
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
| UHID `TOUCH_MAJOR` max | 255 | **2399** (the panel's) | Hearthstone silently ignored every tap on a mulligan card. |
| `contact.pressure_floor` | 0.22 | **0.52** | Under half the real touch-down pressure. |
| `contact_major_peak` | 0.55 | **0.12** | ~6× a real contact. |
| `in_game` anchor | End Turn *plate* | End Turn **housing** | Plate reads "ENEMY TURN" on their turn — i.e. exactly when a going-second game starts. |

### Clone the panel's *axes*, not just its name

Android loads a touch device's calibration (`<name>.idc`) **by name** and applies it
to our reports. Cloning `goodix_ts0`'s name while declaring `TOUCH_MAJOR` max 255
(the panel says 2399) means the framework sizes our contact against the wrong
scale. Hearthstone then ignored every mulligan-card tap while still honouring every
button tap — and the tap reached the kernel with correct coordinates, a clean
`BTN_TOUCH DOWN/UP` and a `TRACKING_ID -1` lift. `adb input tap` on the same pixel
worked, because it injects a MotionEvent above `/dev/input`.

`hop` now reads the panel's ranges from `getevent -pl` at `open()`. If you ever
change `[uhid] device_name`, re-check them:

```sh
adb shell getevent -pl | sed -n '/goodix/,/input props/p'   # TOUCH_MAJOR/MINOR/PRESSURE maxima
```

Beware the self-clone: once registered, the virtual device carries the panel's name
too, and `getevent` lists it first. `hop` takes the lowest event number.

### Open: mulligan card taps are accepted only ~1 time in 3

After the axis fix, Hearthstone accepts a UHID tap on a mulligan **card** about 33%
of the time (0% before). Buttons are accepted every time, and `adb input tap` on the
same pixel works every time. Ruled out by experiment on live mulligans:

| Suspect | Test | Result |
|---|---|---|
| Position / FFitts spread | dead-centre (offset 4 px) vs 39 px off | centre failed, off-centre succeeded |
| Dwell | 0.15 s vs 0.50 s vs the lognormal draw | no relationship; real human dwell is 65–535 ms |
| Micro-slip (drag detection) | `tap_micro_slip_px` 2.5 vs 0 | 1/3 vs 1/3 |
| Contact scale | the axis clone | **0/7 → 33%** — this was the big one |
| Event delivery | `getevent` on the virtual node | clean `DOWN`…`UP`, `TRACKING_ID -1`, correct coords |
| Deal-in animation not finished | 10 fps analysis of a `screenrecord` of the deal | **refuted** — see below |
| Stale card centres | ditto | **refuted** — aim error 2/5/8 px vs a 176 px card half-width |

Two explanations have since died. A *post-toggle animation gate* was the first: it
came from a probe that retapped every 1.2 s, but the engine's real taps are **12–17 s
apart** (measured from `journal.jsonl`) and still see ~33%, so no gate that short can
be responsible. The second was **"the cards are still being dealt"** — the natural
reading, since `_classify_settled()` settles the screen's *identity*, not its
*content*, and the mulligan is read off the very first frame that matches the banner.

That one is measurable. Recording the queue→mulligan transition with `screenrecord`
(native fps, analysed offline at 10 fps with the real classifier and the real
centre-finder) gives:

| Quantity | Measured |
|---|---|
| First frame the classifier calls `MULLIGAN` | t = 30.80 s |
| Card-row motion (`mean_abs_diff`) falls below 1.0 | t = 31.00 s — **0.2 s later** |
| Centres on that first frame vs the settled hand | **2 px, 5 px, 8 px** (half-width 176 px) |

So the hand is at rest almost immediately, the latched centres are good to 8 px, and
the engine taps ~10 s after that regardless. Waiting longer cannot help, and neither
can re-measuring. **The cause is still unidentified.** What is known: the tap is
delivered (kernel-verified), lands on the card, at a plausible dwell, with a settled
target — and Hearthstone ignores it ~2 times in 3, while ignoring no button, ever.

Note that verification here is *loose*, not strict: the card tap passes
`expected_change=None`, so `Verifier` only requires `classify_change != "none"` inside
the card's rectangle. An ignored tap is therefore a genuine no-op, not a missed
detection — which is what rules out "it toggled but we failed to see it".

`mulligan_card_tap_attempts = 3` brings the per-card success rate to ~70%; a card that
never takes is simply kept, and `stats.ignored_card_taps` surfaces the rate. Nothing
about the *committing* action (the concede) depends on this.

Next suspects, untested: the contact's **azimuth/orientation** channel (buttons may
not raycast on it while a draggable card does); Unity's drag threshold applied to the
*first* report rather than the trajectory; and a `Contact Count Maximum` feature
report we never expose, which could make Android's input classifier treat our single
contact as unconfirmed on hit-tests that consult it.

---

## What the post-game path taught us (Pixel 7a, 2026-07-08)

Driven live from a real victory screen, with the real UHID transport and motor.

The `victory` anchor is sound: **0.9301** against a 0.72 accept, sole anchor above
threshold, runner-up `in_game` at 0.6293. A genuine cross-game match — the stored
`victory_full.png` differs from the live frame by 61.9 mean-abs in the hero-portrait
region. The template is the word "Victory!" on its plaque: no XP bar, no level number,
no hero art. Same doctrine as the class label.

Everything downstream of it was wrong.

| Constant / assumption | Shipped | Measured / found | How it failed |
|---|---|---|---|
| `concede_confirm` | `(0.50, 0.56, 0.06)` | **it is the Quit button** | The Game Menu is Concede / Options / Quit. There is no confirm dialog. So the recovery for "the Concede tap was ignored" was *quit Hearthstone*. |
| `end_dismiss` radius | `0.10` (= **240 px**) | `0.0125` (30 px) | `radius_f` is a fraction of the **width** (2400) applied on a screen 1080 tall. 13.1% of 3000 sampled endpoints fell off the panel; 39.8% into Android's bottom `mandatorySystemGestures` inset (`y >= 996`). |
| `_clear_end_screens` terminal set | `PLAY_SCREEN/QUEUE/MENU` | **+ `DECK_SELECT`** | Hearthstone drops back to the deck *list* after a game. Without this the loop taps `end_dismiss` there — on "My Collection", and on a deck box (which would queue the **wrong deck**). |
| `end_dismiss` tap sites | any state outside a 5-case whitelist | **only the 4 end screens** | At (1200, 972) the old 216 px disc is 27 px from the mulligan's Confirm and 205 px from the reconnect dialog's **Cancel** — the failure mode `_reconnect()` calls "the worst available". |
| `_clear_end_screens` budget | one counter for taps *and* waits | two counters | A slow board fade exhausted it, whereupon the loop **returned normally** — indistinguishable from reaching home. `_execute_reject` then booked a completed game and requeued. |
| non-repetition gate | checked *after* `backend.emit` | resample **before** emit | See below. A gate that runs after the action cannot prevent anything. |

`end_dismiss` is now `(0.50, 0.87, 0.0125)`, within 6 px of the pixel measured to
dismiss the victory screen first try. It still *overlaps the mulligan's Confirm button*
(they are ~11 px apart) — both controls live at the bottom centre of their own screen
and no coordinate can separate them. Only the whitelist does.

### The live run that found it

```
victory (0.9280) --end_dismiss--> rewards (0.9528)   full_transition, verify OK
rewards (0.7798) --end_dismiss--> UNKNOWN (0.0000)   -> Halt
```

That UNKNOWN was the **Collection behind a "Ban Notice" modal** — a screen no anchor
covers. Fail-closed did its job. But the frame was *discarded* by the halt, so the
anchor that would prevent the next one could never be built. `hop` now keeps every
UNKNOWN frame, in colour, in `~/.config/hop/unknowns/` — a sibling of `runs/` that
`keep_runs` never touches. **That folder is empty when the hunt is healthy**; its
non-emptiness is the signal. It is the one exception to "delete captured media".

Note also that the Ban Notice *masked* the deck-list bug: the loop halted before it
ever tapped "My Collection".

### The Collection is now a recoverable state, not a halt

Navigated into live (deck list → "My Collection"), captured, and driven back out.
`collection` classifies at **0.9215**, and cross-validates 15/15: its "My Decks" banner
anchor scores ≤ 0.43 against every other stored screen, and no other anchor clears on
the collection frame (next-highest `reconnect_dialog` at 0.59, below the 0.72 accept).
The anchor is the "My Decks" banner — chrome, present whatever cards or class filter are
showing — not any card, which changes.

`_dispatch` now taps the bottom-right back arrow (`collection_back`, LIVE-VERIFIED
collection 0.92 → deck_select 0.91) instead of halting. hop is never *meant* to be here
— the `end_dismiss` geometry and the `END_SCREENS` whitelist keep it off the "My
Collection" plate — but a stray navigation self-heals rather than stopping the hunt.

Incidental measurement: **discrete taps below Android's y=996 gesture inset ARE
delivered on this device** (`input tap 1200 1020` navigated fine). That inset reserves
the home-swipe *gesture*, not taps — which is why `collection_back` at y≈1012 is safe
even though `end_dismiss`'s *spread* into that band was not (the latter also spilled
off-screen and wasn't on a control).

### Two dead ends the loop could never leave

Both were silent — no tap, no cap, no Halt, no alert. Both were found by running.

* **A live board.** `_execute_reject` can only start from the mulligan, so a board
  reached any other way (a slow capture, a reconnect, an app restart) had no matchup
  read and no exit: `_dispatch` slept and re-looped while a game hop queued roped out,
  turn by turn. Conceding is *not* special to the mulligan — the gear is in the
  top-right corner of every board, and `_concede()` classifies before it taps. hop now
  polls `in_game_wait_attempts`, then abandons the game. It abandons only a game **it
  started** (`_own_game`, set when it taps Play); a board it merely found itself in may
  be the user's target game, and fails closed.

* **The matchmaking queue.** Tapping there cancels the queue, so waiting is the only
  safe move — and *every* cap was checked inside `_tap`, so a loop that never taps could
  never trip one, however long it ran. Bounded by `queue_wait_attempts`, and
  `session_seconds` is now honest wall clock accrued by `run()` (it was `think + settle`
  from inside `_tap`, so the 12 s requeue delays, `between_actions`, `read_consider` and
  every settle poll ran for free). `Engine.clock` was stored by `__init__` and never
  read.

### The non-repetition gate saturates, and the blind retap was hiding it

`trajectory_fingerprint` has ~3 effective degrees of freedom, not 6: `dur ==
(n_samples - 1) / report_rate_hz` to **2.8e-17**. Over the real per-game tap sequence,
300 seeded runs × `max_actions_per_run = 30`:

| resamples | emitted duplicates | runs that would Halt |
|---|---|---|
| **1** (shipped) | **14.21%** | **293/300**, median tap #13 |
| 2 | 2.40% | 150/300 |
| 4 | 0.12% | 11/300 |
| 6 | 0.01% | 1/300 |
| 10 | 0.00% | 0/300 |

The loop only survived because `_tap`'s correction retap re-emitted with
`near_duplicate=False` hardcoded — so a bug (blind retry) was load-bearing for another
bug (a post-hoc gate) not firing. Fixed together: resample before emit, 8 draws.

**Do not "fix" the fingerprint.** Dropping the collinear `dur` feature *raises* the flag
rate to 15.50%.

### What measurement killed this round

* **"`state_change_threshold = 9.0` is calibrated on a 24×24 signature but applied to
  full-resolution thirds, so `verify()` fails open."** Two independent adversarial
  reviews reached for this. It is **false**. Measured on the phone:

  | | 24×24 signature | full-res thirds |
  |---|---|---|
  | noise (two captures, static screen) | 0.34–0.44 | 0.88–0.93 |
  | four real transitions | 19.4–30.9 | 26.8–50.0 |

  9.0 straddles both scales with ~44× and ~29× margins over noise. Downsampling
  suppresses uncorrelated noise (~√N) far more than a structured whole-screen change,
  which is why one constant serves both. Only the comment was wrong.

* **"The second `end_dismiss` tap pressed 'My Collection' behind the rewards popup."**
  Refuted by where we landed: `deck_select`, not the Collection. The tap behaved
  correctly; the modal caused the halt. The collision is real but *latent*, reachable
  through the missing `DECK_SELECT` terminal case.

* **"Add a panel-bounds check to `motor._endpoint_inside`."** Unnecessary once
  `end_dismiss`'s radius is sane: re-measured over 5000 draws, **0.00%** of endpoints
  leave the panel for any control in `GameLayout`.

* **"Derive the dismiss point from the winning anchor's glyph (`Classification.at`)
  rather than a fixed `Point`."** Tempting — it is `templates.py`'s own doctrine, and
  it guarantees the tap lands on the popup rather than on whatever is behind it. It is
  also **wrong here**, and the pixels say so:

  | | brightness at the fixed point (1200, 940) | at the anchor centre |
  |---|---|---|
  | `quest_popup` | 21.6 | 132.0 |
  | the `play_screen` behind it | 70.5 | 49.5 |

  The fixed point sits on the modal's **dim+blur scrim** (ratio 0.31, matching the
  far-left scrim at 0.34). The anchor centre sits on the *popup's content* — the "Your
  Quests" title banner, above a quest card that is plausibly interactive (detail view,
  claim). All four end screens dismiss by tapping the scrim / anywhere neutral, which is
  exactly what the fixed point does and what was live-verified at (1208, 941).

  `end_dismiss` is **not a control**. "Locate actions by vision" is a rule about
  *buttons*, whose position moves. Vision-locating a glyph returns the glyph — precisely
  the pixel you must not tap on a quest popup. The fixed point's disc (27 px, y 913–967)
  was checked against both underlying screens: it lands on inert wooden frame, clear of
  "My Collection" at y ≥ 990.

  `Classification.at` stays: it is what made the confidence bug fixable, and it enriches
  the journal.

Two hazards worth carrying to any future device:

* **A card marked for replacement loses its keep-glow.** On a 4-card hand that
  leaves three interiors: a plausible reading that silently flips `we_go_second`.
  Guarded by contiguity + hand-centre symmetry (`hearthstone.hand_is_coherent`).
* **Synthetic fixtures certify nothing here.** Three separate card counters passed
  full unit-test suites built on drawn gems and bars. Every one was wrong on the
  first real frame. `tests/data/frames/` holds real captures for this reason.
