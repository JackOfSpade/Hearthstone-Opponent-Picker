"""Configuration loading and the typed views the code consumes.

Standard, cross-cutting: *"Config over constants. No hardcoded ADB paths,
screen-fraction coordinates, thresholds, or caps in code - everything in
config, overridable per app. Code ships calibrated defaults; live values are
verified on-device."*

The defaults live in ``config.default.toml`` (the §8 priors). A user config
(``~/.config/hop/config.toml`` by default, or an explicit path) is deep-merged
over them. Typed dataclass "views" (:class:`Criteria`, :class:`MotorConfig`,
...) give the rest of the code attribute access without stringly-typed dict
lookups, and are the single place a missing/renamed key is caught.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hero_classes import HeroClass, parse_class

_DEFAULT_PATH = Path(__file__).with_name("config.default.toml")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ─────────────────────────── typed views ───────────────────────────

@dataclass(frozen=True)
class Criteria:
    target_classes: tuple[HeroClass, ...]
    avoid_classes: tuple[HeroClass, ...]
    require_second: bool
    mode: str

    def accepts(self, opponent: HeroClass, we_go_second: bool) -> bool:
        """Is this matchup one we want to KEEP (True) or concede (False)?"""
        if self.require_second and not we_go_second:
            return False
        if self.target_classes:
            return opponent in self.target_classes
        if self.avoid_classes:
            return opponent not in self.avoid_classes
        return True  # no class filter -> class always acceptable

    def pass_rate_estimate(self, meta_weights: dict[HeroClass, float] | None = None) -> float:
        """Rough fraction of games that will pass, for the dashboard risk meter.

        Uses a uniform class prior unless a meta weighting is supplied. The
        go-second factor is ~0.5 (coin is a coin flip). This is intentionally
        approximate - it exists to warn about barcode-shaped criteria (§ the
        anti-barcode design), not to be exact.
        """
        weights = meta_weights or {c: 1.0 / len(HeroClass) for c in HeroClass}
        if self.target_classes:
            class_p = sum(weights.get(c, 0.0) for c in self.target_classes)
        elif self.avoid_classes:
            class_p = 1.0 - sum(weights.get(c, 0.0) for c in self.avoid_classes)
        else:
            class_p = 1.0
        second_p = 0.5 if self.require_second else 1.0
        return max(0.0, min(1.0, class_p * second_p))


@dataclass(frozen=True)
class DeviceConfig:
    adb_address: str
    opponent_corner: str
    touch_backend: str
    posture: str


@dataclass(frozen=True)
class UhidConfig:
    device_name: str
    vendor_id: int
    product_id: int
    bus: str
    min_report_interval_ms: int
    register_settle_ms: int
    #: Contact-channel logical maxima; 0 = clone them from the real panel at open().
    #: See :class:`hop.transport.hid_descriptor.PanelAxes` for why this matters.
    touch_major_max: int = 0
    touch_minor_max: int = 0
    pressure_max: int = 0
    orientation_max: int = 0


@dataclass(frozen=True)
class AlertConfig:
    mac_notification: bool
    mac_sound: bool
    mac_speak: bool
    sound_name: str
    ntfy_topic: str
    ntfy_server: str


@dataclass(frozen=True)
class TimingConfig:
    sigma: float
    between_action_anchor_s: float
    read_consider_dwell_s: float
    think_commit_shift: float
    think_commit_mu: float
    think_commit_sigma: float
    think_reject_shift: float
    think_reject_mu: float
    think_reject_sigma: float


@dataclass(frozen=True)
class MotorConfig:
    ffitts_a: float
    ffitts_b: float
    finger_precision_mm: float
    default_target_w_px: float
    velocity_peak_frac: float
    velocity_sigma: float
    submovement_min: int
    submovement_max: int
    tremor_amplitude_mm: float
    tremor_theta: float
    report_rate_hz: float
    tap_dwell_median_s: float
    tap_dwell_sigma: float
    tap_dwell_min_s: float
    tap_dwell_max_s: float
    tap_micro_slip_px: float


@dataclass(frozen=True)
class ContactConfig:
    pressure_semantics: str
    pressure_alpha: float
    pressure_beta: float
    pressure_peak: float
    pressure_floor: float
    contact_major_peak: float
    contact_minor_ratio: float
    orientation_bias_rad: float


@dataclass(frozen=True)
class ScrollConfig:
    fling_friction: float
    ppi_reference: float
    reading_pause_s: float


@dataclass(frozen=True)
class SensorConfig:
    tap_impulse_g: float
    swipe_torque_dps: float
    poll_rate_hz: float


@dataclass(frozen=True)
class CapsConfig:
    """Config for the non-repetition gate. The volume/session/time caps and
    mandatory breaks that used to live here were removed: a hunt now runs until it
    finds a target, an error halts it, or the user stops it."""

    non_repetition_threshold: float
    #: Independent gesture draws before a near-duplicate becomes a Halt. The gate is
    #: pre-action: resample rather than emit-then-notice. See config.default.toml.
    non_repetition_resamples: int = 8


@dataclass(frozen=True)
class VisionConfig:
    state_change_threshold: float
    dedupe_signature_size: int
    ocr_max_edit_distance: int
    unknown_settle_attempts: int
    #: `wait_until` bounds. Both apply; whichever trips first ends the wait. The
    #: attempt count is what keeps a frozen-clock unit test terminating.
    screen_wait_attempts: int
    screen_wait_timeout_s: float
    screen_wait_poll_s: float
    #: How many times to re-attempt a SINGLE screencap that failed before surfacing it -- a
    #: stalled/timed-out wireless-ADB frame, or an empty/undecodable one. Each retry forces a
    #: fresh wireless link (disconnect+reconnect). A transient drop (screen lock, Doze, a Wi-Fi
    #: power-save blip, the Mac briefly sleeping) recovers here instead of crashing the whole
    #: hunt on one bad frame; a link that stays down exhausts these and fails closed cleanly
    #: (stats intact). 1 = no retry (the old crash-on-first-stall behaviour). See
    #: :meth:`hop.engine.Engine._capture_resilient`.
    capture_retry_attempts: int
    #: Slice length for the stop-aware sleep. A stop request is honoured within about
    #: one slice instead of after the full (up to 12 s) delay, so "Stop" feels instant.
    stop_poll_s: float
    #: Extra looks for motion after an unscoped tap, before calling it a missed tap.
    motion_wait_attempts: int
    #: Polls of a live board at the top of the loop before abandoning the game.
    in_game_wait_attempts: int
    #: Polls of the matchmaking queue before declaring it soft-locked.
    queue_wait_attempts: int
    #: Polls of the deck LIST (waiting for the user to re-select their deck) before failing
    #: closed. hop lands here after a dismissed "error starting your game" and cannot itself
    #: reopen a deck (it can't tell which one was in play), so it PAUSES -- alerting and
    #: waiting, never tapping the list -- and resumes when the user is back on a Play screen.
    #: Bounded so a walked-away session still stops cleanly rather than polling forever.
    deck_select_wait_attempts: int
    reconnecting_wait_attempts: int
    reconnect_attempt_cap: int
    #: How many times to (re-)tap a generic, low-stakes dispatch dialog button (the ERROR_DIALOG
    #: "OK" and the Collection back arrow) before failing closed -- the same silently-dropped-tap
    #: signature `_tap_play`/`_replace_card`/`_concede` retry against, on two screens plain
    #: enough (single button, no committing consequence, no special wait-budget needs) to share
    #: ONE knob rather than a dedicated one each. 1 restores the old single-tap-then-halt
    #: behaviour. See :meth:`hop.engine.Engine._tap_dispatch_button`.
    dispatch_tap_attempts: int
    #: How many times to (re-)tap 'No' on the "Complete deck automatically?" dialog before
    #: failing closed -- same dropped-tap signature as Play/Concede, retried the same way
    #: (positively-still-``INCOMPLETE_DECK`` gate before every retap, never a blind tap towards
    #: 'Yes'). Uses the generic ``screen_wait_attempts``/``screen_wait_timeout_s`` budget per
    #: attempt (this dialog has no roping-opponent-style reason to wait longer), so 3 attempts
    #: costs about the same worst case as Concede's. 1 restores the old single-tap-then-halt
    #: behaviour. See :meth:`hop.engine.Engine._decline_incomplete_deck`.
    deck_decline_tap_attempts: int
    #: How many times to (re-)tap Play before failing closed. Play is the loop's FIRST
    #: committing tap, on a static screen with no ambient animation to mask a miss -- so a
    #: dropped Play tap shows literally zero pixel change, the same `Halt.NO_CHANGE` signature
    #: `_replace_card` already retries against, not a wrong coordinate. Unlike a mulligan card
    #: there is no "keep it and carry on": exhausting the budget still fails closed, since Play
    #: is how every game in the hunt starts. 1 restores the old single-tap-then-halt behaviour
    #: (which failed the whole hunt closed on one dropped Play tap). See
    #: :meth:`hop.engine.Engine._tap_play`.
    play_tap_attempts: int
    mulligan_card_tap_attempts: int
    #: How many times to (re-)tap Concede before failing closed. Like a mulligan card, a
    #: Concede BUTTON tap can be silently dropped under a congested wireless link, and its
    #: own change-check can't catch it (the board animates behind the semi-transparent Game
    #: Menu, so verify passes on ambient motion). Only the menu LEAVING is proof it took, so
    #: retry the SAME Concede coordinate while the menu is positively still up. 1 restores
    #: the old single-tap-then-halt behaviour. See :meth:`hop.engine.Engine._concede`.
    concede_tap_attempts: int
    #: The wait for the mulligan to leave after we tap Confirm is NOT a normal button
    #: transition: it must also absorb the OPPONENT finishing THEIR mulligan. While they
    #: deliberate, this client swaps the "Starting Hand" banner (our mulligan anchor) for
    #: an "Opponent Still Choosing..." banner that carries no anchor -> the frame classifies
    #: UNKNOWN. The generic ``screen_wait_timeout_s`` (a ~20 s transition budget) is far too
    #: short for that, so an opponent who ropes their mulligan tripped a FALSE "Confirm did
    #: not dismiss the mulligan" halt even though the Confirm registered (its own verify_ok
    #: fired). These size that one wait to the mulligan rope instead. The wait NEVER taps, so
    #: a generous budget only delays failing closed -- it can never cause a misdirected tap.
    mulligan_resolve_timeout_s: float
    mulligan_resolve_attempts: int
    #: How many times to (re-)tap mulligan Confirm before failing closed. Unlike Concede/Play,
    #: EVERY attempt here waits the full ``mulligan_resolve_attempts``/``mulligan_resolve_timeout_s``
    #: budget above (sized to outlast a roping opponent) before deciding whether to retap --
    #: shrinking it would reintroduce the false halt that budget exists to prevent, since a
    #: genuinely slow (not dropped) opponent shows the SAME "still on `MULLIGAN`" signature for a
    #: short wait as a dropped tap does. That makes each attempt here far more expensive than
    #: Concede's (~4x by default), so this defaults to 2, not 3: enough to recover the one-dropped-
    #: tap case this whole family targets, without tripling an already-large worst-case wait on a
    #: tap that turns out to be un-recoverable. 1 restores the old single-tap-then-halt behaviour.
    #: See :meth:`hop.engine.Engine._confirm_mulligan`.
    mulligan_confirm_tap_attempts: int
    #: Total tries to READ the opponent's class off the mulligan before failing closed.
    #: The classifier has already confirmed the mulligan is up; a blank class is a transient
    #: (most likely the nameplate still drawing in), so we re-read a few times, not just once,
    #: before giving up. (>=1; 1 restores the old single-read-no-retry behaviour.)
    mulligan_read_attempts: int
    #: Consecutive GAMES whose CLASS never reads (blank/garbled after every re-read) before the
    #: hunt fails closed. One unreadable game is a transient hop recovers from -- it concedes +
    #: requeues, like a human who can't ID the matchup -- rather than halting the whole
    #: unattended run. A *run* of them in a row means the reader is broken RIGHT NOW (a shifted
    #: nameplate region, a UI change breaking every read), so at this many hop stops for a human.
    #: The streak resets on any usable read, so this catches an ACUTE, class-independent break,
    #: NOT a failure correlated to one class's word (that one is surfaced by the report's "class
    #: NEVER read" line + the "?" games in the observed distribution -- visible, not auto-halted,
    #: since auto-halting on a scattered rate would false-stop a healthy hunt). The trade this
    #: makes is deliberate: an unreadable game that happens to be a target is conceded rather
    #: than halting the hunt on it. (>=1; 1 restores the old halt-on-first-unreadable-class.)
    mulligan_unreadable_halt_streak: int
    glow_green_bias: int
    glow_min_green: int
    glow_col_min_frac: float
    glow_min_strip_frac: float
    card_width_tolerance: float
    card_min_gray: float
    card_max_separation_f: float
    hand_center_tolerance_f: float
    min_count_frame_width: int


@dataclass(frozen=True)
class DebugConfig:
    #: Bounded so a long unattended hunt cannot fill the disk with anomaly frames.
    #: The journal (a few KB) is what diagnoses a halt; the PNGs are ~1 MB apiece.
    keep_runs: int
    max_anomaly_frames: int
    #: UNKNOWN-screen frames are the exception: they are the only capture that
    #: cannot be re-taken (nobody knows how to get back to a screen nobody named),
    #: and they are what the next anchor is built from. Kept outside the run dirs,
    #: in a folder that is empty when the hunt is healthy. This is a runaway
    #: backstop, not a retention policy; <= 0 disables pruning entirely.
    max_unknown_frames: int = 30


@dataclass(frozen=True)
class Config:
    criteria: Criteria
    device: DeviceConfig
    uhid: UhidConfig
    alerts: AlertConfig
    timing: TimingConfig
    motor: MotorConfig
    contact: ContactConfig
    scroll: ScrollConfig
    sensor: SensorConfig
    caps: CapsConfig
    vision: VisionConfig
    debug: DebugConfig
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _classes(values: list[str]) -> tuple[HeroClass, ...]:
    return tuple(parse_class(v) for v in values)


def load_config(path: str | Path | None = None) -> Config:
    """Load defaults, merge a user config over them, return typed views."""
    with open(_DEFAULT_PATH, "rb") as f:
        merged = tomllib.load(f)

    user_path = Path(path) if path else _default_user_path()
    if user_path and user_path.exists():
        with open(user_path, "rb") as f:
            merged = _deep_merge(merged, tomllib.load(f))

    c = merged["criteria"]
    d = merged["device"]
    u = merged["uhid"]
    a = merged["alerts"]
    t = merged["timing"]
    m = merged["motor"]
    ct = merged["contact"]
    sc = merged["scroll"]
    sn = merged["sensor"]
    cp = merged["caps"]
    v = merged["vision"]
    dbg = merged["debug"]

    return Config(
        criteria=Criteria(
            target_classes=_classes(c["target_classes"]),
            avoid_classes=_classes(c["avoid_classes"]),
            require_second=bool(c["require_second"]),
            mode=str(c["mode"]),
        ),
        device=DeviceConfig(
            adb_address=str(d["adb_address"]),
            opponent_corner=str(d["opponent_corner"]),
            touch_backend=str(d["touch_backend"]),
            posture=str(d["posture"]),
        ),
        uhid=UhidConfig(
            device_name=str(u["device_name"]),
            vendor_id=int(u["vendor_id"]),
            product_id=int(u["product_id"]),
            bus=str(u["bus"]),
            min_report_interval_ms=int(u["min_report_interval_ms"]),
            register_settle_ms=int(u["register_settle_ms"]),
            touch_major_max=int(u.get("touch_major_max", 0)),
            touch_minor_max=int(u.get("touch_minor_max", 0)),
            pressure_max=int(u.get("pressure_max", 0)),
            orientation_max=int(u.get("orientation_max", 0)),
        ),
        alerts=AlertConfig(
            mac_notification=bool(a["mac_notification"]),
            mac_sound=bool(a["mac_sound"]),
            mac_speak=bool(a["mac_speak"]),
            sound_name=str(a["sound_name"]),
            ntfy_topic=str(a["ntfy_topic"]),
            ntfy_server=str(a["ntfy_server"]),
        ),
        timing=TimingConfig(**{k: t[k] for k in TimingConfig.__annotations__}),
        motor=MotorConfig(**{k: m[k] for k in MotorConfig.__annotations__}),
        contact=ContactConfig(**{k: ct[k] for k in ContactConfig.__annotations__}),
        scroll=ScrollConfig(**{k: sc[k] for k in ScrollConfig.__annotations__}),
        sensor=SensorConfig(**{k: sn[k] for k in SensorConfig.__annotations__}),
        caps=CapsConfig(**{k: cp[k] for k in CapsConfig.__annotations__}),
        vision=VisionConfig(**{k: v[k] for k in VisionConfig.__annotations__}),
        debug=DebugConfig(**{k: dbg[k] for k in DebugConfig.__annotations__}),
        raw=merged,
    )


def _default_user_path() -> Path:
    return Path.home() / ".config" / "hop" / "config.toml"


def save_criteria(
    *,
    target_classes: tuple[HeroClass, ...] | list[HeroClass],
    require_second: bool,
    mode: str | None = None,
    path: str | Path | None = None,
) -> Path:
    """Persist the user-facing criteria to the user config, in place.

    The control app and the dashboard both set these, and a setting that vanishes on
    restart is worse than no setting at all. Writes through
    :func:`hop.tomledit.upsert_toml_scalar`, so the file's comments - including every
    LIVE-VERIFIED provenance note - survive, which a tomllib parse/re-emit would not.
    """
    from .tomledit import upsert_toml_scalar

    user_path = Path(path) if path else _default_user_path()
    user_path.parent.mkdir(parents=True, exist_ok=True)
    text = user_path.read_text() if user_path.exists() else ""

    names = ", ".join(f'"{c.name}"' for c in target_classes)
    text = upsert_toml_scalar(text, "criteria", "target_classes", f"[{names}]")
    text = upsert_toml_scalar(text, "criteria", "avoid_classes", "[]")
    text = upsert_toml_scalar(text, "criteria", "require_second",
                              "true" if require_second else "false")
    if mode is not None:
        text = upsert_toml_scalar(text, "criteria", "mode", f'"{mode}"')

    user_path.write_text(text)
    return user_path


def clear_target_classes(path: str | Path | None = None) -> None:
    """Erase any persisted target classes so the app opens with a clean slate.

    Target classes are treated as *session-only*: the control app clears them on launch
    so every start has nothing selected and you re-pick each time. They still persist
    *within* a session (the running hunt and a bug report see the picks you make), but a
    restart does not carry them over. ``require_second``, ``avoid_classes`` and every
    config comment are left untouched -- only the one scalar is rewritten to ``[]``, via
    :func:`hop.tomledit.upsert_toml_scalar`, so provenance notes survive.

    Best-effort: no config file (targets already default to ``[]``) or an unwritable one
    just means there is nothing to clear.
    """
    from .tomledit import upsert_toml_scalar

    user_path = Path(path) if path else _default_user_path()
    if not user_path.exists():
        return
    try:
        text = user_path.read_text()
        new_text = upsert_toml_scalar(text, "criteria", "target_classes", "[]")
        # Never persist a config we cannot read back. tomledit is line-based, so a
        # hand-edited *multi-line* target_classes array would be corrupted into invalid
        # TOML (only its first line is replaced), and this runs automatically on every
        # app launch -- a corrupted write there would brick every later `load_config`.
        # If the rewrite doesn't round-trip, skip the disk write; the caller still drops
        # the picks from the in-memory config, so the clean-slate behavior is preserved.
        tomllib.loads(new_text)
        user_path.write_text(new_text)
    except (OSError, tomllib.TOMLDecodeError):
        pass
