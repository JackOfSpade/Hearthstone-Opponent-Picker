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
    max_actions_per_run: int
    max_actions_per_day: int
    committing_action_cap: int
    max_games_per_session: int
    max_session_minutes: int
    mandatory_break_every_games: int
    break_min_minutes: int
    break_max_minutes: int
    non_repetition_threshold: float


@dataclass(frozen=True)
class VisionConfig:
    state_change_threshold: float
    dedupe_signature_size: int
    ncc_match_threshold: float
    ocr_max_edit_distance: int
    weak_match_margin: float
    unknown_settle_attempts: int
    reconnecting_wait_attempts: int
    reconnect_attempt_cap: int
    mulligan_card_tap_attempts: int
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
    risk_profile: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


# ─────────────────────── risk-profile multipliers ───────────────────────
# Applied on top of [caps] and the concede journey timing. cautious = full
# anti-barcode strength; aggressive = old-AHK-style fast concedes.
RISK_PROFILES: dict[str, dict[str, float]] = {
    "cautious":   {"cap_scale": 1.0, "delay_scale": 1.0, "min_play_turns": 1.0},
    "balanced":   {"cap_scale": 1.4, "delay_scale": 0.65, "min_play_turns": 0.5},
    "aggressive": {"cap_scale": 2.5, "delay_scale": 0.3, "min_play_turns": 0.0},
}


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

    profile = merged["risk"]["profile"]
    if profile not in RISK_PROFILES:
        raise ValueError(f"Unknown risk profile {profile!r}; choose one of {list(RISK_PROFILES)}")

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
        risk_profile=profile,
        raw=merged,
    )


def _default_user_path() -> Path:
    return Path.home() / ".config" / "hop" / "config.toml"


def profile_multipliers(cfg: Config) -> dict[str, float]:
    return RISK_PROFILES[cfg.risk_profile]


def save_criteria(
    *,
    target_classes: tuple[HeroClass, ...] | list[HeroClass],
    require_second: bool,
    mode: str | None = None,
    risk_profile: str | None = None,
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
    if risk_profile is not None:
        if risk_profile not in RISK_PROFILES:
            raise ValueError(f"unknown risk profile: {risk_profile!r}")
        text = upsert_toml_scalar(text, "risk", "profile", f'"{risk_profile}"')

    user_path.write_text(text)
    return user_path
