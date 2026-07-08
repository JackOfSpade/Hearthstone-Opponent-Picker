"""Layer 3 - motor synthesis (the master principle at the motor level).

Standard: taps and swipes must come from a motor-control model calibrated to
real ``getevent`` traces, not from ``uniform(a, b)``. Concretely (Rev 2):

* **FFitts, not vanilla Fitts.** Movement time and endpoint spread use the
  Bi & Zhai variant, which adds an *absolute finger-precision* term (in mm) to
  the relative, difficulty-dependent variance. (§8 ``ffitts_a/b``,
  ``finger_precision_mm``.)
* **Submovements, not smooth-curve-plus-noise.** A reach is a primary
  minimum-jerk ballistic stroke plus 0-N corrective minimum-jerk submovements
  concentrated near the target, superposed. Correlated OU tremor is a secondary
  texture, not the primary source of path variability. Endpoints stay exact;
  only the transit is noisy.
* **Contact-patch evolution** on every sample (via :mod:`hop.humanize.contact`).
* **Tap dwell** is a clamped lognormal down->up, not a fixed press duration.

Everything here is pure and ``rng``-injectable, and reads its situational
modulation off :class:`hop.humanize.state.HumanState` so paths, tremor and dwell
co-vary with fatigue/attention/familiarity instead of being independently random.

On a touchscreen a tap has no visible on-screen approach path (the finger just
lands), so :func:`synth_tap` emits the contact lifecycle (down, pressure ramp,
micro-slip, release) at a spread-sampled endpoint, and the FFitts movement time
is consumed as pre-contact reach time by the timing layer. Swipes/scrolls, which
*do* move while in contact, use the submovement path synthesizer.
"""

from __future__ import annotations

import math
from random import Random

from ..config import MotorConfig
from ..geometry import PanelGeometry
from ..touchstream import Gesture, TouchSample
from .contact import ContactModel
from .state import HumanState

# A control's nominal width maps to endpoint sigma as W ~= 4.133 * sigma (96%
# capture for a Gaussian). Used to split relative vs absolute precision.
_W_TO_SIGMA = 1.0 / 4.133


def fitts_index_of_difficulty(distance_px: float, width_px: float) -> float:
    """Shannon-form index of difficulty: log2(D/W + 1)."""
    width_px = max(1.0, width_px)
    return math.log2(distance_px / width_px + 1.0)


def ffitts_movement_time(
    rng: Random, distance_px: float, width_px: float, cfg: MotorConfig, state: HumanState
) -> float:
    """FFitts movement time in seconds, modulated by HumanState.

    ``MT = a + b * ID`` with a lognormal multiplier for organic spread, then
    scaled by attention/fatigue (a tired or distracted reach is slower). The
    absolute finger-precision term enters the *endpoint spread*
    (:func:`endpoint_spread`), which is FFitts's distinguishing contribution
    over vanilla Fitts.
    """
    idf = fitts_index_of_difficulty(distance_px, width_px)
    mt = cfg.ffitts_a + cfg.ffitts_b * idf
    mt *= math.exp(rng.gauss(0.0, 0.12))  # organic spread
    mt *= 1.0 + 0.35 * state.fatigue - 0.15 * state.attention + 0.2 * (1 - state.familiarity)
    return max(0.05, mt)


def endpoint_spread(
    rng: Random, width_px: float, panel: PanelGeometry, cfg: MotorConfig, state: HumanState
) -> tuple[float, float]:
    """FFitts endpoint offset (px), sigma = sqrt(sigma_relative^2 + sigma_abs^2).

    ``sigma_relative`` comes from the target width; ``sigma_abs`` is the
    absolute finger precision in **mm** (§8), added in quadrature - the term
    vanilla Fitts omits. Fatigue inflates the spread. Returned as a 2D offset;
    callers that must land inside a control clamp it to the hit area.
    """
    sigma_rel_mm = panel.px_to_mm(width_px * _W_TO_SIGMA)
    sigma_abs_mm = cfg.finger_precision_mm
    sigma_mm = math.hypot(sigma_rel_mm, sigma_abs_mm) * (1.0 + 0.4 * state.fatigue)
    sigma_px = panel.mm_to_px(sigma_mm)
    return (rng.gauss(0.0, sigma_px), rng.gauss(0.0, sigma_px))



_ENDPOINT_RESAMPLES = 8


def _endpoint_inside(rng: Random, hit_radius_px: float, panel: PanelGeometry,
                     cfg: MotorConfig, state: HumanState) -> tuple[float, float]:
    """FFitts endpoint offset, conditioned on landing inside the hit area.

    **Resample, don't clamp.** For a small control the FFitts sigma exceeds the hit
    radius, so scaling every out-of-bounds draw back onto the ``0.9 * r`` circle
    piles the distribution up on that circle: measured 52% of taps at a 30px radius
    landed at exactly 27px from centre, and 21% even at 120px. A ring of identical
    offsets is a signature, and no finger produces one.

    Rejection sampling keeps the Gaussian shape *inside* the disc, which is what
    "aim for the control, land somewhere plausible on it" actually means. If the
    sigma is so much larger than the target that we can't land in a few tries, we
    fall back to a uniformly-random point in the disc rather than a fixed ring -
    still bounded, still not degenerate.
    """
    limit = hit_radius_px * 0.9
    for _ in range(_ENDPOINT_RESAMPLES):
        ox, oy = endpoint_spread(rng, 2 * hit_radius_px, panel, cfg, state)
        if math.hypot(ox, oy) <= limit:
            return ox, oy
    r = limit * math.sqrt(rng.random())          # sqrt => uniform over the area
    theta = rng.uniform(0.0, 2 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)

def _minjerk_phase(s: float) -> float:
    """Minimum-jerk position profile in [0, 1] for normalized time s in [0, 1]."""
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def tap_dwell(rng: Random, cfg: MotorConfig, state: HumanState) -> float:
    """Clamped lognormal down->up dwell (seconds), scaled by HumanState."""
    median = cfg.tap_dwell_median_s * state.dwell_scale()
    val = median * math.exp(rng.gauss(0.0, cfg.tap_dwell_sigma))
    return max(cfg.tap_dwell_min_s, min(cfg.tap_dwell_max_s, val))


def synth_tap(
    rng: Random,
    target: tuple[float, float],
    hit_radius_px: float,
    panel: PanelGeometry,
    cfg: MotorConfig,
    contact: ContactModel,
    state: HumanState,
    pointer_id: int = 0,
) -> Gesture:
    """Synthesize the contact lifecycle of a single tap.

    The endpoint is sampled *within the control's hit area* (not its exact
    center - the standard explicitly wants this), using the FFitts spread
    truncated to ``hit_radius_px`` so the tap reliably lands. Pressure/size/
    orientation ramp per :class:`ContactModel`; the centroid micro-slips as the
    pad compresses; the final sample is an explicit release (tip=False,
    pressure 0).
    """
    ox, oy = _endpoint_inside(rng, hit_radius_px, panel, cfg, state)
    ex, ey = target[0] + ox, target[1] + oy

    dwell = tap_dwell(rng, cfg, state)
    dt = 1.0 / cfg.report_rate_hz
    n = max(3, int(round(dwell / dt)))

    # micro-slip: a small settle-and-slide of the centroid during contact
    slip = cfg.tap_micro_slip_px * (0.5 + rng.random())
    slip_ang = rng.uniform(0, 2 * math.pi)
    sx, sy = math.cos(slip_ang) * slip, math.sin(slip_ang) * slip

    samples: list[TouchSample] = []
    for i in range(n):
        u = i / (n - 1)  # 0..1 across the dwell
        ch = contact.channels(u, travel_dir=None)
        # centroid slides in over the first ~40% then holds (damped settle)
        slip_prog = _minjerk_phase(min(1.0, u / 0.4))
        x = ex + sx * slip_prog
        y = ey + sy * slip_prog
        samples.append(TouchSample(
            t=i * dt, x=x, y=y,
            pressure=ch.pressure, size=ch.size, major=ch.major, minor=ch.minor,
            orientation=ch.orientation, tip=True, pointer_id=pointer_id,
        ))
    # explicit release
    samples.append(TouchSample(
        t=n * dt, x=ex + sx, y=ey + sy,
        pressure=0.0, size=0.0, major=0.0, minor=0.0,
        orientation=samples[-1].orientation, tip=False, pointer_id=pointer_id,
    ))
    return Gesture(kind="tap", samples=samples, target=(ex, ey))


def _plan_submovements(
    rng: Random,
    start: tuple[float, float],
    target: tuple[float, float],
    n_correct: int,
    state: HumanState,
    mt: float,
) -> list[tuple[tuple[float, float], float, float]]:
    """Return [(displacement, t_start_frac, dur_frac), ...] whose displacements
    sum EXACTLY to (target - start).

    Primary ballistic stroke aims at target + overshoot; correctives near the
    end each remove a fraction of the residual, and the final corrective takes
    the exact residual so the endpoint is exact.
    """
    tx, ty = target[0] - start[0], target[1] - start[1]
    dist = math.hypot(tx, ty)
    ux, uy = (tx / dist, ty / dist) if dist > 0 else (0.0, 0.0)

    # overshoot along travel direction, shrinking with familiarity
    over = dist * 0.06 * state.overshoot_scale() * rng.uniform(0.3, 1.0)
    aim = (tx + ux * over, ty + uy * over)

    subs: list[tuple[tuple[float, float], float, float]] = []
    primary_dur_frac = rng.uniform(0.62, 0.78)
    subs.append((aim, 0.0, primary_dur_frac))
    remaining = (tx - aim[0], ty - aim[1])
    t_cursor = primary_dur_frac * rng.uniform(0.75, 0.9)  # correctives overlap the primary tail
    
    mt_safe = max(0.01, mt)
    for k in range(n_correct):
        last = k == n_correct - 1
        if last:
            disp = remaining
        else:
            frac = rng.uniform(0.45, 0.7)
            disp = (remaining[0] * frac, remaining[1] * frac)
            remaining = (remaining[0] - disp[0], remaining[1] - disp[1])
        dur_seconds = rng.uniform(0.06, 0.13)
        dur_frac = dur_seconds / mt_safe
        subs.append((disp, min(0.95, t_cursor), dur_frac))
        t_cursor = min(0.98, t_cursor + dur_frac * rng.uniform(0.6, 0.9))
    return subs


def synth_swipe(
    rng: Random,
    start: tuple[float, float],
    end: tuple[float, float],
    panel: PanelGeometry,
    cfg: MotorConfig,
    contact: ContactModel,
    state: HumanState,
    duration_s: float | None = None,
    kind: str = "swipe",
    pointer_id: int = 0,
) -> Gesture:
    """Synthesize a swipe/drag as superposed minimum-jerk submovements plus OU
    tremor, with contact channels and the finger pad rotating toward travel.

    ``duration_s`` overrides FFitts timing (used by the scroll layer, which sets
    duration from a target release velocity). Endpoints are exact; tremor tapers
    to zero over the last 15% so the release lands precisely.
    """
    dist = math.hypot(end[0] - start[0], end[1] - start[1])
    width = cfg.default_target_w_px
    mt = duration_s if duration_s is not None else ffitts_movement_time(rng, dist, width, cfg, state)

    n_correct = int(round(_clamp(
        (1.0 - state.straightness()) * (cfg.submovement_min + cfg.submovement_max) / 2.0
        + state.submovement_bias()
        + fitts_index_of_difficulty(dist, width) * 0.15,
        cfg.submovement_min, cfg.submovement_max,
    )))
    subs = _plan_submovements(rng, start, end, n_correct, state, mt)

    dt = 1.0 / cfg.report_rate_hz
    n = max(4, int(round(mt / dt)))
    travel_dir = math.atan2(end[1] - start[1], end[0] - start[0])

    # OU tremor state (correlated), amplitude in mm -> px, scaled by HumanState
    tremor_px = panel.mm_to_px(cfg.tremor_amplitude_mm) * state.tremor_scale()
    ox = oy = 0.0

    samples: list[TouchSample] = []
    for i in range(n + 1):
        s_time = i / n  # normalized gesture time 0..1
        # asymmetric velocity: warp time so speed peaks early (velocity_peak_frac)
        warped = _velocity_warp(s_time, cfg.velocity_peak_frac)
        # superpose submovements
        px = start[0]
        py = start[1]
        for disp, ts, dur in subs:
            ph = _minjerk_phase((warped - ts) / dur) if dur > 0 else (1.0 if warped >= ts else 0.0)
            px += disp[0] * ph
            py += disp[1] * ph
        # correlated tremor, speed-scaled, tapered to 0 at the end (exact endpoint)
        speed_scale = math.sin(math.pi * s_time)  # 0 at ends, 1 mid
        taper = max(0.0, 1.0 - max(0.0, (s_time - 0.85) / 0.15))
        ox = _ou(rng, ox, cfg.tremor_theta, tremor_px, dt)
        oy = _ou(rng, oy, cfg.tremor_theta, tremor_px, dt)
        jx = ox * speed_scale * taper
        jy = oy * speed_scale * taper

        u = s_time  # contact progress across the gesture
        ch = contact.channels(u, travel_dir=travel_dir)
        samples.append(TouchSample(
            t=i * dt, x=px + jx, y=py + jy,
            pressure=ch.pressure, size=ch.size, major=ch.major, minor=ch.minor,
            orientation=ch.orientation, tip=True, pointer_id=pointer_id,
        ))
    # force exact endpoint on last in-contact sample, then explicit release
    last = samples[-1]
    samples[-1] = TouchSample(
        t=last.t, x=end[0], y=end[1], pressure=last.pressure, size=last.size,
        major=last.major, minor=last.minor, orientation=last.orientation,
        tip=True, pointer_id=pointer_id,
    )
    # release velocity (for fling physics) from the last two in-contact samples
    rv = None
    if len(samples) >= 2:
        a, b = samples[-2], samples[-1]
        ddt = max(1e-4, b.t - a.t)
        rv = ((b.x - a.x) / ddt, (b.y - a.y) / ddt)
    samples.append(TouchSample(
        t=last.t + dt, x=end[0], y=end[1], pressure=0.0, size=0.0, major=0.0,
        minor=0.0, orientation=last.orientation, tip=False, pointer_id=pointer_id,
    ))
    return Gesture(kind=kind, samples=samples, target=end, release_velocity=rv)


def _velocity_warp(s: float, peak_frac: float) -> float:
    """Warp normalized time so tangential velocity peaks near ``peak_frac``.

    A symmetric min-jerk peaks at 0.5; humans peak earlier (~0.35) with a long
    deceleration. We remap s through a smooth monotone curve that front-loads
    progress. Identity at the endpoints so total displacement is unchanged.
    """
    peak_frac = min(max(peak_frac, 0.15), 0.85)
    # gamma < 1 front-loads; choose gamma so that s=peak_frac maps to 0.5
    if peak_frac <= 0 or peak_frac >= 1:
        return s
    gamma = math.log(0.5) / math.log(peak_frac)
    return s ** gamma


def _ou(rng: Random, prev: float, theta: float, sigma: float, dt: float) -> float:
    return prev - theta * prev * dt + sigma * math.sqrt(max(dt, 0.0)) * rng.gauss(0.0, 1.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
