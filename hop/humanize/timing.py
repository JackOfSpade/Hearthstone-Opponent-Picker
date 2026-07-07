"""Layer 4 - timing (when actions happen), stateful.

Standard: *"every wait is an anchor delay x a log-normal multiplier
(seconds * exp(N(0,sigma)*sigma), reference sigma ~= 0.22), not a uniform
range."* Two flavors:

* :func:`human_delay`   - may be shorter or longer than the anchor (ordinary pauses)
* :func:`human_cooldown`- never below the anchor (minimum waits, backoffs)

Per-decision **think time** is a separate, higher-level model: a shifted
lognormal whose parameters differ by decision type (a committing action is a
faster reaction than a rejecting one), and - crucially - it must be *stateful*:
scaled by :class:`HumanState` (fatigue, familiarity, urgency, confidence) and by
situational factors (visual complexity, novelty, error recovery). *"A single
log-normal sigma standing in for all think time"* is a §9 anti-pattern.

The §383 rule: *no bare `sleep(constant)` and no uniform delay on any app-facing
action path* - everything the app can observe routes through here. Short internal
poll cadences the app never sees (the screencap loop) are exempt and use plain
sleeps elsewhere.

These functions return **durations in seconds**; the engine is what actually
sleeps, so this module stays pure and testable.
"""

from __future__ import annotations

import math
from random import Random

from ..config import TimingConfig
from .state import HumanState


def human_delay(rng: Random, anchor_s: float, cfg: TimingConfig) -> float:
    """Ordinary pause: ``anchor * exp(N(0,sigma)*sigma)`` - may be shorter or
    longer, with an organic right-skewed tail. Never negative."""
    return max(0.0, anchor_s * math.exp(rng.gauss(0.0, cfg.sigma) * cfg.sigma))


def human_cooldown(rng: Random, anchor_s: float, cfg: TimingConfig) -> float:
    """Minimum-wait/backoff pause: like :func:`human_delay` but clamped to never
    fall below the anchor (the anchor is a floor, e.g. a mandated cooldown)."""
    return anchor_s * (1.0 + abs(math.exp(rng.gauss(0.0, cfg.sigma) * cfg.sigma) - 1.0))


def between_actions(rng: Random, cfg: TimingConfig, state: HumanState) -> float:
    """The typical inter-action pause, spread (never bursted), HumanState-scaled."""
    anchor = cfg.between_action_anchor_s * state.think_time_scale()
    return human_delay(rng, anchor, cfg)


def think_time(
    rng: Random,
    decision_type: str,
    cfg: TimingConfig,
    state: HumanState,
    *,
    visual_complexity: float = 0.0,
    novelty: float = 0.0,
    error_recovery: bool = False,
    text_len: int = 0,
) -> float:
    """Per-decision think time (seconds), shifted-lognormal, stateful.

    ``decision_type`` is ``"commit"`` (faster reaction; e.g. confirming a
    mulligan keep, tapping Play) or ``"reject"`` (slower; deliberating a bad
    matchup / a replace). The base shifted-lognormal is then modulated by:

    * ``HumanState.think_time_scale()`` - fatigue/familiarity/urgency/confidence
    * ``visual_complexity`` in [0,1] - a busier screen is read more slowly
    * ``novelty`` in [0,1] - an unfamiliar screen takes longer the first times
    * ``error_recovery`` - a slowdown after a failed validation/missed tap
    * ``text_len`` - reading longer text costs time

    This is what makes the *conditional* checks a detector runs ("time from
    price display to purchase", "time after a failed validation") look human,
    not just the marginal histogram.
    """
    if decision_type == "commit":
        shift, mu, sigma = cfg.think_commit_shift, cfg.think_commit_mu, cfg.think_commit_sigma
    else:
        shift, mu, sigma = cfg.think_reject_shift, cfg.think_reject_mu, cfg.think_reject_sigma

    base = shift + math.exp(rng.gauss(mu, sigma))
    scale = state.think_time_scale()
    scale *= 1.0 + 0.5 * max(0.0, min(1.0, visual_complexity))
    scale *= 1.0 + 0.8 * max(0.0, min(1.0, novelty)) * (1.0 - state.familiarity)
    if error_recovery:
        scale *= 1.6
    reading = 0.025 * max(0, text_len)  # ~40 chars/s glance reading
    return base * scale + reading


def read_consider(rng: Random, cfg: TimingConfig, state: HumanState) -> float:
    """A per-item 'reading the board' consideration pause, HumanState-scaled."""
    anchor = cfg.read_consider_dwell_s * (1.0 + 0.5 * state.fatigue)
    return human_delay(rng, anchor, cfg)
