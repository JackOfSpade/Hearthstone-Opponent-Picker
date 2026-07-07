"""Layer 5 - HumanState, the latent spine.

Standard: *"Maintain a persistent per-session state that biases every
downstream distribution, producing correlated behavior instead of independent
randomness."* This is the single most important idea in the standard's Rev 2:
motor paths, tremor, think time, dwell, hesitation and cancellation must all be
*projections of one latent state*, so they co-vary the way real behavior does.
The engine advances this state once per action; the motor (L3) and timing (L4)
layers read multipliers off it instead of drawing independent randomness.

Nothing here touches the clock or global RNG: :meth:`HumanState.tick` receives
elapsed seconds from the caller and every stochastic nudge takes an injectable
``rng``. That keeps the whole spine reproducible for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from ..rng import clamp


@dataclass
class HumanState:
    """Latent cognitive/physical state, all fields in [0, 1] unless noted.

    Semantics (from the standard's HumanState table):

    * ``attention``  high -> fast reactions, small tremor, direct paths
    * ``confidence`` set from perception match quality -> inspect/commit/abandon
    * ``urgency``    compresses think time and dwell
    * ``fatigue``    accumulates over the session -> longer pauses, more corrections
    * ``familiarity``grows with repetition -> think time drops, paths straighten
    * ``memory``     visited regions, last action, last failed target
    """

    attention: float = 0.85
    confidence: float = 0.7
    urgency: float = 0.25
    fatigue: float = 0.0
    familiarity: float = 0.0

    # Memory (§ Layer 5). Coarse visited-region grid keys, plus recency.
    visited_regions: set[tuple[int, int]] = field(default_factory=set)
    last_action: str | None = None
    last_failed_target: tuple[float, float] | None = None
    recent_signatures: list[str] = field(default_factory=list)

    # Inertia: consecutive gestures drift rather than resetting. Tracks the last
    # normalized gesture "pace" so the next one autocorrelates.
    pace: float = 0.5

    actions_taken: int = 0
    session_seconds: float = 0.0

    # ── evolution ──────────────────────────────────────────────────────────

    def tick(self, rng: Random, dt: float, action: str | None = None) -> None:
        """Advance the state by one action separated by ``dt`` seconds.

        Fatigue accumulates with time-on-task (with a small stochastic nudge);
        familiarity grows with repetition and saturates; attention decays as
        fatigue rises. These are deliberately gentle so a session *evolves*
        rather than lurching - the standard's "behavior is not stationary".
        """
        self.session_seconds += max(0.0, dt)
        self.actions_taken += 1

        # Fatigue: ~0-1 over a couple of hours of steady play, + noise.
        self.fatigue = clamp(
            self.fatigue + dt / 7200.0 + 0.002 * rng.gauss(0.0, 1.0), 0.0, 1.0
        )
        # Familiarity: fast early growth, saturating (repetition of the loop).
        self.familiarity = clamp(self.familiarity + 0.03 * (1.0 - self.familiarity), 0.0, 1.0)
        # Attention: high, pulled down by fatigue, small jitter, mean-reverting.
        target_attn = clamp(0.9 - 0.5 * self.fatigue, 0.2, 0.95)
        self.attention = clamp(
            self.attention + 0.3 * (target_attn - self.attention) + 0.02 * rng.gauss(0, 1),
            0.1, 1.0,
        )
        # Pace autocorrelates (inertia across actions).
        self.pace = clamp(0.8 * self.pace + 0.2 * rng.random(), 0.05, 0.95)

        if action is not None:
            self.last_action = action

    def observe_confidence(self, match_quality: float) -> None:
        """Set confidence from a perception match (L2 feeds this).

        ``match_quality`` is a 0..1 template/OCR quality. Confidence is a
        smoothed follow of it so a single weak frame doesn't whipsaw behavior.
        """
        mq = clamp(match_quality, 0.0, 1.0)
        self.confidence = clamp(0.5 * self.confidence + 0.5 * mq, 0.0, 1.0)

    def remember_region(self, region_key: tuple[int, int]) -> None:
        self.visited_regions.add(region_key)

    def note_signature(self, sig: str, keep: int = 24) -> None:
        self.recent_signatures.append(sig)
        if len(self.recent_signatures) > keep:
            self.recent_signatures.pop(0)

    def note_failed_target(self, xy: tuple[float, float]) -> None:
        self.last_failed_target = xy

    # ── derived modulation the other layers read ─────────────────────────────

    def think_time_scale(self) -> float:
        """Multiplier on per-decision think time (L4).

        Fatigue lengthens it, familiarity and urgency shorten it, and low
        confidence lengthens it (you inspect before acting when unsure).
        """
        s = 1.0
        s *= 1.0 + 0.6 * self.fatigue
        s *= 1.0 - 0.45 * self.familiarity
        s *= 1.0 - 0.5 * self.urgency
        s *= 1.0 + 0.5 * (1.0 - self.confidence)
        return clamp(s, 0.35, 3.0)

    def dwell_scale(self) -> float:
        """Multiplier on tap dwell (L3). Urgency shortens, fatigue lengthens."""
        return clamp((1.0 + 0.4 * self.fatigue) * (1.0 - 0.3 * self.urgency), 0.6, 1.8)

    def tremor_scale(self) -> float:
        """Multiplier on tremor amplitude (L3). Low attention + fatigue -> more."""
        return clamp((1.0 + 0.8 * self.fatigue) * (1.0 + 0.6 * (1.0 - self.attention)), 0.7, 2.5)

    def submovement_bias(self) -> float:
        """Additive bias (in submovement count) from fatigue + low confidence."""
        return 1.5 * self.fatigue + 1.0 * (1.0 - self.confidence)

    def straightness(self) -> float:
        """0..1 - how straight/ballistic the path is. Familiarity straightens."""
        return clamp(0.4 + 0.5 * self.familiarity + 0.2 * self.attention, 0.0, 1.0)

    def overshoot_scale(self) -> float:
        """Multiplier on endpoint overshoot. Familiarity shrinks it."""
        return clamp(1.2 - 0.6 * self.familiarity, 0.4, 1.3)

    def cancel_probability(self) -> float:
        """Probability of a hesitate-then-cancel on a low-confidence action."""
        return clamp(0.12 * (1.0 - self.confidence) - 0.05 * self.familiarity, 0.0, 0.2)

    def decision(self, rng: Random) -> str:
        """Confidence gates action: commit / inspect / hover / abandon.

        High confidence -> immediate commit; medium -> inspect first; low ->
        hover, sometimes abandon. Used by the engine to decide whether to act
        straight away or insert an inspect/hover beat (which the timing layer
        then costs as think time).
        """
        c = self.confidence
        if c >= 0.75:
            return "commit"
        if c >= 0.5:
            return "inspect"
        # low confidence
        if rng.random() < self.cancel_probability() * 5:  # scaled into low band
            return "abandon"
        return "hover"
