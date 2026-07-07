"""Layer 5 - the Hearthstone journey grammar.

Standard: *"real sessions have structure a flat ratio misses ... Define a
per-vertical journey grammar."* The vertical here is Hearthstone laddering, and
the grammar's whole job is to make a *rejected* game look like an impatient
human quitting a bad matchup, not a barcode bot insta-conceding at class reveal.

The load-bearing anti-barcode rule (from the design discussion): **detect-time
is decoupled from concede-time.** We know at the mulligan whether to keep or
reject, but a reject still: performs the mulligan like a human (keep/replace a
plausible subset), enters the game, plays a beat or two, and concedes at a
*randomly chosen* point. Instant concedes are the single most detectable thing
this tool could do, so the grammar never emits one on the cautious profile.

This module plans the cognitive *shape* (which cards to replace, which turn to
bail on, where to hesitate). The concrete screen taps live in the engine; the
timing of each beat is costed by :mod:`hop.humanize.timing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class MulliganDecision:
    """One card slot's keep/replace choice, with the think-time flavor it earns.

    ``decision_type`` feeds the timing layer: keeping a card you like is a
    faster "commit" reaction; deliberating a replace is a slower "reject".
    """

    slot: int
    replace: bool
    decision_type: str  # "commit" | "reject"


# Concede points, and their base weights. min_play_turns (from the risk profile)
# pushes the distribution later (cautious) or earlier (aggressive).
_CONCEDE_POINTS = ("mulligan", "turn1", "turn2")


def plan_mulligan(rng: Random, num_cards: int, keeping: bool) -> list[MulliganDecision]:
    """Decide which mulligan cards to replace and how each is deliberated.

    We can't know the game-theoretic-optimal mulligan, and we don't need to -
    we need *plausible*. Heuristic: usually keep 1-3 cards, replace the rest,
    biased toward keeping cheaper (left-most tends to be lower cost after the
    engine sorts, but we don't rely on that) with per-slot noise. A game we're
    keeping deliberates a touch more carefully (real stakes); a reject is a bit
    more careless but still *interacts* - never a zero-touch mulligan.
    """
    decisions: list[MulliganDecision] = []
    # target number to keep: keepers ~ Binomial-ish, clamped to [1, num_cards].
    base_keep = 2 if keeping else rng.choice([1, 2, 2, 3])
    keep_target = max(1, min(num_cards, base_keep + rng.choice([-1, 0, 0, 1])))
    kept = 0
    for slot in range(num_cards):
        remaining_slots = num_cards - slot
        need = keep_target - kept
        # probability of keeping this slot so we land near keep_target
        p_keep = 0.0 if need <= 0 else min(1.0, need / remaining_slots + rng.uniform(-0.1, 0.1))
        replace = rng.random() > p_keep
        if not replace:
            kept += 1
        dtype = "reject" if replace else "commit"
        decisions.append(MulliganDecision(slot=slot, replace=replace, decision_type=dtype))
    return decisions


def choose_concede_point(rng: Random, min_play_turns: float) -> str:
    """Pick where in a rejected game to concede.

    ``min_play_turns`` in [0, 1]: 1.0 (cautious) heavily favors playing into the
    game before conceding; 0.0 (aggressive) allows a fast mulligan-stage concede.
    """
    # Weight later points more as min_play_turns rises.
    w_mull = 1.0 + 3.0 * (1.0 - min_play_turns)   # aggressive inflates early bail
    w_t1 = 1.2 + 0.6 * min_play_turns
    w_t2 = 0.4 + 1.6 * min_play_turns
    weights = [w_mull, w_t1, w_t2]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for point, w in zip(_CONCEDE_POINTS, weights):
        acc += w
        if r <= acc:
            return point
    return _CONCEDE_POINTS[-1]


@dataclass(frozen=True)
class RejectPlan:
    """The full plan for exiting a rejected game plausibly."""

    concede_point: str                 # "mulligan" | "turn1" | "turn2"
    mulligan: list[MulliganDecision]
    hesitate_before_concede: bool      # insert a hesitation beat (real quitters pause)
    extra_reads: int                   # number of idle "reading the board" pauses


def plan_reject(rng: Random, num_cards: int, min_play_turns: float) -> RejectPlan:
    """Compose a complete plausible-exit plan for a matchup we will concede."""
    point = choose_concede_point(rng, min_play_turns)
    mull = plan_mulligan(rng, num_cards, keeping=False)
    hesitate = rng.random() < (0.5 + 0.3 * min_play_turns)
    extra = {"mulligan": 0, "turn1": rng.choice([0, 1]), "turn2": rng.choice([1, 1, 2])}[point]
    return RejectPlan(
        concede_point=point,
        mulligan=mull,
        hesitate_before_concede=hesitate,
        extra_reads=extra,
    )


def plan_keep(rng: Random, num_cards: int) -> list[MulliganDecision]:
    """For a target matchup we're keeping: a careful, real mulligan, then we
    hand control to the human (we alert and stop touching the game)."""
    return plan_mulligan(rng, num_cards, keeping=True)
