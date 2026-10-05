"""Layer 5 - the Hearthstone journey grammar.

For this fixed-phone workflow, a rejected matchup follows one short, predictable
path: leave the starting hand untouched and concede directly from the positively
named mulligan. The engine owns the screen transitions and actual tap timing; this
module only describes that path and the target-match mulligan.
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


def plan_mulligan(rng: Random, num_cards: int, keeping: bool) -> list[MulliganDecision]:
    """Decide which mulligan cards to replace and how each is deliberated.

    We can't know the game-theoretic-optimal mulligan, and we don't need to -
    we need *plausible*. Heuristic: usually keep 1-3 cards, replace the rest,
    biased toward keeping cheaper (left-most tends to be lower cost after the
    engine sorts, but we don't rely on that) with per-slot noise. Target games
    receive the normal, careful mulligan; the ``keeping`` argument remains for
    compatibility with callers that request the older generic behavior.
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


def choose_concede_point(rng: Random) -> str:
    """Return the immediate post-mulligan concede point.

    ``rng`` is retained in the public signature so existing callers need no
    migration; this fixed policy intentionally makes no random draw.
    """
    del rng
    return "mulligan"


@dataclass(frozen=True)
class RejectPlan:
    """The immediate, no-extra-actions plan for a rejected matchup.

    The fields stay stable for the engine's journal and dispatch code.  Their
    fixed values prevent replacement taps, board-play beats, and extra pauses.
    """

    concede_point: str                 # always "mulligan"
    mulligan: list[MulliganDecision]
    hesitate_before_concede: bool      # always False
    extra_reads: int                   # always 0


def plan_reject(rng: Random, num_cards: int) -> RejectPlan:
    """Plan a direct concede from a named, untouched mulligan.

    ``rng`` and ``num_cards`` remain accepted for engine compatibility, but a
    rejected matchup deliberately spends neither on decorative interactions.
    """
    del num_cards
    return RejectPlan(
        concede_point=choose_concede_point(rng),
        mulligan=[],
        hesitate_before_concede=False,
        extra_reads=0,
    )


def plan_keep(rng: Random, num_cards: int) -> list[MulliganDecision]:
    """For a target matchup we're keeping: a careful, real mulligan, then we
    hand control to the human (we alert and stop touching the game)."""
    return plan_mulligan(rng, num_cards, keeping=True)
