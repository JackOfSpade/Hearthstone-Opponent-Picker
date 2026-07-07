"""Seedable randomness helpers.

Standard, cross-cutting: *"Every motion/timing/state primitive takes an
injectable RNG (`rng=Random(seed)`) so behavior is unit-testable and
reproducible, defaulting to real randomness in production."*

Every pure function in :mod:`hop.humanize` accepts an ``rng`` argument of type
:class:`random.Random`. Production code passes a fresh unseeded ``Random()``;
tests pass ``Random(seed)`` for determinism. Nothing in the humanization layers
may call the module-level ``random.*`` functions or ``time``/``Date`` for
randomness - that would break both testability and reproducibility.
"""

from __future__ import annotations

import math
from random import Random

__all__ = [
    "default_rng",
    "lognormal_multiplier",
    "shifted_lognormal",
    "clamp",
    "ou_step",
    "beta_sample",
]


def default_rng() -> Random:
    """A fresh, entropy-seeded RNG for production use."""
    return Random()


def clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, value))


def lognormal_multiplier(rng: Random, sigma: float) -> float:
    """A positive multiplier centered near 1 with a right-skewed long tail.

    This is the ``exp(N(0, sigma) * sigma)`` form used by the timing layer
    (§8 reference ``sigma`` = 0.22). The result is always > 0, usually close to
    1, occasionally much larger - the organic long tail the standard requires
    instead of ``uniform(a, b)``.
    """
    return math.exp(rng.gauss(0.0, sigma) * sigma)


def shifted_lognormal(rng: Random, shift: float, mu: float, sigma: float) -> float:
    """Shifted-lognormal reaction-time sample: ``shift + exp(N(mu, sigma))``.

    Used for per-decision think time (§8): a non-negative floor (``shift`` -
    minimum perceptual+motor latency) plus a right-skewed lognormal body. The
    mean is roughly ``shift + exp(mu + sigma**2/2)``.
    """
    return shift + math.exp(rng.gauss(mu, sigma))


def ou_step(rng: Random, prev: float, theta: float, sigma: float, dt: float) -> float:
    """One step of an Ornstein-Uhlenbeck (mean-reverting) process, mean 0.

    Produces temporally *correlated* noise - the texture the standard wants for
    tremor, as opposed to white noise. ``theta`` is the reversion rate, ``sigma``
    the volatility, ``dt`` the timestep.
    """
    return (
        prev
        - theta * prev * dt
        + sigma * math.sqrt(max(dt, 0.0)) * rng.gauss(0.0, 1.0)
    )


def beta_sample(rng: Random, alpha: float, beta: float) -> float:
    """Sample from a Beta(alpha, beta) distribution on [0, 1].

    ``random.Random`` has no ``betavariate`` that accepts our exact call shape
    uniformly across versions, but it does - we wrap it so the humanize layer
    depends only on this module. Falls back to a gamma-ratio construction if a
    future Random lacks it.
    """
    try:
        return rng.betavariate(alpha, beta)
    except AttributeError:  # pragma: no cover - defensive
        ga = rng.gammavariate(alpha, 1.0)
        gb = rng.gammavariate(beta, 1.0)
        return ga / (ga + gb) if (ga + gb) > 0 else 0.5
