"""The 11 Hearthstone classes and the OCR-snap that resolves a noisy read of
the mulligan class label to exactly one of them.

Detection design (settled over the design discussion): at the mulligan screen
the game prints the opponent's *class word* ("MAGE", "WARLOCK", ...) as plain
static text under the bottom-left portrait - immune to hero skins, multi-class
skins, and golden-hero animation (all of which only affect the portrait/hero
power). So class detection is: OCR that region, then snap the result to the
nearest of these 11 fixed strings by edit distance. Because the vocabulary is
only 11 words, the snap is effectively a deterministic classifier - no trained
model, no per-run cost.
"""

from __future__ import annotations

from enum import Enum


class HeroClass(str, Enum):
    WARRIOR = "WARRIOR"
    SHAMAN = "SHAMAN"
    ROGUE = "ROGUE"
    PALADIN = "PALADIN"
    HUNTER = "HUNTER"
    DRUID = "DRUID"
    WARLOCK = "WARLOCK"
    MAGE = "MAGE"
    PRIEST = "PRIEST"
    DEMONHUNTER = "DEMONHUNTER"
    DEATHKNIGHT = "DEATHKNIGHT"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# The exact uppercase strings Hearthstone renders on the mulligan nameplate.
# Demon Hunter and Death Knight render as two words on screen; we match against
# the spaced forms too. Keys are what OCR might return; values are the canonical
# enum member.
_SCREEN_LABELS: dict[str, HeroClass] = {
    "WARRIOR": HeroClass.WARRIOR,
    "SHAMAN": HeroClass.SHAMAN,
    "ROGUE": HeroClass.ROGUE,
    "PALADIN": HeroClass.PALADIN,
    "HUNTER": HeroClass.HUNTER,
    "DRUID": HeroClass.DRUID,
    "WARLOCK": HeroClass.WARLOCK,
    "MAGE": HeroClass.MAGE,
    "PRIEST": HeroClass.PRIEST,
    "DEMON HUNTER": HeroClass.DEMONHUNTER,
    "DEMONHUNTER": HeroClass.DEMONHUNTER,
    "DEATH KNIGHT": HeroClass.DEATHKNIGHT,
    "DEATHKNIGHT": HeroClass.DEATHKNIGHT,
}

DISPLAY_NAMES: dict[HeroClass, str] = {
    HeroClass.WARRIOR: "Warrior",
    HeroClass.SHAMAN: "Shaman",
    HeroClass.ROGUE: "Rogue",
    HeroClass.PALADIN: "Paladin",
    HeroClass.HUNTER: "Hunter",
    HeroClass.DRUID: "Druid",
    HeroClass.WARLOCK: "Warlock",
    HeroClass.MAGE: "Mage",
    HeroClass.PRIEST: "Priest",
    HeroClass.DEMONHUNTER: "Demon Hunter",
    HeroClass.DEATHKNIGHT: "Death Knight",
}

# User-facing aliases accepted from config/CLI.
_ALIASES: dict[str, HeroClass] = {
    "WAR": HeroClass.WARRIOR, "WARRIOR": HeroClass.WARRIOR,
    "SHAM": HeroClass.SHAMAN, "SHAMAN": HeroClass.SHAMAN,
    "ROGUE": HeroClass.ROGUE,
    "PALADIN": HeroClass.PALADIN, "PALLY": HeroClass.PALADIN, "PAL": HeroClass.PALADIN,
    "HUNTER": HeroClass.HUNTER, "HUNT": HeroClass.HUNTER,
    "DRUID": HeroClass.DRUID,
    "WARLOCK": HeroClass.WARLOCK, "LOCK": HeroClass.WARLOCK,
    "MAGE": HeroClass.MAGE,
    "PRIEST": HeroClass.PRIEST,
    "DH": HeroClass.DEMONHUNTER, "DEMONHUNTER": HeroClass.DEMONHUNTER,
    "DEMON": HeroClass.DEMONHUNTER,
    "DK": HeroClass.DEATHKNIGHT, "DEATHKNIGHT": HeroClass.DEATHKNIGHT,
    "DEATH": HeroClass.DEATHKNIGHT,
}

ALL_CLASSES: tuple[HeroClass, ...] = tuple(HeroClass)


def _normalize(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch.isalnum())


def parse_class(name: str) -> HeroClass:
    """Parse a user-supplied class name/abbreviation (case/space-insensitive)."""
    key = _normalize(name)
    if key in _ALIASES:
        return _ALIASES[key]
    for member in HeroClass:
        if _normalize(member.value) == key:
            return member
    valid = ", ".join(sorted(DISPLAY_NAMES.values()))
    raise ValueError(f"Unknown Hearthstone class {name!r}. Valid: {valid}")


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance. Small strings, so the simple DP is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def snap_ocr_to_class(text: str, max_distance: int | None = None) -> tuple[HeroClass | None, int]:
    """Snap a raw OCR reading to the nearest of the 11 class labels.

    Returns ``(class_or_None, distance)``. ``distance`` is the edit distance to
    the chosen label (0 == exact). If ``max_distance`` is given and the best
    match exceeds it, returns ``(None, distance)`` so the caller can treat a
    junk read as "unknown" rather than force a wrong class - the fail-closed
    posture the standard requires at the perception boundary.

    The comparison is on the alnum-normalized uppercase form, so spacing,
    punctuation and case noise never matter; only genuine glyph errors count
    toward the distance.
    """
    norm = _normalize(text)
    if not norm:
        return (None, 10**9)

    best_class: HeroClass | None = None
    best_dist = 10**9
    for label, cls in _SCREEN_LABELS.items():
        d = _levenshtein(norm, _normalize(label))
        if d < best_dist:
            best_dist, best_class = d, cls
            if d == 0:
                break

    if max_distance is not None and best_dist > max_distance:
        return (None, best_dist)
    return (best_class, best_dist)
