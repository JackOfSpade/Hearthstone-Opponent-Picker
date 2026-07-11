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

#: Distinctive ONE-WORD fragments of the two-word class labels, for the failure path of
#: :func:`snap_ocr_to_class`. Hearthstone renders "DEATH KNIGHT" and "DEMON HUNTER" as two
#: words, and a slow or slightly-misframed OCR of that long (10-11 char) nameplate often
#: catches only ONE word -- sometimes plus a stray neighbouring glyph, e.g. "G DEATH" for a
#: Death Knight whose hero-name tail bled into the class crop. Such a fragment sits 5-7 edits
#: from the full label, so the edit-distance snap (max ~3) rejects it and the engine halts on a
#: "could not read mulligan" even though the class is obvious (this exact halt was seen live:
#: class='G DEATH', cards=4). The single-word classes need no such entry: their whole name is
#: short enough that a partial read stays within the edit-distance budget already.
#:
#: Each fragment is UNIQUE to its class among all 11 and >=4 edits from every OTHER class's
#: label ("DEATH"/"KNIGHT" appear in no other class name, "DEMON" in no other), so snapping to
#: it can never mis-ID a different class within the max-3 budget. "HUNTER" is deliberately
#: ABSENT: it is the Hunter class's own whole word (in _SCREEN_LABELS), so a lone "HUNTER" must
#: stay Hunter, never Demon Hunter. These are consulted ONLY when the full-label snap fails, so
#: a good full-word match is never overridden.
_PARTIAL_LABELS: dict[str, HeroClass] = {
    "DEATH": HeroClass.DEATHKNIGHT,
    "KNIGHT": HeroClass.DEATHKNIGHT,
    "DEMON": HeroClass.DEMONHUNTER,
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


def _best_label_match(norm: str, labels: dict[str, HeroClass]) -> tuple[HeroClass | None, int]:
    """Nearest ``(class, edit-distance)`` over a label->class map, on the normalized string."""
    best_class: HeroClass | None = None
    best_dist = 10**9
    for label, cls in labels.items():
        d = _levenshtein(norm, _normalize(label))
        if d < best_dist:
            best_dist, best_class = d, cls
            if d == 0:
                break
    return best_class, best_dist


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

    If the full-label snap FAILS the ``max_distance`` cutoff, one more attempt is made against
    the distinctive one-word fragments of the two-word classes (:data:`_PARTIAL_LABELS`): a long
    "DEATH KNIGHT"/"DEMON HUNTER" nameplate frequently OCRs as just one of its words (+/- a stray
    glyph, e.g. "G DEATH"), which sits far outside the budget from the full label but 0-1 edits
    from its fragment. This only runs on the failure path, so a good full-word match is never
    overridden, and each fragment is unique to its class (see :data:`_PARTIAL_LABELS`).
    """
    norm = _normalize(text)
    if not norm:
        return (None, 10**9)

    best_class, best_dist = _best_label_match(norm, _SCREEN_LABELS)

    if max_distance is not None and best_dist > max_distance:
        # Full-label snap missed. Try the distinctive fragments before failing closed.
        frag_class, frag_dist = _best_label_match(norm, _PARTIAL_LABELS)
        if frag_dist < best_dist:
            best_class, best_dist = frag_class, frag_dist

    if max_distance is not None and best_dist > max_distance:
        return (None, best_dist)
    return (best_class, best_dist)


def nearest_class(text: str) -> tuple[HeroClass | None, int]:
    """The most likely class for a raw OCR reading, over BOTH the full labels and the
    distinctive fragments, with NO distance cutoff. For DIAGNOSTICS only (naming the probable
    class of an unresolvable read in a bug report) -- not the fail-closed snap. Returns
    ``(class_or_None, distance)``; the distance is to whichever of the two label sets matched
    closest, so a fragment hit (e.g. "G DEATH" -> Death Knight, distance 1) is reported as the
    fragment distance, not the misleading full-label distance (which here would be Druid, 5)."""
    norm = _normalize(text)
    if not norm:
        return (None, 10**9)
    fc, fd = _best_label_match(norm, _SCREEN_LABELS)
    pc, pd = _best_label_match(norm, _PARTIAL_LABELS)
    return (pc, pd) if pd < fd else (fc, fd)
