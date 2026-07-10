"""The Hearthstone screen-state classifier.

Standard: closed-loop navigation requires always knowing *which screen we're on*,
so clicks get feedback and an unrecognized screen can HALT rather than blind-tap
(Layer 6 fail-closed). We classify by NCC-matching small anchor glyphs (the
"Starting Hand / Keep or Replace Cards" banner for mulligan, the end-screen
victory/defeat marks, the settings gear, etc.) in their expected regions.

Anchors come from a template pack captured on *your* phone via ``hop capture``;
without a pack, classification returns UNKNOWN and the engine refuses auto mode
(a deliberate, safe degradation). :class:`ScreenClassifier` also exposes the
match confidence so a weak/uncertain read feeds ``HumanState`` (perceptual
fallibility: a weak match means inspect/hesitate, not instant commit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .image import Frame
from .templates import Region, Template, best_match, best_score


class ScreenState(str, Enum):
    MENU = "menu"                 # main menu / mode wheel (NOT where we queue from)
    DECK_SELECT = "deck_select"   # the deck list; tap a deck to reach PLAY_SCREEN
    # The deck-detail screen carrying the big Play button. This is where the hunt
    # loop queues from, and where Hearthstone returns to after a game ends, so it
    # is the loop's home state (verified on-device).
    PLAY_SCREEN = "play_screen"
    QUEUE = "queue"               # searching for opponent (do NOT tap: cancels)
    # "There was an error starting your game." A transient server/network blip
    # that Hearthstone throws often; dismissing it and requeueing works.
    ERROR_DIALOG = "error_dialog"
    # "You are currently offline / It's been a while since your last Hearthstone
    # action and your connection was shut down." Hearthstone drops idle sessions,
    # which the hunt loop can trigger while it waits. Tap Reconnect, not Cancel:
    # Cancel leaves the client offline and every later tap is a no-op.
    RECONNECT_DIALOG = "reconnect_dialog"
    # The same dialog *mid-reconnect*: body reads "Reconnecting..." and both
    # buttons are REMOVED (verified: zero gold-button pixels). Tapping here hits
    # dead space, so this must be a distinct, wait-only state - and it must outrank
    # RECONNECT_DIALOG, whose title-banner anchor still matches during it.
    RECONNECTING = "reconnecting"
    # The "match found" VS intro (hero vs hero, a red "VS" between the portraits). Whether
    # it appears is client-dependent: some fade queue->black->mulligan with no distinct
    # splash (then `_classify_settled` absorbs the black frames as a transient UNKNOWN), but
    # others -- the Pixel 7a among them (LIVE-VERIFIED 2026-07-10) -- hold a distinct VS
    # splash for several seconds, long enough to outlast the settle re-looks and halt as
    # UNKNOWN if it is unanchored. The state and its wait-branch are always present; ship an
    # anchor on the invariant red "VS" glyph (`hop capture --state vs_splash --glyph ...`) and
    # the existing wait-for-the-board branch just works.
    VS_SPLASH = "vs_splash"
    # The card Collection / deck manager. hop is never *supposed* to be here - the
    # end_dismiss geometry + END_SCREENS whitelist keep it from tapping "My Collection"
    # on the deck list - but a stray navigation must be recoverable, not a halt: back
    # out to the deck list. Anchored on the "My Decks" banner, which is chrome (present
    # whatever cards or class filter are showing), not content.
    COLLECTION = "collection"
    # "Incomplete Deck - You are N cards short of a full deck. Complete deck
    # automatically? [Yes] [No]" - a modal over the deck list, thrown when you select a
    # deck missing cards. hop must NEVER auto-complete (tap No, never Yes) and instead
    # pick a complete deck. A modal, so it needs a higher priority than DECK_SELECT,
    # which still matches through it. Anchored on the invariant "Incomplete Deck" title
    # (the card-count in the body varies by deck).
    INCOMPLETE_DECK = "incomplete_deck"
    MULLIGAN = "mulligan"         # starting hand / keep or replace
    IN_GAME = "in_game"           # board visible, our turn or theirs
    VICTORY = "victory"
    DEFEAT = "defeat"
    REWARDS = "rewards"           # post-game rewards popups
    # "Your Quests" - Hearthstone throws this over the play screen after a game.
    # An overlay, so the screen beneath still matches: it needs a higher priority.
    QUEST_POPUP = "quest_popup"
    CONCEDE_MENU = "concede_menu" # the settings/gear overlay with Concede
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Anchor:
    state: ScreenState
    template: Template
    #: Higher wins when several anchors match. A modal dialog does not hide the
    #: screen beneath it, so both anchors can clear their thresholds at once; the
    #: dialog must take precedence or we would act on the occluded screen.
    priority: int = 0


@dataclass(frozen=True)
class Classification:
    state: ScreenState
    #: NCC score of the **winning** anchor, i.e. of ``state`` - not of whichever
    #: anchor happened to score highest. ``0.0`` when ``state`` is UNKNOWN, because
    #: :func:`best_match` returns ``None`` below an anchor's threshold and so a
    #: sub-threshold score is never observed at all.
    confidence: float
    #: Centre of the winning anchor's match, in device px, or ``None`` for UNKNOWN.
    #: The glyph is on the thing it identifies, so this is where that screen *is*.
    at: tuple[int, int] | None = None


#: Anchors that can legitimately draw *over* another screen: modals/dialogs and the
#: end-of-game banners that fade in while the board is still drawn. Every scoped check
#: (:meth:`ScreenClassifier.classify_expected`) scans these UNCONDITIONALLY on top of the
#: states it expects, so narrowing the scan can never blind the loop to an interruption a
#: full :meth:`classify` would have caught -- a disconnect dialog over the queue, a
#: reconnect dialog over the open Game Menu, an opponent-concede banner over the board.
#: This fixed floor is what makes scoping safe: the original "scan anything strictly
#: higher-priority than the expected set" rule scanned the EMPTY set when the expected
#: anchor was itself top-priority (e.g. concede_menu), and would have tapped Concede's
#: fixed coordinate straight into a co-drawn reconnect dialog's Cancel.
INTERRUPT_FLOOR = frozenset({
    ScreenState.RECONNECT_DIALOG, ScreenState.RECONNECTING, ScreenState.ERROR_DIALOG,
    ScreenState.INCOMPLETE_DECK, ScreenState.QUEST_POPUP, ScreenState.CONCEDE_MENU,
    ScreenState.VICTORY, ScreenState.DEFEAT,
})


class ScreenClassifier:
    def __init__(self, anchors: list[Anchor]):
        self.anchors = anchors

    def classify_expected(self, frame: Frame, expected) -> Classification:
        """A cost-scoped :meth:`classify` for sites with a strong prior on the next screen.

        A full ``classify`` NCC-scans every anchor (~3.8 s on this phone). Where the loop
        already knows what should be on screen (still-queueing, the mulligan about to
        leave to the board, the gear about to open the Game Menu), we needn't re-ask
        "which of all 19 screens is this?". Scan only ``expected`` (the states we trust a
        hit on here) plus the fixed :data:`INTERRUPT_FLOOR`, and resolve by the SAME
        ``(priority, score)`` argmax ``classify`` uses over exactly those anchors.

        **Result-identical to** :meth:`classify` **by construction**, which is what lets
        scoping be a pure speed-up that leaves the loop's decisions, journal and RNG
        stream bit-identical (see the co-match fixtures in the tests):

        * A win by a trusted ``expected`` state is returned directly. It is ``classify``'s
          winner too: anchor glyphs sit in disjoint screen regions, so no *un-scanned*
          base anchor can out-score it on a frame showing this screen, and every
          higher-priority modal that could is in the floor and *is* scanned. (The known
          cross-fade where two base anchors co-clear -- mulligan+in_game, defeat+in_game --
          is handled by the caller putting BOTH in ``expected``, so the argmax matches.)
        * ANY other outcome -- a floor modal wins, nothing clears, or UNKNOWN -- forces a
          full :meth:`classify` on the SAME frame (no second screencap, no RNG, no sleep).

        So a wrong or missing ``expected`` only costs the full scan back; it can never
        return a screen ``classify`` wouldn't have. ``expected`` is any iterable of
        :class:`ScreenState`; an empty one degrades to a plain ``classify``.
        """
        expected = frozenset(expected)
        scan = expected | INTERRUPT_FLOOR
        best_state = ScreenState.UNKNOWN
        best_rank: tuple[int, float] | None = None
        winner = None
        for anchor in self.anchors:
            if anchor.state not in scan:
                continue
            m = best_match(frame, anchor.template)
            if not m:
                continue
            rank = (anchor.priority, m.score)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_state = anchor.state
                winner = m
        if winner is not None and best_state in expected:
            return Classification(best_state, winner.score, (winner.x, winner.y))
        # Floor-modal win / nothing cleared / UNKNOWN: we cannot be sure the scoped scan
        # saw the real winner, so pay the authoritative full scan on this same frame.
        return self.classify(frame)

    def classify(self, frame: Frame) -> Classification:
        """Return the best-matching screen state and *its* confidence.

        Among anchors that clear their own threshold, the winner is the highest
        ``priority`` and then the highest score - so a modal dialog beats the
        screen it is drawn over. If nothing clears its threshold the state is
        UNKNOWN; the engine re-looks a bounded number of times and then halts,
        never blind-tapping.

        The confidence is the **winner's** score. It used to be ``max`` over every
        anchor that cleared, which is the same number only when priority does not
        decide the winner - and priority deciding the winner is exactly the
        interesting case. A concede menu (priority 10, score 0.75) drawn over a
        board whose ``in_game`` anchor still matches at 0.95 reported
        ``Classification(CONCEDE_MENU, 0.95)``: the confidence of a *different*
        screen. It is read by the halt message, the debug journal and the dashboard
        (:mod:`hop.runner`), i.e. by everything a human uses to decide whether the
        classifier is trustworthy - so it had better describe the screen we picked.

        (The module docstring above promises this number feeds ``HumanState``. It
        does not, yet: only the OCR's ``class_confidence`` does. Fixing that is only
        safe now that this reports the right anchor's score.)
        """
        best_state = ScreenState.UNKNOWN
        best_rank: tuple[int, float] | None = None
        winner = None
        for anchor in self.anchors:
            m = best_match(frame, anchor.template)
            if not m:
                continue
            rank = (anchor.priority, m.score)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_state = anchor.state
                winner = m
        if winner is None:
            return Classification(ScreenState.UNKNOWN, 0.0, None)
        return Classification(best_state, winner.score, (winner.x, winner.y))

    def rank(self, frame: Frame) -> list[tuple[ScreenState, float, float]]:
        """Every anchor's best score *ignoring its threshold*, highest first, as
        ``(state, score, threshold)`` triples.

        Recorded when a frame comes back UNKNOWN so the journal says which known screen
        it was CLOSEST to. A near-miss ("in_game 0.539, threshold 0.72") points straight
        at the fix -- that state needs another visual face, or a looser threshold --
        where a bare "unknown, confidence 0.0" said nothing at all.
        """
        out: list[tuple[ScreenState, float, float]] = []
        for anchor in self.anchors:
            m = best_score(frame, anchor.template)
            if m is not None:
                out.append((anchor.state, m.score, anchor.template.threshold))
        out.sort(key=lambda r: -r[1])
        return out

    @property
    def has_templates(self) -> bool:
        return bool(self.anchors)


def load_template_pack(directory: str | Path) -> ScreenClassifier:
    """Load anchors from a template-pack directory.

    Layout: ``<dir>/screens.json`` describes each anchor as
    ``{"state": "mulligan", "image": "mulligan_banner.png",
    "region": [xf, yf, wf, hf], "threshold": 0.75}``, with PNGs alongside.
    Produced by ``hop capture``. Returns an empty classifier if absent.
    """
    directory = Path(directory)
    meta_path = directory / "screens.json"
    if not meta_path.exists():
        return ScreenClassifier([])
    meta = json.loads(meta_path.read_text())
    anchors: list[Anchor] = []
    for entry in meta.get("anchors", []):
        img = Frame.from_png((directory / entry["image"]).read_bytes())
        r = entry.get("region", [0, 0, 1, 1])
        tmpl = Template(
            name=entry["state"],
            image=img,
            region=Region(*r),
            threshold=float(entry.get("threshold", 0.72)),
        )
        anchors.append(Anchor(ScreenState(entry["state"]), tmpl,
                              int(entry.get("priority", 0))))
    return ScreenClassifier(anchors)
