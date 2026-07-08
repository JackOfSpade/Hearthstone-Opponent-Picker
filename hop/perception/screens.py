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
from .templates import Region, Template, best_match


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
    VS_SPLASH = "vs_splash"       # the VS intro
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
    confidence: float             # NCC score of the winning anchor (0..1)


class ScreenClassifier:
    def __init__(self, anchors: list[Anchor], accept: float = 0.72):
        self.anchors = anchors
        self.accept = accept

    def classify(self, frame: Frame) -> Classification:
        """Return the best-matching screen state and its confidence.

        Among anchors that clear their own threshold, the winner is the highest
        ``priority`` and then the highest score - so a modal dialog beats the
        screen it is drawn over. If nothing clears its threshold the state is
        UNKNOWN with the best score seen; the engine re-looks a bounded number of
        times and then halts, never blind-tapping.
        """
        best_state = ScreenState.UNKNOWN
        best_score = 0.0
        best_rank: tuple[int, float] | None = None
        for anchor in self.anchors:
            m = best_match(frame, anchor.template)
            if not m:
                continue
            best_score = max(best_score, m.score)
            rank = (anchor.priority, m.score)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_state = anchor.state
        return Classification(best_state, best_score)

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
