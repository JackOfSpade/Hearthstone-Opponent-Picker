"""Self-improving bug reports.

The user almost never has a terminal open, so when something goes wrong the evidence -
the per-run journal, the app log, the device/anchor state - is invisible to them. This
bundles all of it, plus the user's own description, into one shareable Markdown file.

**Self-improving.** The report opens with an instruction to Claude Code: after fixing
the issue, judge whether the report was *missing* anything that would have made the
diagnosis faster, and if so, improve *this harness* so the next report carries it. A
report that gets richer every time it is used is the goal - modelled on the same idea
in ~/Desktop/My Apps/infinite-canvas.

Pure assembly (:func:`assemble`) is separated from I/O (:func:`collect`) so the report
format is testable without a device, a run, or a filesystem.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path


# The meta-instruction that makes the report self-improving. Kept blockquoted and FIRST
# so it survives any truncation and is the first thing Claude Code reads on paste-back.
SELF_IMPROVE_PROMPT = """\
> **To Claude Code, on paste-back.** Two jobs, in order:
>
> 1. **Diagnose and fix** the issue described below, using the logs and state attached.
> 2. **Then improve this bug report harness.** Assess whether the report was *missing*
>    any information that would have let you find the cause faster or more certainly -
>    a counter never surfaced, a log line never captured, a device fact never probed.
>    If so, add that collection to `hop/bugreport.py` (and any logging it needs) so the
>    **next** report includes it automatically. The report should get richer every time
>    it is used. If nothing was missing, say so explicitly and skip step 2.
"""


#: Hard ceiling on the whole report's CHARACTER count. The report is pasted straight into
#: Claude Code, which truncates a pasted message at ~50k characters -- and that truncation
#: lands mid-journal, cutting the NEWEST lines (the transition into the halt, the ones that
#: actually diagnose it). We must trim ourselves, oldest-first, so the tail survives. This
#: is characters, not lines, on purpose: a 500-line journal is only ~1000 lines but ~100k
#: characters, so a line cap never fired and let the external truncator cut the diagnosis.
#: A touch under 50k for margin (the note we add, and any downstream wrapping, cost a little).
MAX_REPORT_CHARS = 48_000

#: Sections trimmed (in this order, oldest-line-first) when a report exceeds the cap: the
#: app log first (mostly launch noise), then the run journal -- and because the journal
#: section is emitted BEFORE the app log, even the final backstop truncation eats app-log
#: tail, never the journal's newest lines.
_TRIMMABLE_TITLES = ("App log (~/Library/Logs/hop.log)", "Latest run journal (tail)")


@dataclass(frozen=True)
class Section:
    """One titled block of the report. ``fenced`` wraps the body in a code fence."""

    title: str
    body: str
    fenced: bool = False
    lang: str = ""


def assemble(description: str, sections: list[Section], *, meta: dict) -> str:
    """Build the report Markdown. Pure: no I/O, so the format is unit-testable.

    ``meta`` supplies the header facts (version, platform, when). ``sections`` are the
    gathered evidence, rendered in order; an empty-body section is dropped so a report
    is never padded with "(none)" noise.
    """
    lines: list[str] = [SELF_IMPROVE_PROMPT, "", "# hop bug report", ""]
    lines.append(f"- **hop**: {meta.get('version', '?')}")
    lines.append(f"- **when**: {meta.get('when', '?')}")
    lines.append(f"- **platform**: {meta.get('platform', '?')}")
    lines.append("")
    lines.append("## What went wrong (from the user)")
    lines.append("")
    lines.append(description.strip() or "_(no description given)_")
    lines.append("")

    for s in sections:
        body = (s.body or "").strip()
        if not body:
            continue
        lines.append(f"## {s.title}")
        lines.append("")
        if s.fenced:
            lines.append(f"```{s.lang}")
            lines.append(body)
            lines.append("```")
        else:
            lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── redaction ────────────────────────────────────────────────────────────────

#: Config keys whose values are the user's private channels/addresses, redacted from a
#: report meant to be shared. Extend this as new secretish keys appear.
_REDACT_KEYS = ("ntfy_topic", "ntfy_server")


def redact_toml(text: str) -> str:
    """Blank the values of :data:`_REDACT_KEYS` in a TOML config, line by line.

    Line-based (not a parse/re-emit) so the file's comments survive - the same reason
    :mod:`hop.tomledit` is line-based.
    """
    out = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if any(stripped.startswith(k) and "=" in stripped for k in _REDACT_KEYS):
            key = stripped.split("=", 1)[0].rstrip()
            indent = line[: len(line) - len(stripped)]
            out.append(f'{indent}{key} = "<redacted>"')
        else:
            out.append(line)
    return "\n".join(out)


# ── log helpers ──────────────────────────────────────────────────────────────

def tail(text: str, max_lines: int) -> str:
    """Last ``max_lines`` lines, with a note if anything was dropped."""
    all_lines = text.splitlines()
    if len(all_lines) <= max_lines:
        return text.rstrip()
    dropped = len(all_lines) - max_lines
    return f"... ({dropped} earlier lines omitted) ...\n" + "\n".join(all_lines[-max_lines:])


#: Phrases the *button-transition* stuck halts use -- the ``_wait_until_screen_leaves`` waits
#: (mulligan Confirm, gear, Concede, deck dialog) where a verified tap should have moved us to
#: a NEW screen and, positively, did not. A halt matching one of these whose triggering tap had
#: already verified OK is a false halt on the unnamed DESTINATION, not a stuck source (see below).
#: Deliberately NOT "never ...": the engine's "never came back online" / "board never finished
#: dissolving" halts are genuine external/async timeouts (a network outage, a slow fade), not a
#: tap that landed on an unclassifiable screen -- and the destination-anchor advice is wrong for
#: them. All the true targets match via "did not X", so "never " is unneeded and only over-matches.
_STUCK_HALT_PHRASES = ("did not dismiss", "did not open", "did not close")


def _is_stuck_halt(message: str) -> bool:
    m = (message or "").lower()
    return any(p in m for p in _STUCK_HALT_PHRASES)


#: The "could not read mulligan (class=<raw>, cards=<n>)" halt carries BOTH signals the
#: mulligan read needs, so which one failed is decidable from the message alone.
_MULLIGAN_READ_HALT_RE = re.compile(
    r"could not read mulligan \(class=(?P<cls>.*), cards=(?P<cards>-?\d+)\)")


def _diagnose_mulligan_read_halt(message: str) -> str:
    """Name the cause of a 'could not read mulligan' halt from its own message.

    The read needs two signals: the opponent's CLASS (OCR of the bottom-left nameplate)
    and our CARD count (the green keep-glow strips). The halt message carries both, so
    which failed is decidable here instead of by opening the kept frame. A blank
    class no longer reaches this halt: the match watch keeps capturing without input
    until a class is readable. Returns "" for any other halt.
    """
    m = _MULLIGAN_READ_HALT_RE.search(message or "")
    if not m:
        return ""
    cls_raw = m.group("cls").strip()
    blank = cls_raw in ("''", '""', "")
    try:
        cards = int(m.group("cards"))
    except ValueError:
        cards = -1
    cards_ok = cards in (3, 4)
    if blank and cards_ok:
        return ("the opponent-CLASS OCR was blank (class='') while cards counted fine "
                f"(cards={cards}). Current runs do not halt or concede here: they keep the "
                "hands-off match watch active until the nameplate resolves. This halt is from "
                "an older run or an unexpected code path.")
    if cards_ok:   # class present but did not snap to a known class
        base = (f"the class text ({cls_raw}) did not resolve to a known class though the "
                f"cards counted fine (cards={cards}) -- a GARBLED nameplate read, not a blank one.")
        # Name the LIKELY class (over full labels AND the distinctive two-word fragments), so a
        # garbled read is self-identifying instead of leaving the reader to eyeball the raw text.
        # This is exactly what would have made the live 'G DEATH' halt instant: it is a Death
        # Knight whose long two-word label OCR'd as one word -- nearest_class reports Death Knight
        # (distance 1 to the "DEATH" fragment), not the misleading nearest FULL label (Druid, 5).
        hint = ""
        try:
            from .hero_classes import DISPLAY_NAMES, nearest_class
            near, dist = nearest_class(cls_raw.strip().strip("'\""))
            if near is not None:
                hint = (f" It is closest to {DISPLAY_NAMES.get(near, near)} (edit distance "
                        f"{dist}) -- if that looks right, the OCR caught only part of a long "
                        "two-word class label (DEATH KNIGHT / DEMON HUNTER); this build now snaps "
                        "such fragments, so a recurrence means the read missed even the fragment "
                        "(check the opponent-class region alignment / the kept frame).")
        except Exception:
            pass
        return (base + hint + " Suspect the opponent-class region or "
                "vision.ocr_max_edit_distance rather than the still-choosing transient.")
    return (f"the CARD count was off (cards={cards}, expected 3 or 4) -- the green keep-glow "
            "strip detection miscounted, so this is a card-count/glow failure, not an "
            "opponent-class read. Inspect the saved colour frame and glow-run telemetry in "
            "the Mulligan card-count evidence section (new runs retain both automatically).")


#: Every bounded-retry committing-tap site (`_concede`, `_tap_play`, ...) that exhausts its
#: budget shares this exact phrase in its terminal Halt message -- see
#: :meth:`hop.engine.Engine._concede` / :meth:`hop.engine.Engine._tap_play`. Matching the ONE
#: substring covers every such site, present and future, without a table entry per screen.
_DROPPED_COMMITTING_TAP_PHRASE = "not registering (a dropped committing tap"


def _diagnose_dropped_committing_tap_halt(message: str) -> str:
    """Name the cause of a bounded-retry committing-tap halt (Concede, Play, ...): the tap
    was retried its full configured budget and every attempt showed zero pixel change.
    Returns "" for any other halt.
    """
    if _DROPPED_COMMITTING_TAP_PHRASE not in (message or ""):
        return ""
    return ("hop already retried this exact tap its full configured budget (see the "
            "*_tap_ignored count in 'kinds' and the dropped-taps tally below) and EVERY "
            "attempt showed literally zero pixel change -- not a wrong coordinate, a "
            "genuinely DROPPED committing tap. Corroborate with the capture-time TREND above "
            "(a rising mean = a degrading wireless link) and the dropped-taps tally, though a "
            "FLAT trend does not rule this out: this client drops some taps (mulligan cards "
            "~1 in 3) with no known cause even on a healthy link. Raise "
            "the relevant vision.*_tap_attempts knob, or fix the link, rather than chasing a "
            "coordinate change.")


#: The bare message :meth:`hop.verify.Verifier._fail` raises for ``Halt.NO_CHANGE``. Matched
#: as a SUBSTRING, not an exact equality: every halt this journals is wrapped by
#: :meth:`hop.engine.Engine._notify_halt` as ``f"HALTED: {e.reason}"``, so the message on a
#: real journal is never this literal string alone (an exact-equality check here matched
#: nothing on any real run -- caught only by testing against a `HALTED: `-prefixed message,
#: not the bare one every earlier test in this file happened to use). Still safe against a
#: false match: neither `_concede`'s nor `_tap_play`'s own exhaustion message (they end in
#: `_DROPPED_COMMITTING_TAP_PHRASE` instead) contains this substring anywhere.
_BARE_NO_CHANGE_MESSAGE = "no screen change after action (missed tap / stuck)"


def _diagnose_no_change_halt(message: str, last_tap_what: str) -> str:
    """Name the cause of a bare ``Halt.NO_CHANGE`` straight out of ``_tap`` -- the one halt
    shape ``_is_stuck_halt``'s family does not cover, since it never runs through
    :meth:`hop.engine.Engine._wait_until_screen_leaves` (no ``wait_timeout``/
    ``unknown_after_verify`` to key off). ``_tap`` already tried ONE evidence-based
    correction (Layer 5) before raising, so this means BOTH taps at the same coordinate
    produced zero pixel change -- the same committing-tap-drop `_concede`/`_replace_card`
    already retry against, on a tap site that does not (yet) have that retry. Returns "" for
    any other halt.
    """
    if _BARE_NO_CHANGE_MESSAGE not in (message or ""):
        return ""
    what = last_tap_what or "the last action"
    return (f"the '{what}' tap -- and its one evidence-based correction -- both produced "
            "LITERALLY ZERO pixel change. Not a wrong coordinate (the tap point is a fixed "
            f"fraction of the panel, unaffected by this): a committing tap named {what!r} "
            "that this client silently dropped, the same failure `_concede`/`_replace_card` "
            "already retry against. Check whether this tap site has a bounded retry of its "
            "own (grep engine.py for its `what=` string); a tap with no such budget still "
            "fails the whole hunt closed on ONE drop. Corroborate with the capture-time TREND "
            "above (a rising mean = a degrading link) and the dropped-taps tally, though a "
            "flat trend does not rule this out -- this client drops some taps (mulligan cards "
            "~1 in 3) with no known cause even on a healthy link.")


def _norm_class(s) -> str:
    """Normalize a class token so an enum NAME compares equal to a DISPLAY name: uppercase,
    alphanumerics only. ``_norm_class("Death Knight") == _norm_class("DEATHKNIGHT") ==
    "DEATHKNIGHT"``. Shared by both criteria recomputes (the target_found KEEP check and the
    reject-reason check), which compare ``run_criteria`` enum tokens against ``mulligan_read``
    display names."""
    return "".join(ch for ch in str(s).upper() if ch.isalnum())


def _is_pending_mulligan_read(detail: dict) -> bool:
    """Whether a ``mulligan_read`` is observation-only under the watch contract.

    New engine records write ``pending=True`` while the named mulligan's class
    plate has not appeared. These frames remain useful latency/OCR diagnostics,
    but are not games: they cannot create an opponent, coin result, or reject
    pairing. A missing field deliberately remains *not pending* so journals from
    before this contract retain their existing report semantics.
    """
    return detail.get("pending") is True


def _mulligan_turn(detail: dict) -> bool | None:
    """Return the observed coin/turn, or ``None`` when the card count could not tell.

    Newer journals make this distinction explicit: a class can be readable and actionable
    while the keep-glow card count is not, so ``turn_known=False`` and ``second=None``.  Do
    not turn that into ``False`` ("going first") in a report.  Older journal lines did not
    carry ``turn_known``; preserve their historical bool interpretation so old reports do
    not change merely because the renderer was upgraded.
    """
    if detail.get("turn_known") is False:
        return None
    second = detail.get("second")
    if isinstance(second, bool):
        return second
    # An explicit null is the new unknown-turn encoding even if a transitional writer
    # omitted turn_known.  Missing second on a legacy line retains its old falsey result.
    if "second" in detail:
        return None
    return False


def _turn_label(second: bool | None) -> str:
    """Human label for a tri-state mulligan turn without miscalling an unknown as first."""
    if second is None:
        return "turn unreadable"
    return "going 2nd" if second else "going 1st"


def _mulligan_decision(detail: dict) -> str:
    """Journalled decision for a new mulligan read, or ``""`` for legacy records."""
    decision = detail.get("decision")
    return str(decision) if decision in ("keep", "reject", "unusable") else ""


def _is_actionable_mulligan_read(detail: dict) -> bool:
    """Whether a resolved read can be paired with a target/reject decision.

    A current engine can reject a known non-target even when its coin/card count is
    unreadable: the class can be disallowed regardless of the coin, or the criteria can
    simply not use the coin.  Its explicit ``decision=reject`` is stronger evidence than
    the old ``cards in (3, 4)`` proxy.  Legacy records retain the old proxy so a report
    does not reinterpret their semantics.
    """
    decision = _mulligan_decision(detail)
    if decision:
        return decision in ("keep", "reject")
    return detail.get("cards", 3) in (3, 4)


#: Human labels for the three ways :meth:`hop.config.Criteria.accepts` rejects a matchup.
_REJECT_LABELS = {
    "went_first": "went 1st (require_second is set, so only going-2nd games are kept)",
    "turn_unreadable": ("turn/card count unreadable while require_second is set; the report "
                        "cannot independently verify this reject"),
    "not_targeted": "opponent class not in the target list",
    "avoided": "opponent class is on the avoid list",
}


def _reject_reason(opponent: str, we_go_second: bool | None, require_second: bool,
                   targets: list, avoid: list) -> str:
    """Why a *conceded* matchup was rejected, mirroring :meth:`hop.config.Criteria.accepts`
    precedence EXACTLY for known turns (require_second first and class-independent; then target
    list; then avoid list). A missing turn is special: a class that would fail even with the
    favorable coin (for example, Death Knight when only Priest is wanted) is deterministically
    rejectable, so identify that class reason rather than claiming the unknown turn was decisive.
    Returns a key in :data:`_REJECT_LABELS`, or ``"misfire"`` when ``accepts`` would in fact have
    KEPT the matchup -- i.e. a reject fired on a game the criteria say to keep, a real bug,
    flagged as loudly as the target_found misfire check. Only meaningful for a game that genuinely
    rejected (the caller anchors on ``reject_plan``), so ``"misfire"`` is reached only on a true
    keep/reject inversion."""
    if require_second and we_go_second is None:
        # Favorable-coin check for a malformed hand: an excluded/avoided class cannot become a
        # target by being second, so the engine may safely reject it and the report can prove why.
        if targets:
            return "misfire" if _norm_class(opponent) in {_norm_class(t) for t in targets} else "not_targeted"
        if avoid and _norm_class(opponent) in {_norm_class(a) for a in avoid}:
            return "avoided"
        return "turn_unreadable"
    if require_second and not we_go_second:
        return "went_first"
    if targets:
        return "misfire" if _norm_class(opponent) in {_norm_class(t) for t in targets} else "not_targeted"
    if avoid:
        return "avoided" if _norm_class(opponent) in {_norm_class(a) for a in avoid} else "misfire"
    return "misfire"   # no class filter and the coin gate passed => accepts keeps => a misfire


#: The engine's DELIBERATE fail-closed halts: each is a *designed* stop that needs a specific
#: human action, NOT a malfunction. Matching the terminal halt against this table is the one
#: line that answers "is this a bug?" for the whole guard family -- a reader (or Claude on
#: paste-back) sees "expected stop, do X" instead of re-deriving it from the journal, the way
#: this module already pattern-matches the stuck-tap and mulligan-read halts. Each entry is
#: (lowercase substrings identifying the halt, guidance); the signatures are disjoint. A halt
#: matching NONE falls through to the unknown/mulligan/false-halt diagnostics, which cover the
#: genuinely-unexpected halts -- so silence here never blesses a real fault as designed.
_DELIBERATE_HALTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("deck list", "re-select your deck"),
     "hop dropped back to the deck LIST after a game error (commonly Hearthstone's \"error "
     "starting your game\", which kicks you out of matchmaking). It deliberately will NOT re-pick "
     "a deck -- it can't tell which one you were on, and opening the wrong or an incomplete deck "
     "is worse than stopping. Re-select your deck and restart the hunt from its Play screen; the "
     "hunt resumes on its own the moment you're back on a Play screen."),
    (("main menu",),
     "hop was at the Hearthstone main menu, which is not where it queues from. Open Play, select "
     "a deck, and start the hunt from that deck's Play screen."),
    (("hunt did not start",),
     "a live game was in progress that THIS hunt did not queue, so hop refused to concede it (it "
     "may be YOUR game -- possibly the target it just alerted you to). Finish the game, or stop "
     "and restart the hunt from the deck's Play screen."),
    (("failed to reconnect", "came back online"),
     "Hearthstone could not reconnect within the attempt cap (vision.reconnect_attempt_cap). "
     "External: the phone's network or the account needs attention, not a hop fault."),
    (("no screen templates",),
     "no template pack is loaded, so hop refused to run blind. Run `hop capture` to build the "
     "anchor pack first."),
)


def _classify_halt(message: str) -> str:
    """Guidance for a DELIBERATE fail-closed halt (see :data:`_DELIBERATE_HALTS`), or ``""`` if
    the halt is not a recognized designed stop (and so may be a genuine fault)."""
    m = (message or "").lower()
    for needles, guidance in _DELIBERATE_HALTS:
        if any(n in m for n in needles):
            return guidance
    return ""


def _diagnose_last_error(last_error: str) -> str:
    """Name the cause of a CRASH message the controller caught (``status['last_error']``), the
    way :func:`_diagnose_mulligan_read_halt`/:func:`_classify_halt` name a halt. The one that
    cost a round-trip to diagnose: a raw ``TimeoutExpired: ... screencap ...`` -- a pasted report
    left the reader (and Claude) to work out that this is a wireless-link freeze, not a hop fault,
    and that a locked-but-awake Mac does NOT cause it. Returns "" for an unrecognized error.
    """
    m = (last_error or "").lower()
    if not m:
        return ""
    is_timeout = "timed out" in m or "timeoutexpired" in m
    if "screencap" in m and is_timeout:
        return ("the screencap over wireless ADB HUNG to its timeout -- the link to the phone "
                "froze; this is a transport stall, NOT a hop logic fault. A locked-but-awake Mac "
                "keeps ADB and the network alive, so a bare lock screen does NOT cause this. A "
                "screencap hangs to the full timeout only when the connection actually drops: the "
                "Mac SLEEPING (caffeinate blocks idle sleep only on AC power, and only for the "
                "assertion type it holds -- see the Host power section), the phone's Wi-Fi "
                "entering Doze/power-save, or the AP dropping the phone. Corroborate with the "
                "capture-time trend (a mean that rises through the run = a degrading link) and the "
                "Host power/network section. NOTE: as of this build a stall like this no longer "
                "crashes the hunt -- it reconnects and retries (vision.capture_retry_attempts); "
                "only a link that stays down through every retry stops the run, cleanly "
                "(stop_reason=adb_error, stats intact).")
    if is_timeout and "adb" in m:
        return ("an ADB command timed out -- the wireless link to the phone stalled (device "
                "asleep/Doze, Wi-Fi power-save, or the Mac slept). See the Host power/network "
                "section for which. Transient stalls now self-heal (capture_retry_attempts).")
    if ("could not connect" in m
            and ("network is unreachable" in m or "no route to host" in m)):
        return ("the INITIAL `adb connect` could not route a packet to the configured phone IP -- "
                "it never reached the phone or adbd. This is a network-path failure, NOT a dead "
                "wireless-ADB listener, so re-running `adb tcpip 5555` cannot repair it until the "
                "Mac can reach the phone. Check whether the phone changed Wi-Fi IP, whether the "
                "Mac and phone are on reachable LAN/VLANs, and whether a VPN/split-tunnel owns the "
                "phone subnet. The Host power/network section now records macOS's route plus the "
                "Wi-Fi IPv4 of a sole USB-attached phone, when available, to distinguish those "
                "cases directly. If one authorized phone is attached by USB, hop can use that "
                "transport for this session while leaving the configured wireless address unchanged.")
    if "could not connect" in m and "refused" in m:
        return ("the INITIAL `adb connect` was REFUSED, not timed out -- the phone answered and "
                "closed the socket, meaning adbd's wireless TCP listener isn't running on this "
                "port right now. This is a different failure from a mid-run screencap/ADB stall "
                "above: it happens before the hunt even starts, and no amount of retrying `adb "
                "connect` fixes it, because retrying can't make a closed port listen. THE FIX IS "
                "NOT re-enabling Android's 'Wireless debugging' toggle (Settings > Developer "
                "options) -- that is a SEPARATE feature from this config's fixed-port setup and "
                "opens its own listener on a random port shown on that screen, which does not "
                "match device.adb_address; toggling it again just repeats the same refusal (this "
                "is exactly what a prior report on this same failure tried, and it did not help). "
                "The actual fix: plug the phone into the Mac over USB and run `adb tcpip 5555` "
                "again -- that is the mechanism that pins adbd to the fixed port, and it needs "
                "re-running every time the phone reboots or the listener drops. If the Host "
                "power/network section below has a 'USB-attached device detected' line, it names "
                "the exact command to run. Corroborate with that section generally: an ICMP ping "
                "that IS reachable while `adb get-state` reports the device not found is exactly "
                "this signature (phone is on the network, adbd is not listening on this port) -- "
                "as opposed to the phone being off the network entirely, which would also fail "
                "the ping.")
    return ""


def _one_line(value, limit: int = 280) -> str:
    """Collapse diagnostic text to one bounded, report-safe line."""
    text = " ".join(str(value or "").split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _transport_epoch(status: dict) -> str:
    """Compact ``UHID stream epoch / ADB epoch`` relation, or ``""`` if absent."""
    if not isinstance(status, dict):
        return ""
    stream = status.get("stream_generation")
    adb = status.get("connection_generation")
    if stream is None or adb is None:
        return ""
    suffix = " (STALE)" if stream != adb else ""
    return f"UHID stream generation {stream}; ADB generation {adb}{suffix}"


def _transport_lifecycle_lines(
    reconnects: list[tuple[int, dict]],
    reopens: list[tuple[int, dict]],
    failure: tuple[int, dict] | None,
) -> list[str]:
    """Render the causal lifecycle trail for a persistent transport.

    A normal ADB reconnect tears down its shell channels. A persistent UHID writer
    therefore needs an explicit generation-aware re-registration before its next
    report. The old report had a bare ``BrokenPipeError`` but not these three facts:
    the reconnect happened, the stream was stale afterward, and the next action was
    the first write. Journal events added by :mod:`hop.engine` carry that proof.

    The parser is intentionally tolerant: reports from older backends may have a
    reconnect event with no transport snapshot, or a failure with only an error.
    """
    lines: list[str] = []
    if reconnects:
        _idx, latest = reconnects[-1]
        outcome = str(latest.get("outcome") or "unknown")
        before_gen = latest.get("adb_generation_before")
        after_gen = latest.get("adb_generation_after")
        bits = [f"{len(reconnects)} capture-link reconnect(s); last {outcome}"]
        if before_gen is not None or after_gen is not None:
            bits.append(f"ADB generation {before_gen if before_gen is not None else '?'}→"
                        f"{after_gen if after_gen is not None else '?'}")
        epoch = _transport_epoch(latest.get("transport_after") or {})
        if epoch:
            bits.append(epoch)
        error = _one_line(latest.get("error"), 180)
        if error:
            bits.append(f"error: {error}")
        lines.append("- transport lifecycle: " + "; ".join(bits) + ".")

    if reopens:
        _idx, latest = reopens[-1]
        after = latest.get("transport_after") or {}
        before = latest.get("transport_before") or {}
        what = latest.get("what") or "the next gesture"
        bits = [f"{len(reopens)} UHID stream re-registration(s); latest before {what!r}"]
        before_epoch = _transport_epoch(before)
        after_epoch = _transport_epoch(after)
        if before_epoch or after_epoch:
            bits.append((before_epoch or "UHID stream epoch unavailable") + " → "
                        + (after_epoch or "UHID stream epoch unavailable"))
        count = after.get("reopen_count")
        if count is not None:
            bits.append(f"reopen #{count}")
        reason = _one_line(after.get("last_reopen_reason"), 140)
        if reason:
            bits.append(f"reason: {reason}")
        if after.get("last_reopen_ok") is False:
            bits.append("FAILED")
        lines.append("- transport lifecycle: " + "; ".join(bits) + ".")

    if failure is None:
        return lines

    failure_index, detail = failure
    status = detail.get("transport") or {}
    if not isinstance(status, dict):
        status = {}
    action = detail.get("what") or "?"
    operation = detail.get("operation") or "emit"
    error = _one_line(detail.get("error"), 320) or "unspecified transport error"
    bits = [f"{operation} for {action!r} raised {error}"]
    if status.get("pid") is not None:
        bits.append(f"child pid {status['pid']}")
    if status.get("returncode") is not None:
        bits.append(f"child exit {status['returncode']}")
    command = _one_line(status.get("last_command"), 80)
    if command:
        bits.append(f"last stream command {command!r}")
    epoch = _transport_epoch(status)
    if epoch:
        bits.append(epoch)
    child_failure = _one_line(status.get("failure"), 220)
    if child_failure and child_failure not in error:
        bits.append(f"child failure: {child_failure}")
    stderr = _one_line(status.get("stderr_tail"), 260)
    if stderr:
        bits.append(f"stderr: {stderr}")
    lines.append("- transport failure: " + "; ".join(bits) + ".")

    # The decisive signature of the pasted failure: a capture retry reconnected ADB,
    # leaving the resident `adb shell hid -` stream on its old epoch; the next action
    # wrote to that dead pipe before it had re-registered. Do not make that claim when
    # a successful re-registration is journalled in between, or when the epochs are
    # unavailable -- then the child exit/stderr above is the honest evidence.
    preceding = [item for item in reconnects if item[0] < failure_index]
    if preceding:
        reconnect_index, reconnect = preceding[-1]
        reopens_after = [item for item in reopens
                         if reconnect_index < item[0] < failure_index]
        post = reconnect.get("transport_after") or {}
        stale_after_reconnect = (_transport_epoch(post).endswith("(STALE)"))
        stale_at_failure = _transport_epoch(status).endswith("(STALE)")
        if (stale_after_reconnect or stale_at_failure) and not reopens_after:
            before_gen = reconnect.get("adb_generation_before")
            after_gen = reconnect.get("adb_generation_after")
            transition = (f" (ADB generation {before_gen}→{after_gen})"
                          if before_gen is not None or after_gen is not None else "")
            lines.append(
                "- likely transport cause: the capture-retry reconnect replaced ADB's shell "
                "transport" + transition + ", but the persistent UHID stream stayed on its "
                "old generation and did not re-register before this next write. That stale "
                "writer is why the pipe closed; re-register the UHID stream after every ADB "
                "reconnect.")
    return lines


def _live_transport_status_line(status) -> str:
    """One compact, non-journal transport snapshot for the report header."""
    if not isinstance(status, dict):
        return ""
    bits = []
    kind = _one_line(status.get("kind"), 40)
    if kind:
        bits.append(kind)
    if "opened" in status:
        bits.append("open" if status.get("opened") else "closed")
    if status.get("pid") is not None:
        bits.append(f"pid {status['pid']}")
    if status.get("returncode") is not None:
        bits.append(f"child exit {status['returncode']}")
    epoch = _transport_epoch(status)
    if epoch:
        bits.append(epoch)
    command = _one_line(status.get("last_command"), 80)
    if command:
        bits.append(f"last command {command!r}")
    failure = _one_line(status.get("failure"), 180)
    if failure:
        bits.append(f"failure: {failure}")
    stderr = _one_line(status.get("stderr_tail"), 220)
    if stderr:
        bits.append(f"stderr: {stderr}")
    return "; ".join(bits)


_POST_CONCEDE_TAPS = {
    "concede": ("Concede", "normal"),
    "concede_recovery": ("Concede", "recovery"),
    "concede_now": ("Concede Now", "normal"),
    "concede_now_recovery": ("Concede Now", "recovery"),
    "post_concede_play": ("Play", "burst"),
}


def _point_text(value) -> str:
    """Render a small endpoint payload without assuming one engine schema.

    The post-concede evidence is deliberately additive: old journals only have a
    nominal ``tap.point`` while newer ones may carry requested/display/native
    endpoints.  This helper keeps the report useful across both shapes and never
    lets malformed diagnostic payloads sink a report.
    """
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return f"({float(value[0]):.0f},{float(value[1]):.0f})"
        except (TypeError, ValueError):
            return ""
    if isinstance(value, dict):
        for x, y in (("x", "y"), ("display_x", "display_y"),
                     ("native_x", "native_y")):
            if x in value and y in value:
                try:
                    return f"({float(value[x]):.0f},{float(value[y]):.0f})"
                except (TypeError, ValueError):
                    return ""
    return ""


def _matching_post_concede_terminal(events: list[dict], boundary_index: int,
                                    boundary: dict) -> dict | None:
    """Return only the terminal anomaly written directly for this boundary.

    ``_post_concede_boundary_halt`` records its named boundary and immediately
    saves its terminal frame.  A later terminal anomaly can belong to an entirely
    different control path, so scanning the rest of the journal would attach the
    wrong image or transport snapshot to this report narrative.
    """
    if boundary_index + 1 >= len(events):
        return None
    event = events[boundary_index + 1]
    if not isinstance(event, dict) or event.get("kind") != "anomaly":
        return None
    detail = event.get("detail")
    if not isinstance(detail, dict) or not detail.get("terminal"):
        return None
    reason = str(detail.get("reason") or "").lower()
    shared = [key for key in ("state", "attempt", "recovery")
              if key in detail and key in boundary]
    context_matches = bool(shared) and all(detail[key] == boundary[key] for key in shared)
    context_conflicts = any(detail[key] != boundary[key] for key in shared)
    if ("post-concede" not in reason and not context_matches) or context_conflicts:
        return None
    return detail


def _post_concede_lines(events: list[dict]) -> list[str]:
    """Summarize an unexpected post-concede boundary without extra device work.

    A normal bounded burst has no report line.  When its one boundary capture sees
    an unsafe state, however, the raw sequence is otherwise scattered across 20+
    tap records.  Reconstruct the last such trace from both the old
    ``post_concede_unexpected_boundary`` event and the newer generic
    ``post_concede_boundary`` contract.  The evidence says only that a write was
    accepted by the host input stream -- never that Hearthstone acknowledged it.
    """
    current: dict | None = None
    failures: list[dict] = []

    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        detail = event.get("detail")
        d = detail if isinstance(detail, dict) else {}
        t = event.get("t")
        stamp = float(t) if isinstance(t, (int, float)) else None

        if kind == "tap":
            what = str(d.get("what") or "")
            if what == "gear":
                # A direct reject always opens this compact trace from its named
                # mulligan/board source.  A later gear supersedes an older completed
                # trace, which is what a report about the *latest* halt needs.
                current = {"taps": [], "sleeps": {}, "started": stamp}
            if what in _POST_CONCEDE_TAPS:
                if current is None:
                    current = {"taps": [], "sleeps": {}, "started": stamp}
                label, phase = _POST_CONCEDE_TAPS[what]
                current["taps"].append({"what": what, "label": label, "phase": phase,
                                        "point": d.get("point"), "t": stamp, "detail": d})

        if current is not None and kind == "sleep":
            reason = str(d.get("reason") or "")
            if reason in ("post_concede_start_cooldown", "post_concede_click_cadence",
                          "post_concede_queue_cooldown"):
                seconds = d.get("seconds")
                if isinstance(seconds, (int, float)):
                    current["sleeps"].setdefault(reason, []).append(float(seconds))

        if current is not None and kind == "post_concede_burst_complete":
            current.setdefault("bursts", []).append(d)

        if kind not in ("post_concede_unexpected_boundary", "post_concede_boundary"):
            continue

        # ``post_concede_boundary`` is deliberately logged before deciding whether
        # its named warning/menu state needs the one bounded recovery. It is not a
        # failure by itself. Only the legacy/named terminal event (or an explicitly
        # terminal future boundary contract) earns a report narrative.
        terminal = (kind == "post_concede_unexpected_boundary" or d.get("terminal") is True)
        if not terminal:
            continue
        trace = current if current is not None else {"taps": [], "sleeps": {}, "bursts": [],
                                                      "started": None}
        # Freeze the trace at this terminal record. A generic recovery boundary can
        # precede more taps/bursts; shallow-copying its lists would retroactively
        # make an older trace look like the final one.
        trace = {"taps": list(trace.get("taps", [])),
                 "sleeps": {key: list(values) for key, values in trace.get("sleeps", {}).items()},
                 "bursts": list(trace.get("bursts", [])), "started": trace.get("started")}
        trace["boundary"] = d
        trace["boundary_index"] = index
        trace["boundary_t"] = stamp
        failures.append(trace)

    if not failures:
        return []

    trace = failures[-1]
    boundary = trace["boundary"]
    state = str(boundary.get("state") or boundary.get("boundary_state") or "?")
    confidence = boundary.get("confidence")
    source = str(boundary.get("source") or boundary.get("source_state") or "mulligan")
    state_text = state + (f" (confidence {float(confidence):.3f})"
                          if isinstance(confidence, (int, float)) else "")

    taps = trace.get("taps") or []
    parts = []
    for label in ("Concede", "Concede Now", "Play"):
        hits = [tap for tap in taps if tap.get("label") == label]
        count_key = "concede_taps" if label == "Concede" else (
            "concede_now_taps" if label == "Concede Now" else None)
        authoritative = boundary.get(count_key) if count_key else None
        if not hits and not isinstance(authoritative, int):
            continue
        points = [p for p in (_point_text(hit.get("point")) for hit in hits) if p]
        point = points[0] if points and len(set(points)) == 1 else ""
        count = authoritative if isinstance(authoritative, int) else len(hits)
        recovery_count = sum(tap.get("phase") == "recovery" for tap in hits)
        suffix = f" ({recovery_count} recovery)" if recovery_count else ""
        parts.append(f"{label} {count}" + (f" at {point}" if point else "") + suffix)

    bursts = [b for b in trace.get("bursts", []) if isinstance(b, dict)]
    burst_parts = []
    for burst in bursts:
        configured = burst.get("configured")
        emitted = burst.get("emitted")
        elapsed = burst.get("elapsed_s")
        maximum = burst.get("max_s")
        if not (isinstance(configured, int) or isinstance(emitted, int)):
            continue
        burst_text = (f"Play {emitted if isinstance(emitted, int) else '?'}/"
                      f"{configured if isinstance(configured, int) else '?'}")
        if isinstance(elapsed, (int, float)):
            burst_text += f" in {float(elapsed):.2f}s"
        if isinstance(maximum, (int, float)):
            burst_text += f" (limit {float(maximum):.2f}s)"
        attempt = burst.get("attempt")
        recovery = burst.get("recovery")
        if isinstance(attempt, int):
            burst_text += f" [attempt {attempt}, {recovery or 'normal'}]"
        burst_parts.append(burst_text)
    if burst_parts:
        # Completion records are authoritative for every burst attempt.  They also
        # keep a partial/timed-out Play run legible when raw tap logging was cut off.
        parts = [p for p in parts if not p.startswith("Play ")]
        parts.extend(burst_parts)

    sleeps = trace.get("sleeps") or {}
    wait_parts = []
    labels = (("post_concede_start_cooldown", "start wait"),
              ("post_concede_click_cadence", "burst gaps"),
              ("post_concede_queue_cooldown", "quiet wait"))
    for key, label in labels:
        values = sleeps.get(key) or []
        if values:
            wait_parts.append(f"{label} {sum(values):.2f}s")

    attempt = boundary.get("attempt")
    recovery = boundary.get("recovery")
    initial = boundary.get("initial_state")
    attempt_text = ""
    if isinstance(attempt, int):
        attempt_text = f"; attempt {attempt} ({recovery or 'normal'})"
        if initial:
            attempt_text += f" after initial {initial}"
    line = (f"- post-concede boundary: source {source} -> {state_text}{attempt_text}; "
            + (", ".join(parts) if parts else "tap trace unavailable"))
    if wait_parts:
        line += "; " + ", ".join(wait_parts)
    line += "."
    out = [line]

    # The boundary is deliberately the *only* semantic read after the open-loop
    # Play burst.  The engine records this invariant explicitly rather than making
    # the report reverse-engineer it from a particular timing configuration: the
    # sealed floor is an observed Play->Mulligan minimum, not merely a timeout.
    # Old journals have neither field and remain reportable without speculation.
    mulligan_floor = boundary.get("mulligan_floor_s")
    deadline_exceeded = boundary.get("mulligan_deadline_exceeded")
    boundary_elapsed = boundary.get("elapsed_s")
    if isinstance(mulligan_floor, (int, float)):
        elapsed_text = (f"boundary {float(boundary_elapsed):.2f}s after first Play"
                        if isinstance(boundary_elapsed, (int, float))
                        else "boundary elapsed unavailable")
        status_text = ("exceeded" if deadline_exceeded is True else
                       "within" if deadline_exceeded is False else "unclassified against")
        out.append("- post-concede successor timing: " + elapsed_text + "; calibrated "
                   f"earliest Play->Mulligan arrival {float(mulligan_floor):.2f}s "
                   f"({status_text} floor).")
    if deadline_exceeded is True:
        # This does not assert that every burst tap hit the next mulligan -- the
        # journal has no screen reads inside the deliberately bounded burst. It
        # states the stronger fact the new fields establish: a successor mulligan
        # could have existed before the sole boundary read, so the terminal board
        # cannot be treated as an OCR/criteria decision.
        out.append("- likely post-concede cause: successor mulligan may have been missed: "
                   "the sole boundary read came after the calibrated earliest "
                   "Play->Mulligan arrival. The terminal screen is therefore not evidence "
                   "that hop selected or accepted that opponent class.")

    concede_times = [tap.get("t") for tap in taps if tap.get("label") == "Concede"
                     and isinstance(tap.get("t"), (int, float))]
    now_times = [tap.get("t") for tap in taps if tap.get("label") == "Concede Now"
                 and isinstance(tap.get("t"), (int, float))]
    if concede_times and now_times:
        # Pair each confirmation with its own immediately preceding Concede. A
        # recovery can happen much later, so subtracting the global min/max would
        # hide a second too-fast confirmation behind the first normal-path tap.
        gaps = [now - max(concede for concede in concede_times if concede <= now)
                for now in now_times if any(concede <= now for concede in concede_times)]
        gap = min(gaps) if gaps else None
        if gap is not None and gap < 0.25:
            out.append("- likely post-concede cause: Concede Now followed the last Concede "
                       f"after {gap * 1000:.0f}ms. The confirmation may not have rendered yet, "
                       "so this is a likely confirmation/render race rather than a queue delay.")

    # A tap's ``actual_endpoint`` is synthesized *before* emitting it.  It proves
    # the precise display coordinate that the gesture layer chose, not successful
    # delivery to the app.  Prefer the final Play tap, then the final fixed tap.
    endpoint_bits = []
    endpoint_tap = next((tap for tap in reversed(taps)
                         if tap.get("label") == "Play"), taps[-1] if taps else None)
    if endpoint_tap:
        d = endpoint_tap.get("detail") or {}
        nominal = _point_text(d.get("nominal_point") or endpoint_tap.get("point"))
        synthesized = _point_text(d.get("actual_endpoint"))
        if nominal:
            endpoint_bits.append(f"nominal display {nominal}")
        if synthesized:
            endpoint_bits.append(f"synthesized display endpoint {synthesized}")
    if endpoint_bits:
        out.append("- post-concede tap registration: " + "; ".join(endpoint_bits) + ".")

    # A non-accepted boundary state necessarily came from ScreenClassifier's full
    # fallback, whereas an accepted successor was returned by the scoped scan. The
    # winning anchor location distinguishes a real board/control glyph from a
    # suspicious match behind an overlay without claiming that it proves app input.
    scan = boundary.get("classification_scan")
    anchor_at = _point_text(boundary.get("anchor_at"))
    scan_text = {
        "scoped": "scoped scan (an expected successor matched)",
        # ``full_fallback`` is the production spelling.  Keep ``full`` readable
        # too, in case an early development journal used the shorter label.
        "full_fallback": "full fallback scan (the boundary scope did not accept the result)",
        "full": "full fallback scan (the boundary scope did not accept the result)",
    }.get(scan)
    if scan_text or anchor_at:
        evidence = [scan_text] if scan_text else []
        if anchor_at:
            evidence.append(f"winning {state} anchor at {anchor_at}")
        out.append("- post-concede classifier evidence: " + "; ".join(evidence) + ".")

    terminal_anomaly = _matching_post_concede_terminal(
        events, trace["boundary_index"], boundary)
    transport = boundary.get("transport") or boundary.get("transport_after")
    if not isinstance(transport, dict) and terminal_anomaly:
        transport = terminal_anomaly.get("transport")
    status = _live_transport_status_line(transport)
    if status:
        out.append("- post-concede transport: " + status + ".")
    if isinstance(transport, dict):
        cached_display = _point_text({"display_x": transport.get("last_display_x"),
                                      "display_y": transport.get("last_display_y")})
        cached_native = _point_text({"native_x": transport.get("last_native_x"),
                                     "native_y": transport.get("last_native_y")})
        last_write = transport.get("last_write_at")
        cache_bits = []
        if cached_display:
            cache_bits.append(f"cached display endpoint {cached_display}")
        if cached_native:
            cache_bits.append(f"cached native endpoint {cached_native}")
        if isinstance(last_write, (int, float)):
            cache_bits.append(f"last host write {float(last_write):.3f}")
        if cache_bits:
            proof = ("; host/stream write acceptance only, not Hearthstone/app acknowledgment"
                     if cached_display and isinstance(last_write, (int, float)) else "")
            out.append("- post-concede transport endpoint: " + "; ".join(cache_bits) + proof + ".")
        geometry_bits = []
        if transport.get("rotation") is not None:
            geometry_bits.append(f"rotation {transport['rotation']}")
        panel_w, panel_h = transport.get("panel_width_px"), transport.get("panel_height_px")
        if isinstance(panel_w, (int, float)) and isinstance(panel_h, (int, float)):
            geometry_bits.append(f"panel {panel_w:.0f}x{panel_h:.0f}")
        axes = []
        for name, label in (("axis_touch_major_max", "major"),
                            ("axis_touch_minor_max", "minor"),
                            ("axis_pressure_max", "pressure"),
                            ("axis_orientation_max", "orientation")):
            value = transport.get(name)
            if isinstance(value, (int, float)):
                axes.append(f"{label}={value:.0f}")
        if axes:
            geometry_bits.append("axes " + ", ".join(axes))
        if geometry_bits:
            out.append("- post-concede transport geometry: " + "; ".join(geometry_bits) + ".")

    if terminal_anomaly:
        number = terminal_anomaly.get("index")
        if isinstance(number, int) and number > 0:
            out.append(f"- post-concede terminal evidence: anomaly_{number}_before.png "
                       "(and an after frame when recorded).")
    return out


def summarize_journal(journal_text: str) -> str:
    """A human summary of a run journal: counts, the class distribution, and the tail.

    The journal is the tap-by-tap record a halt is actually diagnosed from, but a raw
    JSONL dump buries the story. This surfaces the shape (how many taps, what opponents
    were seen, how it ended) and then shows the last events verbatim.
    """
    events = []
    for line in journal_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    if not events:
        return ""

    kinds: dict[str, int] = {}
    classes: dict[str, int] = {}
    # Reads that RESOLVED to a class but at a confidence low enough to be a silent misread. The
    # engine acts on these as if certain, so the report must
    # flag them or "opponents seen" reads as ground truth when it may be a garbled two-word label
    # snapped to a valid-but-wrong class -- the Demon-Hunter-read-as-Hunter bug.
    low_conf_reads: list[dict] = []
    LOW_CONF_FLAG = 0.75   # mirror engine._LOW_CONFIDENCE_KEEP: conf < this == a shakily-resolved
    #                        read (>=2 glyph edits at the default ocr_max_edit_distance=3)
    think_credited = 0.0   # think seconds absorbed into perception latency, not stacked
    # Per-tap wall-clock, split think vs perception, so "why is <action> so slow?" is
    # answered inline instead of by hand-tracing timestamps. Everything since the previous
    # tap (screencaps + classifications to *reach* the action, plus its think) is charged
    # to the tap it leads to.
    tap_costs: list[tuple[str, float, float]] = []   # (what, think_s, perception_s)
    seg_think = 0.0
    seg_perception = 0.0
    # The queue/nameplate watch can last arbitrarily long and is deliberately
    # capture-only. Keep its I/O visible, but never roll it into the eventual gear
    # or concede reaction segment. Lifecycle events appeared with the watch; old
    # journals simply retain the pre-contract per-tap segmentation.
    match_watch_perception = 0.0
    match_watch_starts = 0
    match_watch_resolved = 0
    match_watch_interrupted = 0
    match_watch_active = False
    anomalies: list[str] = []
    sleeps: dict[str, float] = {}
    longest_sleeps: list[tuple[float, str]] = []
    captures_ms: list[float] = []
    classify_ms: list[float] = []
    # Transient screencap stalls that were retried instead of crashing the hunt (the new
    # capture-retry resilience). A nonzero count is a wireless-link wobble; a RISING count over
    # a run is a link going bad -- exactly the precursor to the timeout that this harness was
    # improved to explain. Surfaced so a healthy-looking run that quietly limped is visible.
    capture_retry_ms: list[float] = []
    capture_retry_last_err = ""
    nonrep_rejects = 0
    nonrep_reject_last_attempt = 0
    nonrep_reject_last_of = 0
    nonrep_reject_memory_max = 0
    nonrep_collapses = 0
    nonrep_collapse_attempts = 0
    nonrep_collapse_memory = 0
    gaps: list[tuple[float, str, str]] = []
    last_halt = ""
    criteria = ""
    # Structured criteria + the matchup a target_found stopped on, so the summary can
    # RECOMPUTE whether the stop actually satisfied the criteria -- answering "did it stop on
    # the right thing?" ("wrong target found, unless bug?") and flagging a genuine misfire.
    crit_targets: list[str] = []
    crit_avoid: list[str] = []
    crit_require_second = False
    crit_seen = False   # a run_criteria line was present -> the reject recompute is meaningful
    # Why each CONCEDED game was rejected, recomputed against the criteria the SAME way the
    # target_found line recomputes a KEEP -- so a run that only ever conceded (the common shape,
    # and this report's) still answers "were these the right concedes?" instead of leaving the
    # reader to hand-cross the coin split against require_second. reject_plan carries no class or
    # coin (engine `_execute_reject`), so each is paired with the most recent resolved mulligan_read.
    reject_reasons: dict[str, int] = {}
    reject_misfires: list[str] = []   # conceded a matchup the criteria say to KEEP -> a real bug
    last_read_opp = ""                # opponent of the most recent resolved (non-"?") mulligan_read
    last_read_second: bool | None = None
    have_read = False                 # a resolved read is available to pair with a reject_plan
    # Trailing run of the last classified state, so a halt can be shown to have followed a BOUNDED
    # wait (for example, deck_select polled N times) -- the shape that tells a deliberate
    # fail-closed guard (polls its budget, then stops) from a crash (bails mid-action). Distinct
    # from unknown_run_len, which counts only the trailing UNKNOWN looks.
    terminal_state = ""
    terminal_state_run = 0
    tf_opponent = ""
    tf_second: bool | None = None
    closest = ""
    false_halt_note = ""   # "waited for X to leave; X is gone, destination just unnamed"
    verified_then_halt = ""  # the last tap verified OK, then we halted "stuck": a false halt
    # The coin/turn story. A recurring class of question ("does it only concede on MY turn?
    # it waits till turn 2 when I go second?") is unanswerable from taps alone -- it needs
    # the coin split, where the reject grammar chose to bail, and whether the End Turn taps
    # actually did anything. Concede is via the gear menu and works on EITHER turn; only the
    # pass_turn beat is turn-gated (a no-op when End Turn is greyed on the opponent's turn).
    went_second = 0
    went_first = 0
    turn_unreadable = 0
    coin_counted_game = False   # coin already tallied for the current game (once-per-game)
    concede_points: dict[str, int] = {}
    pass_turn_taps = 0
    pass_turn_noops = 0     # tap was a no-op (End Turn greyed -> not our turn); benign
    # last verified tap, for the false-halt-by-verification check below
    last_tap_what = ""
    last_tap_verified = ""      # change_kind of the verify_ok that confirmed the last tap
    last_tap_expected = ""      # the change_kind that tap DEMANDED (its `expected` field)
    unknown_after_verify = False
    # A GENUINE stuck committing tap -- the COMPLEMENT of verified_then_halt. When a button
    # transition's leave-wait times out with the SOURCE screen STILL POSITIVELY NAMED (never
    # UNKNOWN), the tap was DROPPED (never registered), not mis-navigated onto an unnamed
    # destination. Today both false-halt diagnostics stay silent on this shape (false_halt_note
    # needs an unknown_screen near-miss; verified_then_halt needs a trailing UNKNOWN) and the
    # halt isn't in _DELIBERATE_HALTS -- so a genuine dropped-commit fell through to a bare
    # "ended: <message>". Read the terminal wait_timeout{what,state} and how many polls the
    # source persisted; corroborate with the tap's expected-vs-verified change (a `partial`
    # where `full_transition` was demanded is a commit masked by ambient animation behind a
    # semi-transparent overlay). This is exactly the concede-drop this report was pasted for.
    wait_timeout_what = ""
    wait_timeout_state = ""
    leave_poll_what = ""        # contiguous run of leave-wait polls (reset by each tap)
    leave_poll_run = 0
    genuine_stuck_note = ""
    # Taps the client silently ignored and the engine re-sent until honoured. A CLUSTER of these
    # is the wireless link dropping input -- the same failure a terminal stuck-tap halt is the
    # tail of -- so surfacing the count corroborates a dropped-commit root cause.
    dropped_taps = 0
    # Unknown-halt context: which NAMED screen the terminal UNKNOWN run followed, and how many
    # looks it persisted. A near-miss line already says which anchor an unknown came closest to,
    # but not WHERE IN THE FLOW it appeared -- and "an unknown that won't settle right after
    # QUEUE" is the unmistakable signature of the match-start transition (the VS 'match found'
    # splash / board fade-in) that a pack may lack an anchor for. That is derivable from the
    # classify sequence the journal already holds, and it is exactly what turns a useless
    # "closest: deck_select 0.302" into "capture a vs_splash anchor".
    last_named_state = ""
    unknown_run_len = 0
    state_before_unknown_run = ""
    # The sequence of screens the run passed through, consecutive repeats collapsed. On a
    # halt this is emitted as the "state trail into the halt" -- the single line that would
    # have shown, in THIS report, that a dismissed error_dialog dropped to deck_select
    # (queue -> error_dialog -> deck_select). It's what a truncated journal tail hides, and
    # it's derivable from the classify stream the summary already walks.
    state_trail: list[str] = []
    # A persistent UHID writer belongs to one ADB connection generation. Capture recovery can
    # deliberately reconnect ADB; retain the before/after snapshots plus a terminal writer
    # failure so the summary can prove (rather than guess) whether the stream re-registered.
    capture_link_reconnects: list[tuple[int, dict]] = []
    transport_stream_reopens: list[tuple[int, dict]] = []
    transport_failure: tuple[int, dict] | None = None
    first_t: float | None = None
    prev_t: float | None = None
    prev_kind = ""
    for event_index, e in enumerate(events):
        k = e.get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
        d = e.get("detail", {})
        t = e.get("t")
        if isinstance(t, (int, float)):
            if first_t is None:
                first_t = float(t)
            if prev_t is not None:
                gaps.append((float(t) - prev_t, prev_kind, k))
            prev_t = float(t)
            prev_kind = k
        if k == "mulligan_read":
            # Pending watch frames retain their event count/timestamps above (so
            # capture cadence and OCR latency remain diagnosable), but are not a
            # game and must not reset/tally the once-per-game coin guard or pair a
            # later reject with stale data. Missing ``pending`` is legacy-resolved.
            if _is_pending_mulligan_read(d):
                continue
            # "?" is a pending OCR, not an opponent class. Count only real classes.
            opp = d.get("opponent")
            is_reread = bool(d.get("reread"))
            # A current engine can make a class-only decision when the card count is bad but
            # `require_second` is off.  Its explicit decision is authoritative; pre-decision
            # journals retain the old valid-card proxy so history is not reinterpreted.
            actionable = bool(opp and opp != "?" and _is_actionable_mulligan_read(d))
            if actionable:
                classes[opp] = classes.get(opp, 0) + 1
                conf = d.get("conf")
                if isinstance(conf, (int, float)) and conf < LOW_CONF_FLAG:
                    # The engine acted on this class despite the shaky OCR; carry the raw text so
                    # the report can show WHY it may be wrong (a stray glyph, a mangled word).
                    low_conf_reads.append({"opp": opp, "conf": float(conf),
                                           "raw": (d.get("class_raw") or "").strip(),
                                           "reread": is_reread})
                # A readable class is the only kind of mulligan that can pair with
                # a following reject plan.
                last_read_opp, last_read_second, have_read = opp, _mulligan_turn(d), True
            # The coin is derived purely from the card count (we_go_second = cards>=4).  A
            # malformed hand now records its turn as unknown rather than false, so it must not
            # inflate the going-first bucket.  A legacy line lacking `cards` stays trusted.
            if not is_reread:
                coin_counted_game = False
            turn = _mulligan_turn(d)
            decision = _mulligan_decision(d)
            legacy_usable = d.get("cards") is None or d.get("cards") in (3, 4)
            decision_resolved = decision in ("keep", "reject")
            if not coin_counted_game and actionable and (decision_resolved or legacy_usable):
                if turn is None:
                    turn_unreadable += 1
                elif turn:
                    went_second += 1
                else:
                    went_first += 1
                coin_counted_game = True
        if k == "match_watch_start":
            # The capture/classify that first named QUEUE/VS preceded this dispatch,
            # so drain the segment here as well as all later watch observations.
            match_watch_perception += seg_perception
            seg_perception = 0.0
            match_watch_starts += 1
            match_watch_active = True
        if k == "match_watch_resolved":
            # The resolved mulligan's capture/classify is still an observation; the
            # following action starts a fresh reaction segment.
            if match_watch_active:
                match_watch_perception += seg_perception
                seg_perception = 0.0
                match_watch_active = False
            match_watch_resolved += 1
        if k == "match_watch_interrupted":
            # A named modal/reconnect/error takes control back from the watch. Its
            # preceding queue scans remain search observation, not reaction time for
            # the interruption's next tap.
            if match_watch_active:
                match_watch_perception += seg_perception
                seg_perception = 0.0
                match_watch_active = False
            match_watch_interrupted += 1
        if k == "reject_plan":
            # Where the reject grammar chose to bail. Chosen at RANDOM, independent of the
            # coin (journey.choose_concede_point) -- so a lopsided split here is just RNG,
            # and NOT evidence the tool waits for a particular turn.
            cp = str(d.get("concede_point", "?"))
            concede_points[cp] = concede_points.get(cp, 0) + 1
            # Recompute why THIS game was conceded, against the run criteria. Gated on crit_seen:
            # without a run_criteria line the criteria default to "accept everything", under which
            # every reject would spuriously read as a misfire. run_criteria is the journal's first
            # line, so it is always in hand by the time a reject_plan is reached.
            if have_read and crit_seen:
                r = _reject_reason(last_read_opp, last_read_second,
                                   crit_require_second, crit_targets, crit_avoid)
                if r == "misfire":
                    reject_misfires.append(
                        f"{last_read_opp} ({_turn_label(last_read_second)})")
                else:
                    reject_reasons[r] = reject_reasons.get(r, 0) + 1
                have_read = False   # consume, so the next reject pairs with the next game's read
        if k == "anomaly":
            anomalies.append(str(d.get("reason", "?")))
            # A pass_turn (End Turn) tap that changed nothing means End Turn was greyed --
            # i.e. it was NOT our turn (we went second and it is still the opponent's turn).
            # That is benign and expected: the concede itself follows via the gear menu on
            # EITHER turn. Counting these separates "the End Turn beat no-opped" (fine) from
            # a real stuck tap, and is the direct evidence for the "only concedes on my turn?"
            # question -- the beat is turn-gated, the concede is not.
            if last_tap_what == "pass_turn" and d.get("fault") == "no_change":
                pass_turn_noops += 1
        if k == "sleep":
            reason = str(d.get("reason", "?"))
            seconds = d.get("seconds")
            if isinstance(seconds, (int, float)):
                sleeps[reason] = sleeps.get(reason, 0.0) + float(seconds)
                longest_sleeps.append((float(seconds), reason))
            # How much think was *absorbed* into perception latency instead of stacked on
            # top (see engine `_tap` / `timing.credit_latency`). This is the proof the
            # anti-stacking is firing on-device: a big number here means think was reduced
            # to fit inside the screencap+classify time, not added to it.
            credited = d.get("credited_s")
            if reason == "tap_think" and isinstance(credited, (int, float)):
                think_credited += float(credited)
            if reason == "tap_think" and isinstance(seconds, (int, float)):
                seg_think += float(seconds)   # ACTUAL waited think (post latency-credit)
        if k == "capture":
            ms = d.get("ms")
            if isinstance(ms, (int, float)):
                captures_ms.append(float(ms))
                seg_perception += float(ms) / 1000.0
        if k == "capture_retry":
            ms = d.get("ms")
            if isinstance(ms, (int, float)):
                capture_retry_ms.append(float(ms))
                seg_perception += float(ms) / 1000.0   # the hung time WAS wall-clock spent
            if d.get("error"):
                capture_retry_last_err = str(d.get("error"))
        if k == "capture_link_reconnect" and isinstance(d, dict):
            capture_link_reconnects.append((event_index, d))
        if k == "transport_stream_reopened" and isinstance(d, dict):
            transport_stream_reopens.append((event_index, d))
        if k == "transport_failure" and isinstance(d, dict):
            # Keep the final failure only: it is the one nearest the halt, and a prior
            # transient that recovered must not be mistaken for the terminal cause.
            transport_failure = (event_index, d)
        if k == "non_repetition_reject":
            nonrep_rejects += 1
            attempt = d.get("attempt")
            of = d.get("of")
            memory = d.get("memory")
            if isinstance(attempt, int):
                nonrep_reject_last_attempt = attempt
            if isinstance(of, int):
                nonrep_reject_last_of = of
            if isinstance(memory, int):
                nonrep_reject_memory_max = max(nonrep_reject_memory_max, memory)
        if k == "non_repetition_collapse":
            nonrep_collapses += 1
            attempts = d.get("attempts")
            memory = d.get("memory")
            if isinstance(attempts, int):
                nonrep_collapse_attempts = attempts
            if isinstance(memory, int):
                nonrep_collapse_memory = memory
        if k == "classify":
            ms = d.get("ms")
            if isinstance(ms, (int, float)):
                classify_ms.append(float(ms))
                seg_perception += float(ms) / 1000.0
            # Track the trailing run of UNKNOWN looks and the named screen it followed, so a
            # halt on an unknown can say WHERE in the flow it struck (see the context line below).
            state = d.get("state")
            if state:
                # trailing run of the SAME classified state (any state, UNKNOWN included), so a
                # halt can report the bounded wait it followed -- see the terminal_state_run block.
                if state == terminal_state:
                    terminal_state_run += 1
                else:
                    terminal_state, terminal_state_run = state, 1
            if state == "unknown":
                if unknown_run_len == 0:
                    state_before_unknown_run = last_named_state
                unknown_run_len += 1
            elif state:
                last_named_state = state
                unknown_run_len = 0
            if state and (not state_trail or state_trail[-1] != state):
                state_trail.append(state)   # collapse consecutive repeats (queue x8 -> queue)
        if k == "tap":
            what = str(d.get("what", "?"))
            tap_costs.append((what, seg_think, seg_perception))
            seg_think = 0.0
            seg_perception = 0.0
            if what == "pass_turn":
                pass_turn_taps += 1
            # reset the verified-then-halt trail for THIS tap's outcome
            last_tap_what = what
            last_tap_verified = ""
            last_tap_expected = str(d.get("expected", "any"))
            unknown_after_verify = False
            # a new action ends the previous leave-wait's poll run (see the genuine-stuck note)
            leave_poll_what, leave_poll_run = "", 0
        if k == "wait_until_poll":
            # Count the contiguous run of leave-wait polls for one action, so a terminal
            # stuck halt can report "the source screen persisted N looks" (the tell that the
            # tap never took, as opposed to a one-frame transient).
            w = str(d.get("what", ""))
            if w == leave_poll_what:
                leave_poll_run += 1
            else:
                leave_poll_what, leave_poll_run = w, 1
        if k == "wait_timeout":
            # The leave-wait gave up. `state` is the SOURCE screen still on-frame at timeout; a
            # NAMED one (not "unknown") means the awaited transition never happened -- a genuine
            # stuck, resolved into the dropped-commit note below.
            wait_timeout_what = str(d.get("what", ""))
            wait_timeout_state = str(d.get("state", ""))
        if k in ("mulligan_card_tap_ignored", "concede_tap_ignored", "play_tap_ignored",
                 "error_ok_tap_ignored", "collection_back_tap_ignored",
                 "deck_decline_tap_ignored", "mulligan_confirm_tap_ignored",
                 "reconnect_tap_ignored", "end_dismiss_tap_ignored"):
            dropped_taps += 1
        if k == "verify_ok":
            # the tap's OWN change-check passed: the screen provably moved after it.
            last_tap_verified = str(d.get("change_kind", "?"))
        if k == "classify" and last_tap_verified:
            # verified a change, then couldn't name where we landed: the tell of a false
            # halt whose *destination* is unrecognised (not a stuck source screen). Track the
            # MOST RECENT post-verify look: a later NAMED frame means we DID recognise the
            # destination (e.g. genuinely stuck back on the mulligan), so the latch must mean
            # "the last look since the verify was UNKNOWN", not "at least one UNKNOWN ever".
            unknown_after_verify = d.get("state") == "unknown"
        if k in ("stop", "halt") and d.get("message"):
            last_halt = d["message"]
            # A button-transition stuck halt ("X did not dismiss/open/close") whose LAST tap
            # verified OK and was then followed only by UNKNOWN frames is a FALSE halt: the
            # action worked and we simply could not classify the screen it produced. The
            # existing near-miss check (below) catches this only when the destination is a
            # *marginal* miss (a known screen, new face); this catches the other case -- a
            # genuinely un-anchored destination (all near-misses far away), which is exactly
            # the "Opponent Still Choosing..." halt -- where the near-miss line stays silent.
            if last_tap_verified and unknown_after_verify and _is_stuck_halt(last_halt):
                verified_then_halt = (
                    f"the last action before the halt ('{last_tap_what}') recorded verify_ok "
                    f"({last_tap_verified}) -- its OWN change-check PASSED, so the tap WORKED. "
                    f"The halt then came from being unable to NAME the screen it produced "
                    f"(UNKNOWN), not from a stuck '{last_tap_what}'. Treat as a classification "
                    f"miss on the DESTINATION: capture that screen's anchor (hop capture "
                    f"--from-file), don't chase a missed tap.")
            # The COMPLEMENT: a stuck-halt whose leave-wait timed out with the SOURCE screen
            # still POSITIVELY NAMED (never UNKNOWN) is a GENUINE stuck -- the awaited transition
            # never happened, i.e. the committing tap was DROPPED (never registered), not
            # mis-navigated to an unnamed screen. Mutually exclusive with verified_then_halt by
            # construction (that needs a trailing UNKNOWN; this needs a named source). Reads only
            # the terminal wait_timeout, so a mid-run wait that later succeeded can't trip it.
            elif (_is_stuck_halt(last_halt) and wait_timeout_state
                  and wait_timeout_state != "unknown"):
                polls = (leave_poll_run + 1 if leave_poll_what == wait_timeout_what
                         else leave_poll_run or 1)
                # The masked-drop tell: the tap's OWN verify logged a WEAKER change than it
                # demanded (partial < full_transition, admitted only by _compatible), which
                # happens when ambient animation behind a semi-transparent overlay -- an enemy
                # turn behind the open Game Menu -- moves pixels the tap did not.
                weaker = ""
                if (last_tap_verified and last_tap_expected
                        and last_tap_expected not in ("", "any")
                        and last_tap_verified != last_tap_expected):
                    weaker = (f" That '{last_tap_what}' tap's own verify logged change_kind "
                              f"'{last_tap_verified}', WEAKER than the '{last_tap_expected}' it "
                              f"demanded (admitted by _compatible) -- the signature of a commit "
                              f"masked by ambient animation behind a semi-transparent overlay, so "
                              f"the tap read OK yet never registered.")
                genuine_stuck_note = (
                    f"the '{wait_timeout_what or last_tap_what}' leave-wait timed out with the "
                    f"screen still positively '{wait_timeout_state}' for {polls} look(s) and "
                    f"never UNKNOWN -- so the committing tap was DROPPED (never registered), not "
                    f"mis-navigated to an unnamed screen.{weaker} The fix is a bounded RE-TAP of "
                    f"the button's own coordinate while the frame still reads "
                    f"'{wait_timeout_state}' (like the mulligan-card retry), NOT a new anchor or "
                    f"a coordinate change. A congested/degrading wireless link (see the "
                    f"capture-time TREND) is what drops even a button tap.")
        if k == "unknown_screen" and d.get("near_misses"):
            # the anchor an unknown screen came CLOSEST to: a near-miss below its
            # threshold usually means "known screen, new visual face" -- the single
            # most actionable line for an unrecognised-screen halt. Guard the shape:
            # a journal written before near_misses round-tripped as dicts holds strings
            # here, and a malformed entry must not sink the whole report.
            top = d["near_misses"][0]
            if isinstance(top, dict):
                st, sc, th = top.get("state"), top.get("score"), top.get("thr")
                closest = f"{st} {sc} (thr {th})"
                # A near-miss just under threshold is not a novel screen - it is a known
                # one wearing a face the anchor doesn't cover (e.g. `in_game` while going
                # second, END TURN greyed). Spell that out with the exact margin so the
                # halt is self-diagnosing: it needs another anchor or a lower threshold,
                # not head-scratching. Only for a *small* miss; a far-away top score is a
                # genuinely unhandled screen and must not be mislabelled as "almost known".
                if isinstance(sc, (int, float)) and isinstance(th, (int, float)) and 0 <= th - sc <= 0.1:
                    closest += (f" - MARGINAL: missed by {th - sc:.3f}; that screen needs "
                                "another anchor (hop capture --from-file) or a lower threshold")
                    # Which of the two fixes? The deciding fact is the gap to the nearest
                    # DIFFERENT screen: if the next state down is far below, the board owns
                    # this score-band alone and simply LOWERING the threshold is safe (no
                    # other screen can sneak in). A close runner-up means the band is
                    # contested -> lowering risks a mis-ID, so add an anchor instead. Adding
                    # anchors is whack-a-mole (one per visual variant); the gap is what tells
                    # you when the one-line threshold fix will actually hold.
                    other = next((m for m in d["near_misses"]
                                  if isinstance(m, dict) and m.get("state") != st
                                  and isinstance(m.get("score"), (int, float))), None)
                    if other is not None:
                        gap = sc - other["score"]
                        verdict = ("wide gap -> lowering the threshold is safe (the board owns "
                                   "this band)" if gap >= 0.2 else
                                   "narrow gap -> prefer a new anchor over lowering the threshold")
                        closest += (f"; nearest DIFFERENT screen is {other['state']} "
                                    f"{other['score']:.3f} ({gap:.3f} below) -- {verdict}")
                    # If we were waiting for a screen to LEAVE and it isn't even among the
                    # near-misses -- while some OTHER known screen is the marginal top --
                    # then the screen already changed: the action succeeded and we simply
                    # cannot name where it landed. The halt message then blames the wrong
                    # thing ("Confirm did not dismiss the mulligan" when Confirm worked and
                    # the *board* is the unrecognised screen). Flag it so a false halt is
                    # not chased as a stuck tap. Only when the top miss is marginal (above),
                    # i.e. a known screen wearing a new face -- not a genuinely novel screen.
                    waiting = d.get("waiting_to_leave")
                    near_states = {ns.get("state") for ns in d["near_misses"]
                                   if isinstance(ns, dict)}
                    if waiting and waiting not in near_states:
                        false_halt_note = (
                            f"waited for '{waiting}' to leave; '{waiting}' is GONE (not among the "
                            f"near-misses) and the top match '{st}' is only a marginal miss -- so the "
                            f"action almost certainly SUCCEEDED and this is a classification miss on "
                            f"the *destination*, not a stuck '{waiting}'. Fix the '{st}' anchor/threshold.")
            else:
                closest = str(top)
        if k == "run_criteria":
            targets = d.get("target_classes") or []
            criteria = (f"targets={targets or 'ANY'} require_second={d.get('require_second')} "
                        f"mode={d.get('mode')}")
            crit_targets = list(targets)
            crit_avoid = list(d.get("avoid_classes") or [])
            crit_require_second = bool(d.get("require_second"))
            crit_seen = True
        if k == "target_found":
            tf_opponent = str(d.get("opponent") or "?")
            tf_second = _mulligan_turn(d)

    # A user stop during an unusually long queue has no resolution event. Its
    # observation latency is still real and must remain visible, just not attached
    # to some unrelated future tap.
    if match_watch_active:
        match_watch_perception += seg_perception
        seg_perception = 0.0

    post_concede_lines = _post_concede_lines(events)

    out = [f"- events: {len(events)}"]
    if criteria:
        out.append(f"- criteria (this run): {criteria}")
    if tf_opponent:
        # Recompute whether the matchup the hunt STOPPED on actually satisfied the criteria,
        # so "did it stop on the right thing?" is answered inline -- and a genuine misfire
        # (a target_found the criteria should have rejected) is flagged loudly instead of
        # read as correct. The common confusion this answers: an EMPTY class filter means ANY
        # class is a target, so a stop on a class you didn't tick is correct, not a bug.
        _norm = _norm_class   # shared with the reject-reason recompute (see _reject_reason)
        coin = _turn_label(tf_second)
        ok, why = True, []
        if crit_require_second:
            if tf_second is None:
                ok = False; why.append("require-going-2nd was set but the turn/card count was unreadable")
            elif not tf_second:
                ok = False; why.append("require-going-2nd was set but this matchup went 1st")
        if crit_targets:
            if _norm(tf_opponent) in {_norm(t) for t in crit_targets}:
                why.append(f"{tf_opponent} is in the target list")
            else:
                ok = False; why.append(f"{tf_opponent} is NOT in the target list {crit_targets}")
        elif crit_avoid:
            if _norm(tf_opponent) in {_norm(a) for a in crit_avoid}:
                ok = False; why.append(f"{tf_opponent} is on the AVOID list")
            else:
                why.append(f"{tf_opponent} is not on the avoid list (any other class is a target)")
        else:
            why.append("no class filter is set, so ANY class is a target")
        if crit_require_second and tf_second is True:
            why.append("and the going-2nd requirement was met")
        verdict = ("matches the criteria — working as configured, NOT a misfire" if ok
                   else "does NOT match the criteria — a real MISFIRE; check evaluate_matchup/accepts")
        out.append(f"- target found: {tf_opponent} ({coin}) — {verdict}. Why: "
                   + "; ".join(why) + ".")
    if first_t is not None and prev_t is not None and prev_t > first_t:
        # The headline a "why is it slow?" report needs: where the wall-clock actually
        # went. Humanized waits are journalled (sleeps) and, since the capture-timing
        # change, so is screencap I/O (a `capture` ms per frame); the remainder is
        # classification/OCR/logic. The split matters because the remedies differ - the
        # humanized waits are deliberate anti-detection pacing and must NOT be trimmed,
        # whereas screencap is pure transport latency (Wi-Fi ~1.4 s vs USB ~1.0 s/frame).
        wall = prev_t - first_t
        slept = sum(sleeps.values())
        if captures_ms:
            cap_s = sum(captures_ms) / 1000.0
            # Retry stalls are hung TRANSPORT time (a screencap that timed out), not OCR/logic.
            # Keep them out of captures_ms (so the healthy per-frame mean/max stay clean) but
            # subtract them here as their own term, else a 20 s stall lands in the "OCR/logic"
            # residual and points a reader at the wrong subsystem -- the very transport stall
            # this report exists to diagnose. Emitted only when nonzero, so healthy runs read
            # exactly as before.
            retry_s = sum(capture_retry_ms) / 1000.0
            stalled = f" + {retry_s:.0f}s stalled I/O" if retry_s >= 0.5 else ""
            if classify_ms:
                # Classification is journalled separately, so split it out of the residual:
                # it is the second-biggest cost and lumping it into "logic" hid that.
                cls_s = sum(classify_ms) / 1000.0
                other = max(0.0, wall - slept - cap_s - cls_s - retry_s)
                out.append(f"- time: {wall:.0f}s wall = {slept:.0f}s humanized waits + "
                           f"{cap_s:.0f}s screencap I/O + {cls_s:.0f}s classification{stalled} + "
                           f"{other:.0f}s OCR/logic")
            else:
                other = max(0.0, wall - slept - cap_s - retry_s)
                out.append(f"- time: {wall:.0f}s wall = {slept:.0f}s humanized waits + "
                           f"{cap_s:.0f}s screencap I/O{stalled} + {other:.0f}s classify/OCR/logic")
        else:
            out.append(f"- time: {wall:.0f}s wall, {slept:.0f}s of it humanized waits "
                       "(the rest is screencap + classify/OCR; a pre-capture-journal run)")
    out.append("- kinds: " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))
    if classes:
        out.append("- opponents seen: " + ", ".join(f"{c}×{n}" for c, n in sorted(classes.items(), key=lambda kv: -kv[1])))
    if low_conf_reads:
        # A read that RESOLVED but at low confidence is the silent-misread risk: it is where a
        # garbled TWO-word class lands on a valid-but-WRONG one -- "DEMON HUNTER" snapping to its
        # own tail word HUNTER. Show the confidence + RAW OCR so "opponents seen" above is not
        # taken as ground truth (the raw text shows the actual corruption); the class-region
        # pixels are also kept (an `anomaly` frame + a `mulligan_low_confidence` journal line) for
        # the deciding look. Report the confidence itself, NOT a decoded edit count: the OCR
        # distance->confidence scale depends on vision.ocr_max_edit_distance, so a reverse-
        # engineered "N glyph edits" would be wrong whenever that knob is off its default.
        bits = [f"{r['opp']} conf {r['conf']:.2f}"
                + (f" (raw {r['raw']!r})" if r['raw'] else "") for r in low_conf_reads]
        out.append("- low-confidence class reads -- VERIFY (a garbled two-word label can snap to "
                   "a valid-but-wrong class): " + "; ".join(bits))
    if went_first or went_second or turn_unreadable or concede_points:
        # The coin/turn story, so "does it only concede on my turn / wait till turn 2 going
        # second?" is answered from the report instead of by hand-tracing. Concede is via the
        # gear menu and fires on EITHER turn; only the pass_turn *beat* is turn-gated (a no-op
        # when End Turn is greyed on the opponent's turn). A no-op End Turn going second is
        # therefore expected, not a stuck tap, and the concede still follows it.
        bits = [f"coin: {went_second} went 2nd / {went_first} went 1st"]
        if turn_unreadable:
            bits.append(f"{turn_unreadable} turn unreadable (the class was resolved, but the "
                        "keep-glow card count could not establish the coin)")
        if concede_points:
            bits.append("concede points " + ", ".join(
                f"{p}×{n}" for p, n in sorted(concede_points.items(), key=lambda kv: -kv[1]))
                + " (random, independent of the coin)")
        if pass_turn_taps:
            bits.append(f"End Turn beats {pass_turn_taps}, of which {pass_turn_noops} no-op'd "
                        "(End Turn did nothing -- typically greyed because it was not our turn; "
                        "benign, and the concede that follows is NOT turn-gated)")
        out.append("- coin/turn: " + "; ".join(bits))
    if reject_reasons or reject_misfires:
        # Why the conceded games were conceded, recomputed against the criteria -- the mirror of
        # the target_found "did it stop on the right thing?" line, for the far commoner run that
        # only ever concedes. This is the line that answers "were these the right concedes, or is
        # the class read / require_second logic broken?" directly, instead of leaving the reader to
        # cross the coin split against require_second by hand (the reasoning THIS report needed).
        # A misfire -- a reject the criteria say to KEEP -- is flagged as loudly as a target_found
        # misfire, since it means the tool conceded a matchup it was told to stop on.
        total = sum(reject_reasons.values()) + len(reject_misfires)
        parts = [f"{n}× {_REJECT_LABELS.get(r, r)}"
                 for r, n in sorted(reject_reasons.items(), key=lambda kv: -kv[1])]
        if reject_misfires:
            out.append(
                f"- rejects vs criteria: {total} conceded, but MISFIRED on {len(reject_misfires)} "
                f"({'; '.join(reject_misfires)}) -- the criteria say to KEEP these, so this is a "
                "real MISFIRE; check evaluate_matchup/accepts. Correct rejects: "
                + ("; ".join(parts) if parts else "none") + ".")
        else:
            out.append(f"- rejects vs criteria: {total} conceded, all correct per the criteria -- "
                       + "; ".join(parts) + ".")
    if anomalies:
        out.append("- anomalies: " + "; ".join(anomalies[-5:]))
    # A failed open-loop exit is the rare case where the report must make the
    # exact fixed-coordinate trace legible without asking the reader to count
    # two dozen raw tap records.  This is journal-only reconstruction: it adds no
    # capture, delay, or healthy-path I/O.
    out.extend(post_concede_lines)
    if dropped_taps:
        # Taps the client silently ignored, re-sent until honoured. A cluster of these is the
        # wireless link dropping INPUT (not just frames) -- the same failure a terminal
        # stuck-tap halt is the tail of, so it corroborates a dropped-commit root cause.
        out.append(f"- dropped taps re-sent: {dropped_taps} (the client silently ignored these "
                   "and hop re-tapped until they took -- a cluster means the link is dropping "
                   "input; the same failure mode as a terminal stuck committing tap)")
    if nonrep_rejects or nonrep_collapses:
        parts = []
        if nonrep_rejects:
            parts.append(f"{nonrep_rejects} rejected draw(s)")
            if nonrep_reject_last_of:
                parts.append(f"last rejected attempt {nonrep_reject_last_attempt}/{nonrep_reject_last_of}")
            if nonrep_reject_memory_max:
                parts.append(f"memory peaked at {nonrep_reject_memory_max}")
        if nonrep_collapses:
            attempts = nonrep_collapse_attempts or nonrep_reject_last_of
            mem = nonrep_collapse_memory or nonrep_reject_memory_max
            detail = f"{nonrep_collapses} collapse(s)"
            if attempts:
                detail += f" after {attempts} attempts"
            if mem:
                detail += f" with memory={mem}"
            parts.append(detail)
        out.append("- non-repetition gate: " + "; ".join(parts) +
                   " -- repeated recent tap fingerprints exhausted the resample budget")
    if sleeps:
        biggest = sorted(sleeps.items(), key=lambda kv: -kv[1])[:6]
        out.append("- intentional sleeps: " + ", ".join(
            f"{reason}={seconds:.1f}s" for reason, seconds in biggest))
        long_one = sorted(longest_sleeps, key=lambda x: -x[0])[:3]
        out.append("- longest single sleeps: " + ", ".join(
            f"{reason} {seconds:.1f}s" for seconds, reason in long_one))
    if think_credited > 0:
        # The anti-stacking at work: think time that was absorbed into perception latency
        # rather than waited on top of it. 0 here on a slow wireless run would mean the
        # credit is NOT firing (a regression); a large number means it is.
        out.append(f"- think absorbed into latency: {think_credited:.1f}s not stacked on "
                   "top of screencap+classify (would otherwise be added to the reaction)")
    if match_watch_starts:
        status_parts = []
        if match_watch_resolved:
            status_parts.append(f"{match_watch_resolved} resolved")
        if match_watch_interrupted:
            status_parts.append(f"{match_watch_interrupted} interrupted")
        if match_watch_active:
            status_parts.append("still observing at run end")
        status = ", ".join(status_parts) or "still observing at run end"
        out.append(f"- match-search observation: {match_watch_perception:.1f}s "
                   f"screencap/classify across {match_watch_starts} watch(es), {status}; "
                   "excluded from per-tap reaction")
    if tap_costs:
        # Answers "why is <action> so slow?" directly: for the costliest taps, how much of
        # the wall-clock to reach+fire them was humanized THINKING vs perception (screencap
        # + classification). Almost always the think is small and perception dominates --
        # so a step that "feels like it thinks too long" is really waiting on wireless I/O,
        # which no timing change can fix (only fewer/cheaper captures + classifies can).
        slow = sorted(tap_costs, key=lambda c: -(c[1] + c[2]))[:4]
        out.append("- per-tap reaction (wall-clock to reach+fire; think vs perception): " + "; ".join(
            f"{what} {think + perc:.1f}s = {think:.1f}s think + {perc:.1f}s screencap/classify"
            for what, think, perc in slow))
    if captures_ms:
        total = sum(captures_ms) / 1000.0
        mean_ms = total * 1000.0 / len(captures_ms)
        line = (f"- captures: {len(captures_ms)} screencaps, {total:.1f}s total "
                f"(mean {mean_ms:.0f}ms, max {max(captures_ms):.0f}ms) - wireless-ADB "
                "frame I/O; ~1.0s USB / ~1.4s Wi-Fi per frame is normal for this phone")
        if len(captures_ms) >= 6:
            # First-third vs last-third mean: a link degrading through the run is the classic
            # precursor to the stall/timeout that kills a capture. A one-number mean hides it
            # (a run that started fast and ended crawling reads "fine on average"); the trend
            # is what would have foretold THIS report's timeout. Only flagged when the rise is
            # both large in ratio AND absolute, so ordinary jitter stays quiet.
            third = len(captures_ms) // 3
            early = sum(captures_ms[:third]) / third
            late = sum(captures_ms[-third:]) / third
            if late > early * 1.4 and late - early > 800:
                line += (f". TREND: mean rose {early:.0f}ms -> {late:.0f}ms (first third -> last "
                         "third) -- the link was DEGRADING through the run, the classic precursor "
                         "to a stall/timeout")
        out.append(line)
    if capture_retry_ms:
        # Transient screencap stalls that the new capture-retry recovered from instead of
        # crashing. Their existence means the link wobbled; their hung time is real wall-clock.
        hung = sum(capture_retry_ms) / 1000.0
        note = (f"- capture retries: {len(capture_retry_ms)} screencap(s) stalled/failed and were "
                f"retried after a forced reconnect ({hung:.1f}s hung total) -- a wireless-link "
                "wobble the hunt rode through; a rising count means the link is going bad")
        if capture_retry_last_err:
            note += f". Last retry error: {capture_retry_last_err}"
        out.append(note)
    # Keep the writer/reconnect story adjacent to the capture-retry line: that is the
    # causal sequence a raw BrokenPipe traceback used to hide (reconnect -> old shell
    # dies -> first later UHID write fails), not merely another generic tap anomaly.
    out.extend(_transport_lifecycle_lines(capture_link_reconnects,
                                          transport_stream_reopens,
                                          transport_failure))
    if classify_ms:
        total = sum(classify_ms) / 1000.0
        mean_ms = total * 1000.0 / len(classify_ms)
        out.append(f"- classification: {len(classify_ms)} scans, {total:.1f}s total "
                   f"(mean {mean_ms:.0f}ms, max {max(classify_ms):.0f}ms) - sliding-window "
                   "NCC over every screen anchor; cost grows with frame size and anchor count")
    if gaps:
        biggest_gaps = [g for g in sorted(gaps, key=lambda x: -x[0])[:5] if g[0] >= 2.0]
        if biggest_gaps:
            out.append("- largest journal gaps: " + "; ".join(
                f"{a}->{b} {seconds:.1f}s" for seconds, a, b in biggest_gaps))
    if closest:
        out.append(f"- closest known screen (unknown near-miss): {closest}")
    if last_halt and "unknown screen" in last_halt and unknown_run_len:
        # Where in the flow the halting unknown struck. The near-miss line says which anchor
        # it came closest to (often meaningless for a genuinely novel screen); this says which
        # KNOWN screen preceded it, which is what actually locates the gap.
        ctx = (f"the halting UNKNOWN first appeared after "
               f"{state_before_unknown_run or 'an unnamed screen'} and persisted "
               f"{unknown_run_len} look(s) before failing closed")
        if state_before_unknown_run == "queue":
            ctx += ("; an UNKNOWN that won't settle right after QUEUE is the match-start "
                    "transition -- the VS 'match found' splash, or the board fading in. Some "
                    "clients show a distinct VS splash (the Pixel 7a does), others fade "
                    "queue->black->mulligan. If yours shows one, capture a vs_splash anchor "
                    "from the saved frame: `hop capture --from-file <that PNG> --state "
                    "vs_splash --glyph xf,yf,wf,hf` (aim the glyph at the red 'VS'). The engine "
                    "already has the vs_splash wait-branch; it only needs the anchor.")
        out.append(f"- unknown-halt context: {ctx}")
    if false_halt_note:
        out.append(f"- likely FALSE halt: {false_halt_note}")
    if verified_then_halt:
        out.append(f"- likely FALSE halt (verified-then-stuck): {verified_then_halt}")
    if genuine_stuck_note:
        out.append(f"- likely GENUINE stuck (DROPPED committing tap): {genuine_stuck_note}")
    # Only when neither of the above already told this exact story: a Concede exhaustion
    # (unlike Play's) DOES run through `_wait_until`, so `genuine_stuck_note` already covers
    # it via `wait_timeout_state` -- firing both would duplicate the same paragraph.
    dropped_tap_note = ("" if (genuine_stuck_note or verified_then_halt)
                        else _diagnose_dropped_committing_tap_halt(last_halt))
    if dropped_tap_note:
        out.append(f"- likely cause: {dropped_tap_note}")
    no_change_note = _diagnose_no_change_halt(last_halt, last_tap_what)
    if no_change_note:
        out.append(f"- likely cause: {no_change_note}")
    mulligan_read_note = _diagnose_mulligan_read_halt(last_halt)
    if mulligan_read_note:
        # Name a "could not read mulligan" halt's cause (class blank vs garbled vs card
        # miscount) inline, so it is self-diagnosing instead of a raw message to decode.
        out.append(f"- likely cause: {mulligan_read_note}")
    halt_guidance = _classify_halt(last_halt)
    if halt_guidance:
        # Name the terminal halt as a DELIBERATE fail-closed stop (not a malfunction) and say what
        # to do -- the single line that answers "is this a bug?" for the whole guard family, so a
        # reader doesn't have to know the engine to tell a designed stop from a crash. When the
        # halt followed a run of the same screen (for example, deck_select polled N times),
        # include that bounded-wait shape: a designed guard polls its budget then stops,
        # where a crash bails mid-action, so "looked N×, then failed closed" is itself the tell.
        prefix = "- halt is a DELIBERATE fail-closed stop, not a malfunction: "
        if terminal_state and terminal_state_run >= 2:
            prefix += (f"hop looked at '{terminal_state}' {terminal_state_run}× (a bounded wait) "
                       "then failed closed -- ")
        out.append(prefix + halt_guidance)
    if last_halt and len(state_trail) >= 2:
        # The screens the run passed through just before the halt. This is what a truncated
        # journal tail hides: in this report it reads "... -> queue -> error_dialog ->
        # deck_select", which names the transient error dialog as the thing that dropped hop
        # to the deck list (the deliberate DECK_SELECT halt) -- no journal archaeology needed.
        out.append("- state trail into the halt: " + " -> ".join(state_trail[-6:]))
    if last_halt:
        out.append(f"- ended: {last_halt}")
    return "\n".join(out)


# ── collection (I/O) ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReportPaths:
    config: Path
    runs_root: Path
    unknowns_dir: Path
    app_log: Path
    templates: Path


def _read(path: Path, limit_bytes: int = 512 * 1024) -> str:
    try:
        data = path.read_bytes()[-limit_bytes:]
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def _ocr_unknown_frames(unknowns_dir: Path, names: list[str], *,
                        max_frames: int = 3, max_chars: int = 260) -> dict[str, str]:
    """Best-effort OCR of the newest unrecognised-screen frames. Returns ``{name: text}``.

    The one fact that actually NAMES an unknown screen -- a banner reading "Opponent Still
    Choosing...", "You are currently offline", "There was an error starting your game" --
    lives in the frame's pixels, invisible in a report that lists only the filename. Reading
    it here (tesseract is already a hop dependency) puts that banner in front of the diagnosis
    instead of behind a "go open the PNG" step: the single highest-value line for an
    unrecognised-screen halt, because it tells you which anchor to build.

    Only the newest ``max_frames`` (the ones nearest the failure), truncated, so this never
    bloats the report or the runtime. Guarded end to end -- no OCR engine, an unreadable
    frame, or a decode error simply yields no text for that file, never a broken report.
    """
    try:  # optional deps; a report must build without them
        from io import BytesIO

        import numpy as np
        import pytesseract
        from PIL import Image
    except Exception:
        return {}

    def _read_text(im, cfg=""):
        return " ".join(pytesseract.image_to_string(im, config=cfg).split())

    out: dict[str, str] = {}
    for name in list(names)[-max_frames:]:
        try:
            png = (Path(unknowns_dir) / name).read_bytes()
            img = Image.open(BytesIO(png)).convert("L")
            # The top banner is what NAMES most screens ("Opponent Still Choosing...", "You
            # are currently offline", "There was an error starting your game"), but it is
            # ornate glowing text a plain full-frame scan misses. Isolate the bright pixels
            # in the top band and read them as one block -- that recovers the banner; then a
            # full-frame pass adds any other identifying text (card names, portraits, dialog
            # bodies). Banner first so it leads the (truncated) line.
            w, h = img.size
            band = np.asarray(img.crop((0, 0, w, int(h * 0.25))))
            banner = Image.fromarray(np.where(band > 180, 255, 0).astype("uint8"))
            banner_text = _read_text(banner, "--psm 6")
            body_text = _read_text(img)
            text = banner_text
            if body_text and body_text != banner_text:
                text = (banner_text + " | " + body_text).strip(" |")
            if text:
                out[name] = text[:max_chars] + ("…" if len(text) > max_chars else "")
        except Exception:
            continue
    return out


def mulligan_card_count_evidence(journal_text: str) -> dict | None:
    """Extract the newest structured mulligan card-count failure from a journal.

    The old ``mulligan_read`` line could say only ``cards=0``.  That tells us *which*
    perception signal failed, but not whether the colour plane vanished, the green strips
    were absent, candidate card spans missed the width/brightness gates, or the hand-coherence
    guard correctly rejected a partial hand.  New engines attach a bounded diagnostic dict to
    their saved anomaly as ``mulligan_card_count``.  Keep the parser deliberately tolerant of
    the short-lived standalone event spellings as well: reporting must never be the fragile part
    of an exceptional halt.

    The immediately preceding mulligan read supplies class/OCR context without duplicating it in
    the diagnostic payload.  The returned values are all journal-safe primitives.
    """
    latest_read: dict = {}
    found: dict | None = None
    for line in journal_text.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("kind")
        detail = event.get("detail")
        if not isinstance(detail, dict):
            continue
        if kind == "mulligan_read":
            latest_read = dict(detail)
            continue

        diagnostic = None
        if kind == "anomaly":
            diagnostic = detail.get("mulligan_card_count")
        elif kind in ("mulligan_card_count", "mulligan_card_count_failure",
                      "mulligan_card_count_unreadable"):
            diagnostic = detail.get("diagnostics", detail.get("mulligan_card_count", detail))

        if isinstance(diagnostic, dict):
            index = detail.get("index")
            decision = _mulligan_decision(detail) or _mulligan_decision(latest_read)
            require_second = detail.get("require_second")
            found = {
                "diagnostics": diagnostic,
                "read": dict(latest_read),
                "anomaly_index": index if isinstance(index, int) and index > 0 else None,
                "reason": str(detail.get("reason") or ""),
                "decision": decision,
                "require_second": require_second if isinstance(require_second, bool) else None,
            }
        elif (kind == "anomaly" and diagnostic is True and found is not None):
            # Transitional writers may emit the metrics in a separate event followed by the
            # anomaly that saved the PNG.  Marry the latter's index to the newest metrics.
            index = detail.get("index")
            if isinstance(index, int) and index > 0:
                found["anomaly_index"] = index
            if detail.get("reason"):
                found["reason"] = str(detail["reason"])
    return found


def mulligan_card_count_evidence_frame(evidence: dict | None) -> str | None:
    """Filename of the bounded colour frame associated with card-count evidence."""
    if not evidence:
        return None
    index = evidence.get("anomaly_index")
    if not isinstance(index, int) or index <= 0:
        return None
    return f"anomaly_{index}_before.png"


def _span_text(value) -> str:
    """Compact x-span rendering for JSON ``[start, end]`` values."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            return f"{start:g}\N{EN DASH}{end:g}"
    return "?"


def _number_text(value) -> str:
    """Compact safe numeric rendering for a diagnostic field."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return "?" if value is None else str(value)


_MULLIGAN_CARD_REJECTIONS = {
    "no_rgb": "no RGB colour plane was available, so green glow cannot be measured",
    "frame_too_narrow": "the capture was below the safe minimum width",
    "empty_card_row": "the calibrated card-row crop was empty",
    "insufficient_glow_strips": "fewer than two qualifying green glow strips were found",
    "no_candidate_interiors": "no span passed both the card-width and brightness gates",
    "incoherent_hand": "candidate cards failed the contiguous/centred whole-hand guard",
    "implausible_card_total": "the coherent candidate total was not three or four",
    "ok": "the detector accepted the hand",
}


def format_mulligan_card_count_evidence(evidence: dict, *, frame_exists: bool) -> str:
    """Render structured glow/card-count telemetry as a compact report section.

    This is intentionally text-first: the important pass/fail measurements survive clipboard
    paste, while the retained *colour* PNG can be attached if visual threshold tuning is still
    needed.  It accepts a partially populated dict so journals from an interrupted write remain
    reportable.
    """
    diagnostics = evidence.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return ""
    read = evidence.get("read")
    read = read if isinstance(read, dict) else {}
    frame = diagnostics.get("frame")
    frame = frame if isinstance(frame, dict) else {}
    row = diagnostics.get("card_row")
    row = row if isinstance(row, dict) else {}
    thresholds = diagnostics.get("thresholds")
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    out = [
        "Hop retained this mulligan's colour frame and detector measurements because its "
        "keep-glow card count was unreadable. The run may have halted or safely made a "
        "class-only decision; the metrics below survive paste. Attach the PNG too if glow "
        "thresholds need tuning.",
        "",
    ]

    frame_name = mulligan_card_count_evidence_frame(evidence)
    if frame_name:
        if frame_exists:
            out.append(f"- exact colour frame: `{frame_name}` (attach this PNG with the report)")
        else:
            out.append(f"- exact frame was journalled as `{frame_name}`, but is no longer on disk "
                       "(anomaly retention may have pruned it)")
    else:
        out.append("- exact frame: unavailable (legacy/incomplete diagnostic record)")

    opponent = read.get("opponent")
    if opponent and opponent != "?":
        raw = read.get("class_raw")
        detail = f"opponent {opponent}"
        if raw:
            detail += f" (raw {raw!r})"
        cards = read.get("cards")
        if cards is not None:
            detail += f"; reported cards={_number_text(cards)}"
        detail += f"; {_turn_label(_mulligan_turn(read))}"
        out.append(f"- mulligan read: {detail}")

    decision = evidence.get("decision") or _mulligan_decision(read)
    require_second = evidence.get("require_second")
    if decision:
        decision_text = f"- engine decision: {decision}"
        if isinstance(require_second, bool):
            decision_text += f"; require_second={str(require_second).lower()}"
        if decision in ("keep", "reject") and _mulligan_turn(read) is None:
            decision_text += " (class-only; no turn was invented)"
        elif decision == "unusable":
            decision_text += " (fail-closed: a valid turn was needed)"
        out.append(decision_text)

    w, h = frame.get("width"), frame.get("height")
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        colour = "RGB present" if frame.get("has_rgb") is True else "NO RGB"
        out.append(f"- capture: {_number_text(w)}×{_number_text(h)}; {colour}")
    if row:
        keys = ("x", "y", "width", "height")
        if all(k in row for k in keys):
            out.append("- card-row crop: " + ", ".join(
                f"{k}={_number_text(row[k])}" for k in keys))

    rejection = str(diagnostics.get("rejection") or "unknown")
    verdict = _MULLIGAN_CARD_REJECTIONS.get(rejection, rejection.replace("_", " "))
    coherent = diagnostics.get("coherent")
    coherent_note = "" if coherent is None else f"; coherent={str(bool(coherent)).lower()}"
    out.append(f"- detector verdict: {verdict} (`{rejection}`){coherent_note}")

    runs = diagnostics.get("glow_runs")
    if isinstance(runs, list):
        spans = ", ".join(_span_text(span) for span in runs[:12]) or "none"
        more = " (truncated)" if diagnostics.get("glow_runs_truncated") else ""
        count = diagnostics.get("glow_run_count")
        out.append(f"- qualifying green strips, card-row-relative x "
                   f"({_number_text(count)}): {spans}{more}")

    candidates = diagnostics.get("candidates")
    if isinstance(candidates, list) and candidates:
        bits = []
        for c in candidates[:8]:
            if not isinstance(c, dict):
                continue
            span = _span_text([c.get("start"), c.get("end")])
            width = _number_text(c.get("width_px"))
            mean = _number_text(c.get("mean_rgb"))
            width_ok = "pass" if c.get("width_ok") else "fail"
            bright_ok = "pass" if c.get("brightness_ok") else "fail"
            chosen = "selected" if c.get("selected") else "rejected"
            bits.append(f"{span} ({width}px, width {width_ok}; mean RGB {mean}, "
                        f"brightness {bright_ok}; {chosen})")
        if bits:
            more = " (truncated)" if diagnostics.get("candidates_truncated") else ""
            out.append("- inter-strip candidates: " + "; ".join(bits) + more)

    interiors = diagnostics.get("candidate_interiors")
    if isinstance(interiors, list):
        spans = ", ".join(_span_text(span) for span in interiors[:12]) or "none"
        out.append(f"- candidates passing both gates (absolute x): {spans}")

    if thresholds:
        glow = (f"G > R/B + {_number_text(thresholds.get('glow_green_bias'))}; "
                f"G > {_number_text(thresholds.get('glow_min_green'))}; "
                f"column >= {_number_text(thresholds.get('glow_column_min_rows'))} rows; "
                f"strip >= {_number_text(thresholds.get('glow_min_strip_width_px'))}px")
        card = (f"card width {_number_text(thresholds.get('expected_width_px'))} "
                f"± {_number_text(thresholds.get('width_tolerance_px'))}px; "
                f"mean RGB >= {_number_text(thresholds.get('min_mean_rgb'))}; "
                f"minimum frame width {_number_text(thresholds.get('min_frame_width'))}px")
        out.append(f"- active thresholds: {glow}; {card}")
    return "\n".join(out)


def terminal_evidence_frame(journal_text: str) -> str | None:
    """Return the saved frame name for the final named-but-stuck screen, if any.

    A state can be *wrongly named* rather than UNKNOWN: the ranked-progress medal in
    particular leaves the board's End Turn anchor visible behind it and is therefore
    classified as ``in_game``. Engine records that terminal frame as an anomaly with
    ``terminal=true``. Keeping the selection here rather than listing every anomaly means a
    pasted report leads with the one image that explains the halt.
    """
    candidate: tuple[int, str] | None = None
    for line in journal_text.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("kind") != "anomaly":
            continue
        detail = event.get("detail") or {}
        if not detail.get("terminal"):
            continue
        index = detail.get("index")
        if isinstance(index, int) and index > 0:
            candidate = (index, str(detail.get("reason") or "terminal screen"))
    if candidate is None:
        return None
    return f"anomaly_{candidate[0]}_before.png"


def latest_run_dir(runs_root: Path) -> Path | None:
    """The newest run directory (names are %Y%m%d-%H%M%S, so lexical == chronological)."""
    try:
        dirs = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return None
    return dirs[-1] if dirs else None


def _recent_run_outcome(journal_text: str) -> str:
    """Return one compact, run-scoped outcome from a journal.

    The live dashboard tally intentionally spans the day, while ``Latest run summary``
    intentionally describes only one directory.  A report that showed both without a
    bridge made a prior target (for example, Priest) look causally related to a later,
    correctly-rejected opponent (for example, Hunter).  This lightweight reconstruction
    gives every retained run one decision/outcome sentence; the detailed latest-run
    summary remains the source for timing and diagnosis.

    Malformed or pre-journal records are deliberately tolerated.  We only state a
    criterion verdict when that run recorded ``run_criteria`` before its decision.
    """
    events = []
    for line in journal_text.splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(event, dict):
            events.append(event)
    if not events:
        return ""

    crit_seen = False
    crit_targets: list = []
    crit_avoid: list = []
    crit_require_second = False
    last_read: tuple[str, bool | None] | None = None
    last_reject: tuple[str, bool | None, str, int] | None = None
    target: tuple[str, bool | None] | None = None
    terminal: tuple[str, int] | None = None

    for index, event in enumerate(events):
        kind = event.get("kind")
        detail = event.get("detail")
        if not isinstance(detail, dict):
            detail = {}
        if kind == "run_criteria":
            targets = detail.get("target_classes")
            avoid = detail.get("avoid_classes")
            crit_targets = list(targets) if isinstance(targets, (list, tuple)) else []
            crit_avoid = list(avoid) if isinstance(avoid, (list, tuple)) else []
            crit_require_second = bool(detail.get("require_second"))
            crit_seen = True
        elif kind == "mulligan_read":
            # A watch observation must not become the latest game for a following
            # reject decision. Legacy records that lack the flag retain their old
            # behavior; see `_is_pending_mulligan_read`.
            if _is_pending_mulligan_read(detail):
                continue
            opponent = detail.get("opponent")
            # A new `decision` makes a class-only read actionable when the coin is
            # intentionally irrelevant.  Legacy records still require 3/4 cards.
            if opponent and opponent != "?" and _is_actionable_mulligan_read(detail):
                last_read = (str(opponent), _mulligan_turn(detail))
        elif kind == "reject_plan" and last_read is not None:
            opponent, second = last_read
            reason = (_reject_reason(opponent, second, crit_require_second,
                                     crit_targets, crit_avoid)
                      if crit_seen else "")
            last_reject = (opponent, second, reason, index)
            last_read = None       # one reject consumes one game/read pairing
        elif kind == "target_found":
            target = (str(detail.get("opponent") or "?"), _mulligan_turn(detail))
        elif kind in ("halt", "stop") and detail.get("message"):
            terminal = (str(detail["message"]), index)

    if crit_seen:
        criteria = (f"targets={crit_targets or 'ANY'} "
                    f"require_second={crit_require_second}")
    else:
        criteria = "criteria not journalled"

    if target is not None:
        opponent, second = target
        decision = f"TARGET FOUND {opponent} ({_turn_label(second)})"
        if crit_seen:
            verdict = _reject_reason(opponent, second, crit_require_second,
                                     crit_targets, crit_avoid)
            if verdict == "misfire":
                decision += " — matched the recorded criteria"
            elif verdict == "turn_unreadable":
                decision += " — turn unreadable, so require_second cannot be independently verified"
            else:
                decision += " — DOES NOT match the recorded criteria (MISFIRE)"
    elif last_reject is not None:
        opponent, second, reason, reject_index = last_reject
        decision = f"REJECTED {opponent} ({_turn_label(second)})"
        if reason == "misfire":
            decision += " — criteria say KEEP (MISFIRE)"
        elif reason == "turn_unreadable":
            decision += " — turn unreadable; cannot independently verify the reject reason"
        elif reason:
            decision += f" correctly: {_REJECT_LABELS[reason]}"
        else:
            decision += " (criteria unavailable)"
        if terminal is not None and terminal[1] > reject_index:
            message = " ".join(terminal[0].split())
            if len(message) > 180:
                message = message[:179] + "…"
            decision += f"; terminal stop came later: {message}"
    elif terminal is not None:
        message = " ".join(terminal[0].split())
        if len(message) > 180:
            message = message[:179] + "…"
        decision = f"terminal stop: {message}"
    else:
        decision = "no resolved matchup decision or terminal outcome journalled"
    return f"{criteria}; {decision}."


def summarize_recent_runs(runs_root: Path, *, max_runs: int = 5) -> str:
    """Summarize the retained run outcomes in chronological order.

    ``DebugLog`` normally retains five run directories.  Keep this report section to the
    same small window and mark the newest entry, which is the run expanded below.  The
    timeline is deliberately per-run rather than a cumulative counter, so a prior
    target cannot be mistaken for the latest run's decision.
    """
    try:
        dirs = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return ""
    if max_runs <= 0:
        return ""
    selected = dirs[-max_runs:]
    lines = []
    for pos, run_dir in enumerate(selected):
        outcome = _recent_run_outcome(_read(run_dir / "journal.jsonl"))
        if not outcome:
            continue
        latest = " (latest; detailed below)" if pos == len(selected) - 1 else ""
        lines.append(f"- {run_dir.name}{latest}: {outcome}")
    return "\n".join(lines)


def format_status(status: dict) -> str:
    """Render a controller ``status()`` snapshot into report lines.

    This is the "what was the hunt actually doing?" section: running/idle, how long,
    how many actions and games in, the budget, the human-state, the class distribution.
    Added because a "Stop takes forever" report couldn't say whether the engine was
    mid-wait or how far in - the journal shows taps, not the live loop state. Best-effort
    over a plain dict so this module never imports the runner.
    """
    if not status:
        return ""
    g = status.get
    lines = [f"- running: {g('running')}   uptime_s: {g('uptime_s')}"]
    # The target-found stop is the ONE common successful exit that records no stop_reason on older
    # builds -- run() breaks on stats.target_found without setting one -- so a reader sees "running:
    # False" with no reason and mistakes the SUCCESS for a crash (this very report's confusion). Say
    # plainly it is the designed win: hop found a class matching your criteria at the mulligan and
    # STOPS touching the game so you can play it. Derived from running/target_found (NOT stop_reason)
    # so it fires identically on an OLD build (stop_reason="") and a NEW one (stop_reason=
    # "target_found"); gated on no last_error so a genuine crash is never blessed as a success.
    if not g("running") and g("target_found") and not g("last_error"):
        opp = g("last_opponent") or "a targeted class"
        lines.append(f"- outcome: SUCCESS — hop found {opp}, which matched your criteria, and "
                     "STOPPED ON PURPOSE so you can play the game (target -> alert + stop touching "
                     "the game). 'running: False' with no error here is the WIN condition, NOT a crash.")
    crit = g("criteria") or {}
    if crit:
        # the criteria the run ACTUALLY used (dashboard/menu overrides included), so a
        # report shows what it hunted for even when the config file on disk is stale.
        targets = crit.get("target_classes") or []
        lines.append(f"- criteria: targets={targets or 'ANY'}   "
                     f"require_second={crit.get('require_second')}   mode={crit.get('mode')}")
    if g("stop_reason"):
        lines.append(f"- stop_reason: {g('stop_reason')}")
    if g("last_error"):
        lines.append(f"- last_error: {g('last_error')}")
        # Name a recognized crash cause inline (screencap timeout = wireless-link freeze, not a
        # hop fault), so the report SAYS what the raw message means instead of leaving it decoded
        # by hand -- the same self-diagnosing treatment the halt messages already get.
        why = _diagnose_last_error(str(g("last_error")))
        if why:
            lines.append(f"  ↳ likely cause: {why}")
    transport_line = _live_transport_status_line(g("transport_status"))
    if transport_line:
        # This is a live/surviving snapshot, complementary to the structured journal
        # lifecycle below. It keeps a child-process exit visible at the TOP of a report
        # even if a crash happened before a run journal could be opened.
        lines.append(f"- input transport: {transport_line}")
    if "games" in status:
        lines.append(f"- games: {g('games')}   concedes: {g('concedes')}   "
                     f"target_found: {g('target_found')}   last_opponent: {g('last_opponent')}")
    b = g("budget") or {}
    if b:
        lines.append(f"- actions: {b.get('actions_run')}   "
                     f"concedes: {b.get('concedes_run')}   "
                     f"games: {b.get('games_session')}   "
                     f"session_min: {b.get('session_minutes')}")
    hs = g("human_state") or {}
    if hs:
        lines.append(f"- human_state: attention={hs.get('attention')} confidence={hs.get('confidence')} "
                     f"fatigue={hs.get('fatigue')} familiarity={hs.get('familiarity')} actions={hs.get('actions')}")
    # `class_distribution` is the dashboard's observed total, normally day-scoped and
    # persistent across runs.  It was previously printed without a scope, which made a
    # historical ``Priest×1`` beside the latest run's ``Hunter×1`` read like one run had
    # stopped on Hunter.  Controller status now supplies the run-local tally separately;
    # retain a cautious label for older callers that do not yet supply the scope metadata.
    run_dist_present = "run_class_distribution" in status
    run_dist = g("run_class_distribution") or {}
    if run_dist_present:
        lines.append("- class_distribution (this run): " +
                     (", ".join(f"{k}×{v}" for k, v in run_dist.items()) if run_dist else "none yet"))
    dist = g("class_distribution") or {}
    scope = g("class_distribution_scope")
    if dist and not (scope == "run" and run_dist_present):
        if scope == "day":
            label = "observed class_distribution (today, across runs)"
        elif scope == "run":
            label = "class_distribution (this run)"
        else:
            label = "class_distribution (scope unspecified; may include earlier runs)"
        lines.append(f"- {label}: " + ", ".join(f"{k}×{v}" for k, v in dist.items()))
    if scope == "day" and run_dist_present:
        lines.append("- distribution scope: the observed-today total is historical context; "
                     "use the this-run line and latest journal to tell what this hunt decided.")
    tb = g("last_error_traceback")
    if tb:
        # The full traceback of the last crash, fenced so it renders as a code block
        # inside this section. Without it, a bare "TypeError: ..." message is a grep
        # hunt; with it, the report points straight at the failing line.
        lines += ["", "Last error traceback:", "```text", str(tb).rstrip(), "```"]
    return "\n".join(lines)


def perception_env(*, probe=None) -> list[str]:
    """Whether the numpy-accelerated perception path is live -- the single biggest
    determinant of how long a screen classify takes, and until now absent from the report.

    Classification is a sliding-window NCC over ~19 screen anchors on a full-resolution
    capture. WITH numpy it is one vectorized FFT + summed-area pass (~45 ms/frame here);
    WITHOUT it, classify falls back to a pure-Python per-window loop that is ~80x slower --
    multiple seconds per look, which makes every closed-loop state check the dominant cost
    of a run. That is exactly the shape of an "each cycle is unacceptably slow" report, and
    the two cases have opposite fixes (optimize the vectorized path vs. `pip install numpy`),
    so a report that never says which path is live leaves the root cause to guesswork.
    Pillow decodes the screencap PNG at all; without it perception cannot run.

    Pure import probes (no subprocess, cannot hang); ``probe`` is injectable for tests.
    """
    def _default_probe(mod):
        import importlib
        try:
            return getattr(importlib.import_module(mod), "__version__", "present")
        except Exception:
            return None
    probe = probe or _default_probe
    npv, pilv = probe("numpy"), probe("PIL")
    lines = []
    if npv:
        lines.append(f"- numpy: {npv}  (vectorized NCC active — ~45 ms/classify)")
    else:
        lines.append("- numpy: NOT INSTALLED  <-- classify runs the ~80x-slower pure-Python "
                     "per-window loop (seconds/look); `pip install numpy` is the likely fix "
                     "for a slow run")
    lines.append(f"- Pillow: {pilv or 'NOT INSTALLED — screencaps cannot decode; perception is dead'}")
    return lines


def tooling_summary(*, which=None, env=None, perception=None) -> str:
    """The external tools hop shells out to, and whether they resolve on PATH.

    A .app launched from Finder gets a minimal PATH; when ``adb`` or ``tesseract`` fall
    off it, Start Search dies with "No such file: 'adb'" and OCR can't read the class.
    Surfacing the resolved paths + PATH makes that root cause obvious from the report
    rather than something to infer. Pure lookups (no subprocess), so it can't hang;
    ``which``/``env``/``perception`` are injectable for tests.
    """
    import shutil
    import sys
    which = which or shutil.which
    environ = env if env is not None else os.environ
    lines = [f"- python: {sys.executable}"]
    for tool in ("adb", "tesseract"):
        lines.append(f"- {tool} on PATH: {which(tool) or 'NOT FOUND'}")
    # The perception accelerator governs classify cost far more than any PATH lookup does.
    lines.extend(perception_env() if perception is None else perception)
    lines.append(f"- PATH: {environ.get('PATH', '')}")
    return "\n".join(lines)


def _bounded_run(args: list[str], timeout: float, run) -> str | None:
    """Run a diagnostic command, returning its decoded stdout+stderr, or ``None`` on any
    failure. Never raises and never hangs past ``timeout`` -- a report probe must be safe even
    when the tool is missing, slow, or the device is exactly the thing that's wedged. stderr is
    folded in because the useful text for a *failed* probe (``adb get-state`` on a dead link:
    ``error: device offline``) lives there."""
    try:
        proc = run(args, capture_output=True, timeout=timeout)
    except Exception:
        return None
    parts = []
    for stream in (getattr(proc, "stdout", b""), getattr(proc, "stderr", b"")):
        if not stream:
            continue
        if isinstance(stream, (bytes, bytearray)):
            stream = bytes(stream).decode("utf-8", errors="replace")
        parts.append(str(stream))
    return "\n".join(p for p in parts if p.strip())


def _adb_host_port(adb_address: str) -> tuple[str, str]:
    """Extract an ADB host and optional numeric port without mangling IPv6.

    Wireless ADB normally uses ``IPv4:port``.  Bracketed IPv6 is also unambiguous;
    an unbracketed multi-colon value is deliberately rejected rather than reporting a
    truncated pseudo-host in a diagnostic.
    """
    address = (adb_address or "").strip()
    if not address:
        return "", ""
    bracketed = re.fullmatch(r"\[([^\]]+)\](?::(\d+))?", address)
    if bracketed:
        return bracketed.group(1), bracketed.group(2) or ""
    if ":" not in address:
        return address, ""
    if address.count(":") == 1:
        host, port = address.split(":", 1)
        if host and port.isdigit():
            return host, port
    return "", ""


def _host_diagnostics(adb_address: str = "", *, run=None, system=None) -> str:
    """macOS power / sleep / network facts, so a report can answer 'was it the Mac?' itself.

    The exact question a screencap-timeout report could NOT answer -- was the Mac asleep, was
    ``caffeinate`` actually holding a sleep assertion (and the right *type*), is the phone even
    reachable now -- lives in ``pmset`` and a ping, in no hop artifact. A wireless screencap
    hangs to its timeout only when the link genuinely drops, and the top cause on an unattended
    Mac is the machine sleeping: ``caffeinate`` prevents *idle* sleep only on AC power and only
    for the assertion type it holds, so "caffeinate was on" is NOT proof the Mac stayed awake.
    This gathers the deciding facts best-effort: every command is timeout-bounded and guarded
    (see :func:`_bounded_run`), so a slow or missing tool drops its line rather than hanging or
    breaking the report. ``run``/``system`` are injectable so tests need no real pmset/ping.

    Emits nothing off macOS (``pmset`` is macOS-only; the tool ships as a mac ``.app``).
    """
    import subprocess as _sp
    run = run or _sp.run
    system = system or platform.system
    if system() != "Darwin":
        return ""
    lines: list[str] = []

    # Power source: caffeinate -s prevents sleep ONLY on AC, so AC-vs-battery is load-bearing
    # for "could it have slept despite caffeinate?".
    batt = _bounded_run(["pmset", "-g", "batt"], 3, run)
    if batt:
        src = next((ln.strip() for ln in batt.splitlines() if "drawing from" in ln.lower()), "")
        if src:
            lines.append(f"- power source: {src}")

    # Sleep timers (minutes; 0 = never). displaysleep is the lock/screensaver-adjacent one.
    settings = _bounded_run(["pmset", "-g"], 3, run)
    if settings:
        picks = [f"{key}={m.group(1)}m"
                 for key in ("sleep", "displaysleep", "disksleep")
                 for m in [re.search(rf"^\s*{key}\s+(\d+)", settings, re.MULTILINE)] if m]
        if picks:
            lines.append("- sleep timers: " + " ".join(picks) + " (0 = never)")

    # Sleep-preventing assertions held right now. If nothing holds PreventUserIdleSystemSleep,
    # an unattended Mac WILL idle-sleep and freeze wireless ADB -- the exact failure mode. A
    # caffeinate holder confirms the assertion is (still) live at report time.
    assertions = _bounded_run(["pmset", "-g", "assertions"], 3, run)
    if assertions:
        held = [f"{key}={m.group(1)}"
                for key in ("PreventUserIdleSystemSleep", "PreventSystemSleep",
                            "PreventUserIdleDisplaySleep")
                for m in [re.search(rf"\b{key}\s+(\d+)", assertions)] if m]
        if held:
            lines.append("- sleep assertions: " + " ".join(held)
                         + " (1 = held/blocked, 0 = NOT blocked)")
        if "caffeinate" in assertions.lower():
            lines.append("- caffeinate: an assertion is held by a caffeinate process (active now)")
        else:
            lines.append("- caffeinate: no caffeinate assertion visible at report time "
                         "(it may have exited, or never held a system-sleep assertion)")

    # Recent sleep/wake history: a Sleep/Wake pair straddling the crash time is the smoking gun
    # for "the Mac slept". pmset -g log is large, so bound it hard and keep only sleep/wake rows.
    plog = _bounded_run(["pmset", "-g", "log"], 6, run)
    if plog:
        evts = [ln.strip() for ln in plog.splitlines()
                if re.search(r"\b(Sleep|Wake|DarkWake)\b", ln)
                and not ln.lstrip().startswith("*")]
        tail = [ln[:160] for ln in evts[-10:]]
        if tail:
            lines.append("- recent power events (pmset -g log; a Sleep/Wake around the crash "
                         "time = the Mac slept):")
            lines.extend("    " + ln for ln in tail)

    # Is the phone reachable NOW? Distinguishes "still down" (network/phone) from "recovered"
    # (a transient stall, or a Mac-side sleep that has since woken).  Ping says whether a
    # packet got a reply, but a fast ``Network is unreachable`` needs one earlier fact: did
    # macOS even have a route for the configured phone IP?  A route through ``utun*`` also
    # identifies a VPN/split-tunnel path without disclosing the user's SSID or host address.
    host, configured_port = _adb_host_port(adb_address)
    route_unreachable = False
    if host:
        route = _bounded_run(["route", "-n", "get", host], 3, run)
        if route:
            route_l = route.lower()
            route_unreachable = any(needle in route_l for needle in (
                "network is unreachable", "no route to host", "not in table"))
            if route_unreachable:
                lines.append("- macOS route to phone: NO ROUTE (the configured IP is not "
                             "reachable from this Mac right now -- check the phone's current "
                             "Wi-Fi IP, LAN/VLAN reachability, or VPN/split-tunnel routing)")
            else:
                interface_m = re.search(r"^\s*interface:\s*(\S+)", route, re.MULTILINE)
                if interface_m:
                    interface = interface_m.group(1)
                    note = (" (VPN tunnel -- verify its local-LAN/split-tunnel route)"
                            if interface.startswith("utun") else "")
                    lines.append(f"- macOS route to phone: via {interface}{note}")
        ping = _bounded_run(["ping", "-c", "1", "-t", "2", host], 4, run)
        if ping is not None:
            reachable = "1 packets received" in ping or "1 received" in ping or " 0.0% packet loss" in ping
            lines.append(f"- phone {host} ICMP ping: "
                         + ("reachable" if reachable else "NO reply (unreachable right now)"))
        try:
            from .adb import resolve_adb
            adb_bin = resolve_adb()
            state = _bounded_run([adb_bin, "-s", adb_address, "get-state"], 4, run)
        except Exception:
            adb_bin, state = "", None
        if state is not None:
            lines.append(f"- adb get-state ({adb_address}): "
                         + (state.strip().replace("\n", " ") or "(no output / not connected)"))

        # A wireless refusal is only fixable by re-running `adb tcpip 5555` over USB (see
        # _diagnose_last_error). hop's own Adb.connect() now tries this automatically the
        # moment a connect is REFUSED (see hop.adb.Adb._rescue_via_usb), so this line mostly
        # fires for an OLDER run/report, or when the auto-rescue didn't apply (not plugged in
        # at connect time, or more than one USB device made it too ambiguous to guess). Named
        # here too so it's never just "go check" -- `adb devices -l` lists any USB-attached
        # device, whose id has no ip:port colon and whose state is "device" (not
        # "offline"/"unauthorized"); if one is present right now, this is a single
        # copy-pasteable line, no guesswork. Skipped once the wireless link is already fine
        # (state == "device"): nothing to fix.
        if adb_bin and (state or "").strip() != "device":
            devices = _bounded_run([adb_bin, "devices", "-l"], 4, run)
            if devices is not None:
                from .adb import parse_usb_serials, parse_wlan_ipv4
                usb = parse_usb_serials(devices)
                # More than one USB phone is ambiguous: never pick one (or expose its IP) by
                # accident.  ``Adb._rescue_via_usb`` follows the same sole-device rule.
                if len(usb) == 1:
                    serial = usb[0]
                    wlan = _bounded_run(
                        [adb_bin, "-s", serial, "shell", "ip", "-f", "inet",
                         "addr", "show", "dev", "wlan0"],
                        4, run)
                    phone_ip = parse_wlan_ipv4(wlan or "")
                    if phone_ip:
                        if phone_ip == host:
                            lines.append(f"- USB phone Wi-Fi IPv4: {phone_ip} (matches the "
                                         "configured device.adb_address host; the address is "
                                         "current, so investigate routing or the ADB listener)")
                        else:
                            endpoint = f"{phone_ip}:{configured_port or '5555'}"
                            lines.append(f"- USB phone Wi-Fi IPv4: {phone_ip} (DIFFERS from "
                                         f"configured host {host}; device.adb_address is stale -- "
                                         f"update it to `{endpoint}`)")
                    elif wlan:
                        # ``parse_wlan_ipv4`` deliberately declines ambiguous/non-routable
                        # results.  Say the probe happened without exposing raw shell output.
                        lines.append("- USB phone Wi-Fi IPv4: unavailable or ambiguous (not "
                                     "guessed or included in this report)")
                    if route_unreachable:
                        lines.append(f"- USB-attached device detected right now: {serial} -- a "
                                     "USB connection can keep this session running, but `adb tcpip` "
                                     "cannot fix the NO ROUTE result above; restore network "
                                     "reachability or update device.adb_address first")
                    else:
                        lines.append(f"- USB-attached device detected right now: {serial} -- hop "
                                     "will re-pin the wireless listener automatically on its next "
                                     f"connect attempt, or run `{adb_bin} -s {serial} tcpip 5555` "
                                     "yourself right now (this is the actual fix for a refused "
                                     "wireless connect; Android's 'Wireless debugging' toggle does "
                                     "not do this)")
                elif len(usb) > 1:
                    lines.append("- USB-attached devices: more than one authorized phone is "
                                 "present, so hop will not guess which one to inspect or rescue")

    return "\n".join(lines)


def _fit_report_budget(description: str, sections: list[Section], *, meta: dict,
                       max_chars: int | None = None) -> str:
    """Assemble, and if the report exceeds ``max_chars``, drop the OLDEST lines of the
    biggest log sections (see :data:`_TRIMMABLE_TITLES`) until it fits.

    Budgeted in CHARACTERS, because that is what actually truncates a report pasted into
    Claude Code (a ~50k-character message cap) -- and an external truncation lands
    mid-journal, cutting the NEWEST lines, which are the ones nearest the failure. So we
    trim ourselves, oldest-first: the self-improve prompt, description, live status, config
    and run summary are preserved, and the run journal keeps its tail. A hard-truncation
    backstop guarantees we never hand back something over budget even if the fixed sections
    alone are large (it eats the app-log tail, which is emitted last, never the journal).
    ``max_chars`` defaults to :data:`MAX_REPORT_CHARS`, resolved at call time so it stays
    overridable. Idempotent for a report already under budget.
    """
    if max_chars is None:
        max_chars = MAX_REPORT_CHARS
    report = assemble(description, sections, meta=meta)
    if len(report) <= max_chars:
        return report

    sections = list(sections)  # don't mutate the caller's list
    for title in _TRIMMABLE_TITLES:
        report = assemble(description, sections, meta=meta)
        if len(report) <= max_chars:
            break
        for i, sec in enumerate(sections):
            if sec.title != title:
                continue
            body_lines = sec.body.splitlines()
            note = f"... (oldest lines dropped to fit the {max_chars:,}-character report cap) ..."
            # Shed enough oldest lines (their chars, +1 each for the newline) to get under
            # budget, plus room for the note we add back; keep the last 10 (nearest failure).
            over = len(report) - max_chars + len(note) + 1
            shed = drop = 0
            max_drop = max(0, len(body_lines) - 10)
            while drop < max_drop and shed < over:
                shed += len(body_lines[drop]) + 1
                drop += 1
            if drop > 0:
                sections[i] = Section(sec.title, note + "\n" + "\n".join(body_lines[drop:]),
                                      fenced=sec.fenced, lang=sec.lang)
            break

    report = assemble(description, sections, meta=meta)
    if len(report) > max_chars:
        # Backstop: the fixed sections alone still overflow (a huge config, say). Hard-cut
        # so the pasted report can NEVER be silently truncated mid-line by the 50k limit.
        # This eats the END of the document -- the app log / unknowns, emitted after the
        # journal -- so the summary and the journal tail (the diagnosis) always survive.
        note = "\n\n... (report hard-truncated to the character cap) ...\n"
        report = report[: max(0, max_chars - len(note))] + note
    return report


def collect(description: str, paths: ReportPaths, *, version: str,
            journal_tail_lines: int = 500, log_tail_lines: int = 2000,
            clock=time.time, device_probe=None, status_probe=None, ocr_probe=None,
            host_probe=None) -> str:
    """Gather every artifact and assemble the report. Best-effort: a missing or
    unreadable file drops its section rather than failing the whole report.

    ``device_probe`` is an optional callable returning a device-summary string (adb /
    panel / pack); injected so the report can include live device state without this
    module importing the transport, and so tests can run without a phone.
    ``status_probe`` is an optional callable returning a live engine-status string (see
    :func:`format_status`); the dashboard supplies it since it holds the controller.
    ``ocr_probe(directory, names) -> {name: text}`` reads text off saved evidence frames
    (defaults to :func:`_ocr_unknown_frames`); injectable so tests need no tesseract and no
    real PNGs. It handles both unknown-screen captures and the terminal named-screen capture.
    """
    if ocr_probe is None:
        ocr_probe = _ocr_unknown_frames
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(clock()))
    meta = {"version": version, "when": when,
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})"}

    sections: list[Section] = []

    if status_probe is not None:
        try:
            live = status_probe() or ""
        except Exception as e:  # never let a status probe break the report
            live = f"(status probe failed: {e})"
        sections.append(Section("Live engine status", live))

    if device_probe is not None:
        try:
            probe = device_probe() or ""
        except Exception as e:  # never let a device probe break the report
            probe = f"(device probe failed: {e})"
        sections.append(Section("Device / anchors", probe))

    sections.append(Section("Tooling / environment", tooling_summary()))

    cfg_text = _read(paths.config)
    if cfg_text:
        sections.append(Section("Config (secrets redacted)", redact_toml(cfg_text), fenced=True, lang="toml"))

    # Host power/sleep/network -- answers "was it the Mac (lock/sleep), or the phone/net?" from
    # the report itself, the gap a screencap-timeout crash exposed. The phone's address is
    # derived from the config we just read, so BOTH the CLI and the app path get this with no
    # new call-site wiring. Best-effort; a failed/empty probe drops the section (assemble()).
    if host_probe is None:
        host_probe = _host_diagnostics
    addr_m = re.search(r'^\s*adb_address\s*=\s*"([^"]+)"', cfg_text or "", re.MULTILINE)
    try:
        host = host_probe(addr_m.group(1) if addr_m else "") or ""
    except Exception as e:  # never let a host probe break the report
        host = f"(host diagnostics failed: {e})"
    sections.append(Section("Host power / sleep / network (macOS)", host))

    # A day-scoped dashboard total can legitimately contain a prior target while the newest
    # journal contains a later reject.  Put the retained per-run timeline before the detailed
    # latest summary so that provenance is explicit instead of inferred from mixed scopes.
    recent_runs = summarize_recent_runs(paths.runs_root)
    if recent_runs:
        sections.append(Section("Recent run outcomes (run-scoped; newest is current)", recent_runs))

    run_dir = latest_run_dir(paths.runs_root)
    if run_dir is not None:
        journal_text = _read(run_dir / "journal.jsonl")
        if journal_text:
            summary = summarize_journal(journal_text)
            sections.append(Section(f"Latest run summary ({run_dir.name})", summary))
            card_count = mulligan_card_count_evidence(journal_text)
            card_count_frame = mulligan_card_count_evidence_frame(card_count)
            if card_count:
                sections.append(Section(
                    "Mulligan card-count evidence",
                    format_mulligan_card_count_evidence(
                        card_count,
                        frame_exists=bool(
                            card_count_frame and (run_dir / card_count_frame).is_file()
                        ),
                    ),
                ))
            terminal_frame = terminal_evidence_frame(journal_text)
            # A cards=0 halt has its own colour-specific evidence section above.  Calling it a
            # generic "named screen did not clear" would be both duplicate and wrong: the
            # mulligan was named correctly; only its RGB keep-glow geometry was unreadable.
            if (terminal_frame and terminal_frame != card_count_frame
                    and (run_dir / terminal_frame).is_file()):
                try:
                    ocr = ocr_probe(run_dir, [terminal_frame]) or {}
                except Exception:
                    ocr = {}
                text = ocr.get(terminal_frame)
                evidence = (f"- {terminal_frame} (the final screen recorded at the halt)"
                            + (f'\n    reads: "{text}"' if text else ""))
                sections.append(Section(
                    "Terminal screen evidence",
                    "The final screen was positively classified but did not clear. Its saved "
                    "frame is OCR'd here so a hidden overlay is visible in the report:\n\n" + evidence))
            sections.append(Section("Latest run journal (tail)",
                                    tail(journal_text, journal_tail_lines), fenced=True, lang="json"))

    app_log = _read(paths.app_log)
    if app_log:
        sections.append(Section("App log (~/Library/Logs/hop.log)",
                                tail(app_log, log_tail_lines), fenced=True, lang="text"))

    try:
        unknown_files = sorted(p.name for p in paths.unknowns_dir.glob("unknown_*.png"))
    except OSError:
        unknown_files = []
    if unknown_files:
        try:  # OCR is diagnostic sugar; never let it break the report
            ocr = ocr_probe(paths.unknowns_dir, unknown_files) or {}
        except Exception:
            ocr = {}
        lines = []
        for n in unknown_files:
            lines.append(f"- {n}")
            if ocr.get(n):
                # The banner text that NAMES the screen -- e.g. "Opponent Still Choosing..."
                # -- so the anchor to build is obvious without opening the PNG.
                lines.append(f'    reads: "{ocr[n]}"')
        sections.append(Section(
            "Unrecognized screens (need anchors)",
            "The hunt hit screens no anchor covers. These frames are kept for building "
            "anchors (`hop capture --from-file`); their existence means something is "
            "unhandled. Any OCR'd banner text is shown to name the screen:\n\n"
            + "\n".join(lines)))

    return _fit_report_budget(description, sections, meta=meta)


def copy_to_clipboard(markdown: str, *, runner=None) -> bool:
    """Put the report on the macOS clipboard via ``pbcopy``. Returns True on success.

    This is the default delivery: the user pastes the report straight into Claude Code
    instead of hunting for a file. ``runner`` is injectable for tests; it defaults to
    :func:`subprocess.run`. Any failure (not macOS, ``pbcopy`` absent) returns False
    rather than raising, so a caller can fall back (print it, or write a file).
    """
    import subprocess
    run = runner or subprocess.run
    try:
        proc = run(["pbcopy"], input=markdown.encode("utf-8"))
        return getattr(proc, "returncode", 0) == 0
    except (OSError, ValueError):
        return False


def write_report(markdown: str, dest_dir: Path, *, clock=time.time) -> Path:
    """Write the report to ``dest_dir/hop_bug_report_<timestamp>.md``. Returns the path.

    Kept for the explicit ``hop bugreport --out DIR`` opt-in; the default path is
    :func:`copy_to_clipboard`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(clock()))
    path = dest_dir / f"hop_bug_report_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path
