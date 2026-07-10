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


#: Hard ceiling on the whole report's line count. A report is pasted straight into
#: Claude Code, so it must not balloon: past this, the oldest lines of the biggest log
#: sections (app log first, then the run journal) are dropped — the newest lines are the
#: most diagnostic. Generous enough that a normal report is never touched.
MAX_REPORT_LINES = 50_000

#: Sections trimmed (in this order, oldest-line-first) when a report exceeds the cap.
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
    reread_recovered = 0   # first read unusable, re-read resolved a class
    reread_failed = 0      # re-read still unreadable (this is what precedes a halt)
    unreadable_first = 0   # first reads that came back "?" (OCR whiffed)
    think_credited = 0.0   # think seconds absorbed into perception latency, not stacked
    # Per-tap wall-clock, split think vs perception, so "why is <action> so slow?" is
    # answered inline instead of by hand-tracing timestamps. Everything since the previous
    # tap (screencaps + classifications to *reach* the action, plus its think) is charged
    # to the tap it leads to.
    tap_costs: list[tuple[str, float, float]] = []   # (what, think_s, perception_s)
    seg_think = 0.0
    seg_perception = 0.0
    anomalies: list[str] = []
    sleeps: dict[str, float] = {}
    longest_sleeps: list[tuple[float, str]] = []
    captures_ms: list[float] = []
    classify_ms: list[float] = []
    gaps: list[tuple[float, str, str]] = []
    last_halt = ""
    criteria = ""
    closest = ""
    false_halt_note = ""   # "waited for X to leave; X is gone, destination just unnamed"
    first_t: float | None = None
    prev_t: float | None = None
    prev_kind = ""
    for e in events:
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
            # "?" is a failed OCR, not a class -- counting it as an "opponent seen"
            # made a recovered run read like a total failure. Count only real classes;
            # track the whiff-then-recover story (reread=True) on its own line so the
            # report answers "did class detection work?" instead of implying it didn't.
            opp = d.get("opponent")
            is_reread = bool(d.get("reread"))
            if opp and opp != "?":
                classes[opp] = classes.get(opp, 0) + 1
                if is_reread:
                    reread_recovered += 1
            elif opp == "?":
                if is_reread:
                    reread_failed += 1
                else:
                    unreadable_first += 1
        if k == "anomaly":
            anomalies.append(str(d.get("reason", "?")))
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
        if k == "classify":
            ms = d.get("ms")
            if isinstance(ms, (int, float)):
                classify_ms.append(float(ms))
                seg_perception += float(ms) / 1000.0
        if k == "tap":
            tap_costs.append((str(d.get("what", "?")), seg_think, seg_perception))
            seg_think = 0.0
            seg_perception = 0.0
        if k in ("stop", "halt") and d.get("message"):
            last_halt = d["message"]
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

    out = [f"- events: {len(events)}"]
    if criteria:
        out.append(f"- criteria (this run): {criteria}")
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
            if classify_ms:
                # Classification is journalled separately, so split it out of the residual:
                # it is the second-biggest cost and lumping it into "logic" hid that.
                cls_s = sum(classify_ms) / 1000.0
                other = max(0.0, wall - slept - cap_s - cls_s)
                out.append(f"- time: {wall:.0f}s wall = {slept:.0f}s humanized waits + "
                           f"{cap_s:.0f}s screencap I/O + {cls_s:.0f}s classification + "
                           f"{other:.0f}s OCR/logic")
            else:
                other = max(0.0, wall - slept - cap_s)
                out.append(f"- time: {wall:.0f}s wall = {slept:.0f}s humanized waits + "
                           f"{cap_s:.0f}s screencap I/O + {other:.0f}s classify/OCR/logic")
        else:
            out.append(f"- time: {wall:.0f}s wall, {slept:.0f}s of it humanized waits "
                       "(the rest is screencap + classify/OCR; a pre-capture-journal run)")
    out.append("- kinds: " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))
    if classes:
        out.append("- opponents seen: " + ", ".join(f"{c}×{n}" for c, n in sorted(classes.items(), key=lambda kv: -kv[1])))
    if reread_recovered or reread_failed or unreadable_first:
        # The whiff-then-recover story. A first-read miss that a re-read resolves is
        # normal perception fallibility, NOT a fault -- say so, so a recovered run
        # isn't mistaken for a broken one. A re-read that itself failed is the line
        # that precedes a "could not read mulligan" halt; call it out.
        parts = []
        if reread_recovered:
            parts.append(f"{reread_recovered} recovered by a re-read (first OCR whiffed, "
                         "class resolved on the retry -- expected, not a fault)")
        if unreadable_first and not reread_recovered and not reread_failed:
            parts.append(f"{unreadable_first} first read(s) came back '?'")
        if reread_failed:
            parts.append(f"{reread_failed} still unreadable after the re-read (this is what precedes a halt)")
        out.append("- mulligan reads: " + "; ".join(parts))
    if anomalies:
        out.append("- anomalies: " + "; ".join(anomalies[-5:]))
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
        out.append(f"- captures: {len(captures_ms)} screencaps, {total:.1f}s total "
                   f"(mean {mean_ms:.0f}ms, max {max(captures_ms):.0f}ms) - wireless-ADB "
                   "frame I/O; ~1.0s USB / ~1.4s Wi-Fi per frame is normal for this phone")
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
    if false_halt_note:
        out.append(f"- likely FALSE halt: {false_halt_note}")
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


def latest_run_dir(runs_root: Path) -> Path | None:
    """The newest run directory (names are %Y%m%d-%H%M%S, so lexical == chronological)."""
    try:
        dirs = sorted((p for p in runs_root.iterdir() if p.is_dir()), key=lambda p: p.name)
    except OSError:
        return None
    return dirs[-1] if dirs else None


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
    dist = g("class_distribution") or {}
    if dist:
        lines.append("- class_distribution: " + ", ".join(f"{k}×{v}" for k, v in dist.items()))
    tb = g("last_error_traceback")
    if tb:
        # The full traceback of the last crash, fenced so it renders as a code block
        # inside this section. Without it, a bare "TypeError: ..." message is a grep
        # hunt; with it, the report points straight at the failing line.
        lines += ["", "Last error traceback:", "```text", str(tb).rstrip(), "```"]
    return "\n".join(lines)


def tooling_summary(*, which=None, env=None) -> str:
    """The external tools hop shells out to, and whether they resolve on PATH.

    A .app launched from Finder gets a minimal PATH; when ``adb`` or ``tesseract`` fall
    off it, Start Search dies with "No such file: 'adb'" and OCR can't read the class.
    Surfacing the resolved paths + PATH makes that root cause obvious from the report
    rather than something to infer. Pure lookups (no subprocess), so it can't hang;
    ``which``/``env`` are injectable for tests.
    """
    import shutil
    import sys
    which = which or shutil.which
    environ = env if env is not None else os.environ
    lines = [f"- python: {sys.executable}"]
    for tool in ("adb", "tesseract"):
        lines.append(f"- {tool} on PATH: {which(tool) or 'NOT FOUND'}")
    lines.append(f"- PATH: {environ.get('PATH', '')}")
    return "\n".join(lines)


def _fit_line_budget(description: str, sections: list[Section], *, meta: dict,
                     max_lines: int | None = None) -> str:
    """Assemble, and if the report exceeds ``max_lines``, drop the oldest lines of the
    biggest log sections (see :data:`_TRIMMABLE_TITLES`) until it fits.

    Oldest-first because the newest log/journal lines are the ones nearest the failure.
    Everything else (the self-improve prompt, description, live status, config) is
    preserved. Idempotent for a report already under budget. ``max_lines`` defaults to
    :data:`MAX_REPORT_LINES`, resolved at call time so it stays overridable.
    """
    if max_lines is None:
        max_lines = MAX_REPORT_LINES
    report = assemble(description, sections, meta=meta)
    over = len(report.splitlines()) - max_lines
    if over <= 0:
        return report

    sections = list(sections)  # don't mutate the caller's list
    for title in _TRIMMABLE_TITLES:
        if over <= 0:
            break
        for i, sec in enumerate(sections):
            if sec.title != title:
                continue
            body_lines = sec.body.splitlines()
            # +1: the omission note we add back is itself a line, so to shed `over`
            # net lines we must drop `over + 1` of the body's oldest lines.
            drop = min(over + 1, max(0, len(body_lines) - 10))  # keep the last few lines
            if drop > 0:
                note = (f"... ({drop} more oldest lines dropped to fit the "
                        f"{max_lines:,}-line report cap) ...")
                sections[i] = Section(sec.title, note + "\n" + "\n".join(body_lines[drop:]),
                                      fenced=sec.fenced, lang=sec.lang)
                over -= drop
            break
    return assemble(description, sections, meta=meta)


def collect(description: str, paths: ReportPaths, *, version: str,
            journal_tail_lines: int = 500, log_tail_lines: int = 2000,
            clock=time.time, device_probe=None, status_probe=None) -> str:
    """Gather every artifact and assemble the report. Best-effort: a missing or
    unreadable file drops its section rather than failing the whole report.

    ``device_probe`` is an optional callable returning a device-summary string (adb /
    panel / pack); injected so the report can include live device state without this
    module importing the transport, and so tests can run without a phone.
    ``status_probe`` is an optional callable returning a live engine-status string (see
    :func:`format_status`); the dashboard supplies it since it holds the controller.
    """
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

    run_dir = latest_run_dir(paths.runs_root)
    if run_dir is not None:
        journal_text = _read(run_dir / "journal.jsonl")
        if journal_text:
            summary = summarize_journal(journal_text)
            sections.append(Section(f"Latest run summary ({run_dir.name})", summary))
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
        sections.append(Section(
            "Unrecognized screens (need anchors)",
            "The hunt hit screens no anchor covers. These frames are kept for building "
            "anchors (`hop capture --from-file`); their existence means something is "
            "unhandled:\n\n" + "\n".join(f"- {n}" for n in unknown_files)))

    return _fit_line_budget(description, sections, meta=meta)


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
