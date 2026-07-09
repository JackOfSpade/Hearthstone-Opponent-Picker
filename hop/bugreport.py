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
    anomalies: list[str] = []
    last_halt = ""
    for e in events:
        k = e.get("kind", "?")
        kinds[k] = kinds.get(k, 0) + 1
        d = e.get("detail", {})
        if k == "mulligan_read" and d.get("opponent"):
            classes[d["opponent"]] = classes.get(d["opponent"], 0) + 1
        if k == "anomaly":
            anomalies.append(str(d.get("reason", "?")))
        if k in ("stop", "halt") and d.get("message"):
            last_halt = d["message"]

    out = [f"- events: {len(events)}"]
    out.append("- kinds: " + ", ".join(f"{k}={n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])))
    if classes:
        out.append("- opponents seen: " + ", ".join(f"{c}×{n}" for c, n in sorted(classes.items(), key=lambda kv: -kv[1])))
    if anomalies:
        out.append("- anomalies: " + "; ".join(anomalies[-5:]))
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
    if g("stop_reason"):
        lines.append(f"- stop_reason: {g('stop_reason')}")
    if g("last_error"):
        lines.append(f"- last_error: {g('last_error')}")
    if "games" in status:
        lines.append(f"- games: {g('games')}   concedes: {g('concedes')}   "
                     f"target_found: {g('target_found')}   last_opponent: {g('last_opponent')}")
    b = g("budget") or {}
    if b:
        lines.append(f"- actions: {b.get('actions_run')}/{b.get('actions_run_cap')}   "
                     f"concedes: {b.get('concedes_run')}/{b.get('concedes_cap')}   "
                     f"games: {b.get('games_session')}/{b.get('games_cap')}   "
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
