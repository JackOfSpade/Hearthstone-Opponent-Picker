"""The self-improving bug report.

Pure assembly and the log/journal/redaction helpers are unit-tested here; the I/O
`collect`/`write_report` are exercised against a tmp filesystem.
"""

from pathlib import Path

from hop import bugreport as br


def test_the_self_improve_prompt_is_first_and_present():
    """The meta-instruction must be the first thing Claude Code reads on paste-back."""
    md = br.assemble("it broke", [], meta={"version": "2.0.0"})
    assert md.startswith(br.SELF_IMPROVE_PROMPT)
    assert "improve this bug report harness" in md.lower()
    assert "hop/bugreport.py" in md


def test_assemble_includes_description_and_meta():
    md = br.assemble("mulligan never read", [], meta={
        "version": "9.9", "when": "2026-07-08 22:00:00", "platform": "Darwin"})
    assert "mulligan never read" in md
    assert "9.9" in md and "Darwin" in md


def test_empty_sections_are_dropped_not_padded():
    md = br.assemble("x", [br.Section("Empty", "  "), br.Section("Real", "content")],
                     meta={"version": "1"})
    assert "## Empty" not in md
    assert "## Real" in md and "content" in md


def test_fenced_section_wraps_in_a_code_block():
    md = br.assemble("x", [br.Section("Log", "line1\nline2", fenced=True, lang="text")],
                     meta={"version": "1"})
    assert "```text\nline1\nline2\n```" in md


def test_a_missing_description_is_labelled_not_blank():
    md = br.assemble("", [], meta={"version": "1"})
    assert "no description given" in md


# ── redaction ────────────────────────────────────────────────────────────────

def test_redact_blanks_secretish_values_keeps_comments():
    toml = (
        "# my alert settings\n"
        "[alerts]\n"
        "ntfy_topic = \"my-private-topic\"\n"
        "mac_sound = true   # beep on target\n"
        "[device]\n"
        "adb_address = \"192.168.1.5:5555\"\n"
    )
    red = br.redact_toml(toml)
    assert "my-private-topic" not in red
    assert 'ntfy_topic = "<redacted>"' in red
    assert "# my alert settings" in red      # comments survive
    assert "# beep on target" in red         # inline comments on non-secret lines survive
    assert "192.168.1.5:5555" in red         # adb address is not a secret
    assert "mac_sound = true" in red


# ── log helpers ──────────────────────────────────────────────────────────────

def test_tail_keeps_the_last_lines_and_notes_the_drop():
    text = "\n".join(str(i) for i in range(100))
    out = br.tail(text, 10)
    assert "90 earlier lines omitted" in out
    assert out.strip().endswith("99")
    assert "89" not in out.splitlines()


def test_tail_is_a_noop_below_the_cap():
    assert br.tail("a\nb", 10) == "a\nb"


def test_summarize_journal_surfaces_class_distribution_and_ending():
    import json
    events = [
        {"kind": "tap", "detail": {"what": "play"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Mage"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Mage"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Warrior"}},
        {"kind": "anomaly", "detail": {"reason": "missed tap"}},
        {"kind": "halt", "detail": {"message": "unknown screen"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    summary = br.summarize_journal(text)
    assert "events: 6" in summary
    assert "Mage×2" in summary and "Warrior×1" in summary
    assert "missed tap" in summary
    assert "unknown screen" in summary


def test_summarize_empty_journal_is_empty():
    assert br.summarize_journal("") == ""
    assert br.summarize_journal("not json\n{bad") == ""


def test_summarize_journal_surfaces_the_runs_active_criteria():
    """A run journal's first line records what it hunted for; the summary must show it."""
    import json
    events = [
        {"kind": "run_criteria", "detail": {"target_classes": ["PALADIN"],
                                            "require_second": False, "mode": "casual"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Priest"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    summary = br.summarize_journal(text)
    assert "criteria (this run): targets=['PALADIN']" in summary
    assert "require_second=False" in summary


def test_summarize_journal_surfaces_the_unknown_screen_near_miss():
    """The closest known screen is the single most actionable line for an unknown halt."""
    import json
    events = [
        {"kind": "unknown_screen", "detail": {"where": "mulligan_confirm", "near_misses": [
            {"state": "in_game", "score": 0.539, "thr": 0.72},
            {"state": "collection", "score": 0.29, "thr": 0.72}]}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    s = br.summarize_journal(text)
    assert "closest known screen" in s
    assert "in_game 0.539 (thr 0.72)" in s


def test_summarize_journal_shows_any_when_no_target_classes():
    import json
    text = json.dumps({"kind": "run_criteria",
                       "detail": {"target_classes": [], "require_second": True, "mode": "casual"}})
    assert "targets=ANY" in br.summarize_journal(text)


# ── collection (I/O) ─────────────────────────────────────────────────────────

def _paths(tmp: Path) -> br.ReportPaths:
    return br.ReportPaths(
        config=tmp / "config.toml",
        runs_root=tmp / "runs",
        unknowns_dir=tmp / "unknowns",
        app_log=tmp / "hop.log",
        templates=tmp / "templates",
    )


def test_collect_bundles_config_journal_log_and_unknowns(tmp_path):
    (tmp_path / "config.toml").write_text('[alerts]\nntfy_topic = "hunter2-topic-zzz"\n')
    run = tmp_path / "runs" / "20260708-120000"
    run.mkdir(parents=True)
    (run / "journal.jsonl").write_text(
        '{"kind":"mulligan_read","detail":{"opponent":"Rogue"}}\n'
        '{"kind":"halt","detail":{"message":"stuck"}}\n')
    (tmp_path / "hop.log").write_text("launching\nboom\n")
    (tmp_path / "unknowns").mkdir()
    (tmp_path / "unknowns" / "unknown_x_dispatch.png").write_bytes(b"x")

    md = br.collect("halp", _paths(tmp_path), version="2.0.0",
                    clock=lambda: 0.0, device_probe=lambda: "device: fake")

    assert md.startswith(br.SELF_IMPROVE_PROMPT)
    assert "halp" in md
    assert "device: fake" in md
    assert "hunter2-topic-zzz" not in md and "<redacted>" in md   # config redacted
    assert "Rogue×1" in md and "stuck" in md              # journal summarised
    assert "boom" in md                                   # app log tailed
    assert "unknown_x_dispatch.png" in md                 # unknowns listed


def test_collect_survives_missing_everything(tmp_path):
    """An offline machine with no runs/logs still yields a valid report, not a crash."""
    md = br.collect("nothing here", _paths(tmp_path), version="2.0.0", clock=lambda: 0.0)
    assert md.startswith(br.SELF_IMPROVE_PROMPT)
    assert "nothing here" in md


def test_a_failing_device_probe_never_breaks_the_report(tmp_path):
    def boom():
        raise RuntimeError("adb exploded")
    md = br.collect("x", _paths(tmp_path), version="1", clock=lambda: 0.0, device_probe=boom)
    assert "device probe failed" in md and "adb exploded" in md


def test_write_report_names_by_timestamp(tmp_path):
    p = br.write_report("# report\n", tmp_path / "out", clock=lambda: 0.0)
    assert p.exists() and p.name.startswith("hop_bug_report_") and p.suffix == ".md"
    assert p.read_text() == "# report\n"


# ── clipboard delivery ────────────────────────────────────────────────────────

def test_copy_to_clipboard_pipes_markdown_to_pbcopy():
    seen = {}

    class Done:
        returncode = 0

    def fake_run(argv, input=None):
        seen["argv"] = argv
        seen["input"] = input
        return Done()

    assert br.copy_to_clipboard("# report\n", runner=fake_run) is True
    assert seen["argv"] == ["pbcopy"]
    assert seen["input"] == b"# report\n"      # bytes, on stdin


def test_copy_to_clipboard_returns_false_on_failure():
    def boom(argv, input=None):
        raise OSError("pbcopy not found")
    assert br.copy_to_clipboard("x", runner=boom) is False

    class Fail:
        returncode = 1
    assert br.copy_to_clipboard("x", runner=lambda *a, **k: Fail()) is False


# ── live status snapshot ──────────────────────────────────────────────────────

def test_format_status_surfaces_running_actions_and_distribution():
    s = {
        "running": True, "uptime_s": 12.3, "stop_reason": "", "last_error": "",
        "games": 2, "concedes": 1, "target_found": False, "last_opponent": "Mage",
        "budget": {"actions_run": 5, "actions_run_cap": 100, "concedes_run": 1,
                   "concedes_cap": 10, "games_session": 2, "games_cap": 50,
                   "session_minutes": 0.4},
        "class_distribution": {"Mage": 2, "Rogue": 1},
    }
    out = br.format_status(s)
    assert "running: True" in out
    assert "actions: 5/100" in out
    assert "Mage×2" in out and "Rogue×1" in out


def test_format_status_empty_is_empty():
    assert br.format_status({}) == ""


def test_format_status_shows_the_active_run_criteria():
    out = br.format_status({
        "running": True, "uptime_s": 5,
        "criteria": {"target_classes": ["MAGE"], "require_second": True, "mode": "ranked"},
    })
    assert "criteria: targets=['MAGE']" in out
    assert "require_second=True" in out and "mode=ranked" in out


def test_format_status_criteria_reads_any_when_empty():
    out = br.format_status({
        "running": True, "uptime_s": 5,
        "criteria": {"target_classes": [], "require_second": False, "mode": "casual"},
    })
    assert "targets=ANY" in out


def test_collect_attaches_live_status_when_probed(tmp_path):
    md = br.collect("x", _paths(tmp_path), version="1", clock=lambda: 0.0,
                    status_probe=lambda: "- running: True")
    assert "## Live engine status" in md and "running: True" in md


def test_a_failing_status_probe_never_breaks_the_report(tmp_path):
    def boom():
        raise RuntimeError("kaboom")
    md = br.collect("x", _paths(tmp_path), version="1", clock=lambda: 0.0, status_probe=boom)
    assert "status probe failed" in md and "kaboom" in md


# ── tooling / environment ─────────────────────────────────────────────────────

def test_tooling_summary_flags_missing_tools():
    out = br.tooling_summary(which=lambda name: None, env={"PATH": "/usr/bin:/bin"})
    assert "adb on PATH: NOT FOUND" in out
    assert "tesseract on PATH: NOT FOUND" in out
    assert "/usr/bin:/bin" in out


def test_tooling_summary_reports_resolved_paths():
    out = br.tooling_summary(which=lambda n: "/opt/homebrew/bin/" + n, env={"PATH": "x"})
    assert "adb on PATH: /opt/homebrew/bin/adb" in out
    assert "tesseract on PATH: /opt/homebrew/bin/tesseract" in out


def test_collect_includes_the_tooling_section(tmp_path):
    md = br.collect("x", _paths(tmp_path), version="1", clock=lambda: 0.0)
    assert "## Tooling / environment" in md


# ── last-error traceback ──────────────────────────────────────────────────────

def test_format_status_renders_the_traceback_when_present():
    tb = "Traceback (most recent call last):\n  File ...\nTypeError: boom"
    out = br.format_status({"running": False, "uptime_s": 1, "last_error": "TypeError: boom",
                            "last_error_traceback": tb})
    assert "Last error traceback:" in out
    assert "```text" in out and "TypeError: boom" in out


def test_format_status_omits_the_traceback_block_when_absent():
    out = br.format_status({"running": True, "uptime_s": 1})
    assert "Last error traceback" not in out


# ── 50k-line cap ──────────────────────────────────────────────────────────────

def test_fit_line_budget_trims_oldest_log_lines_first():
    big = "\n".join(f"log{i}" for i in range(200))
    secs = [br.Section("Live engine status", "- running: False"),
            br.Section("App log (~/Library/Logs/hop.log)", big, fenced=True, lang="text")]
    out = br._fit_line_budget("desc", secs, meta={"version": "1"}, max_lines=40)
    assert len(out.splitlines()) <= 40
    assert "log199" in out                    # newest log line survives
    assert "\nlog0\n" not in out              # oldest log line is dropped
    assert "- running: False" in out          # non-log sections are untouched
    assert "oldest lines dropped" in out


def test_fit_line_budget_is_a_noop_under_the_cap():
    secs = [br.Section("App log (~/Library/Logs/hop.log)", "a\nb\nc", fenced=True, lang="text")]
    out = br._fit_line_budget("d", secs, meta={"version": "1"}, max_lines=1000)
    assert "oldest lines dropped" not in out and "a\nb\nc" in out


def test_collect_never_exceeds_the_line_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "MAX_REPORT_LINES", 60)
    (tmp_path / "hop.log").write_text("\n".join(f"line{i}" for i in range(500)) + "\n")
    md = br.collect("halp", _paths(tmp_path), version="2.0.0", clock=lambda: 0.0,
                    log_tail_lines=500)
    assert len(md.splitlines()) <= 60
    assert "line499" in md                    # the most recent log line is kept
    assert md.startswith(br.SELF_IMPROVE_PROMPT)
