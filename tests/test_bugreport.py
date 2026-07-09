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
