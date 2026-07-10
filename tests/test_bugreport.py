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


def test_summarize_journal_surfaces_sleep_budget_and_gaps():
    import json
    events = [
        {"t": 0.0, "kind": "mulligan_read", "detail": {"opponent": "Druid"}},
        {"t": 0.1, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 2.4}},
        {"t": 2.5, "kind": "tap", "detail": {"what": "mulligan_card[1]"}},
        {"t": 3.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 1.6}},
        {"t": 4.6, "kind": "tap", "detail": {"what": "mulligan_confirm"}},
        {"t": 9.2, "kind": "tap", "detail": {"what": "gear"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    summary = br.summarize_journal(text)
    assert "intentional sleeps" in summary
    assert "tap_think=4.0s" in summary
    assert "longest single sleeps: tap_think 2.4s" in summary
    assert "largest journal gaps" in summary and "tap->tap 4.6s" in summary


def test_summarize_journal_accounts_for_screencap_latency():
    """Screencap I/O is the dominant, previously-invisible cost of a run. Once the engine
    journals a `capture` ms per frame, the summary must split the wall-clock into humanized
    waits vs screencap vs the rest, and stat the frames - so "why is it slow" is answered
    at a glance instead of by subtracting timestamps."""
    import json
    events = [
        {"t": 0.0, "kind": "run_criteria",
         "detail": {"target_classes": ["MAGE"], "require_second": False, "mode": "casual"}},
        {"t": 0.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 2.0}},
        {"t": 2.0, "kind": "capture", "detail": {"ms": 1400}},
        {"t": 3.4, "kind": "tap", "detail": {"what": "play"}},
        {"t": 3.4, "kind": "capture", "detail": {"ms": 1600}},
        {"t": 10.0, "kind": "verify_ok", "detail": {"change_kind": "full_transition"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    s = br.summarize_journal(text)
    # wall 10s = 2s humanized + 3s screencap + 5s other
    assert "time: 10s wall = 2s humanized waits + 3s screencap I/O + 5s classify/OCR/logic" in s
    assert "captures: 2 screencaps, 3.0s total" in s
    assert "mean 1500ms" in s and "max 1600ms" in s


def test_summarize_journal_splits_classification_out_of_the_residual():
    """Once the engine journals `classify` ms, the summary must break it out of the
    'classify/OCR/logic' bucket - it is the second-biggest cost and lumping it hid that."""
    import json
    events = [
        {"t": 0.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 2.0}},
        {"t": 2.0, "kind": "capture", "detail": {"ms": 1500}},
        {"t": 3.5, "kind": "classify", "detail": {"ms": 3500, "state": "queue"}},
        {"t": 7.0, "kind": "capture", "detail": {"ms": 1500}},
        {"t": 8.5, "kind": "classify", "detail": {"ms": 3500, "state": "unknown"}},
        {"t": 12.0, "kind": "tap", "detail": {"what": "play"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    s = br.summarize_journal(text)
    # wall 12s = 2s humanized + 3s screencap + 7s classification + 0s OCR/logic
    assert "time: 12s wall = 2s humanized waits + 3s screencap I/O + 7s classification + 0s OCR/logic" in s
    assert "classification: 2 scans, 7.0s total (mean 3500ms, max 3500ms)" in s


def test_summarize_journal_flags_a_marginal_unknown_as_needing_an_anchor():
    """The exact halt this report caught: the board classified UNKNOWN at in_game 0.686
    vs a 0.72 threshold. A miss that small is a known screen with an uncovered face; the
    summary must say so with the margin, not leave it as a bare near-miss."""
    import json
    events = [
        {"kind": "unknown_screen", "detail": {"where": "mulligan_confirm", "near_misses": [
            {"state": "in_game", "score": 0.686, "thr": 0.72}]}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    s = br.summarize_journal(text)
    assert "MARGINAL: missed by 0.034" in s
    assert "another anchor" in s


def test_summarize_journal_does_not_flag_a_far_off_unknown_as_marginal():
    """A genuinely novel screen (top anchor far below threshold) must NOT be mislabelled
    'almost known' - that would send the fix in the wrong direction."""
    import json
    events = [{"kind": "unknown_screen", "detail": {"near_misses": [
        {"state": "collection", "score": 0.20, "thr": 0.72}]}}]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "collection 0.2 (thr 0.72)" in s
    assert "MARGINAL" not in s


def test_summarize_journal_time_split_degrades_without_capture_events():
    """A journal written before capture-timing has no `capture` events; the summary still
    reports the wall/humanized split rather than a misleading 0s of screencap."""
    import json
    events = [
        {"t": 0.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 2.0}},
        {"t": 9.0, "kind": "tap", "detail": {"what": "play"}},
    ]
    text = "\n".join(json.dumps(e) for e in events)
    s = br.summarize_journal(text)
    assert "time: 9s wall, 2s of it humanized waits" in s
    assert "screencap I/O +" not in s          # no fabricated screencap term
    assert "captures:" not in s


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


def test_summarize_journal_credits_a_reread_that_recovered_the_class():
    """The whiff-then-recover case: the first OCR reads opponent '?' (conf 0.0), the
    engine's single re-read resolves the real class and the run concedes correctly.
    The report must show the resolved class and say the miss was recovered -- NOT count
    '?' as an opponent, which made a healthy run read like class detection had died."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "?", "conf": 0.0}},
        {"kind": "mulligan_read", "detail": {"opponent": "Druid", "conf": 0.61, "reread": True}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "opponents seen: Druid×1" in s   # the resolved class, not "?×1"
    assert "?×" not in s                      # the failed first read is not an opponent
    assert "recovered by a re-read" in s      # the story is told, framed as not-a-fault


def test_summarize_journal_flags_a_reread_that_stayed_unreadable():
    """A re-read that itself comes back '?' is the line that precedes a
    'could not read mulligan' halt -- the summary must call it out, not swallow it."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "?", "conf": 0.0}},
        {"kind": "mulligan_read", "detail": {"opponent": "?", "conf": 0.0, "reread": True}},
        {"kind": "halt", "detail": {"message": "could not read mulligan (class='', cards=0)"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "still unreadable after the re-read" in s
    assert "opponents seen" not in s          # no class was ever resolved


def test_summarize_journal_reports_think_absorbed_into_latency():
    """The latency-credit's on-device proof: think seconds absorbed into perception
    latency (credited_s on tap_think sleeps) are totalled so a slow wireless run can be
    checked for the anti-stacking actually firing."""
    import json
    events = [
        {"t": 0.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 0.75, "credited_s": 2.1}},
        {"t": 1.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 0.45, "credited_s": 1.3}},
        {"t": 2.0, "kind": "sleep", "detail": {"reason": "tap_settle", "seconds": 0.6}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "think absorbed into latency: 3.4s" in s


def test_summarize_journal_omits_absorbed_line_when_nothing_credited():
    """No credit (e.g. a fast USB run, or the credit disabled) -> the line is absent, not
    a misleading '0.0s'."""
    import json
    events = [{"t": 0.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 1.7, "credited_s": 0.0}}]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "think absorbed" not in s


def test_summarize_journal_near_miss_reports_the_gap_to_decide_threshold_vs_anchor():
    """A marginal near-miss can be fixed by lowering the threshold OR adding an anchor; the
    deciding fact is the gap to the nearest DIFFERENT screen. A wide gap means the board owns
    the score band, so lowering is safe -- the report should say so, not just 'add an anchor
    or lower threshold'."""
    import json
    # in_game 0.69 missed by 0.03; next different screen concede_menu 0.28 -> 0.41 gap
    events = [
        {"kind": "unknown_screen", "detail": {"where": "mulligan_confirm", "waiting_to_leave": "mulligan",
            "near_misses": [{"state": "in_game", "score": 0.690, "thr": 0.72},
                            {"state": "in_game", "score": 0.484, "thr": 0.72},
                            {"state": "concede_menu", "score": 0.282, "thr": 0.72}]}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "nearest DIFFERENT screen is concede_menu 0.282 (0.408 below)" in s
    assert "lowering the threshold is safe" in s


def test_summarize_journal_near_miss_warns_when_the_gap_is_narrow():
    """A close runner-up means lowering the threshold could mis-ID -> recommend an anchor."""
    import json
    events = [
        {"kind": "unknown_screen", "detail": {"where": "dispatch",
            "near_misses": [{"state": "in_game", "score": 0.700, "thr": 0.72},
                            {"state": "concede_menu", "score": 0.640, "thr": 0.72}]}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "prefer a new anchor" in s


def test_summarize_journal_flags_a_false_halt_when_the_awaited_screen_is_gone():
    """The misleading-halt case: we tapped Confirm, waited for 'mulligan' to leave, and it
    DID -- the board (in_game) is up but scores just under threshold, so it reads UNKNOWN
    and the halt message wrongly blames Confirm. The summary must flag it as a false halt
    on the destination, since 'mulligan' isn't even among the near-misses."""
    import json
    events = [
        {"kind": "unknown_screen", "detail": {"where": "mulligan_confirm", "waiting_to_leave": "mulligan",
            "near_misses": [{"state": "in_game", "score": 0.662, "thr": 0.72},
                            {"state": "in_game", "score": 0.504, "thr": 0.72},
                            {"state": "concede_menu", "score": 0.216, "thr": 0.72}]}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "likely FALSE halt" in s
    assert "'mulligan' is GONE" in s
    assert "in_game" in s     # names the destination to fix


def test_summarize_journal_does_not_flag_false_halt_when_awaited_screen_is_present():
    """If the screen we're waiting to leave IS still a top near-miss, it may genuinely be
    stuck -- do NOT cry false halt."""
    import json
    events = [
        {"kind": "unknown_screen", "detail": {"where": "mulligan_confirm", "waiting_to_leave": "mulligan",
            "near_misses": [{"state": "mulligan", "score": 0.71, "thr": 0.72},
                            {"state": "in_game", "score": 0.30, "thr": 0.72}]}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "FALSE halt" not in s


def test_summarize_journal_breaks_down_per_tap_think_vs_perception():
    """The recurring 'why is <step> so slow?' answer: for each tap, split the wall-clock to
    reach+fire it into humanized think vs perception (screencap+classify). Perception should
    dominate, so a step that 'feels like it thinks too long' is exposed as I/O-bound."""
    import json
    events = [
        # reaching the concede: a classify + captures, then a tiny credited think, then the tap
        {"kind": "classify", "detail": {"ms": 3800, "state": "concede_menu"}},
        {"kind": "sleep", "detail": {"reason": "tap_think", "seconds": 0.75, "credited_s": 2.78}},
        {"kind": "capture", "detail": {"ms": 2100}},
        {"kind": "tap", "detail": {"what": "concede"}},
        # a cheaper tap for contrast
        {"kind": "capture", "detail": {"ms": 1200}},
        {"kind": "sleep", "detail": {"reason": "tap_think", "seconds": 0.45, "credited_s": 1.4}},
        {"kind": "tap", "detail": {"what": "end_dismiss"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "per-tap reaction" in s
    # concede: 0.75s think + (3.8+2.1)=5.9s perception -> think is the small part
    assert "concede 6.7s = 0.8s think + 5.9s screencap/classify" in s


def test_summarize_journal_shows_any_when_no_target_classes():
    import json
    text = json.dumps({"kind": "run_criteria",
                       "detail": {"target_classes": [], "require_second": True, "mode": "casual"}})
    assert "targets=ANY" in br.summarize_journal(text)


def test_near_misses_survive_the_real_journal_write_into_the_report(tmp_path):
    """Regression: near_misses (a list of dicts) went through DebugLog._jsonable, which
    stringified each dict, so the journal held reprs and summarize_journal crashed on
    top.get('state') -- the exact unknown-screen halt the report most needed to describe.

    The older near-miss test hand-built the journal JSON and so never exercised the
    lossy write path; this one writes through DebugLog and reads it back.
    """
    from hop.debuglog import DebugLog

    dbg = DebugLog(tmp_path / "run", unknown_dir=tmp_path / "unknowns")
    dbg.record("unknown_screen", where="dispatch",
               near_misses=[{"state": "in_game", "score": 0.539, "thr": 0.72}])
    journal = (tmp_path / "run" / "journal.jsonl").read_text()

    summary = br.summarize_journal(journal)                 # must not raise
    # the *formatted* form only appears if the dict round-tripped (a stringified repr
    # would render as "{'state': 'in_game', ...}", not "in_game 0.539 (thr 0.72)")
    assert "in_game 0.539 (thr 0.72)" in summary


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
        "budget": {"actions_run": 5, "concedes_run": 1, "games_session": 2,
                   "committing_ratio": 0.2, "session_minutes": 0.4},
        "class_distribution": {"Mage": 2, "Rogue": 1},
    }
    out = br.format_status(s)
    assert "running: True" in out
    assert "actions: 5" in out and "concedes: 1" in out
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
