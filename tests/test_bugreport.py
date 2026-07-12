"""The self-improving bug report.

Pure assembly and the log/journal/redaction helpers are unit-tested here; the I/O
`collect`/`write_report` are exercised against a tmp filesystem.
"""

from pathlib import Path

import pytest

from hop import bugreport as br

#: The real host probe, captured before the autouse stub below replaces it, so the tests that
#: exercise the probe itself can still reach it while collect() tests stay hermetic.
_REAL_HOST_DIAGNOSTICS = br._host_diagnostics


@pytest.fixture(autouse=True)
def _stub_host_diagnostics(monkeypatch):
    """collect() shells out to pmset/ping by default (macOS power/sleep/network facts). Stub it
    so collection tests stay fast and hermetic on any host; tests that assert on the real probe
    call :data:`_REAL_HOST_DIAGNOSTICS` (or pass an explicit ``host_probe``)."""
    monkeypatch.setattr(br, "_host_diagnostics", lambda *a, **k: "")


# ── host diagnostics / last-error diagnosis (answer "was it the Mac lock screen?") ──────

class _Proc:
    def __init__(self, stdout=b"", stderr=b""):
        self.stdout = stdout
        self.stderr = stderr


def _router(rules):
    """A fake subprocess.run dispatching on the arg list. ``rules`` is a list of
    (predicate(args)->bool, _Proc); a command matching nothing raises (tool absent)."""
    def run(args, capture_output=True, timeout=None):
        for pred, proc in rules:
            if pred(args):
                return proc
        raise FileNotFoundError(" ".join(args))
    return run


def test_diagnose_last_error_names_a_screencap_timeout():
    why = br._diagnose_last_error(
        "TimeoutExpired: Command '[... exec-out screencap -p]' timed out after 20.0 seconds")
    assert why
    assert "wireless" in why.lower() and "lock screen does not cause this" in why.lower()
    assert "capture_retry_attempts" in why      # points at the new self-heal + knob


def test_diagnose_last_error_blank_for_unrecognized():
    assert br._diagnose_last_error("") == ""
    assert br._diagnose_last_error("ValueError: bad thing") == ""


def test_format_status_surfaces_the_timeout_cause_inline():
    status = {"running": False, "uptime_s": 651,
              "last_error": "TimeoutExpired: adb ... screencap -p timed out after 20.0 seconds"}
    out = br.format_status(status)
    assert "last_error:" in out
    assert "likely cause:" in out and "wireless" in out.lower()


def test_host_diagnostics_reads_power_sleep_and_reachability():
    rules = [
        (lambda a: a[:3] == ["pmset", "-g", "batt"],
         _Proc(b"Now drawing from 'Battery Power'\n -InternalBattery-0 (id=...) 88%; discharging\n")),
        (lambda a: a[:3] == ["pmset", "-g", "assertions"],
         _Proc(b"Assertion status system-wide:\n   PreventUserIdleSystemSleep    0\n"
               b"   PreventUserIdleDisplaySleep   0\n   PreventSystemSleep            0\n")),
        (lambda a: a[:3] == ["pmset", "-g", "log"],
         _Proc(b"2026-07-10 15:18:39 -0600 Sleep               Entering Sleep state\n"
               b"2026-07-10 15:22:10 -0600 Wake                Wake from Normal Sleep\n")),
        (lambda a: a[:2] == ["pmset", "-g"] and len(a) == 2,
         _Proc(b" System-wide power settings:\n sleep                1\n"
               b" displaysleep         2\n disksleep            10\n")),
        (lambda a: a and a[0] == "ping",
         _Proc(b"1 packets transmitted, 1 packets received, 0.0% packet loss\n")),
        (lambda a: "get-state" in a, _Proc(stderr=b"error: device offline\n")),
    ]
    out = _REAL_HOST_DIAGNOSTICS("192.168.99.139:5555",
                                 run=_router(rules), system=lambda: "Darwin")
    assert "Battery Power" in out                       # AC-vs-battery is load-bearing
    assert "PreventUserIdleSystemSleep=0" in out        # nothing was blocking idle sleep
    assert "sleep=1m" in out and "displaysleep=2m" in out
    assert "Entering Sleep state" in out                # the smoking-gun sleep/wake pair
    assert "ICMP ping: reachable" in out                # ping parsed (not a substring of "unreachable")
    assert "device offline" in out                      # adb get-state stderr folded in


def test_host_diagnostics_flags_a_held_caffeinate_and_unreachable_phone():
    """The two branches that directly answer the user's question ('caffeinate was on' / is the
    phone up): a HELD caffeinate assertion, and a phone that does NOT answer ping."""
    rules = [
        (lambda a: a[:3] == ["pmset", "-g", "assertions"],
         _Proc(b"   PreventUserIdleSystemSleep    1\n"
               b"   pid 742(caffeinate): PreventUserIdleSystemSleep named: 'caffeinate'\n")),
        (lambda a: a and a[0] == "ping",
         _Proc(b"2 packets transmitted, 0 packets received, 100.0% packet loss\n")),
        (lambda a: "get-state" in a, _Proc(stderr=b"error: device offline\n")),
    ]
    out = _REAL_HOST_DIAGNOSTICS("192.168.99.139:5555",
                                 run=_router(rules), system=lambda: "Darwin")
    assert "PreventUserIdleSystemSleep=1" in out                     # idle system sleep WAS blocked
    assert "assertion is held by a caffeinate process" in out        # caffeinate-present branch
    assert "NO reply (unreachable right now)" in out                 # unreachable branch


def test_diagnose_last_error_names_a_non_screencap_adb_timeout():
    """The second branch: an adb command that timed out but was NOT the screencap."""
    why = br._diagnose_last_error("TimeoutExpired: adb -s 1.2.3.4 shell dumpsys timed out after 20s")
    assert why and "adb command timed out" in why.lower()
    assert "screencap over wireless" not in why.lower()             # not the first branch


def test_host_diagnostics_is_empty_off_macos():
    assert _REAL_HOST_DIAGNOSTICS("1.2.3.4:5555", run=_router([]), system=lambda: "Linux") == ""


def test_host_diagnostics_survives_missing_tools():
    # every command raises (tools absent) -> no lines, never an exception
    out = _REAL_HOST_DIAGNOSTICS("1.2.3.4:5555", run=_router([]), system=lambda: "Darwin")
    assert out == ""


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


def test_could_not_read_mulligan_halt_is_self_diagnosed():
    """The 'class=\'\' cards=3' halt (the one this feature was built for) must name its own
    cause -- class blank but cards fine -> point at the `region_gray` datum that decides blank
    vs mis-aligned OCR, and at the recover-not-halt streak knob -- so the next such report is
    actionable without decoding the raw message. (The still-choosing banner is NOT the cause:
    a real still-choosing frame OCRs the class fine, so the diagnosis must not claim it is.)"""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "?", "cards": 3, "conf": 0.0}},
        {"kind": "mulligan_read", "detail": {"reread": True, "attempt": 1,
                                             "opponent": "?", "cards": 3, "class_raw": ""}},
        {"kind": "halt", "detail": {"message": "HALTED: could not read mulligan (class='', cards=3)"}},
    ]
    summary = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "likely cause:" in summary
    assert "BLANK" in summary and "region_gray" in summary
    assert "mulligan_unreadable_halt_streak" in summary   # points at the recover-not-halt knob


def test_mulligan_read_halt_diagnosis_distinguishes_the_three_failure_modes():
    d = br._diagnose_mulligan_read_halt
    assert "BLANK" in d("could not read mulligan (class='', cards=3)")          # class missing
    assert "GARBLED" in d("could not read mulligan (class='WARRIQR', cards=4)")  # class present, unresolved
    assert "CARD count" in d("could not read mulligan (class='MAGE', cards=1)")  # card miscount
    assert d("some other halt entirely") == ""                                   # not our halt


def test_mulligan_garbled_diagnosis_names_the_likely_class():
    """A garbled-class halt should name the LIKELY class (over full labels + two-word
    fragments), so the live 'G DEATH' halt reads 'closest to Death Knight' -- not the
    misleading nearest full label (Druid). This is what makes such a halt self-identifying."""
    out = br._diagnose_mulligan_read_halt(
        "could not read mulligan (class='G DEATH', cards=4)")
    assert "GARBLED" in out
    assert "Death Knight" in out          # named the real class via the "DEATH" fragment
    assert "Druid" not in out             # NOT the misleading nearest full label


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


def test_summarize_journal_attributes_retry_stalls_to_stalled_io_not_ocr():
    """A screencap that HUNG (capture_retry) is transport time, not OCR/logic. It must appear as
    a distinct 'stalled I/O' term, so a 20s stall isn't misread as slow classification/logic."""
    import json
    events = [
        {"t": 0.0, "kind": "sleep", "detail": {"reason": "tap_think", "seconds": 2.0}},
        {"t": 2.0, "kind": "capture", "detail": {"ms": 1400}},
        {"t": 3.4, "kind": "capture_retry", "detail": {"attempt": 1, "of": 3, "ms": 20000,
                                                        "error": "adb ... screencap timed out"}},
        {"t": 23.4, "kind": "capture", "detail": {"ms": 1500}},
        {"t": 24.9, "kind": "tap", "detail": {"what": "play"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    # wall 25s = 2s humanized + 3s screencap + 20s stalled I/O + 0s classify/OCR/logic
    assert "20s stalled I/O" in s
    assert "0s classify/OCR/logic" in s          # the stall is NOT dumped into the residual


def test_summarize_journal_surfaces_capture_retries():
    """The new capture-retry resilience: a transient stall the hunt rode through must be
    visible (it means the link wobbled), with the hung time and the last error."""
    import json
    events = [
        {"t": 0.0, "kind": "capture", "detail": {"ms": 1400}},
        {"t": 1.4, "kind": "capture_retry",
         "detail": {"attempt": 1, "of": 3, "ms": 20000, "error": "adb ... timed out after 20s"}},
        {"t": 21.4, "kind": "capture", "detail": {"ms": 1500}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "capture retries: 1 screencap(s) stalled/failed" in s
    assert "20.0s hung total" in s
    assert "timed out after 20s" in s


def test_summarize_journal_surfaces_non_repetition_rejects_and_collapse():
    """The live non-repetition halt needed the rejected-draw count and memory size. A raw
    terminal halt says only '8 attempts'; the journal now shows whether the gate saturated."""
    import json
    events = [
        {"kind": "non_repetition_reject", "detail": {"attempt": i, "of": 8, "memory": 32}}
        for i in range(1, 9)
    ] + [
        {"kind": "non_repetition_collapse", "detail": {"attempts": 8, "memory": 32}},
        {"kind": "halt", "detail": {"message": "HALTED: could not draw a non-repeating "
                                    "gesture in 8 attempts; the motor generator has collapsed"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "non-repetition gate: 8 rejected draw(s)" in s
    assert "last rejected attempt 8/8" in s
    assert "memory=32" in s


def test_summarize_journal_flags_a_degrading_capture_trend():
    """A one-number mean hides a link that started fast and ended crawling. When the last
    third's mean is much higher than the first third's, the summary must flag the TREND -
    the precursor that would have foretold a stall/timeout."""
    import json
    fast = [{"t": float(i), "kind": "capture", "detail": {"ms": 1000}} for i in range(6)]
    slow = [{"t": float(6 + i), "kind": "capture", "detail": {"ms": 5000}} for i in range(6)]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in fast + slow))
    assert "TREND" in s and "DEGRADING" in s


def test_summarize_journal_does_not_flag_a_steady_capture_stream():
    """Ordinary jitter must stay quiet: no TREND flag when captures are steady."""
    import json
    steady = [{"t": float(i), "kind": "capture", "detail": {"ms": 1400}} for i in range(12)]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in steady))
    assert "TREND" not in s


def test_summarize_journal_trend_needs_a_large_ABSOLUTE_rise():
    """The compound guard is `late > early*1.4 AND late - early > 800`. A big RATIO but a small
    absolute rise (100->300ms) must NOT flag -- else fast phones trip on sub-ms jitter. Pins the
    absolute conjunct independently (deleting it would turn this green->flag)."""
    import json
    evs = ([{"t": float(i), "kind": "capture", "detail": {"ms": 100}} for i in range(6)]
           + [{"t": float(6 + i), "kind": "capture", "detail": {"ms": 300}} for i in range(6)])
    s = br.summarize_journal("\n".join(json.dumps(e) for e in evs))
    assert "TREND" not in s                          # ratio 3.0 but rise only 200ms (< 800)


def test_summarize_journal_trend_needs_a_large_RATIO():
    """A big absolute rise but a shallow ratio (3000->4000ms) must NOT flag -- an already-slow
    link creeping is not the DEGRADING signature. Pins the ratio conjunct independently."""
    import json
    evs = ([{"t": float(i), "kind": "capture", "detail": {"ms": 3000}} for i in range(6)]
           + [{"t": float(6 + i), "kind": "capture", "detail": {"ms": 4000}} for i in range(6)])
    s = br.summarize_journal("\n".join(json.dumps(e) for e in evs))
    assert "TREND" not in s                          # rise 1000ms (> 800) but ratio 1.33 (< 1.4)


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


def test_summarize_journal_reads_out_unreadable_mulligan_region_stats():
    """The `mulligan_unreadable` events carry the class-region pixel stats -- the datum a
    'could not read mulligan' report never had. The summary must read them out (recovered vs
    halted, and the last region_gray with a blank-vs-mis-aligned interpretation) so the cause
    is IN the report, not left to saved-frame archaeology."""
    import json
    events = [
        {"kind": "mulligan_unreadable", "detail": {"class_raw": "", "cards": 4, "streak": 1,
            "cap": 4, "recovering": True, "region_gray": {"mean": 12.0, "min": 12, "max": 12, "std": 0.0}}},
        {"kind": "mulligan_unreadable", "detail": {"class_raw": "", "cards": 4, "streak": 2,
            "cap": 4, "recovering": True, "region_gray": {"mean": 44.0, "min": 0, "max": 255, "std": 61.8}}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "class NEVER read" in s
    assert "2 recovered" in s                 # both conceded + requeued, hunt kept running
    assert "std=61.8" in s                     # the last one's region stats are read out
    assert "high std" in s                     # ...with the "text was present -> OCR/region" reading


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


def test_summarize_journal_flags_a_verified_then_stuck_halt():
    """The false halt the near-miss check CAN'T see: the destination is a genuinely
    un-anchored screen (all near-misses far away), so the marginal-miss heuristic stays
    silent. But the Confirm tap's OWN verify_ok fired, then only UNKNOWN frames followed --
    proof the tap worked and only the destination is unnamed. The real 'Opponent Still
    Choosing...' halt, which had NO near-miss line to catch it."""
    import json
    events = [
        {"kind": "tap", "detail": {"what": "mulligan_confirm"}},
        {"kind": "verify_ok", "detail": {"change_kind": "full_transition"}},
        {"kind": "classify", "detail": {"ms": 6000, "state": "unknown"}},
        {"kind": "classify", "detail": {"ms": 6000, "state": "unknown"}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "FALSE halt (verified-then-stuck)" in s
    assert "mulligan_confirm" in s and "verify_ok" in s


def test_summarize_journal_does_not_cry_false_halt_without_a_verify():
    """A tap that did NOT verify (a genuine missed tap / stuck source) has no verify_ok, so
    it must not be mislabelled a false halt -- the tap did not provably work."""
    import json
    events = [
        {"kind": "tap", "detail": {"what": "mulligan_confirm"}},
        {"kind": "classify", "detail": {"ms": 6000, "state": "unknown"}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "verified-then-stuck" not in s


def test_summarize_journal_does_not_cry_false_halt_on_a_reconnect_timeout():
    """Regression: the 'verified-then-stuck' detector must NOT fire on genuine external
    failures. Reconnect taps Reconnect (verify_ok), its wait tolerates UNKNOWN mid-reconnect
    redraws, then halts 'Hearthstone never came back online' -- a network outage, not an
    unnamed destination. The 'capture the anchor' advice would be actively wrong here."""
    import json
    events = [
        {"kind": "tap", "detail": {"what": "reconnect"}},
        {"kind": "verify_ok", "detail": {"change_kind": "full_transition"}},
        {"kind": "classify", "detail": {"ms": 6000, "state": "unknown"}},
        {"kind": "halt", "detail": {"message": "stuck on 'Reconnecting...'; Hearthstone never came back online"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "verified-then-stuck" not in s


def test_summarize_journal_false_halt_ignores_a_named_destination_after_the_verify():
    """If a NAMED frame follows the verify (we DID recognise where we landed -- e.g. genuinely
    stuck back on the mulligan), a trailing transient UNKNOWN must not flip it to a false halt.
    The latch means 'the last look since the verify was UNKNOWN', not 'any UNKNOWN ever'."""
    import json
    events = [
        {"kind": "tap", "detail": {"what": "mulligan_confirm"}},
        {"kind": "verify_ok", "detail": {"change_kind": "full_transition"}},
        {"kind": "classify", "detail": {"ms": 6000, "state": "unknown"}},   # transient
        {"kind": "classify", "detail": {"ms": 3000, "state": "mulligan"}},  # NAMED, genuinely stuck
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "verified-then-stuck" not in s


def test_summarize_journal_flags_a_genuine_stuck_dropped_commit():
    """The COMPLEMENT of verified-then-stuck: the leave-wait timed out with the SOURCE screen
    still positively NAMED (never UNKNOWN) -- a DROPPED committing tap, not an unnamed
    destination. This is the concede-drop this whole diagnosis was added for: the tap's own
    verify passed on a WEAKER change (partial < the full_transition it demanded) because the
    board animated behind the semi-transparent Game Menu, masking the drop."""
    import json
    events = (
        [{"kind": "tap", "detail": {"what": "concede", "expected": "full_transition"}},
         {"kind": "verify_ok", "detail": {"change_kind": "partial"}}]
        + [e for _ in range(6) for e in (
            {"kind": "classify", "detail": {"ms": 30, "state": "concede_menu"}},
            {"kind": "wait_until_poll", "detail": {"what": "concede", "state": "concede_menu"}})]
        + [{"kind": "classify", "detail": {"ms": 30, "state": "concede_menu"}},
           {"kind": "wait_timeout", "detail": {"what": "concede", "state": "concede_menu"}},
           {"kind": "halt", "detail": {"message": "Concede did not dismiss the Game Menu after 3 tap(s)"}}]
    )
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "GENUINE stuck (DROPPED committing tap)" in s
    assert "still positively 'concede_menu' for 7 look(s)" in s   # 6 polls + the timeout look
    assert "'partial', WEAKER than the 'full_transition'" in s    # the masked-drop tell
    assert "FALSE halt (verified-then-stuck)" not in s            # mutually exclusive


def test_summarize_journal_genuine_stuck_yields_to_the_unknown_destination_false_halt():
    """When the SAME stuck halt is preceded by a trailing UNKNOWN (the destination is unnamed),
    it is the verified-then-stuck FALSE halt, NOT a genuine dropped commit. The two are mutually
    exclusive: a genuine stuck needs a positively-NAMED source at timeout, never UNKNOWN."""
    import json
    events = [
        {"kind": "tap", "detail": {"what": "mulligan_confirm", "expected": "full_transition"}},
        {"kind": "verify_ok", "detail": {"change_kind": "full_transition"}},
        {"kind": "classify", "detail": {"ms": 6000, "state": "unknown"}},
        {"kind": "wait_timeout", "detail": {"what": "mulligan_confirm", "state": "unknown"}},
        {"kind": "halt", "detail": {"message": "mulligan Confirm did not dismiss the mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "FALSE halt (verified-then-stuck)" in s
    assert "GENUINE stuck (DROPPED committing tap)" not in s


def test_summarize_journal_tallies_dropped_taps():
    """A cluster of silently-ignored-then-re-sent taps (mulligan cards, and now concede) is the
    wireless link dropping INPUT -- surfaced as one count so it corroborates a dropped-commit."""
    import json
    events = [
        {"kind": "mulligan_card_tap_ignored", "detail": {"slot": 1, "attempt": 1, "of": 3}},
        {"kind": "mulligan_card_tap_ignored", "detail": {"slot": 2, "attempt": 1, "of": 3}},
        {"kind": "concede_tap_ignored", "detail": {"attempt": 1, "of": 3}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "dropped taps re-sent: 3" in s


def test_summarize_journal_surfaces_the_coin_and_concede_timing():
    """Answers 'does it only concede on MY turn / wait till turn 2 going second?': the coin
    split, where the reject grammar chose to bail (random, not coin-driven), and how many
    End Turn beats no-op'd because the button was greyed (not our turn). The concede itself
    is via the gear menu and fires on EITHER turn -- the report must make that legible."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "Mage", "second": True}},
        {"kind": "reject_plan", "detail": {"concede_point": "turn2"}},
        {"kind": "tap", "detail": {"what": "pass_turn"}},
        {"kind": "anomaly", "detail": {"reason": "no screen change", "fault": "no_change"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Rogue", "second": False}},
        {"kind": "reject_plan", "detail": {"concede_point": "mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "1 went 2nd / 1 went 1st" in s
    assert "turn2×1" in s and "mulligan×1" in s
    assert "End Turn beats 1, of which 1 no-op" in s
    assert "NOT turn-gated" in s


def test_summarize_journal_coin_ignores_a_reread_so_the_split_is_once_per_game():
    """The coin is tallied on the FIRST read of a game only; a re-read (perception retry)
    must not double-count it."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "?", "second": True}},
        {"kind": "mulligan_read", "detail": {"opponent": "Mage", "second": True, "reread": True}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "1 went 2nd / 0 went 1st" in s   # one game, not two


def test_summarize_journal_opponents_seen_counts_a_whiff_then_recover_game_once():
    """Regression: a first read that OCRs the class fine (Mage) but MISCOUNTS the cards (0 ->
    unusable) is journalled, then a re-read recovers (Mage, cards=3). The class must be counted
    ONCE -- mirroring the engine's own class_distribution, which counts a game once -- not twice
    ('Mage×2') off the unusable first read plus the recovering re-read."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "Mage", "second": False, "cards": 0}},
        {"kind": "mulligan_read", "detail": {"reread": True, "opponent": "Mage",
                                             "second": False, "cards": 3}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "opponents seen: Mage×1" in s
    assert "Mage×2" not in s


def test_summarize_journal_coin_uses_the_first_valid_card_read_not_a_whiffed_one():
    """Regression: the coin is derived from the card count, so a first read whose cards whiffed
    to 0 records second=False regardless of the true coin. A game that actually went 2nd (its
    re-read counts 4 cards) must be tallied '2nd', not inverted to '1st' off the unusable first
    read -- the same card-validity gate the engine's own going_second tally uses."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "?", "second": False, "cards": 0}},
        {"kind": "mulligan_read", "detail": {"reread": True, "opponent": "Mage",
                                             "second": True, "cards": 4}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "1 went 2nd / 0 went 1st" in s   # the RESOLVED coin, not the whiffed first read's


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


def test_collect_includes_the_host_power_section(tmp_path):
    """The host power/sleep/network section is added from the config's adb_address with no
    extra call-site wiring, so both the CLI and the app path get it."""
    (tmp_path / "config.toml").write_text('[device]\nadb_address = "10.0.0.5:5555"\n')
    seen = {}

    def probe(addr):
        seen["addr"] = addr
        return "- power source: Now drawing from 'Battery Power'\n- sleep assertions: none held"

    md = br.collect("died on a screencap timeout", _paths(tmp_path), version="2.0.0",
                    clock=lambda: 0.0, host_probe=probe)
    assert "Host power / sleep / network" in md
    assert "Battery Power" in md
    assert seen["addr"] == "10.0.0.5:5555"                 # address parsed out of the config


def test_summarize_journal_names_the_state_before_an_unknown_halt():
    """A halt on an unknown that first appeared right after QUEUE is the match-start
    transition (VS splash / board fade-in). The summary must say so and point at the
    vs_splash anchor -- the near-miss line alone ('deck_select 0.302') doesn't locate it."""
    journal = "\n".join([
        '{"kind":"classify","detail":{"ms":40,"state":"queue"}}',
        '{"kind":"classify","detail":{"ms":40,"state":"queue"}}',
        '{"kind":"classify","detail":{"ms":80,"state":"unknown"}}',
        '{"kind":"classify","detail":{"ms":80,"state":"unknown"}}',
        '{"kind":"classify","detail":{"ms":80,"state":"unknown"}}',
        '{"kind":"unknown_screen","detail":{"where":"dispatch","near_misses":['
        '{"state":"deck_select","score":0.302,"thr":0.72}]}}',
        '{"kind":"halt","detail":{"message":"HALTED: unknown screen (best confidence 0.00)"}}',
    ])
    out = br.summarize_journal(journal)
    assert "unknown-halt context:" in out
    assert "after queue" in out and "persisted 3 look(s)" in out
    assert "vs_splash" in out and "match found" in out.lower()


def test_summarize_journal_unknown_context_names_a_non_queue_predecessor():
    """The context is general: an unknown halt after some other screen names THAT screen
    (and withholds the queue-specific vs_splash hint, which wouldn't apply)."""
    journal = "\n".join([
        '{"kind":"classify","detail":{"ms":40,"state":"in_game"}}',
        '{"kind":"classify","detail":{"ms":80,"state":"unknown"}}',
        '{"kind":"halt","detail":{"message":"unknown screen (best confidence 0.00)"}}',
    ])
    out = br.summarize_journal(journal)
    assert "unknown-halt context:" in out and "after in_game" in out
    assert "vs_splash" not in out


def test_perception_env_flags_the_active_numpy_fast_path():
    """With numpy present the report says the vectorized NCC path is live -- the fact that
    tells a 'slow cycle' reader the classify cost is the FFT pass, not a missing dependency."""
    lines = br.perception_env(probe=lambda mod: {"numpy": "2.4.4", "PIL": "12.2.0"}.get(mod))
    body = "\n".join(lines)
    assert "numpy: 2.4.4" in body and "vectorized NCC active" in body
    assert "Pillow: 12.2.0" in body


def test_perception_env_calls_out_a_missing_numpy_as_the_slow_path():
    """Without numpy, classify is the ~80x-slower pure-Python loop; the report must name
    that as the likely cause and point at the fix rather than leave it to be inferred."""
    lines = br.perception_env(probe=lambda mod: None)
    body = "\n".join(lines)
    assert "numpy: NOT INSTALLED" in body
    assert "pure-Python" in body and "pip install numpy" in body


def test_tooling_summary_includes_the_perception_accelerator():
    out = br.tooling_summary(which=lambda t: f"/usr/bin/{t}", env={"PATH": "/usr/bin"},
                             perception=["- numpy: 2.4.4  (vectorized NCC active)"])
    assert "adb on PATH: /usr/bin/adb" in out
    assert "numpy: 2.4.4" in out          # the accelerator sits alongside the PATH tools


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


def test_collect_ocrs_unknown_frames_and_shows_the_banner(tmp_path):
    """The banner text that NAMES an unrecognised screen is put in front of the diagnosis,
    so 'Opponent Still Choosing...' is legible without opening the PNG."""
    (tmp_path / "unknowns").mkdir()
    (tmp_path / "unknowns" / "unknown_x_mulligan_confirm.png").write_bytes(b"x")
    md = br.collect(
        "x", _paths(tmp_path), version="1", clock=lambda: 0.0,
        ocr_probe=lambda d, names: {"unknown_x_mulligan_confirm.png": "Opponent Still Choosing..."})
    assert "unknown_x_mulligan_confirm.png" in md
    assert 'reads: "Opponent Still Choosing..."' in md


def test_a_failing_ocr_probe_never_breaks_the_report(tmp_path):
    """OCR is diagnostic sugar; a broken engine drops the text, never the report or the
    filename list."""
    (tmp_path / "unknowns").mkdir()
    (tmp_path / "unknowns" / "unknown_x_dispatch.png").write_bytes(b"x")
    def boom(d, names):
        raise RuntimeError("ocr down")
    md = br.collect("x", _paths(tmp_path), version="1", clock=lambda: 0.0, ocr_probe=boom)
    assert "unknown_x_dispatch.png" in md   # still listed, just without the banner text


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


def test_format_status_names_a_target_found_stop_as_success_not_a_crash():
    """The most common successful exit -- hop found a targeted class at the mulligan and STOPPED
    touching the game so you can play it -- records no stop_reason on older builds, so 'running:
    False' with no reason reads as a crash. format_status must name it as the WIN condition, robust
    to an OLD build (stop_reason='') AND a NEW one (stop_reason='target_found')."""
    for stop_reason in ("", "target_found"):
        out = br.format_status({
            "running": False, "uptime_s": 903.2, "stop_reason": stop_reason, "last_error": "",
            "games": 0, "concedes": 0, "target_found": True, "last_opponent": "Priest",
        })
        assert "outcome: SUCCESS" in out
        assert "NOT a crash" in out
        assert "Priest" in out


def test_format_status_target_found_success_line_stays_silent_otherwise():
    """The success line must fire ONLY for a stopped, target-found run with no error -- never on a
    running hunt, a genuine crash, a deliberate halt, or a crash that happened to have found a
    target earlier (last_error must win)."""
    running = br.format_status({"running": True, "uptime_s": 5, "target_found": True})
    assert "outcome: SUCCESS" not in running          # still hunting, not a stop
    crash = br.format_status({"running": False, "uptime_s": 5, "target_found": False,
                              "last_error": "TimeoutExpired: screencap timed out after 20s"})
    assert "outcome: SUCCESS" not in crash
    halt = br.format_status({"running": False, "uptime_s": 5, "target_found": False,
                             "stop_reason": "halt:deck list; re-select your deck"})
    assert "outcome: SUCCESS" not in halt
    crash_after_found = br.format_status({"running": False, "uptime_s": 5, "target_found": True,
                                          "last_error": "TimeoutExpired: screencap timed out"})
    assert "outcome: SUCCESS" not in crash_after_found   # a real crash is never blessed


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


# ── 50k-CHARACTER cap (what actually truncates a pasted report) ─────────────────

def test_fit_report_budget_trims_oldest_log_lines_first():
    big = "\n".join(f"log-line-number-{i:04d}-padding-padding" for i in range(400))
    secs = [br.Section("Live engine status", "- running: False"),
            br.Section("App log (~/Library/Logs/hop.log)", big, fenced=True, lang="text")]
    out = br._fit_report_budget("desc", secs, meta={"version": "1"}, max_chars=1500)
    assert len(out) <= 1500                   # the CHARACTER budget is honoured
    assert "0399" in out                      # newest log line survives
    assert "log-line-number-0000" not in out  # oldest log line is dropped
    assert "- running: False" in out          # non-log sections are untouched
    assert "oldest lines dropped" in out


def test_fit_report_budget_is_a_noop_under_the_cap():
    secs = [br.Section("App log (~/Library/Logs/hop.log)", "a\nb\nc", fenced=True, lang="text")]
    out = br._fit_report_budget("d", secs, meta={"version": "1"}, max_chars=10_000)
    assert "oldest lines dropped" not in out and "a\nb\nc" in out


def test_fit_report_budget_hard_truncates_when_fixed_sections_alone_overflow():
    """The last-resort backstop: when even the UN-trimmable sections exceed the cap (a huge
    live-status block, say), trimming the log/journal cannot get under budget, so the report
    is hard-cut. This guarantees the pasted report is NEVER silently truncated mid-line by the
    external 50k limit -- and the cut eats the END of the document (emitted after the summary
    and journal), so the diagnosis survives. Nothing else exercises this path."""
    big_status = "\n".join(f"status-line-{i:04d}" for i in range(500))   # non-trimmable
    secs = [br.Section("Live engine status", big_status)]
    out = br._fit_report_budget("desc", secs, meta={"version": "1"}, max_chars=1_200)
    assert len(out) <= 1_200                       # the cap is honoured even here
    assert "hard-truncated" in out                 # the backstop fired and said so
    assert out.startswith(br.SELF_IMPROVE_PROMPT)  # the fixed head (the diagnosis) survives


def test_collect_never_exceeds_the_char_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "MAX_REPORT_CHARS", 4_000)
    # a journal whose tail alone is far over budget: the newest lines must survive, oldest go
    (tmp_path / "hop.log").write_text("\n".join(f"applog{i}" for i in range(50)) + "\n")
    run = tmp_path / "runs" / "20260710-020000"
    run.mkdir(parents=True)
    run.joinpath("journal.jsonl").write_text(
        "\n".join('{"kind":"classify","detail":{"ms":40,"state":"queue","i":%d}}' % i
                  for i in range(400)) + "\n")
    md = br.collect("halp", _paths(tmp_path), version="2.0.0", clock=lambda: 0.0,
                    journal_tail_lines=400, log_tail_lines=50)
    assert len(md) <= 4_000                    # never exceeds the character cap
    assert md.startswith(br.SELF_IMPROVE_PROMPT)   # the fixed head always survives
    assert '"i":399' in md                     # the most recent journal line is kept
    assert "oldest lines dropped" in md


def test_summarize_journal_confirms_a_target_found_matches_empty_class_criteria():
    """The 'wrong target found, unless bug?' answer: with no class filter set, ANY class is a
    target, so a stop on Paladin going 2nd is CORRECT. The summary must say so, not leave it
    to be re-derived."""
    journal = "\n".join([
        '{"kind":"run_criteria","detail":{"target_classes":[],"avoid_classes":[],"require_second":true,"mode":"casual"}}',
        '{"kind":"target_found","detail":{"opponent":"Paladin","second":true}}',
    ])
    out = br.summarize_journal(journal)
    assert "target found: Paladin (going 2nd)" in out
    assert "NOT a misfire" in out
    assert "no class filter is set" in out and "going-2nd requirement was met" in out


def test_summarize_journal_flags_a_target_found_that_violates_the_criteria():
    """A genuine misfire (stopped on a class NOT in the target list) must be flagged loudly
    as a real bug, not blessed as correct."""
    journal = "\n".join([
        '{"kind":"run_criteria","detail":{"target_classes":["MAGE"],"avoid_classes":[],"require_second":false,"mode":"casual"}}',
        '{"kind":"target_found","detail":{"opponent":"Warrior","second":false}}',
    ])
    out = br.summarize_journal(journal)
    assert "real MISFIRE" in out and "Warrior is NOT in the target list" in out


def test_summarize_journal_shows_the_state_trail_into_a_halt():
    """The single line that would have surfaced THIS report's cause without journal
    archaeology: a dismissed error dialog dropped hop to the deck list."""
    journal = "\n".join([
        '{"kind":"classify","detail":{"ms":40,"state":"play_screen"}}',
        '{"kind":"classify","detail":{"ms":40,"state":"queue"}}',
        '{"kind":"classify","detail":{"ms":40,"state":"queue"}}',
        '{"kind":"classify","detail":{"ms":88,"state":"error_dialog"}}',
        '{"kind":"classify","detail":{"ms":53,"state":"deck_select"}}',
        '{"kind":"halt","detail":{"message":"HALTED: dropped back to the deck list; re-select your deck"}}',
    ])
    out = br.summarize_journal(journal)
    assert "state trail into the halt:" in out
    # consecutive repeats collapse, so the trigger reads cleanly
    assert "queue -> error_dialog -> deck_select" in out


# ── why were the games conceded? (reject-reason recompute) ─────────────────────

def test_summarize_journal_explains_why_conceded_games_were_rejected():
    """THIS report's missing line: with targets=ANY + require_second, two games that went 1st
    were conceded. The summary must recompute and state that -- 'all correct: went 1st' -- so a
    reader isn't left hand-crossing the coin split against require_second to tell a correct
    concede from a class-read/logic bug."""
    import json
    events = [
        {"kind": "run_criteria", "detail": {"target_classes": [], "avoid_classes": [],
                                            "require_second": True, "mode": "casual"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Druid", "second": False}},
        {"kind": "reject_plan", "detail": {"concede_point": "turn1"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Death Knight", "second": False}},
        {"kind": "reject_plan", "detail": {"concede_point": "mulligan"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "rejects vs criteria: 2 conceded, all correct" in s
    assert "2× went 1st" in s and "require_second is set" in s
    assert "MISFIRE" not in s


def test_summarize_journal_reject_reason_names_a_class_not_targeted():
    """A class filter that rejects: the opponent class simply isn't on the target list."""
    import json
    events = [
        {"kind": "run_criteria", "detail": {"target_classes": ["MAGE"], "avoid_classes": [],
                                            "require_second": False, "mode": "casual"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Warrior", "second": True}},
        {"kind": "reject_plan", "detail": {"concede_point": "turn2"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "rejects vs criteria: 1 conceded, all correct" in s
    assert "opponent class not in the target list" in s


def test_summarize_journal_flags_a_reject_the_criteria_say_to_keep_as_a_misfire():
    """The mirror of the target_found misfire check: a game the criteria would KEEP (Mage is in
    the target list, coin not required) that was conceded anyway is a real bug -- flag it loudly,
    naming the matchup, instead of blessing it as a correct reject."""
    import json
    events = [
        {"kind": "run_criteria", "detail": {"target_classes": ["MAGE"], "avoid_classes": [],
                                            "require_second": False, "mode": "casual"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Mage", "second": True}},
        {"kind": "reject_plan", "detail": {"concede_point": "turn1"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "real MISFIRE" in s
    assert "Mage (going 2nd)" in s
    assert "evaluate_matchup/accepts" in s


def test_summarize_journal_reject_reason_matches_enum_name_to_display_name():
    """run_criteria records enum tokens ("DEATHKNIGHT", no underscore); mulligan_read records the
    display name ("Death Knight"). The recompute must normalize both so a Death Knight target is
    KEPT (=> conceding it is a misfire), exactly as the target_found recompute matches names."""
    import json
    events = [
        {"kind": "run_criteria", "detail": {"target_classes": ["DEATHKNIGHT"], "avoid_classes": [],
                                            "require_second": False, "mode": "casual"}},
        {"kind": "mulligan_read", "detail": {"opponent": "Death Knight", "second": True}},
        {"kind": "reject_plan", "detail": {"concede_point": "turn1"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "real MISFIRE" in s and "Death Knight" in s


def test_summarize_journal_reject_reason_needs_run_criteria_to_avoid_false_misfires():
    """Without a run_criteria line the criteria default to 'accept everything', under which every
    reject would spuriously read as a misfire. Gate on it: no criteria => no reject line at all."""
    import json
    events = [
        {"kind": "mulligan_read", "detail": {"opponent": "Mage", "second": False}},
        {"kind": "reject_plan", "detail": {"concede_point": "turn1"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "rejects vs criteria" not in s
    assert "MISFIRE" not in s


def test_reject_reason_mirrors_accepts_precedence():
    """Unit-check the recompute against Criteria.accepts: require_second is checked FIRST and is
    class-independent; then the target list; then the avoid list; a keep that was rejected is a
    misfire."""
    r = br._reject_reason
    assert r("Druid", False, True, [], []) == "went_first"          # coin required, went 1st
    assert r("Druid", False, True, ["MAGE"], []) == "went_first"    # ...checked before the class
    assert r("Druid", True, True, [], []) == "misfire"              # coin ok, no filter -> keep
    assert r("Warrior", True, False, ["MAGE"], []) == "not_targeted"
    assert r("Mage", True, False, ["MAGE"], []) == "misfire"        # in targets -> keep
    assert r("Rogue", False, False, [], ["ROGUE"]) == "avoided"
    assert r("Mage", False, False, [], ["ROGUE"]) == "misfire"      # not avoided -> keep
    assert r("Death Knight", True, False, ["DEATHKNIGHT"], []) == "misfire"  # name-normalized keep


# ── is the halt a deliberate fail-closed stop, or a bug? ───────────────────────

def test_summarize_journal_labels_the_deck_select_halt_as_a_deliberate_stop():
    """THIS report's halt: hop dropped to the deck list and fail-closed after polling it. The
    summary must certify it as a DELIBERATE stop (not a malfunction) with the action to take, and
    show the bounded-poll shape (looked at deck_select N times) that distinguishes a guard from a
    crash."""
    import json
    events = [
        {"kind": "classify", "detail": {"ms": 50, "state": "deck_select"}},
        {"kind": "classify", "detail": {"ms": 50, "state": "deck_select"}},
        {"kind": "classify", "detail": {"ms": 50, "state": "deck_select"}},
        {"kind": "halt", "detail": {"message": "HALTED: dropped back to the deck list and nobody "
                                    "re-selected a deck; hop won't reopen one. Re-select your deck "
                                    "and restart the hunt from its Play screen."}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "DELIBERATE fail-closed stop, not a malfunction" in s
    assert "looked at 'deck_select' 3× (a bounded wait)" in s
    assert "Re-select your deck" in s


def test_summarize_journal_does_not_label_an_unknown_screen_halt_as_deliberate():
    """A genuinely-unexpected halt (an unrecognized screen) must NOT be blessed as a deliberate
    stop -- that would hide a real fault. It falls through to the unknown-halt diagnostics."""
    import json
    events = [
        {"kind": "classify", "detail": {"ms": 80, "state": "unknown"}},
        {"kind": "halt", "detail": {"message": "unknown screen (best confidence 0.00)"}},
    ]
    s = br.summarize_journal("\n".join(json.dumps(e) for e in events))
    assert "DELIBERATE fail-closed stop" not in s


def test_classify_halt_recognizes_the_deliberate_family_and_rejects_faults():
    """The deliberate fail-closed halts each map to actionable guidance; a real fault (or an
    unrecognized message) maps to '' so it is not mislabelled as designed."""
    c = br._classify_halt
    assert "Re-select your deck" in c("dropped back to the deck list; Re-select your deck")
    assert "main menu" in c("at the Hearthstone main menu; open Play first").lower()
    assert "did not queue" in c("a game is in progress that this hunt did not start")
    assert "soft-lock" in c("still queueing after 60 polls; matchmaking never matched")
    assert "reconnect" in c("stuck on 'Reconnecting...'; Hearthstone never came back online").lower()
    assert c("could not draw a non-repeating gesture in 8 attempts") == ""   # a real fault
    assert c("") == ""
