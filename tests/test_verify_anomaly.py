"""A verify FAILURE must journal a clean anomaly, not crash the recorder.

Regression for: ``TypeError: DebugLog.record() got multiple values for argument 'kind'``.
``Verifier._fail`` forwarded the Halt fault code as ``kind=`` into ``DebugLog.anomaly``,
which forwards **context into ``record(kind="anomaly", ...)`` -- so the fault-code
``kind`` collided with the journal-entry ``kind`` and every verify failure raised,
stopping the whole search instead of logging the anomaly it was trying to record.

Two levels are pinned: the recorder itself (positional-only ``kind`` can't collide) and
the real ``Verifier`` path that first hit it.
"""

import json
import types

from hop.debuglog import DebugLog
from hop.verify import Halt, VerifyResult, Verifier


def _journal(tmp_path):
    return (tmp_path / "journal.jsonl").read_text()


def test_record_tolerates_a_kind_key_in_the_detail(tmp_path):
    """The defensive half: a caller whose detail carries ``kind`` must not crash record.

    (positional-only ``kind`` routes the stray key into ``detail`` instead of colliding.)
    """
    log = DebugLog(tmp_path, clock=lambda: 0.0)
    log.record("anomaly", kind="no_change", reason="x")   # would raise before the fix
    entry = json.loads(_journal(tmp_path).splitlines()[-1])
    assert entry["kind"] == "anomaly"                     # the journal kind is intact
    assert entry["detail"]["kind"] == "no_change"         # the stray key landed in detail


def test_anomaly_records_the_fault_code_without_crashing(tmp_path):
    log = DebugLog(tmp_path, clock=lambda: 0.0)
    log.anomaly("no screen change", fault="no_change", change_kind="none")
    entry = json.loads(_journal(tmp_path).splitlines()[-1])
    assert entry["kind"] == "anomaly"
    assert entry["detail"]["fault"] == "no_change"
    assert entry["detail"]["reason"] == "no screen change"


def test_verify_failure_journals_an_anomaly_instead_of_raising_typeerror(tmp_path, monkeypatch):
    """The end-to-end path: a size mismatch drives ``_fail`` -> ``anomaly`` -> ``record``."""
    # keep the frame-saving path out of it: fake frames have no pixels, and this test is
    # about the journal entry, not the PNGs.
    monkeypatch.setattr("hop.debuglog.pil_available", lambda: False)
    log = DebugLog(tmp_path, clock=lambda: 0.0)
    v = Verifier(change_threshold=25.0, sensorimotor=None, debug=log)

    before = types.SimpleNamespace(width=1920, height=1080)
    after = types.SimpleNamespace(width=1080, height=1920)   # rotated: size mismatch
    result = v.verify(before, after, raise_on_fail=False)    # must NOT raise TypeError

    assert isinstance(result, VerifyResult) and result.ok is False
    assert result.kind == Halt.SIZE_MISMATCH
    entry = json.loads(_journal(tmp_path).splitlines()[-1])
    assert entry["kind"] == "anomaly"
    assert entry["detail"]["fault"] == Halt.SIZE_MISMATCH   # fault code recorded cleanly
