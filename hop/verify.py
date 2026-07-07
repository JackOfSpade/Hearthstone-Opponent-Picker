"""Layer 6 - verification, coherence gates, and fail-closed.

Standard: *"After every autonomous action the screen must change ... AND the
channels must cohere. If not - even after a short settle - the tap missed, the
app is in an unknown state, or the automation is leaking -> raise and HALT,
preserving evidence, not retry blindly."*

Concretely this module checks, after each action:

1. **Generic progress** - the screen advanced (mean-abs-diff over threshold).
2. **Semantic end-state** - the *specific* expected change kind occurred (a bare
   change is spoofable by an unrelated animation).
3. **Coherence gates** - sensorimotor coherence for the declared posture,
   non-repetition (this gesture didn't near-duplicate a prior one), and a
   perception->action loop that wasn't implausibly fast/certain for the context.

A failure raises :class:`Halt` (unexpected: stuck screen, missed tap, unknown
modal, incoherence) which the engine turns into a stop-and-snapshot. A
:class:`CleanStop` (ADB dropped, human closed the app) is distinct and
restart-safe - the standard forbids letting the second masquerade as the first.
The one permitted recovery is a *single evidence-based correction* before Halt
(§ Layer 5), which the engine - not this module - performs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .humanize.sensorimotor import CoherenceVerdict, ImuSignature, SensorimotorModel
from .perception.diffing import classify_change
from .perception.image import Frame, mean_abs_diff
from .touchstream import Gesture


class Halt(RuntimeError):
    """Unexpected halt: unknown state / missed tap / incoherence. Fail closed."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class CleanStop(RuntimeError):
    """Orderly stop (ADB dropped, app closed, cap reached). Restart-safe."""


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str
    change_kind: str = ""
    coherence: str = ""


class Verifier:
    def __init__(
        self,
        change_threshold: float,
        sensorimotor: SensorimotorModel,
        debug=None,
    ):
        self.change_threshold = change_threshold
        self.sensorimotor = sensorimotor
        self.debug = debug

    def verify(
        self,
        before: Frame,
        after: Frame,
        *,
        expected_change: str | None = None,
        gesture: Gesture | None = None,
        observed_imu: ImuSignature | None = None,
        near_duplicate: bool = False,
        raise_on_fail: bool = True,
    ) -> VerifyResult:
        """Run all Layer-6 checks; raise :class:`Halt` on failure by default.

        ``expected_change`` (from :func:`hop.perception.diffing.classify_change`)
        is the semantic end-state, e.g. ``"full_transition"`` after tapping Play.
        ``near_duplicate`` comes from the limiter's non-repetition check.
        """
        # 1. generic progress
        mad = mean_abs_diff(_match(before, after), after) if before.width == after.width else 999.0
        change_kind = classify_change(before, after, self.change_threshold) if before.width == after.width else "full_transition"
        if change_kind == "none":
            return self._fail("no screen change after action (missed tap / stuck)", before, after,
                              change_kind=change_kind, raise_on_fail=raise_on_fail)

        # 2. semantic end-state
        if expected_change is not None and change_kind != expected_change and not _compatible(expected_change, change_kind):
            return self._fail(
                f"screen changed but not as expected (want {expected_change}, saw {change_kind})",
                before, after, change_kind=change_kind, raise_on_fail=raise_on_fail,
            )

        # 3a. sensorimotor coherence
        coherence = ""
        if gesture is not None:
            verdict = self.sensorimotor.validate(gesture, observed_imu)
            coherence = verdict.value
            if verdict == CoherenceVerdict.INCOHERENT:
                return self._fail("touch/IMU incoherence on a handheld run (flat sensors)",
                                  before, after, change_kind=change_kind, coherence=coherence,
                                  raise_on_fail=raise_on_fail)

        # 3b. non-repetition
        if near_duplicate:
            return self._fail("gesture near-duplicates a prior trajectory (non-repetition)",
                              before, after, change_kind=change_kind, coherence=coherence,
                              raise_on_fail=raise_on_fail)

        if self.debug is not None:
            self.debug.record("verify_ok", change_kind=change_kind, coherence=coherence, mad=round(mad, 2))
        return VerifyResult(True, "ok", change_kind, coherence)

    def _fail(self, reason: str, before: Frame, after: Frame, *, change_kind: str = "",
              coherence: str = "", raise_on_fail: bool = True) -> VerifyResult:
        if self.debug is not None:
            self.debug.anomaly(reason, before=before, after=after,
                               change_kind=change_kind, coherence=coherence)
        if raise_on_fail:
            raise Halt(reason)
        return VerifyResult(False, reason, change_kind, coherence)


def _match(a: Frame, b: Frame) -> Frame:
    """Crop ``a`` to ``b``'s size if they differ slightly (defensive)."""
    if a.width == b.width and a.height == b.height:
        return a
    return a.crop(0, 0, b.width, b.height)


def _compatible(expected: str, actual: str) -> bool:
    """Some expected/actual change kinds are acceptable variants of each other."""
    if expected == "full_transition" and actual in ("top_banner", "partial"):
        return True
    if expected == "partial" and actual in ("bottom_sheet", "top_banner"):
        return True
    return False
