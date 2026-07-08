"""Layer 6 - verification, coherence gates, and fail-closed.

Standard: *"After every autonomous action the screen must change ... AND the
channels must cohere. If not - even after a short settle - the tap missed, the
app is in an unknown state, or the automation is leaking -> raise and HALT,
preserving evidence, not retry blindly."*

Concretely this module checks, after each action:

0. **Comparability** - the two frames are the same size. A different size means the
   display rotated or the app restarted, so no pixel comparison between them means
   anything and every screen fraction the engine holds is stale. Fail closed.
1. **Generic progress** - the screen advanced. Operationally this is
   ``classify_change() != "none"``, i.e. *some* third of the screen moved by more
   than ``change_threshold``. (The whole-frame ``mad`` is journaled, not gated on -
   a tap that redraws one card moves the whole frame by 4.80 against a 9.0
   threshold, which is why scoped verification exists.)
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
    """Unexpected halt: unknown state / missed tap / incoherence. Fail closed.

    ``kind`` names *why*, because one caller legitimately tolerates exactly one of
    these and must not tolerate the rest. ``_replace_card`` retries a mulligan card
    the game ignored (``no_change``) - we are still on the mulligan, we know where we
    are, and Hearthstone drops ~2 card taps in 3 for reasons nobody has found. But a
    coherence failure or a non-repetition failure on that same tap says the
    *automation* is wrong, not the game, and swallowing those as "ignored tap" both
    hides them and inflates ``stats.ignored_card_taps``.
    """

    #: the screen did not move: a missed tap, or an input the app dropped
    NO_CHANGE = "no_change"
    #: it moved, but not into the end-state we demanded
    WRONG_CHANGE = "wrong_change"
    #: touch/IMU incoherence for the declared posture
    INCOHERENT = "incoherent"
    #: this gesture near-duplicates a prior trajectory
    NEAR_DUPLICATE = "near_duplicate"
    #: the display geometry changed under us; frames are not comparable
    SIZE_MISMATCH = "size_mismatch"
    #: anything raised by the engine rather than by a verification check
    UNEXPECTED = "unexpected"

    def __init__(self, reason: str, kind: str = UNEXPECTED):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


class CleanStop(RuntimeError):
    """Orderly stop (ADB dropped, app closed, cap reached). Restart-safe."""


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    reason: str
    change_kind: str = ""
    coherence: str = ""
    #: which check failed; see :class:`Halt`. Empty on success.
    kind: str = ""


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
        # 0. the two frames must be comparable at all.
        #
        # A frame-size change means the DISPLAY changed underneath us: the phone
        # rotated, or Hearthstone restarted into portrait. Every screen fraction the
        # engine is about to act on was resolved against the old geometry, so the one
        # thing we must not do here is call it a successful transition and tap on.
        #
        # It used to do exactly that. When the widths differed it set `mad = 999.0`
        # and *fabricated* `change_kind = "full_transition"` - which is precisely the
        # kind every navigation tap expects (Play, the gear, Concede, end_dismiss), so
        # a rotation verified OK and the engine carried on aiming at coordinates from
        # a screen that no longer existed. A height-only mismatch was worse: the width
        # guard passed it through to `classify_change`, which compares each frame's
        # own thirds and so silently correlated misaligned pixels.
        if before.width != after.width or before.height != after.height:
            return self._fail(
                f"frame size changed under us ({before.width}x{before.height} -> "
                f"{after.width}x{after.height}); the display rotated or the app "
                "restarted - refusing to verify across a geometry change",
                before, after, change_kind="size_mismatch", kind=Halt.SIZE_MISMATCH,
                raise_on_fail=raise_on_fail)

        # 1. generic progress
        mad = mean_abs_diff(before, after)
        change_kind = classify_change(before, after, self.change_threshold)
        if change_kind == "none":
            return self._fail("no screen change after action (missed tap / stuck)", before, after,
                              change_kind=change_kind, kind=Halt.NO_CHANGE,
                              raise_on_fail=raise_on_fail)

        # 2. semantic end-state
        if expected_change is not None and change_kind != expected_change and not _compatible(expected_change, change_kind):
            return self._fail(
                f"screen changed but not as expected (want {expected_change}, saw {change_kind})",
                before, after, change_kind=change_kind, kind=Halt.WRONG_CHANGE,
                raise_on_fail=raise_on_fail,
            )

        # 3a. sensorimotor coherence
        coherence = ""
        if gesture is not None:
            verdict = self.sensorimotor.validate(gesture, observed_imu)
            coherence = verdict.value
            if verdict == CoherenceVerdict.INCOHERENT:
                return self._fail("touch/IMU incoherence on a handheld run (flat sensors)",
                                  before, after, change_kind=change_kind, coherence=coherence,
                                  kind=Halt.INCOHERENT, raise_on_fail=raise_on_fail)

        # 3b. non-repetition
        if near_duplicate:
            return self._fail("gesture near-duplicates a prior trajectory (non-repetition)",
                              before, after, change_kind=change_kind, coherence=coherence,
                              kind=Halt.NEAR_DUPLICATE, raise_on_fail=raise_on_fail)

        if self.debug is not None:
            self.debug.record("verify_ok", change_kind=change_kind, coherence=coherence, mad=round(mad, 2))
        return VerifyResult(True, "ok", change_kind, coherence)

    def _fail(self, reason: str, before: Frame, after: Frame, *, change_kind: str = "",
              coherence: str = "", kind: str = Halt.UNEXPECTED,
              raise_on_fail: bool = True) -> VerifyResult:
        if self.debug is not None:
            self.debug.anomaly(reason, before=before, after=after,
                               change_kind=change_kind, coherence=coherence, kind=kind)
        if raise_on_fail:
            raise Halt(reason, kind)
        return VerifyResult(False, reason, change_kind, coherence, kind)


def _compatible(expected: str, actual: str) -> bool:
    """Some expected/actual change kinds are acceptable variants of each other."""
    if expected == "full_transition" and actual in ("top_banner", "partial"):
        return True
    if expected == "partial" and actual in ("bottom_sheet", "top_banner"):
        return True
    return False
