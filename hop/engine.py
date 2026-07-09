"""The opponent-picker hunt loop.

Ties every layer together into the state machine that hunts for your target
matchup. The anti-barcode design is realized here as the interaction of Layer 5
(journey grammar + caps) with the timing layer: we *detect* the matchup at the
mulligan (class OCR + card count) but *decide when to concede* on a human-like
schedule, never insta-conceding.

Flow per game:

    classify screen
      PLAY_SCREEN    -> tap Play (queue). This is the loop's home state: the
                        deck-detail screen Hearthstone returns to after a game.
      QUEUE          -> wait (tapping while searching CANCELS the queue)
      MENU           -> HALT (main menu; the user must open Play and pick a deck)
      VS_SPLASH      -> wait for the board
      MULLIGAN       -> read class + card-count
                          target?  -> ALERT + stop touching the game (you play)
                          reject?  -> run the plausible-exit journey, then requeue
      VICTORY/DEFEAT/REWARDS -> dismiss -> back to PLAY_SCREEN -> requeue
      CONCEDE_MENU   -> tap Concede
      UNKNOWN        -> HALT + alert (fail closed; never blind-tap)

Every tap goes through :meth:`_tap`, which sequences: stateful think time (L4) ->
cap check (L5) -> capture-before -> synthesize gesture (L3) -> emit through the
persistent transport (L1) -> capture-after -> verify + coherence (L6) -> advance
HumanState (L5). Real sleeps and captures are injected so the decision logic is
testable off-device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from random import Random

from .config import Config, profile_multipliers
from .geometry import PanelGeometry
from .hearthstone import GameLayout, MulliganRead, Point, read_mulligan
from .hero_classes import DISPLAY_NAMES
from .humanize import journey, motor, timing
from .humanize.contact import ContactModel
from .humanize.limiter import CapReached, Limiter
from .humanize.sensorimotor import Posture, SensorimotorModel
from .humanize.state import HumanState
from .perception.capture import CaptureError, Capturer, FrameDeduper
from .perception.image import Frame
from .perception.ocr import ClassReader
from .perception.screens import Classification, ScreenClassifier, ScreenState
from .perception.templates import Region
from .touchstream import Gesture
from .verify import CleanStop, Halt, Verifier


class StopRequested(Exception):
    """The operator asked the hunt to stop (Stop button / hotkey). Raised from the
    stop-aware sleep so a stop interrupts a wait immediately; caught in :meth:`Engine.run`.
    """


#: Friendly, human-readable label per screen state, surfaced as the live "phase" so the
#: dashboard shows what the hunt is doing right now instead of a bare "running".
_PHASE_LABELS = {
    ScreenState.MENU: "at main menu",
    ScreenState.DECK_SELECT: "at deck select",
    ScreenState.PLAY_SCREEN: "at Play — queuing",
    ScreenState.QUEUE: "in queue — waiting for a match",
    ScreenState.ERROR_DIALOG: "clearing an error dialog",
    ScreenState.RECONNECT_DIALOG: "reconnecting…",
    ScreenState.RECONNECTING: "reconnecting…",
    ScreenState.VS_SPLASH: "match starting…",
    ScreenState.COLLECTION: "backing out of Collection",
    ScreenState.INCOMPLETE_DECK: "incomplete-deck dialog",
    ScreenState.MULLIGAN: "reading opponent (mulligan)",
    ScreenState.IN_GAME: "in game",
}


def _phase_label(state) -> str:
    return _PHASE_LABELS.get(state, state.value.replace("_", " "))


@dataclass
class RunStats:
    games: int = 0
    concedes: int = 0
    target_found: bool = False
    stop_reason: str = ""
    last_opponent: str = ""
    #: mulligan replace-taps the game ignored. Non-fatal (we stay on the mulligan
    #: and simply keep the card), but a rising count means the touch profile is
    #: drifting from what Hearthstone accepts - worth surfacing, not swallowing.
    ignored_card_taps: int = 0
    #: Opponents seen this session, by class display-name -> count. The hunt is a
    #: sampler of the ladder, so this is the observed class distribution - the thing
    #: the dashboard charts. Counted once per mulligan read (every game, target or not).
    class_distribution: dict[str, int] = field(default_factory=dict)
    #: how many mulligans we read going first / second (we go second = we have the coin
    #: = 4 cards). Lets the dashboard show the coin-flip split we actually drew.
    going_first: int = 0
    going_second: int = 0
    #: concedes that had happened when the target finally appeared (== `concedes` at
    #: that moment, since the hunt stops on a target). None until a target is found.
    concedes_until_target: int | None = None


def evaluate_matchup(read: MulliganRead, cfg: Config) -> str:
    """Pure decision: 'keep' (target), 'reject', or 'unusable' (can't read)."""
    if not read.usable:
        return "unusable"
    return "keep" if cfg.criteria.accepts(read.opponent_class, read.we_go_second) else "reject"


class Engine:
    def __init__(
        self,
        cfg: Config,
        adb,
        backend,
        panel: PanelGeometry,
        classifier: ScreenClassifier,
        reader: ClassReader,
        *,
        alerts=None,
        debug=None,
        limiter: Limiter | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
        rng: Random | None = None,
        layout: GameLayout | None = None,
        capturer=None,
    ):
        self.cfg = cfg
        self.adb = adb
        self.backend = backend
        self.panel = panel
        self.classifier = classifier
        self.reader = reader
        self.alerts = alerts
        self.debug = debug
        #: raw injected sleep; callers use self.sleep, the stop-aware wrapper below
        self._raw_sleep = sleep
        self.sleep = self._interruptible_sleep
        self.clock = clock
        self.rng = rng or Random()
        self.layout = layout or GameLayout()

        self.state = HumanState()
        self.contact = ContactModel(cfg.contact)
        self.sensor = SensorimotorModel(cfg.sensor, panel, Posture(cfg.device.posture))
        self.verifier = Verifier(cfg.vision.state_change_threshold, self.sensor, debug=debug)
        self.capturer = capturer or Capturer(adb)
        self.deduper = FrameDeduper(cfg.vision.dedupe_signature_size, cfg.vision.state_change_threshold)
        # read_mulligan is bound here so tests can substitute a scripted reader
        self._read_mulligan = read_mulligan

        mult = profile_multipliers(cfg)
        self.limiter = limiter or Limiter(cfg.caps, scale=mult["cap_scale"])
        self.delay_scale = mult["delay_scale"]
        self.min_play_turns = mult["min_play_turns"]

        self.stats = RunStats()
        self._stop = False
        self._reconnect_attempts = 0
        #: consecutive top-level dispatches that saw a live board
        self._in_game_polls = 0
        #: consecutive top-level dispatches that saw the matchmaking queue
        self._queue_polls = 0
        #: did THIS hunt queue the game currently in progress? Only a game we started
        #: may be conceded from the board - a board we found ourselves in is the user's.
        self._own_game = False
        #: live, human-readable phase for the dashboard, refreshed each loop iteration
        self.phase = "starting"

    # ── stop control (hotkeys/panic) ─────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep, but honour a stop request within ~one slice rather than after the
        full delay. A single hunt iteration can wait through a 12 s requeue delay or a
        match-start poll; without this, pressing Stop is not noticed until that wait
        ends, so "immediate stop" felt like it hung. Raising unwinds any wait loop or
        dispatch handler straight to :meth:`run`, which stops cleanly.

        Sliced by arithmetic, never by the clock, so it still terminates under the
        frozen clock the unit tests inject.
        """
        if self._stop:
            raise StopRequested()
        slice_s = self.cfg.vision.stop_poll_s
        remaining = float(seconds)
        while remaining > slice_s:
            self._raw_sleep(slice_s)
            if self._stop:
                raise StopRequested()
            remaining -= slice_s
        if remaining > 0:
            self._raw_sleep(remaining)
        if self._stop:
            raise StopRequested()

    # ── the tap primitive (L3 -> L1 -> L6) ────────────────────────────────────

    def _capture(self) -> Frame:
        self.adb.ensure_connected()
        return self.capturer.capture()

    def _tap(
        self,
        point: Point,
        *,
        committing: bool,
        decision_type: str,
        expected_change: str | None = None,
        visual_complexity: float = 0.2,
        novelty: float = 0.0,
        allow_correction: bool = True,
        verify_region: Region | None = None,
        what: str = "",
    ) -> None:
        """One humanized, verified tap. Raises Halt/CapReached/CleanStop upward.

        ``verify_region`` scopes the "did the screen change?" check to a sub-region.
        Some taps are *supposed* to change only a small part of the screen - marking
        one mulligan card for replacement redraws that card and nothing else, and a
        whole-frame mean delta of that is below the noise threshold. Asking "did the
        thing I touched change?" is both the honest question and a stricter one: a
        tap that misses the card leaves the card unchanged, and still fails.
        """
        # L4 think time (stateful), scaled by the risk profile's delay_scale
        think = timing.think_time(
            self.rng, decision_type, self.cfg.timing, self.state,
            visual_complexity=visual_complexity, novelty=novelty,
        ) * self.delay_scale
        self.sleep(think)

        # L5 cap gate BEFORE the action (never substitute; raise to stop)
        self.limiter.check_before_action(committing)

        tx, ty, radius = point.to_px(self.panel)
        before = self._capture()
        gesture = self._synth_non_repeating_tap(tx, ty, radius)

        if self.debug:
            # Which control did we aim at? Without this the journal is a list of
            # anonymous verify_ok lines and a failure can't be attributed.
            self.debug.record("tap", what=what or "?", point=(round(tx), round(ty)),
                              expected=expected_change or "any",
                              scoped=verify_region is not None)
        self.backend.emit(gesture)
        self.limiter.register_action(committing)
        self.limiter.remember_trajectory(gesture)

        # settle, then verify. (Session time is wall clock, accrued by `run()`; adding
        # think+settle here as well would double-count it.)
        settle = timing.human_delay(self.rng, 0.6, self.cfg.timing)
        self.state.tick(self.rng, dt=think + settle, action=("commit" if committing else decision_type))
        self.sleep(settle)
        after = self._capture()
        if verify_region is None:
            after = self._await_screen_motion(before, after)

        b_v, a_v = self._verify_frames(before, after, verify_region)
        try:
            self.verifier.verify(b_v, a_v, expected_change=expected_change,
                                 gesture=gesture, near_duplicate=False)
        except Halt as e:
            # ONE evidence-based correction before failing (L5), not a blind retry -
            # and only for the ONE failure that means "the tap missed". The others all
            # say the screen has ALREADY moved, or that we cannot see it at all:
            #
            #   NO_CHANGE       nothing moved  -> the tap missed. Correcting is valid.
            #   WRONG_CHANGE    it moved, differently -> the tap landed on *something*.
            #                   Re-tapping the old coordinates puts a second touch into
            #                   a screen we have not re-read: exactly the blind tap
            #                   closed-loop navigation exists to prevent.
            #   NEAR_DUPLICATE  the gesture, not the target, was wrong.
            #   INCOHERENT      the automation is wrong.
            #   SIZE_MISMATCH   the display rotated; `tx, ty` are stale coordinates.
            if not (allow_correction and e.kind == Halt.NO_CHANGE):
                raise
            self.state.note_failed_target((tx, ty))
            if self.debug:
                self.debug.record("single_correction", point=(round(tx), round(ty)))
            self.sleep(timing.human_delay(self.rng, 0.5, self.cfg.timing))
            # the correction is a real action: it must clear the cap gate, be counted,
            # and be remembered - the old code registered it but never gated or
            # remembered it, so a committing correction could step past the cap.
            self.limiter.check_before_action(committing)
            before2 = self._capture()
            g2 = self._synth_non_repeating_tap(tx, ty, radius)
            self.backend.emit(g2)
            self.limiter.register_action(committing)
            self.limiter.remember_trajectory(g2)
            self.sleep(timing.human_delay(self.rng, 0.6, self.cfg.timing))
            after2 = self._capture()
            b2, a2 = self._verify_frames(before2, after2, verify_region)
            self.verifier.verify(b2, a2, expected_change=expected_change,
                                 gesture=g2, near_duplicate=False)

    def _await_screen_motion(self, before: Frame, after: Frame) -> Frame:
        """Return the first post-tap frame that differs from ``before``, or ``after``.

        "The screen has not moved" and "the screen has not moved *yet*" are the same
        picture. `_tap` used to settle for a fixed 0.6 s and then treat the second as
        the first - and the remedy for a missed tap is another tap. **Looking again is
        strictly better than tapping again**, and a fixed settle is a bet on the
        network: screencap costs ~1.0 s over USB and ~1.4 s over Wi-Fi on this phone,
        and Hearthstone's transitions stretch with it.

        This is what :class:`FrameDeduper` is for. It was constructed in ``__init__``
        and never called; ``advanced()`` is exactly "did this frame change beyond
        noise", on a 24x24 signature that suppresses capture noise far better than the
        full-resolution comparison the verifier does.

        Only for **unscoped** taps. A card tap's whole-frame delta is 4.80 against a
        9.0 threshold (measured), so the signature cannot see it, and `_replace_card`
        already owns that retry budget.

        If nothing ever moves we return ``after`` unchanged, so the verifier still
        raises ``NO_CHANGE`` and the caller's single correction still fires.
        """
        self.deduper.reset()
        self.deduper.advanced(before)          # seed the signature with the pre-tap frame
        if self.deduper.advanced(after):
            return after                       # already moved; nothing to wait for
        for _ in range(max(0, self.cfg.vision.motion_wait_attempts)):
            self.sleep(timing.human_delay(self.rng, self.cfg.vision.screen_wait_poll_s,
                                          self.cfg.timing))
            frame = self._capture()
            if self.deduper.advanced(frame):
                if self.debug:
                    self.debug.record("late_motion", after_extra_looks=True)
                return frame
        return after

    def _synth_non_repeating_tap(self, tx: float, ty: float, radius: float) -> Gesture:
        """Draw a tap gesture that does not near-duplicate a prior trajectory.

        Non-repetition is a **pre-action** gate. It used to be a post-action check:
        the gesture was synthesized, tested, and emitted *anyway*, and the duplicate
        only surfaced when :meth:`Verifier.verify` raised afterwards - by which point
        the touch was on the wire. A gate that runs after the action cannot prevent
        anything; it can only stop the run.

        And it very nearly did, always. Measured over the real per-game tap sequence
        (300 seeded runs x 30 actions): **14.2% of taps near-duplicate a prior one**,
        and 293/300 runs flag at least once, median at tap #13. The engine survives
        today only because the correction retap re-emits with ``near_duplicate=False``
        hardcoded - i.e. the blind retap this method's caller is about to stop doing
        was load-bearing for the loop not halting.

        The remedy is the one :func:`hop.humanize.motor._endpoint_inside` already uses
        for out-of-disc endpoints: **resample, don't clamp**. Each draw is a fresh
        endpoint, dwell and micro-slip, so a redraw is genuinely a different gesture,
        not a nudged one. Eight draws take the emitted-duplicate rate to 0.01%; if all
        eight still collide, the generator is degenerate and we fail closed.
        """
        draws = max(1, self.cfg.caps.non_repetition_resamples)
        for _ in range(draws):
            gesture = motor.synth_tap(self.rng, (tx, ty), radius, self.panel,
                                      self.cfg.motor, self.contact, self.state)
            if not self.limiter.is_near_duplicate(gesture):
                return gesture
        raise Halt(f"could not draw a non-repeating gesture in {draws} attempts; "
                   "the motor generator has collapsed", Halt.NEAR_DUPLICATE)

    def _verify_frames(self, before: Frame, after: Frame,
                       region: Region | None) -> tuple[Frame, Frame]:
        """Crop both frames to ``region`` so verification asks about the right pixels."""
        if region is None:
            return before, after
        rx, ry, rw, rh = region.to_px(before)
        return before.crop(rx, ry, rw, rh), after.crop(rx, ry, rw, rh)

    # ── screen handling ──────────────────────────────────────────────────────

    def _classify(self) -> tuple[Classification, Frame]:
        frame = self._capture()
        return self.classifier.classify(frame), frame

    def _classify_settled(self) -> tuple[Classification, Frame]:
        """Classify, tolerating transient animation frames.

        Hearthstone animates between screens (the board blurs and fades out after
        a concede, banners slide in, ...), so a single UNKNOWN frame usually means
        we looked mid-transition rather than that we are lost. Re-look a bounded
        number of times - **never tapping** - and only then let UNKNOWN stand so
        the caller can fail closed.
        """
        attempts = max(1, self.cfg.vision.unknown_settle_attempts)
        cls, frame = self._classify()
        for _ in range(attempts - 1):
            if cls.state != ScreenState.UNKNOWN:
                break
            self.sleep(timing.human_delay(self.rng, 0.8, self.cfg.timing))
            cls, frame = self._classify()
        return cls, frame

    # ── waiting by looking, never by sleeping ────────────────────────────────

    def _record_unknown(self, frame: Frame | None, *, where: str, **context) -> None:
        """Keep the pixels of a screen we could not name. See :mod:`hop.debuglog`.

        This is the one capture the tool retains on purpose: it cannot be re-taken
        (nobody knows how to navigate back to a screen nobody identified) and it is
        what the anchor that prevents the next halt gets built from.

        We also record the NEAR-MISSES: the top anchors and how close they came. An
        unknown that is really a known screen just under threshold (in_game 0.539 vs
        0.72 -> "the board, but END TURN was glowing") is diagnosed from that line
        alone; a truly novel screen shows every anchor far away instead.
        """
        if self.debug:
            if frame is not None and hasattr(self.classifier, "rank"):
                try:  # near-misses are diagnostic only; never let them break the capture
                    near = self.classifier.rank(frame)[:4]
                    context.setdefault("near_misses", [
                        {"state": st.value, "score": round(sc, 3), "thr": th}
                        for st, sc, th in near])
                except Exception:
                    pass
            self.debug.unknown_screen(frame, where=where, **context)

    def _wait_until(self, predicate, *, what: str,
                    attempts: int | None = None,
                    timeout_s: float | None = None,
                    poll_s: float | None = None) -> tuple[bool, Classification, Frame]:
        """Poll the screen until ``predicate(cls, frame)`` holds. **Never taps.**

        Returns ``(satisfied, last_classification, last_frame)``; the caller decides
        whether a timeout is a Halt, because sometimes it isn't.

        Waiting is done by *looking*, not by sleeping a fixed amount. A screencap
        costs ~1.0 s over USB and ~1.4 s over Wi-Fi on this phone, and Hearthstone's
        transitions stretch with the network - so a constant delay is a guess that is
        either wasted time or, worse, a tap into a screen that has not arrived. Two
        bounds apply and the first to trip wins: a poll count (which is also what
        makes this terminate under a frozen test clock) and a wall-clock deadline.
        """
        v = self.cfg.vision
        attempts = max(1, v.screen_wait_attempts if attempts is None else attempts)
        timeout_s = v.screen_wait_timeout_s if timeout_s is None else timeout_s
        poll_s = v.screen_wait_poll_s if poll_s is None else poll_s
        deadline = self.clock() + timeout_s
        cls, frame = self._classify()
        for _ in range(attempts - 1):
            if self._stop:                       # bail before another ~1.4 s screencap
                raise StopRequested()
            if predicate(cls, frame):
                return True, cls, frame
            if self.clock() >= deadline:
                break
            self.sleep(timing.human_delay(self.rng, poll_s, self.cfg.timing))
            cls, frame = self._classify()
        satisfied = predicate(cls, frame)
        if not satisfied and self.debug:
            self.debug.record("wait_timeout", what=what, state=cls.state.value)
        return satisfied, cls, frame

    def _wait_until_screen_leaves(self, state: ScreenState, *, where: str, stuck: str) -> Classification:
        """Wait until the screen *positively* shows something other than ``state``.

        **UNKNOWN is never proof that we left.** A frame we could not read is a frame
        we could not read - and confirming a mulligan or a concede is asynchronous, so
        the frames right after the tap are exactly the ones the classifier cannot name.
        Accepting UNKNOWN here would let the next action be aimed at a screen nobody
        identified, which is the whole thing closed-loop navigation exists to prevent.

        So UNKNOWN keeps us waiting, and if the budget runs out we fail closed and
        keep the pixels.
        """
        ok, cls, frame = self._wait_until(
            lambda c, _f: c.state not in (state, ScreenState.UNKNOWN), what=where)
        if ok:
            return cls
        if cls.state == ScreenState.UNKNOWN:
            self._record_unknown(frame, where=where, confidence=round(cls.confidence, 3),
                                 waiting_to_leave=state.value)
        raise Halt(stuck)

    def run(self, max_iterations: int | None = None) -> RunStats:
        """Main hunt loop. Returns when a target is found, a cap is hit, the app
        closes, an unexpected halt occurs, or stop is requested.

        Session time is **wall clock**, accrued here. It used to be accrued only inside
        `_tap` (``think + settle``), so ``max_session_minutes`` gated tap-adjacent time
        while the 12 s requeue delays, `between_actions`, `read_consider` and every
        settle poll ran for free - and a loop that never tapped, like a soft-locked
        queue, accrued nothing at all and could never trip a cap.

        ``self.clock`` was taken and stored by ``__init__`` and never read. Read it.
        """
        iters = 0
        last = self.clock()
        try:
            if self.debug is not None:
                # First line of every run journal: exactly what THIS run was told to hunt
                # for. Ties the criteria to the run dir, so a bug report is never left
                # guessing whether the config file matches the run's real selection.
                c = self.cfg.criteria
                self.debug.record("run_criteria",
                                  target_classes=[hc.name for hc in c.target_classes],
                                  require_second=c.require_second, mode=c.mode)
            if not self.classifier.has_templates:
                raise Halt("no screen templates loaded; run `hop capture` first (refusing to run blind)")
            while not self._stop:
                if max_iterations is not None and iters >= max_iterations:
                    self.stats.stop_reason = "max_iterations"
                    break
                iters += 1
                now = self.clock()
                self.limiter.register_time(now - last)
                last = now
                # the caps that time alone can breach, checked whether or not we tap
                self.limiter.check_elapsed_caps()
                cls, frame = self._classify_settled()
                self.phase = _phase_label(cls.state)
                self._dispatch(cls, frame)
                # inter-action spacing (never burst)
                self.sleep(timing.between_actions(self.rng, self.cfg.timing, self.state) * self.delay_scale)
                if self.stats.target_found:
                    break
            if self._stop and not self.stats.stop_reason:
                self.stats.stop_reason = "user_stop"
        except StopRequested:
            self.stats.stop_reason = "user_stop"
        except CapReached as e:
            self.stats.stop_reason = f"cap:{e.which}"
            self._notify_stop(f"Stopped: {e}")
        except CleanStop as e:
            self.stats.stop_reason = "clean_stop"
            self._notify_stop(f"Clean stop: {e}")
        except CaptureError as e:
            self.stats.stop_reason = "capture_error"
            self._notify_halt(f"Capture failed: {e}")
        except Halt as e:
            self.stats.stop_reason = f"halt:{e.reason}"
            self._notify_halt(f"HALTED: {e.reason}")
        finally:
            self.backend.close()
        return self.stats

    def _dispatch(self, cls: Classification, frame: Frame) -> None:
        st = cls.state
        if st != ScreenState.IN_GAME:
            self._in_game_polls = 0
        if st != ScreenState.QUEUE:
            self._queue_polls = 0
        if st == ScreenState.ERROR_DIALOG:
            # "There was an error starting your game." - a transient network blip.
            # Dismiss and let the loop requeue; no long backoff is warranted, the
            # journey's own requeue delay supplies the human pacing.
            if self.debug:
                self.debug.record("error_dialog_dismissed")
            self._tap(self.layout.error_ok, committing=False, decision_type="commit",
                      expected_change="full_transition", what="error_ok")
        elif st == ScreenState.RECONNECT_DIALOG:
            self._reconnect()
        elif st == ScreenState.RECONNECTING:
            # Mid-reconnect: the buttons are gone, so any tap hits dead space.
            # Wait it out; _reconnect() owns the bounded polling.
            self.sleep(timing.human_delay(self.rng, 1.5, self.cfg.timing))
        elif st == ScreenState.DECK_SELECT:
            # Dropped back to the deck list (e.g. after an error). hop does NOT reopen a
            # deck: it cannot reliably tell which deck was in play (the grid names render
            # too soft/stylised to OCR, and any other pick could open the WRONG deck or
            # an incomplete one), and re-selecting a *different* deck than the user chose
            # is worse than stopping. Pause and let the user re-select. See ScreenState.
            raise Halt("dropped back to the deck list; hop won't reopen a deck (it can't "
                       "tell which one you were playing). Re-select your deck and restart "
                       "the hunt from its Play screen.")
        elif st == ScreenState.INCOMPLETE_DECK:
            # The "Complete deck automatically?" dialog (hop started on it, say). NEVER
            # auto-complete: decline it (No, never Yes). That returns to the deck list,
            # where the branch above pauses.
            self._decline_incomplete_deck()
        elif st == ScreenState.PLAY_SCREEN:
            # the loop's home state: the deck's Play button queues a game
            self._tap(self.layout.play_button, committing=False, decision_type="commit",
                      expected_change="full_transition", novelty=0.1, what="play")
            # ...and from here the game in progress is OURS to abandon. See
            # :meth:`_handle_in_game`.
            self._own_game = True
        elif st == ScreenState.QUEUE:
            # Searching for an opponent - tapping here CANCELS the queue, so the only
            # safe move is to wait. That made this the loop's one reachable infinite
            # spin: a matchmaking soft-lock holds a static screen, and no cap could ever
            # fire because every cap used to be checked inside `_tap`. Bound it.
            self._queue_polls += 1
            if self._queue_polls > max(1, self.cfg.vision.queue_wait_attempts):
                raise Halt(f"still queueing after {self._queue_polls} polls; "
                           "matchmaking never matched")
            self.sleep(timing.human_delay(self.rng, 1.5, self.cfg.timing))
        elif st == ScreenState.COLLECTION:
            # hop is never meant to be here; a stray navigation got us in. Back out to
            # the deck list (a home screen) rather than halt. See ScreenState.COLLECTION.
            if self.debug:
                self.debug.record("collection_backout")
            self._tap(self.layout.collection_back, committing=False, decision_type="commit",
                      expected_change="full_transition", what="collection_back")
        elif st == ScreenState.MENU:
            raise Halt("at the Hearthstone main menu; open Play and select a deck first "
                       "(the hunt loop queues from that deck's Play screen)")
        elif st == ScreenState.VS_SPLASH:
            self.sleep(timing.human_delay(self.rng, 1.2, self.cfg.timing))
        elif st == ScreenState.MULLIGAN:
            self._handle_mulligan(frame)
        elif st in (ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.REWARDS,
                    ScreenState.QUEST_POPUP):
            self._clear_end_screens()
        elif st == ScreenState.CONCEDE_MENU:
            self._tap(self.layout.concede_button, committing=True, decision_type="reject",
                      expected_change="full_transition", what="concede")
        elif st == ScreenState.IN_GAME:
            self._handle_in_game()
        else:  # UNKNOWN -> fail closed
            self._record_unknown(frame, where="dispatch", confidence=round(cls.confidence, 3))
            raise Halt(f"unknown screen (best confidence {cls.confidence:.2f})")

    def _handle_in_game(self) -> None:
        """A live board at the *top* of the hunt loop, which should never last.

        `_execute_reject` can only start from the mulligan, so a board reached any
        other way had no matchup read and no exit. The old branch slept and re-looped -
        forever. A game we queued and then lost the mulligan window on (a slow capture,
        a reconnect, an app restart) would rope out, turn by turn, with nobody at the
        controls. That is not a wait; that is a dead end.

        **Conceding is not special to the mulligan.** The gear sits in the top-right
        corner of every board, and :meth:`_concede` classifies before it taps. So poll a
        bounded number of times - the board may just be the tail of a concede animation,
        or a reject-play beat - and then abandon the game exactly as a rejected matchup
        is abandoned.

        The one game we must never concede is **the user's**. `_own_game` is set only
        when this hunt tapped Play, so a board we merely found ourselves in - hop
        restarted during a game the user is playing, quite possibly the target game it
        alerted them about - fails closed instead.
        """
        self._in_game_polls += 1
        if self._in_game_polls <= max(1, self.cfg.vision.in_game_wait_attempts):
            self.sleep(timing.read_consider(self.rng, self.cfg.timing, self.state))
            return
        if not self._own_game:
            raise Halt("a game is in progress that this hunt did not start; refusing to "
                       "concede it. Finish the game, or stop the hunt and restart it "
                       "from the deck's Play screen.")
        if self.debug:
            self.debug.record("abandoning_unread_game", polls=self._in_game_polls)
        # we never got to read this mulligan, so we never got to hesitate over it
        self.sleep(timing.think_time(self.rng, "reject", self.cfg.timing, self.state,
                                     visual_complexity=0.6) * self.delay_scale)
        conceded = self._concede()
        self._clear_end_screens()
        self._book_game(conceded)

    def _decline_incomplete_deck(self) -> None:
        """Tap 'No' on 'Complete deck automatically?' - never 'Yes'.

        Auto-completing a deck spends the user's dust/cards on cards Hearthstone picks;
        hop must never do that. Tap No and wait for the dialog to leave (an UNKNOWN frame
        is never proof it left - see :meth:`_wait_until_screen_leaves`).
        """
        self._tap(self.layout.deck_decline, committing=False, decision_type="commit",
                  expected_change=None, allow_correction=False, what="deck_decline")
        self._wait_until_screen_leaves(
            ScreenState.INCOMPLETE_DECK, where="deck_decline",
            stuck="the 'Complete deck automatically?' dialog did not close after No")

    def _book_game(self, conceded: bool) -> None:
        """Count a finished game, take any mandatory break, then pace the requeue."""
        self.limiter.register_game()
        self.stats.games += 1
        # A game that ended on its own was not conceded. `concedes` is the numerator of
        # the concede/commit ratio the caps exist to keep human - do not inflate it.
        self.stats.concedes += int(conceded)
        self._own_game = False
        self._in_game_polls = 0

        brk = self.limiter.needs_break(self.rng)
        if brk is not None:
            if self.debug:
                self.debug.record("break", seconds=round(brk))
            self.sleep(brk)   # wall clock; `run()` accrues it on the next iteration

        # randomized requeue delay (humans don't requeue instantly)
        self.sleep(timing.human_delay(self.rng, 12.0, self.cfg.timing) * self.delay_scale)

    def _reconnect(self) -> None:
        """Tap Reconnect once, then wait out the asynchronous reconnect.

        Hearthstone shuts down idle connections, and this loop's own anti-barcode
        pacing is what makes us idle - so this is expected traffic, not an anomaly.

        Three things make this different from every other tap:

        * Tap **Reconnect**, never Cancel. Cancelling leaves the client offline,
          after which every subsequent tap is a silent no-op - the worst failure
          mode available, because the loop looks alive while doing nothing.
        * The effect is **asynchronous**. The immediate result is not a screen
          transition but an in-place body swap to "Reconnecting...", so we cannot
          demand ``full_transition``; we only demand that *something* changed
          (``expected_change=None``), which still catches a missed tap.
        * **Never correct.** During "Reconnecting..." both buttons are removed, so
          the standard single-correction retap would hit dead space, see no change,
          and halt on a reconnect that was working fine.
        """
        cap = self.cfg.vision.reconnect_attempt_cap
        if self._reconnect_attempts >= cap:
            raise Halt(f"Hearthstone failed to reconnect after {cap} attempts; "
                       "check the phone's network or sign in again")
        self._reconnect_attempts += 1
        if self.debug:
            self.debug.record("reconnect_tapped", attempt=self._reconnect_attempts)

        self._tap(self.layout.reconnect_button, committing=False,
                  decision_type="commit", expected_change=None, allow_correction=False, what="reconnect")

        for _ in range(max(1, self.cfg.vision.reconnecting_wait_attempts)):
            cls, _frame = self._classify()
            if cls.state == ScreenState.UNKNOWN:
                # Mid-reconnect the client redraws; an unreadable frame is not
                # evidence the reconnect resolved. Keep waiting (never tapping)
                # rather than book a success we did not observe.
                self.sleep(timing.human_delay(self.rng, 1.5, self.cfg.timing))
                continue
            if cls.state not in (ScreenState.RECONNECTING, ScreenState.RECONNECT_DIALOG):
                if self.debug:
                    self.debug.record("reconnect_resolved", state=cls.state.value)
                return
            if cls.state == ScreenState.RECONNECT_DIALOG:
                # The attempt finished and failed: the buttons are back. Let the
                # outer loop re-enter _reconnect(), which re-checks the cap.
                return
            self.sleep(timing.human_delay(self.rng, 1.5, self.cfg.timing))
        raise Halt("stuck on 'Reconnecting...'; Hearthstone never came back online")

    def _handle_mulligan(self, frame: Frame) -> None:
        read = self._read_mulligan(frame, self.layout, self.reader, self.cfg.vision)
        self.state.observe_confidence(read.class_confidence)
        self.stats.last_opponent = DISPLAY_NAMES.get(read.opponent_class, "?") if read.opponent_class else "?"
        if self.debug:
            self.debug.record("mulligan_read", opponent=self.stats.last_opponent,
                              second=read.we_go_second, cards=read.num_cards,
                              conf=round(read.class_confidence, 3), method=read.method)

        decision = evaluate_matchup(read, self.cfg)
        if decision == "unusable":
            # a single re-read before halting (perception fallibility, one correction)
            self.sleep(timing.human_delay(self.rng, 0.8, self.cfg.timing))
            frame2 = self._capture()
            read = self._read_mulligan(frame2, self.layout, self.reader, self.cfg.vision)
            decision = evaluate_matchup(read, self.cfg)
            if decision == "unusable":
                raise Halt(f"could not read mulligan (class={read.class_raw!r}, cards={read.num_cards})")

        # session stats (once per game, on the resolved read): the observed class
        # distribution and the coin split the dashboard charts.
        name = DISPLAY_NAMES.get(read.opponent_class, "?") if read.opponent_class else "?"
        self.stats.last_opponent = name
        self.stats.class_distribution[name] = self.stats.class_distribution.get(name, 0) + 1
        if read.we_go_second:
            self.stats.going_second += 1
        else:
            self.stats.going_first += 1

        if decision == "keep":
            self.stats.target_found = True
            # the hunt stops on a target, so `concedes` right now IS "concedes until target".
            self.stats.concedes_until_target = self.stats.concedes
            self._alert_target(read)
            return
        self._execute_reject(read)

    # ── the plausible-exit journey (anti-barcode core) ───────────────────────

    def _execute_reject(self, read: MulliganRead) -> None:
        plan = journey.plan_reject(self.rng, read.num_cards, self.min_play_turns)
        # perform a plausible mulligan: replace the chosen cards, deliberating each
        for d in plan.mulligan:
            if d.replace and d.slot < len(read.card_centers_f):
                self._replace_card(d.slot, read.card_centers_f[d.slot], d.decision_type)
        self._confirm_mulligan()

        # play into the game to the chosen concede point (never insta-concede)
        if plan.concede_point in ("turn1", "turn2"):
            self._play_beats(plan)

        if plan.hesitate_before_concede:
            self.sleep(timing.think_time(self.rng, "reject", self.cfg.timing, self.state,
                                         visual_complexity=0.6) * self.delay_scale)

        conceded = self._concede()
        self._clear_end_screens()
        self._book_game(conceded)

    def _confirm_mulligan(self) -> None:
        """Tap Confirm, then wait for the mulligan to actually go away.

        Confirming is **asynchronous**: the cards fly off and the board draws in over
        a second or more. Within the tap's settle window only the bottom of the
        screen has moved, so demanding ``full_transition`` fails on a confirm that
        worked (measured: ``bottom_sheet``). And the single-correction retap then
        fires *after* the mulligan is already confirmed - a blind tap into a live
        game, which is precisely what closed-loop navigation exists to prevent.

        So: require only that something changed, never correct, and then verify the
        real semantic end-state - that we are no longer on the mulligan - by looking.

        And *looking* means seeing a screen we can name. The old loop accepted
        ``cls.state != MULLIGAN``, which an UNKNOWN frame satisfies - so a single
        unreadable frame (of which a confirm animation produces several) was taken as
        proof the mulligan was gone, and ``_play_beats``/``_concede`` then tapped on
        that premise. UNKNOWN is not an observation; it is the absence of one.
        """
        self._tap(self.layout.mulligan_confirm, committing=False, decision_type="commit",
                  expected_change=None, allow_correction=False, what="mulligan_confirm")
        self._wait_until_screen_leaves(
            ScreenState.MULLIGAN, where="mulligan_confirm",
            stuck="mulligan Confirm did not dismiss the mulligan")

    def _replace_card(self, slot: int, center_xf: float, decision_type: str) -> bool:
        """Mark one mulligan card for replacement. Returns whether it took.

        Two things make this unlike every other tap.

        **It changes only that card.** Marking it redraws the card and nothing else:
        measured whole-frame mean-abs-diff 4.80 against a 9.0 threshold, versus 24.24
        inside the card's own rectangle. So verification is scoped to the pixels the
        tap was aimed at - the honest question, and a stricter one, because a tap that
        misses the card leaves the card unchanged and still fails.

        **Hearthstone accepts it only intermittently** - about 1 tap in 3, while
        accepting every button tap. The cause is *unknown*; see CALIBRATION.md for the
        suspects that measurement has killed (contact scale, position, dwell,
        micro-slip, event delivery, an unfinished deal-in animation, and stale latched
        centres). The kernel sees a clean DOWN/UP at the right coordinates on a hand
        that has been at rest for ten seconds, and the game ignores it anyway. So retry
        a bounded number of times rather than pretend to understand it.

        If it still won't take, **keep the card and carry on**. A card that refuses to
        toggle is not an unknown state - we are still on the mulligan and know exactly
        where we are - so halting the hunt would be fail-closed applied where nothing
        is closed. A human who fumbles a card simply keeps it. The *committing* action,
        the concede, remains fully verified.

        **Only ``Halt.NO_CHANGE`` means "the game ignored it."** The loop used to
        swallow every Halt, so a coherence failure or a non-repetition failure - both
        of which say the *automation* is wrong, not the game - were retried three
        times, silently discarded, and counted as ignored card taps. That both hid the
        real fault and poisoned the one statistic that would have revealed it.
        """
        point, region = self.layout.card_point(center_xf), self.layout.card_region(center_xf)
        attempts = max(1, self.cfg.vision.mulligan_card_tap_attempts)
        for attempt in range(attempts):
            try:
                self._tap(point, committing=False, decision_type=decision_type,
                          visual_complexity=0.5, verify_region=region,
                          allow_correction=False, what=f"mulligan_card[{slot}]")
                return True
            except Halt as e:
                if e.kind != Halt.NO_CHANGE:
                    raise
                if self.debug:
                    self.debug.record("mulligan_card_tap_ignored",
                                      slot=slot, attempt=attempt + 1, of=attempts)
                self.sleep(timing.human_delay(self.rng, 0.7, self.cfg.timing))
        self.stats.ignored_card_taps += 1
        return False

    def _play_beats(self, plan: journey.RejectPlan) -> None:
        """A few plausible 'reading the board / passing turn' beats before bailing.

        Precisely tracking whose turn it is would need more templates; for a game
        we're about to concede, a bounded set of reading pauses plus an occasional
        end-turn tap is a sufficiently human exit. Bounded so it can't spin.
        """
        beats = 1 if plan.concede_point == "turn1" else 2
        for _ in range(beats + plan.extra_reads):
            self.sleep(timing.read_consider(self.rng, self.cfg.timing, self.state))
            if self.rng.random() < 0.5:
                try:
                    self._tap(self.layout.pass_turn_button, committing=False, decision_type="commit",
                              expected_change="partial", allow_correction=False, what="pass_turn")
                except Halt as e:
                    # Two benign readings, and they are not the same thing:
                    #   NO_CHANGE     it wasn't our turn, so End Turn did nothing.
                    #   WRONG_CHANGE  the board moved differently than `partial` - the
                    #                 game ENDED (a full_transition to the end banner).
                    # Both mean "stop playing beats". Neither means "keep tapping".
                    # Everything else - a rotation, an incoherent gesture, a degenerate
                    # generator - is the automation being wrong, and was being swallowed
                    # under the same comment before we walked into _concede()'s taps.
                    if e.kind not in (Halt.NO_CHANGE, Halt.WRONG_CHANGE):
                        raise
                    return  # _concede() re-classifies and will skip a finished game

    def _concede(self) -> bool:
        """gear -> Concede, then wait for the Game Menu to actually go away.

        **There is no confirm button, and there must be no blind tap here.** The
        Game Menu reads Concede / Options / Quit top to bottom. The old code, if the
        menu was still classified afterwards, tapped a fixed "concede_confirm" point
        at y=0.56 - which is *dead centre on Quit* (measured against the real
        concede_menu capture: the 0.9x144 px truncation disc lies almost entirely on
        the Quit plate). So the recovery path for "the Concede tap was ignored" was
        "quit Hearthstone" - and this client is known to ignore taps.

        Conceding is also **asynchronous**: the board dissolves over a second or
        more, and the menu can still be drawn on the frame right after the tap. So
        wait for the menu to leave, exactly as :meth:`_confirm_mulligan` waits for
        the mulligan, and fail closed if it never does. An UNKNOWN frame is never
        proof that we left - see :meth:`_wait_until_screen_leaves`.

        Returns whether a concede actually happened. The game can end *before* we get
        here - the opponent concedes, or a lethal lands during ``_play_beats`` - and
        the board's gear icon is drawn on the victory screen too, so tapping it there
        opens something we never anchored. There is nothing to concede on an end
        screen; say so, and let the caller clear it instead of booking a concede that
        never occurred.
        """
        cls, _frame = self._classify_settled()
        if cls.state in self.END_SCREENS:
            if self.debug:
                self.debug.record("concede_skipped", reason="game already over",
                                  state=cls.state.value)
            return False
        if cls.state != ScreenState.IN_GAME:
            raise Halt(f"asked to concede from {cls.state.value!r}, not a live game")

        # The gear opens a menu that dims the whole board, so this is a
        # full_transition, not a bottom_sheet (measured: thirds 21.5/21.1/3.7).
        self._tap(self.layout.gear_button, committing=False, decision_type="commit",
                  expected_change="full_transition", what="gear")
        # ...but `full_transition` does not mean "the Game Menu is up". `_compatible`
        # admits `top_banner` and `partial` too, so that assertion really only says
        # "something other than the bottom third moved" - which an opponent's turn
        # animating behind a *dropped* gear tap also satisfies. Then the loop's single
        # most consequential tap, the concede, would fire on an unclassified screen at
        # a coordinate that on the board is not a button at all. Look first.
        ok, cls, frame = self._wait_until(
            lambda c, _f: c.state == ScreenState.CONCEDE_MENU, what="gear")
        if not ok:
            if cls.state == ScreenState.UNKNOWN:
                self._record_unknown(frame, where="gear", confidence=round(cls.confidence, 3))
            raise Halt(f"the gear did not open the Game Menu (saw {cls.state.value!r}); "
                       "refusing to tap Concede on a screen we did not identify")

        self._tap(self.layout.concede_button, committing=True, decision_type="reject",
                  expected_change="full_transition", what="concede")
        self._wait_until_screen_leaves(
            ScreenState.CONCEDE_MENU, where="concede",
            stuck="Concede did not dismiss the Game Menu; refusing to tap again "
                  "(the entry below Concede is Quit)")
        return True

    #: Screens ``_clear_end_screens`` is allowed to tap ``end_dismiss`` on. The tap is
    #: a fixed point, so the set of screens it may land on has to be closed and named.
    END_SCREENS = (ScreenState.VICTORY, ScreenState.DEFEAT,
                   ScreenState.REWARDS, ScreenState.QUEST_POPUP)
    #: Screens that mean "the post-game stack is cleared". DECK_SELECT belongs here:
    #: Hearthstone drops back to the deck LIST after a game, and dispatch knows how to
    #: reopen the deck from there. Leaving it out is what let the loop tap end_dismiss
    #: on the deck list - i.e. on "My Collection" and the deck boxes.
    HOME_SCREENS = (ScreenState.PLAY_SCREEN, ScreenState.QUEUE,
                    ScreenState.MENU, ScreenState.DECK_SELECT)

    def _clear_end_screens(self, max_taps: int = 8) -> None:
        """Tap through victory/defeat/rewards/quest popups until back at a home screen.

        Hearthstone stacks several of these after a game - the end banner, a rewards
        screen, and "Your Quests" - in an order that varies, so this is a loop rather
        than a fixed sequence. Bounded and closed-loop: each tap is verified and we
        re-classify, so a stuck or unrecognized popup halts instead of looping.

        **``end_dismiss`` is a fixed point, so it may only be tapped on screens we
        have positively identified as end screens.** The old fallthrough tapped it on
        *every* state outside a five-case whitelist, and (1200, 972) with a 216 px
        truncation disc is 27 px from the mulligan's Confirm button, 205 px from the
        reconnect dialog's *Cancel* - "the worst failure mode available", per
        :meth:`_reconnect` - and squarely on the deck list's "My Collection" plate.
        Anything not on the list is an unexpected screen: fail closed, keep the pixels.

        The tap budget and the wait budget are also **separate**. Waiting for the board
        to finish dissolving is not a tap, and it used to consume the same counter - so
        a slow fade could exhaust the budget, whereupon the loop *returned normally*,
        indistinguishable from having reached a home screen. Exhausting either budget
        is a Halt. Both are bounded and every iteration spends one or leaves, so the
        loop terminates.
        """
        taps = waits = 0
        max_waits = max(1, self.cfg.vision.screen_wait_attempts)
        while True:
            cls, frame = self._classify_settled()
            if cls.state in self.HOME_SCREENS:
                return
            if cls.state == ScreenState.UNKNOWN:
                self._record_unknown(frame, where="clear_end_screens",
                                     confidence=round(cls.confidence, 3))
                raise Halt(f"unknown screen while clearing end screens "
                           f"(best confidence {cls.confidence:.2f})")
            if cls.state == ScreenState.IN_GAME:
                # Just after a concede the board is still fading, so the End Turn
                # housing that anchors IN_GAME is still drawn while the Defeat
                # banner is not yet. Wait it out; do not tap the dissolving board.
                if waits >= max_waits:
                    raise Halt("board never finished dissolving after the concede")
                waits += 1
                self.sleep(timing.human_delay(self.rng, 1.2, self.cfg.timing))
                continue
            if cls.state not in self.END_SCREENS:
                raise Halt(f"unexpected screen {cls.state.value!r} while clearing end "
                           f"screens; refusing to blind-tap end_dismiss there")
            if taps >= max_taps:
                raise Halt(f"end screens did not clear after {taps} dismiss taps")
            self._tap(self.layout.end_dismiss, committing=False, decision_type="commit",
                      expected_change="full_transition", allow_correction=False, what="end_dismiss")
            taps += 1

    # ── alerts ───────────────────────────────────────────────────────────────

    def _alert_target(self, read: MulliganRead) -> None:
        name = DISPLAY_NAMES.get(read.opponent_class, "?")
        second = "2nd" if read.we_go_second else "1st"
        msg = f"Target matchup: {name} (going {second}) - your turn!"
        if self.debug:
            self.debug.record("target_found", opponent=name, second=read.we_go_second)
        if self.alerts:
            self.alerts.target_found(name, read.we_go_second)
        else:
            print(msg)

    def _notify_stop(self, msg: str) -> None:
        if self.debug:
            self.debug.record("stop", message=msg)
        if self.alerts:
            self.alerts.info(msg)
        else:
            print(msg)

    def _notify_halt(self, msg: str) -> None:
        if self.debug:
            self.debug.record("halt", message=msg)
        if self.alerts:
            self.alerts.halt(msg)
        else:
            print(msg)
