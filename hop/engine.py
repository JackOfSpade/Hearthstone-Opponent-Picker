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
from dataclasses import dataclass
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
from .touchstream import Gesture
from .verify import CleanStop, Halt, Verifier


@dataclass
class RunStats:
    games: int = 0
    concedes: int = 0
    target_found: bool = False
    stop_reason: str = ""
    last_opponent: str = ""


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
        gem_template=None,
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
        self.sleep = sleep
        self.clock = clock
        self.rng = rng or Random()
        self.layout = layout or GameLayout()
        self.gem_template = gem_template

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

    # ── stop control (hotkeys/panic) ─────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

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
    ) -> None:
        """One humanized, verified tap. Raises Halt/CapReached/CleanStop upward."""
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
        gesture = motor.synth_tap(self.rng, (tx, ty), radius, self.panel,
                                  self.cfg.motor, self.contact, self.state)
        near_dup = self.limiter.is_near_duplicate(gesture)

        self.backend.emit(gesture)
        self.limiter.register_action(committing)
        self.limiter.remember_trajectory(gesture)

        # settle, then verify
        settle = timing.human_delay(self.rng, 0.6, self.cfg.timing)
        self.limiter.register_time(think + settle)
        self.state.tick(self.rng, dt=think + settle, action=("commit" if committing else decision_type))
        self.sleep(settle)
        after = self._capture()

        try:
            self.verifier.verify(before, after, expected_change=expected_change,
                                 gesture=gesture, near_duplicate=near_dup)
        except Halt:
            if allow_correction:
                # ONE evidence-based correction before failing (L5), not blind retry
                self.state.note_failed_target((tx, ty))
                if self.debug:
                    self.debug.record("single_correction", point=(round(tx), round(ty)))
                self.sleep(timing.human_delay(self.rng, 0.5, self.cfg.timing))
                before2 = self._capture()
                g2 = motor.synth_tap(self.rng, (tx, ty), radius, self.panel,
                                     self.cfg.motor, self.contact, self.state)
                self.backend.emit(g2)
                self.limiter.register_action(committing)
                self.sleep(timing.human_delay(self.rng, 0.6, self.cfg.timing))
                after2 = self._capture()
                self.verifier.verify(before2, after2, expected_change=expected_change,
                                     gesture=g2, near_duplicate=False)
            else:
                raise

    # ── screen handling ──────────────────────────────────────────────────────

    def _classify(self) -> tuple[Classification, Frame]:
        frame = self._capture()
        return self.classifier.classify(frame), frame

    def run(self, max_iterations: int | None = None) -> RunStats:
        """Main hunt loop. Returns when a target is found, a cap is hit, the app
        closes, an unexpected halt occurs, or stop is requested."""
        iters = 0
        try:
            if not self.classifier.has_templates:
                raise Halt("no screen templates loaded; run `hop capture` first (refusing to run blind)")
            while not self._stop:
                if max_iterations is not None and iters >= max_iterations:
                    self.stats.stop_reason = "max_iterations"
                    break
                iters += 1
                cls, frame = self._classify()
                self._dispatch(cls, frame)
                # inter-action spacing (never burst)
                self.sleep(timing.between_actions(self.rng, self.cfg.timing, self.state) * self.delay_scale)
                if self.stats.target_found:
                    break
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
        if st == ScreenState.PLAY_SCREEN:
            # the loop's home state: the deck's Play button queues a game
            self._tap(self.layout.play_button, committing=False, decision_type="commit",
                      expected_change="full_transition", novelty=0.1)
        elif st == ScreenState.QUEUE:
            # searching for an opponent - tapping here CANCELS the queue, so wait
            self.sleep(timing.human_delay(self.rng, 1.5, self.cfg.timing))
        elif st == ScreenState.MENU:
            raise Halt("at the Hearthstone main menu; open Play and select a deck first "
                       "(the hunt loop queues from that deck's Play screen)")
        elif st == ScreenState.VS_SPLASH:
            self.sleep(timing.human_delay(self.rng, 1.2, self.cfg.timing))
        elif st == ScreenState.MULLIGAN:
            self._handle_mulligan(frame)
        elif st in (ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.REWARDS):
            self._clear_end_screens()
        elif st == ScreenState.CONCEDE_MENU:
            self._tap(self.layout.concede_button, committing=True, decision_type="reject",
                      expected_change="full_transition")
        elif st == ScreenState.IN_GAME:
            # Unexpected here means a reject-play beat; a plausible pass, then re-loop
            self.sleep(timing.read_consider(self.rng, self.cfg.timing, self.state))
        else:  # UNKNOWN -> fail closed
            if self.debug:
                self.debug.anomaly("unknown screen", after=frame, confidence=round(cls.confidence, 3))
            raise Halt(f"unknown screen (best confidence {cls.confidence:.2f})")

    def _handle_mulligan(self, frame: Frame) -> None:
        read = self._read_mulligan(frame, self.layout, self.reader, self.gem_template)
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
            read = self._read_mulligan(frame2, self.layout, self.reader, self.gem_template)
            decision = evaluate_matchup(read, self.cfg)
            if decision == "unusable":
                raise Halt(f"could not read mulligan (class={read.class_raw!r}, cards={read.num_cards})")

        if decision == "keep":
            self.stats.target_found = True
            self._alert_target(read)
            return
        self._execute_reject(read)

    # ── the plausible-exit journey (anti-barcode core) ───────────────────────

    def _execute_reject(self, read: MulliganRead) -> None:
        plan = journey.plan_reject(self.rng, read.num_cards, self.min_play_turns)
        # perform a plausible mulligan: replace the chosen cards, deliberating each
        for d in plan.mulligan:
            if d.replace:
                pt = self.layout.card_slot(d.slot, read.num_cards, self.panel)
                self._tap(pt, committing=False, decision_type=d.decision_type,
                          expected_change="partial", visual_complexity=0.5)
        # confirm the mulligan
        self._tap(self.layout.mulligan_confirm, committing=False, decision_type="commit",
                  expected_change="full_transition")

        # play into the game to the chosen concede point (never insta-concede)
        if plan.concede_point in ("turn1", "turn2"):
            self._play_beats(plan)

        if plan.hesitate_before_concede:
            self.sleep(timing.think_time(self.rng, "reject", self.cfg.timing, self.state,
                                         visual_complexity=0.6) * self.delay_scale)

        self._concede()
        self._clear_end_screens()
        self.limiter.register_game()
        self.stats.games += 1
        self.stats.concedes += 1

        # mandatory break?
        brk = self.limiter.needs_break(self.rng)
        if brk is not None:
            if self.debug:
                self.debug.record("break", seconds=round(brk))
            self.limiter.register_time(brk)
            self.sleep(brk)

        # randomized requeue delay (humans don't requeue instantly)
        self.sleep(timing.human_delay(self.rng, 12.0, self.cfg.timing) * self.delay_scale)

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
                              expected_change="partial", allow_correction=False)
                except Halt:
                    return  # board state moved on; fine, we're conceding anyway

    def _concede(self) -> None:
        # gear -> concede -> confirm; the classifier verifies each transition
        self._tap(self.layout.gear_button, committing=False, decision_type="commit",
                  expected_change="bottom_sheet")
        self._tap(self.layout.concede_button, committing=True, decision_type="reject",
                  expected_change="full_transition")
        # some clients show a confirm; tap it if a concede menu is still up
        cls, _ = self._classify()
        if cls.state == ScreenState.CONCEDE_MENU:
            self._tap(self.layout.concede_confirm, committing=False, decision_type="commit",
                      expected_change="full_transition", allow_correction=False)

    def _clear_end_screens(self, max_taps: int = 8) -> None:
        """Tap through victory/defeat/rewards popups until back at menu/queue.

        Bounded and closed-loop: each tap is verified and we re-classify, so a
        stuck popup halts instead of looping forever."""
        for _ in range(max_taps):
            cls, frame = self._classify()
            # Hearthstone drops back to the deck's Play screen after a game; MENU
            # is also terminal (the caller's next dispatch reports it).
            if cls.state in (ScreenState.PLAY_SCREEN, ScreenState.QUEUE, ScreenState.MENU):
                return
            if cls.state == ScreenState.UNKNOWN:
                raise Halt("unknown screen while clearing end screens")
            self._tap(self.layout.end_dismiss, committing=False, decision_type="commit",
                      expected_change="full_transition", allow_correction=False)

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
