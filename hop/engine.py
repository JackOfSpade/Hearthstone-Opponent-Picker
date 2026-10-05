"""The opponent-picker hunt loop.

Ties every layer together into the state machine that hunts for your target
matchup. It detects the matchup at the mulligan (class OCR + card count): a
target stops all input for the user, while a rejected, positively named
mulligan follows the calibrated direct-concede and bounded click-through path.

Flow per game:

    classify screen
      PLAY_SCREEN    -> tap Play (queue). This is the loop's home state: the
                        deck-detail screen Hearthstone returns to after a game.
      QUEUE/VS_SPLASH -> hands-off capture watch until the opponent class is readable
      MENU           -> HALT (main menu; the user must open Play and pick a deck)
      MULLIGAN       -> read class + card-count
                          target?  -> ALERT + stop touching the game (you play)
                          reject?  -> direct named-mulligan concede, one bounded
                                      Play-location click-through, then requeue
      VICTORY/DEFEAT/REWARDS -> dismiss -> back to PLAY_SCREEN -> requeue
      CONCEDE_MENU   -> tap Concede
      UNKNOWN        -> HALT + alert (fail closed; never blind-tap)

Fixed Pixel-7a controls use :meth:`_tap_fixed`: stateful think time, motor/limiter
accounting, then one calibrated semantic phase check.  Small or exceptional controls
still use :meth:`_tap` with before/after pixel verification.  Real sleeps and captures
are injected so the decision logic is testable off-device.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from random import Random

from .adb import AdbError
from .config import Config, MAX_UI_OPEN_DELAY_S, validate_config
from .geometry import PanelGeometry
from .hearthstone import (
    GameLayout,
    MulliganRead,
    Point,
    mulligan_card_count_diagnostics,
    read_mulligan,
)
from .hero_classes import DISPLAY_NAMES, nearest_class
from .humanize import journey, motor, timing
from .humanize.contact import ContactModel
from .humanize.limiter import Limiter
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
    ScreenState.DECK_SELECT: "at the deck list — re-select your deck to resume",
    ScreenState.PLAY_SCREEN: "at Play — queuing",
    ScreenState.QUEUE: "in queue — waiting for a match",
    ScreenState.ERROR_DIALOG: "clearing an error dialog",
    ScreenState.RECONNECT_DIALOG: "reconnecting…",
    ScreenState.RECONNECTING: "reconnecting…",
    ScreenState.VS_SPLASH: "match starting…",
    ScreenState.COLLECTION: "backing out of Collection",
    ScreenState.INCOMPLETE_DECK: "incomplete-deck dialog",
    ScreenState.CONCEDE_WARNING: "early-concede warning",
    ScreenState.MULLIGAN: "reading opponent (mulligan)",
    ScreenState.IN_GAME: "in game",
}


def _phase_label(state) -> str:
    return _PHASE_LABELS.get(state, state.value.replace("_", " "))


#: Pause between capture retries (see :meth:`Engine._capture_resilient`), before forcing a
#: fresh wireless link. A RAW, fixed sleep -- deliberately NOT a humanized ``timing`` delay --
#: so a transient screencap stall self-heals without drawing from ``self.rng``, keeping the
#: RNG stream bit-identical to a clean capture. Small: it only spaces the reconnect from the
#: retry so the link has a beat to come back before we ask for the frame again.
_CAPTURE_RETRY_BACKOFF_S = 1.0


# Only keep the small, causal subset of a transport's diagnostic snapshot in the
# per-run journal.  A backend owns live subprocesses and could expose arbitrary
# objects or a very large stderr buffer; the journal must stay cheap and safe to
# paste into a bug report.  These fields are enough to answer the important
# question when a persistent ``adb shell hid -`` writer dies: did an ADB reconnect
# advance the connection epoch, did the writer re-register against that epoch, and
# if not, what did its child process say before the pipe closed?
_TRANSPORT_STATUS_FIELDS = (
    "kind", "opened", "pid", "returncode",
    "connection_generation", "stream_generation", "reopen_count",
    "last_reopen_reason", "last_reopen_ok",
    "last_command", "last_write_at", "stderr_tail", "failure",
    # Cached session geometry only: this is evidence for a coordinate diagnosis,
    # never a live per-tap ADB probe.
    "rotation", "panel_width_px", "panel_height_px", "axis_touch_major_max",
    "axis_touch_minor_max", "axis_pressure_max", "axis_orientation_max",
    "last_display_x", "last_display_y", "last_native_x", "last_native_y",
)
_TRANSPORT_STATUS_TEXT_LIMIT = 800


def _transport_snapshot(backend) -> dict:
    """Return a bounded, JSON-safe transport diagnostic snapshot.

    Backends deliberately do not have to implement ``transport_status``: this
    observability layer is additive, so existing test fakes and the ADB-input
    fallback retain their minimal interface.  A status probe is diagnostic only;
    it must never turn a recoverable transport failure into a logging failure.
    """
    probe = getattr(backend, "transport_status", None)
    if not callable(probe):
        return {}
    try:
        raw = probe()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key in _TRANSPORT_STATUS_FIELDS:
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, str):
            out[key] = value[-_TRANSPORT_STATUS_TEXT_LIMIT:]
        elif value is None or isinstance(value, (bool, int, float)):
            out[key] = value
    return out


def _transport_stream_reopened(before: dict, after: dict) -> bool:
    """Whether a successful emit replaced a persistent transport stream.

    ``reopen_count`` is the explicit modern signal; comparing the stream epoch
    also supports a backend that has the generation contract but not a counter.
    Require a value in *both* snapshots so the initial tap on an older backend
    never looks like a recovery event.
    """
    for key in ("reopen_count", "stream_generation"):
        if key in before and key in after and before[key] != after[key]:
            return True
    return False


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
    """Pure decision: ``keep``, ``reject``, or ``unusable``.

    The opponent class is always essential.  The card count answers only one
    criterion -- ``require_second`` -- so making it a universal prerequisite
    needlessly stopped a known non-target on a transient green-glow miss.  If
    the class itself cannot ever pass the criteria, reject it regardless of the
    unknown coin.  If it could pass, require a valid turn only when the criteria
    actually use one.  A class-only target is merely alerted (no card input),
    and the direct reject path deliberately leaves the hand untouched.
    """
    if read.opponent_class is None:
        return "unusable"
    # Ask whether the class could be kept under the favourable coin result.
    # A false answer makes rejection deterministic even for require_second=True.
    if not cfg.criteria.accepts(read.opponent_class, True):
        return "reject"
    if cfg.criteria.require_second and read.num_cards not in (3, 4):
        return "unusable"
    return "keep" if cfg.criteria.accepts(read.opponent_class, read.we_go_second) else "reject"


#: A RESOLVED class read below this confidence is kept for diagnosis even though the engine acts
#: on it. ClassReader maps OCR edit-distance to confidence as 1 - distance/(max_edit+1); at the
#: default ocr_max_edit_distance=3 that is 1.0 = exact, 0.75 = 1 edit, 0.5 = 2 edits, 0.25 = 3
#: edits (the snap's max), so this floor keeps reads >=2 glyphs off their label. (If that knob is
#: raised the scale widens, but the floor still keeps only shakily-resolved reads.) Such a read
#: never reaches the "unusable" re-read path -- it looks certain -- yet 2+ errors is exactly where
#: a garbled two-word label mis-snaps to a valid-but-wrong class (the Demon-Hunter-read-as-Hunter
#: bug: conf 0.25). The frame is otherwise deleted, leaving the report only a bare class name;
#: keeping the pixels makes the next such misread diagnosable. A clean read (distance 0, conf 1.0)
#: saves nothing.
_LOW_CONFIDENCE_KEEP = 0.75


def _region_gray_stats(frame: Frame, region) -> dict:
    """Grayscale mean/min/max/std of a screen region, for diagnosing an unreadable read.

    This is the datum a "could not read mulligan" report never had: a LOW std means the
    class region was near-uniform (the opponent nameplate had not drawn in -- a transient
    that waiting or the recover-and-requeue resolves), while a HIGH std means the text WAS
    there and the OCR/region alignment is the fault (a code problem, not a slow phone).
    Works with either Frame backing (ndarray or flat bytes); returns ``{}`` if unavailable.
    """
    try:
        rx, ry, rw, rh = region.to_px(frame)
        crop = frame.crop(rx, ry, rw, rh)
    except Exception:
        return {}
    data = crop.data
    try:
        import numpy as np
        if not isinstance(data, (bytes, bytearray)):
            a = np.asarray(data, dtype=float)
            return {"mean": round(float(a.mean()), 1), "min": int(a.min()),
                    "max": int(a.max()), "std": round(float(a.std()), 1)}
    except Exception:
        pass
    vals = bytes(data) if isinstance(data, (bytes, bytearray)) else b""
    if not vals:
        return {}
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"mean": round(mean, 1), "min": min(vals), "max": max(vals),
            "std": round(var ** 0.5, 1)}


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
        # Config may arrive from ``dataclasses.replace`` rather than TOML, so make
        # the burst's hard timing proof before constructing any motor/transport state.
        validate_config(cfg)
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

        self.limiter = limiter or Limiter(cfg.caps)

        self.stats = RunStats()
        self._stop = False
        #: monotonic time the most recent frame finished capturing. The reaction clock
        #: a detector sees starts when the screen becomes actionable, so the capture +
        #: classification latency spent perceiving it is credited against the next tap's
        #: think time (see `_tap`) rather than stacked on top. None until the first look.
        self._perceived_at: float | None = None
        #: Duration of the most recent successful screen read.  Polling is a cadence,
        #: not an additional pause on top of a slow wireless screencap, so wait helpers
        #: use this to sleep only the unspent part of their target cadence.
        self._last_capture_s = 0.0
        #: Fixed-coordinate buttons defer proof to the next named screen.  Counts are
        #: deliberately keyed by button/site so a persistent source screen can retry a
        #: bounded number of times without ever tapping through UNKNOWN.
        self._fixed_tap_attempts: dict[str, int] = {}
        #: Scoped prior for the NEXT top-loop classify: the set of screens the branch we
        #: just ran expects to see next (e.g. QUEUE sets {queue, mulligan}). Consumed and
        #: reset to None every iteration, so a branch that sets nothing => full classify.
        #: A wrong guess only costs the full scan back (see classify_expected).
        self._expected_next = None
        #: A deliberate phase-boundary classification performed by a fixed action.
        #: The main loop consumes it instead of immediately taking the same screenshot again.
        self._deferred_screen: tuple[Classification, Frame] | None = None
        self._reconnect_attempts = 0
        #: consecutive top-level dispatches that saw a live board
        self._in_game_polls = 0
        #: Once matchmaking has been seen, the next game is deliberately watched
        #: hands-off until its mulligan has a readable opponent class.  A slow queue
        #: is not a fault on this phone: capture latency is its cadence, and no
        #: timeout may turn a still-searching game into a stop.
        self._awaiting_match_class = False
        #: consecutive top-level dispatches that saw the deck LIST (waiting, never tapping,
        #: for the user to re-select their deck after an error dropped us there)
        self._deck_select_polls = 0
        #: did THIS hunt queue the game currently in progress? Only a game we started
        #: may be conceded from the board - a board we found ourselves in is the user's.
        self._own_game = False
        #: live, human-readable phase for the dashboard, refreshed each loop iteration
        self.phase = "starting"

    # ── stop control (hotkeys/panic) ─────────────────────────────────────────

    def request_stop(self) -> None:
        self._stop = True

    def transport_status(self) -> dict:
        """Bounded backend diagnostics for an in-process status/report snapshot.

        Kept on the engine rather than teaching the controller about every backend,
        and deliberately returns an empty mapping for legacy/degraded transports.
        """
        return _transport_snapshot(self.backend)

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep, but honour a stop request within ~one slice rather than after the
        full delay. A single hunt iteration can wait through a calibrated semantic
        cooldown or recovery poll; without this, pressing Stop is not noticed until that
        wait ends, so "immediate stop" felt like it hung. Raising unwinds any wait loop or
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

    def _sleep_for(self, seconds: float, reason: str, **detail) -> None:
        """Record an intentional wait, then sleep.

        The raw journal already timestamps taps and verification, but a delay report
        should not require subtracting timestamps and guessing whether the gap was a
        deliberate humanization pause, a screen poll, or slow capture I/O.
        """
        seconds = float(seconds)
        if self.debug:
            self.debug.record("sleep", reason=reason, seconds=round(seconds, 3), **detail)
        self.sleep(seconds)

    # ── the tap primitive (L3 -> L1 -> L6) ────────────────────────────────────

    def _capture(self) -> Frame:
        # Do not put a get-state round trip in front of every screencap.  On the Pixel
        # it was an unjournalled extra wireless operation on the hot path, and a dead
        # link is already handled safely by `_capture_resilient`'s bounded reconnect.
        # Record the *start* of the successful perception interval: its latency is part
        # of the reaction a user/app sees, rather than another delay stacked on top.
        started = self.clock()
        frame = self._capture_resilient()
        finished = self.clock()
        if (frame.width, frame.height) != (self.panel.width_px, self.panel.height_px):
            if self.debug:
                self.debug.anomaly("captured frame no longer matches calibrated panel",
                                   before=frame, frame_size=(frame.width, frame.height),
                                   panel_size=(self.panel.width_px, self.panel.height_px))
            raise Halt("display size/orientation changed; refusing fixed-coordinate tap",
                       Halt.SIZE_MISMATCH)
        self._last_capture_s = max(0.0, finished - started)
        self._perceived_at = started
        return frame

    def _paced_poll_sleep(self, anchor_s: float, reason: str, **detail) -> None:
        """Spend only the part of a polling cadence not already spent capturing.

        A Pixel 7a Wi-Fi screencap is normally about two seconds.  Adding a sampled
        one-second poll sleep after every one made the *effective* cadence three
        seconds, without buying another observation or any input safety.  Faster
        devices still receive the ordinary humanized remainder.
        """
        requested = timing.human_delay(self.rng, anchor_s, self.cfg.timing)
        capture_s = self._last_capture_s
        remaining = max(0.0, requested - capture_s)
        self._sleep_for(remaining, reason, requested_s=round(requested, 3),
                        capture_s=round(capture_s, 3), **detail)

    def _semantic_cooldown(self, anchor_s: float, reason: str, **detail) -> None:
        """One randomized, calibrated wait before a semantic boundary check."""
        seconds = timing.human_cooldown(self.rng, anchor_s, self.cfg.timing)
        self._sleep_for(seconds, reason, anchor_s=anchor_s,
                        verification="semantic_deferred", **detail)

    def _ui_open_delay(self, seconds: float, reason: str, **detail) -> None:
        """Wait an exact, short allowance for a user-visible control to render.

        This is deliberately separate from :meth:`_semantic_cooldown`: a menu or
        confirmation popup needs a small predictable render allowance, rather than
        a randomized state-resolution wait.  The configured value is validated at
        engine construction and checked here as a defensive boundary as well.
        """
        seconds = float(seconds)
        if not 0 < seconds <= MAX_UI_OPEN_DELAY_S:
            raise ValueError(
                f"UI-opening delay must be within (0, {MAX_UI_OPEN_DELAY_S:g}] seconds")
        self._sleep_for(seconds, reason, artificial=True,
                        max_s=MAX_UI_OPEN_DELAY_S, **detail)

    def _capture_resilient(self) -> Frame:
        """One good frame, tolerating a *transient* capture failure instead of dying on it.

        A wireless-ADB screencap that stalls or drops -- screen lock, Doze, a Wi-Fi power-save
        blip, the Mac briefly sleeping -- raises ``CaptureError``/``AdbError``. A SINGLE such
        stall used to crash the entire hunt: the raw ``TimeoutExpired`` escaped every handler in
        :meth:`run` (it dispatched from inside a mid-mulligan ``_tap``). Instead we reconnect and
        retry a bounded ``vision.capture_retry_attempts`` times; only a link that stays down
        through all of them surfaces (as the last error, which :meth:`run` now halts on cleanly,
        stats intact). The common transient -- one bad frame in a long session -- self-heals.

        RNG-safety (this codebase is bit-identical-sensitive): the retry path draws NOTHING from
        ``self.rng`` and never ticks ``HumanState``; its backoff is the stop-aware but RNG-free
        ``self.sleep``. So a clean capture (one iteration, the overwhelming case) is byte-for-byte
        the same sequence of RNG draws as before, and even a retried capture inserts zero draws.
        A Stop is honoured between attempts (via the per-iteration ``self._stop`` check and the
        stop-aware backoff), so the bounded recovery window can't swallow a Stop press.

        Journalling: the ``capture`` ms is recorded for the SUCCESSFUL frame only, so the healthy
        per-frame I/O stats the report leans on aren't skewed by a stall; each failed attempt
        instead records its own ``capture_retry`` (how long it hung + the error) -- the exact
        evidence a 'why did it die?' report needs, and the counter that shows a link going bad.
        """
        attempts = max(1, self.cfg.vision.capture_retry_attempts)
        last_err: Exception | None = None
        for i in range(attempts):
            # Honour a Stop between attempts. A dead link makes each capturer.capture() hang the
            # full adb timeout and each reconnect block too; without this a Stop pressed during a
            # retry storm would go unobserved for the whole (bounded) recovery window. The raised
            # StopRequested unwinds to run() -> stop_reason="user_stop" (the same clean unwind the
            # in-tap sleeps use) and draws no RNG, so the clean path is unchanged.
            if self._stop:
                raise StopRequested()
            t0 = self.clock()
            try:
                frame = self.capturer.capture()
            except (CaptureError, AdbError) as e:
                last_err = e
                if self.debug:
                    self.debug.record("capture_retry", attempt=i + 1, of=attempts,
                                      ms=round((self.clock() - t0) * 1000.0),
                                      error=str(e)[:200])
                if i + 1 < attempts:
                    self._recover_capture_link()
                continue
            if self.debug:
                # Screencap latency is the biggest, and once-invisible, cost of a run:
                # ~1 s over USB / ~1.4 s over Wi-Fi on this phone (see `_await_screen_motion`),
                # two-plus frames per tap, dozens per game. One ms line per frame makes the
                # dominant cost attributable, and separable from the humanized delays that must
                # NOT be "optimized" away. Timed on self.clock so it is 0 under the frozen test
                # clock and adds no RNG draw.
                self.debug.record("capture", ms=round((self.clock() - t0) * 1000.0))
            return frame
        # Every attempt failed: hand the last error up. run() maps it to a clean stop.
        assert last_err is not None  # attempts >= 1, so the loop ran and set this
        raise last_err

    def _recover_capture_link(self) -> None:
        """Between capture retries: pause a beat, then force a fresh wireless link. Best-effort
        and RNG-free -- the backoff is the STOP-AWARE ``self.sleep`` (``_interruptible_sleep``
        draws nothing from ``self.rng`` and never ticks ``HumanState``, so it keeps bit-identity
        while still honouring a Stop within one ``stop_poll_s`` slice), and the reconnect is
        guarded so a failed reconnect doesn't abort the retry loop -- we still want the next
        screencap attempt (the link may have come back on its own). ``getattr`` so a duck-typed
        adb without ``reconnect`` (older test fakes) simply skips the reconnect."""
        self.sleep(_CAPTURE_RETRY_BACKOFF_S)
        recover = getattr(self.adb, "reconnect", None)
        if recover is not None:
            # A reconnect replaces ADB's transport underneath any resident ``adb shell``
            # child. Record both epochs and the UHID stream snapshot before it happens, so a
            # later BrokenPipe is mechanically attributable to this transition rather than
            # inferred from timestamps in a truncated journal.
            before_generation = getattr(self.adb, "connection_generation", None)
            before_transport = _transport_snapshot(self.backend)
            try:
                recover()
            except Exception as e:
                self._record_transport_event(
                    "capture_link_reconnect", outcome="failed",
                    adb_generation_before=before_generation,
                    adb_generation_after=getattr(self.adb, "connection_generation", None),
                    transport_before=before_transport,
                    transport_after=_transport_snapshot(self.backend),
                    error=f"{type(e).__name__}: {e}"[:_TRANSPORT_STATUS_TEXT_LIMIT],
                )
            else:
                self._record_transport_event(
                    "capture_link_reconnect", outcome="succeeded",
                    adb_generation_before=before_generation,
                    adb_generation_after=getattr(self.adb, "connection_generation", None),
                    transport_before=before_transport,
                    transport_after=_transport_snapshot(self.backend),
                )

    def _record_transport_event(self, kind: str, **detail) -> None:
        """Best-effort journal record for transport lifecycle evidence.

        A disk/serialization fault while trying to explain a transport problem must
        never replace the real failure. ``DebugLog`` normally handles all values
        here, but this boundary intentionally makes the instrumentation inert when
        a third-party debug sink is unhealthy.
        """
        if self.debug is None:
            return
        try:
            self.debug.record(kind, **detail)
        except Exception:
            pass

    def _emit_gesture(self, gesture: Gesture, *, what: str) -> None:
        """Emit one gesture and preserve persistent-transport lifecycle evidence.

        The reporter needs this *before* :meth:`run` unwinds into ``backend.close()``,
        which necessarily tears down the child process and would otherwise erase its
        exit status/stderr. The wrapper changes no recovery or input behaviour: it
        only snapshots optional diagnostics around the existing emit call and reraises
        the original exception unchanged.
        """
        before = _transport_snapshot(self.backend)
        try:
            self.backend.emit(gesture)
        except Exception as e:
            after = _transport_snapshot(self.backend)
            self._record_transport_event(
                "transport_failure", operation="emit", what=what or "?",
                error=f"{type(e).__name__}: {e}"[:_TRANSPORT_STATUS_TEXT_LIMIT],
                transport=after or before,
            )
            raise
        after = _transport_snapshot(self.backend)
        if _transport_stream_reopened(before, after):
            self._record_transport_event(
                "transport_stream_reopened", what=what or "?",
                transport_before=before, transport_after=after,
            )

    def _tap_fixed(
        self,
        point: Point,
        *,
        committing: bool,
        decision_type: str,
        what: str,
        visual_complexity: float = 0.2,
        novelty: float = 0.0,
    ) -> None:
        """Emit a known fixed button without generic pixel verification.

        The caller may use this only from a positively classified source state and
        must name a later semantic classification that proves success.  It preserves
        the same motor trajectory, limiter bookkeeping, HumanState trajectory and
        latency-aware think sample as :meth:`_tap`, but avoids two redundant wireless
        screencaps for controls whose next named state already supplies that proof.
        """
        think = timing.think_time(
            self.rng, decision_type, self.cfg.timing, self.state,
            visual_complexity=visual_complexity, novelty=novelty,
        )
        latency = (0.0 if self._perceived_at is None
                   else max(0.0, self.clock() - self._perceived_at))
        # A fixed action is already preceded by a named phase/capture whose measured
        # latency is human-scale.  Unlike pixel-verified taps, do not stack another
        # irreducible reaction floor after that latency.
        wait = timing.credit_latency(think, latency, 0.0)
        self._sleep_for(wait, "tap_think", what=what, decision_type=decision_type,
                        committing=committing, think_s=round(think, 3),
                        credited_s=round(think - wait, 3),
                        latency_s=round(latency, 3), verification="semantic_deferred")

        tx, ty, radius = point.to_px(self.panel)
        gesture = self._synth_non_repeating_tap(tx, ty, radius)
        # The in-contact endpoint includes the bounded micro-slip, so report its
        # true geometric envelope rather than claiming it is constrained by the
        # nominal target disc alone. Motor synthesis draws slip below 1.5x its
        # configured scale, so this is a hard inclusive telemetry bound.
        endpoint_radius = radius * 0.9 + 1.5 * self.cfg.motor.tap_micro_slip_px
        if self.debug:
            self.debug.record("tap", what=what, point=(round(tx), round(ty)),
                              nominal_point=(round(tx), round(ty)),
                              actual_endpoint=tuple(round(v) for v in gesture.endpoint()),
                              radius_px=round(endpoint_radius, 3),
                              target_radius_px=round(radius, 3),
                              duration_s=round(gesture.duration, 4),
                              expected="semantic_next_state", scoped=False,
                              verification="semantic_deferred",
                              latency_s=round(latency, 3))
        self._emit_gesture(gesture, what=what)
        self.limiter.register_action(committing)
        self.limiter.remember_trajectory(gesture)

        # Draw and account for the same settle component so HumanState remains a
        # faithful trajectory.  Do not sleep it separately: the mandatory next
        # semantic screencap is already longer than this Pixel's ~0.6 s settle.
        settle = timing.human_delay(self.rng, 0.6, self.cfg.timing)
        self.state.tick(self.rng, dt=think + settle,
                        action=("commit" if committing else decision_type))
        if self.debug:
            self.debug.record("tap_timing", what=what, think_s=round(think, 3),
                              settle_s=round(settle, 3),
                              verification="semantic_deferred",
                              settle_paced_by="next_semantic_capture")

    def _tap_open_loop(self, point: Point, *, committing: bool, what: str) -> None:
        """Emit one tap in a bounded, already-authorized fixed-coordinate burst.

        The reject fast path has no useful visual observation between these taps: the
        post-concede stack consumes the same Play location, and extra taps while the
        next game searches are inert.  Deliberation happened before opening the
        burst; its configured cooldown cadence is the only app-facing wait here.
        In particular, this must never capture or classify.
        """
        # A Stop can race a completed cadence sleep.  Check again at the last
        # possible point, before even synthesizing an input that could be emitted.
        if self._stop:
            raise StopRequested()
        tx, ty, radius = point.to_px(self.panel)
        gesture = self._synth_non_repeating_tap(tx, ty, radius)
        endpoint_radius = radius * 0.9 + 1.5 * self.cfg.motor.tap_micro_slip_px
        if self.debug:
            self.debug.record("tap", what=what, point=(round(tx), round(ty)),
                              nominal_point=(round(tx), round(ty)),
                              actual_endpoint=tuple(round(v) for v in gesture.endpoint()),
                              radius_px=round(endpoint_radius, 3),
                              target_radius_px=round(radius, 3),
                              duration_s=round(gesture.duration, 4),
                              expected="open_loop_burst", scoped=False,
                              verification="burst")
        # Synthesis itself can take long enough for a Stop to race in.  Recheck at
        # the last possible boundary before the gesture reaches the transport.
        if self._stop:
            raise StopRequested()
        self._emit_gesture(gesture, what=what)
        self.limiter.register_action(committing)
        self.limiter.remember_trajectory(gesture)
        # A burst still advances the latent interaction trajectory.  Its real elapsed
        # time is accounted by the cadence sleeps surrounding it; zero here avoids
        # pretending an extra invisible delay occurred.
        self.state.tick(self.rng, dt=0.0, action="commit" if committing else "burst")

    def _fixed_attempt(self, key: str, limit: int) -> int:
        """Reserve a bounded semantic retry; never reserve an extra blind tap."""
        used = self._fixed_tap_attempts.get(key, 0)
        if used >= max(1, limit):
            raise Halt(f"{key} did not reach its semantic destination after {used} tap(s)")
        self._fixed_tap_attempts[key] = used + 1
        return used + 1

    def _clear_fixed_attempt(self, key: str) -> None:
        self._fixed_tap_attempts.pop(key, None)

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
        """One humanized, verified tap. Raises Halt/CleanStop upward.

        ``verify_region`` scopes the "did the screen change?" check to a sub-region.
        Some taps are *supposed* to change only a small part of the screen - marking
        one mulligan card for replacement redraws that card and nothing else, and a
        whole-frame mean delta of that is below the noise threshold. Asking "did the
        thing I touched change?" is both the honest question and a stricter one: a
        tap that misses the card leaves the card unchanged, and still fails.
        """
        # L4 think time (stateful). `think` is the human's whole reaction budget; the
        # perception latency already burned on this frame (capture + classification, the
        # ~5 s that dominate a wireless run) is part of that same reaction to a detector,
        # so we credit it against the wait instead of stacking on top -- otherwise the
        # observable reaction was `latency + think`, inhumanly slow (it overran the
        # mulligan timer). The SAME `think` is sampled (one RNG draw, unchanged) and still
        # feeds `state.tick` below, so HumanState and the whole RNG-dependent stream stay
        # bit-identical; only the realized sleep shrinks, and only when a real clock runs
        # (under the frozen test clock the credit is 0 and the wait is unchanged).
        think = timing.think_time(
            self.rng, decision_type, self.cfg.timing, self.state,
            visual_complexity=visual_complexity, novelty=novelty,
        )
        latency = 0.0 if self._perceived_at is None else max(0.0, self.clock() - self._perceived_at)
        floor = (self.cfg.timing.think_commit_shift if decision_type == "commit"
                 else self.cfg.timing.think_reject_shift)
        wait = timing.credit_latency(think, latency, floor)
        self._sleep_for(wait, "tap_think", what=what or "?",
                        decision_type=decision_type, committing=committing,
                        think_s=round(think, 3), credited_s=round(think - wait, 3),
                        latency_s=round(latency, 3), verification="pixel")

        tx, ty, radius = point.to_px(self.panel)
        before = self._capture()
        gesture = self._synth_non_repeating_tap(tx, ty, radius)

        if self.debug:
            # Which control did we aim at? Without this the journal is a list of
            # anonymous verify_ok lines and a failure can't be attributed.
            self.debug.record("tap", what=what or "?", point=(round(tx), round(ty)),
                              expected=expected_change or "any",
                              scoped=verify_region is not None)
        self._emit_gesture(gesture, what=what)
        self.limiter.register_action(committing)
        self.limiter.remember_trajectory(gesture)

        # settle, then verify. (Session time is wall clock, accrued by `run()`; adding
        # think+settle here as well would double-count it.)
        settle = timing.human_delay(self.rng, 0.6, self.cfg.timing)
        if self.debug:
            self.debug.record("tap_timing", what=what or "?",
                              think_s=round(think, 3), settle_s=round(settle, 3))
        self.state.tick(self.rng, dt=think + settle, action=("commit" if committing else decision_type))
        self._sleep_for(settle, "tap_settle", what=what or "?")
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
            self._sleep_for(timing.human_delay(self.rng, 0.5, self.cfg.timing),
                            "correction_before_retry", what=what or "?")
            # the correction is a real action: count it and remember its trajectory,
            # so it feeds the concede ratio and can't replay an earlier gesture (the
            # old code registered it but never remembered it).
            before2 = self._capture()
            g2 = self._synth_non_repeating_tap(tx, ty, radius)
            self._emit_gesture(g2, what=what)
            self.limiter.register_action(committing)
            self.limiter.remember_trajectory(g2)
            self._sleep_for(timing.human_delay(self.rng, 0.6, self.cfg.timing),
                            "correction_settle", what=what or "?")
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
            self._paced_poll_sleep(self.cfg.vision.screen_wait_poll_s,
                                   "await_screen_motion")
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
        not a nudged one. The limiter compares against recent, location-aware history;
        if all configured draws still collide, the generator is degenerate and we fail closed.
        """
        draws = max(1, self.cfg.caps.non_repetition_resamples)
        for i in range(draws):
            gesture = motor.synth_tap(self.rng, (tx, ty), radius, self.panel,
                                      self.cfg.motor, self.contact, self.state)
            if not self.limiter.is_near_duplicate(gesture):
                return gesture
            if self.debug:
                self.debug.record("non_repetition_reject", attempt=i + 1, of=draws,
                                  memory=self.limiter.trajectory_memory_size)
        if self.debug:
            self.debug.record("non_repetition_collapse", attempts=draws,
                              memory=self.limiter.trajectory_memory_size)
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

    def _classify(self, expected=None) -> tuple[Classification, Frame]:
        frame = self._capture()
        t0 = self.clock()
        # `expected` (a set of ScreenStates) scopes the NCC scan to the screens this site
        # actually expects, plus the fixed interrupt floor -- ~300 ms instead of ~3.8 s,
        # and result-identical to a full classify (a missed expectation just full-scans on
        # the same frame). None => full classify, so every unscoped caller is unchanged.
        cls = (self.classifier.classify(frame) if expected is None
               else self.classifier.classify_expected(frame, expected))
        if self.debug:
            # Classification is a sliding-window NCC scan over every anchor's region and,
            # on this phone, the second-biggest cost after screencap (~3-4 s/call at the
            # current frame size) - and until now it hid inside the delay report's
            # "classify/OCR/logic" residual. Journal its ms and the state it resolved to,
            # so the report can name it the way it now names capture latency, and so a
            # sub-threshold UNKNOWN shows up inline. self.clock => 0 under the frozen test
            # clock and adds no RNG draw.
            self.debug.record("classify", ms=round((self.clock() - t0) * 1000.0),
                              state=cls.state.value)
        return cls, frame

    def _classify_settled(self, expected=None) -> tuple[Classification, Frame]:
        """Classify, tolerating transient animation frames.

        Hearthstone animates between screens (the board blurs and fades out after
        a concede, banners slide in, ...), so a single UNKNOWN frame usually means
        we looked mid-transition rather than that we are lost. Re-look a bounded
        number of times - **never tapping** - and only then let UNKNOWN stand so
        the caller can fail closed.

        ``expected`` scopes each look (see :meth:`_classify`); a transient UNKNOWN
        already full-scans (nothing in the scoped set cleared), so settling is unaffected.
        """
        attempts = max(1, self.cfg.vision.unknown_settle_attempts)
        cls, frame = self._classify(expected)
        for _ in range(attempts - 1):
            if cls.state != ScreenState.UNKNOWN:
                break
            self._semantic_cooldown(self.cfg.vision.screen_wait_poll_s,
                                    "unknown_settle")
            cls, frame = self._classify(expected)
        return cls, frame

    # ── bounded semantic waits and screen reads ──────────────────────────────

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
                    poll_s: float | None = None,
                    expected=None) -> tuple[bool, Classification, Frame]:
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
        cls, frame = self._classify(expected)
        for _ in range(attempts - 1):
            if self._stop:                       # bail before another ~1.4 s screencap
                raise StopRequested()
            if predicate(cls, frame):
                return True, cls, frame
            if self.clock() >= deadline:
                break
            self._paced_poll_sleep(poll_s, "wait_until_poll", what=what,
                                   state=cls.state.value)
            cls, frame = self._classify(expected)
        satisfied = predicate(cls, frame)
        if not satisfied and self.debug:
            self.debug.record("wait_timeout", what=what, state=cls.state.value)
        return satisfied, cls, frame

    def _wait_semantic(self, predicate, *, what: str, cooldown_s: float,
                       attempts: int, expected=None) -> tuple[bool, Classification, Frame]:
        """Wait in full calibrated intervals, then make one named-state check.

        This deliberately replaces screenshot-driven one-second polling on the
        fixed phone.  A persistent source state is useful retry evidence; UNKNOWN
        is not, and receives another full interval rather than a tight capture loop.
        """
        cls = None
        frame = None
        for check in range(max(1, attempts)):
            self._semantic_cooldown(cooldown_s, f"{what}_cooldown",
                                    check=check + 1, of=max(1, attempts))
            cls, frame = self._classify(expected)
            if predicate(cls, frame):
                return True, cls, frame
        assert cls is not None and frame is not None
        if self.debug:
            self.debug.record("wait_timeout", what=what, state=cls.state.value,
                              verification="semantic_deferred")
        return False, cls, frame

    def _wait_until_screen_leaves(self, state: ScreenState, *, where: str, stuck: str,
                                  arrivals=(), attempts: int | None = None,
                                  timeout_s: float | None = None) -> Classification:
        """Wait until the screen *positively* shows something other than ``state``.

        **UNKNOWN is never proof that we left.** A frame we could not read is a frame
        we could not read - and confirming a mulligan or a concede is asynchronous, so
        the frames right after the tap are exactly the ones the classifier cannot name.
        Accepting UNKNOWN here would let the next action be aimed at a screen nobody
        identified, which is the whole thing closed-loop navigation exists to prevent.

        So UNKNOWN keeps us waiting, and if the budget runs out we fail closed and
        keep the pixels.

        ``arrivals`` names the screens we expect to land on, so the wait can scope its
        looks to ``{state} | arrivals`` (+ the interrupt floor). ``state`` (the FROM
        screen) MUST be in the scoped set: while it is still up its anchor is the true
        winner, and during a cross-fade it can out-score the arriving screen -- omitting
        it would let a scoped look call "left" one poll early. Still result-identical to
        an unscoped wait (any un-listed arrival just full-scans on its frame).

        ``attempts``/``timeout_s`` override the generic ``wait_until`` budget for the rare
        wait whose semantics are not a button transition. The mulligan-confirm wait uses
        this: it must also outlast the OPPONENT's mulligan (an anchorless "Opponent Still
        Choosing..." banner that reads UNKNOWN), for which the ~20 s transition budget is
        far too short. Since UNKNOWN never counts as "left" and this never taps, a larger
        budget only postpones failing closed -- it cannot turn a wait into a misdirected tap.
        """
        expected = frozenset({state}) | frozenset(arrivals)
        ok, cls, frame = self._wait_until(
            lambda c, _f: c.state not in (state, ScreenState.UNKNOWN), what=where,
            expected=expected, attempts=attempts, timeout_s=timeout_s)
        if ok:
            return cls
        if cls.state == ScreenState.UNKNOWN:
            self._record_unknown(frame, where=where, confidence=round(cls.confidence, 3),
                                 waiting_to_leave=state.value)
        raise Halt(stuck)

    def run(self, max_iterations: int | None = None) -> RunStats:
        """Main hunt loop. Returns when a target is found, the app closes, an
        unexpected halt occurs, or the user stops it - there are deliberately no
        volume or time caps.

        Session time is **wall clock**, accrued here once per iteration for the
        status display's session minutes; it gates nothing. ``self.clock`` was taken
        and stored by ``__init__`` and never read. Read it.
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
                                  avoid_classes=[hc.name for hc in c.avoid_classes],
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
                # While searching (or while the mulligan nameplate is still drawing),
                # one immediate scoped capture is the entire heartbeat.  In particular,
                # do not use _classify_settled here: an UNKNOWN animation is simply the
                # next watch observation, not a reason to add an artificial sleep.
                if self._awaiting_match_class:
                    cls, frame = self._classify({ScreenState.QUEUE, ScreenState.VS_SPLASH,
                                                 ScreenState.MULLIGAN})
                else:
                    # Consume-once: the previous iteration's branch may have left a scoped
                    # prior for this look; clear it first so a branch that sets nothing this
                    # time falls back to a full classify next time.
                    expected, self._expected_next = self._expected_next, None
                    if self._deferred_screen is not None:
                        cls, frame = self._deferred_screen
                        self._deferred_screen = None
                    else:
                        cls, frame = self._classify_settled(expected)
                self.phase = _phase_label(cls.state)
                self._dispatch(cls, frame)
                # The next loop begins with a wireless screencap (~2 s on the fixed
                # Pixel 7a).  A second unconditional inter-action sleep used to stack
                # on top of both branch-specific pacing and that capture.
                if self.stats.target_found:
                    # The hunt's designed SUCCESS exit: a target matchup appeared, we alerted
                    # (_alert_target) and stop touching the game so the user plays it. Record
                    # WHY we stopped, exactly like every other exit does -- this is the ONLY
                    # common exit that otherwise left stop_reason blank, which made a by-design
                    # stop read as a crash in status/the bug report ("running: False" with no
                    # reason at all). Stats-only: no RNG draw and no HumanState tick, so the
                    # stream stays bit-identical. Guarded so a Stop/Halt that raced onto this
                    # same iteration keeps its own, more specific reason.
                    if not self.stats.stop_reason:
                        self.stats.stop_reason = "target_found"
                    break
            if self._stop and not self.stats.stop_reason:
                self.stats.stop_reason = "user_stop"
        except StopRequested:
            self.stats.stop_reason = "user_stop"
        except CleanStop as e:
            self.stats.stop_reason = "clean_stop"
            self._notify_stop(f"Clean stop: {e}")
        except CaptureError as e:
            self.stats.stop_reason = "capture_error"
            self._notify_halt(f"Capture failed: {e}")
        except AdbError as e:
            # A transport failure that survived the in-capture retry+reconnect
            # (`_capture_resilient`): the Wi-Fi link stayed down through every attempt -- the
            # Mac asleep, the phone off the network, the router gone. Not a hop fault and not a
            # blind-tap risk. End the run cleanly with a named stop_reason and the stats intact,
            # exactly like its CaptureError sibling, instead of letting a raw adb error escape to
            # the controller thread -- which reported only a bare traceback and lost the clean
            # stop. (Before the retry landed, this was the crash the user pasted: a screencap
            # TimeoutExpired thrown from a mid-mulligan _tap, caught by nothing.)
            self.stats.stop_reason = "adb_error"
            self._notify_halt(f"ADB link failed: {e}")
        except Halt as e:
            self.stats.stop_reason = f"halt:{e.reason}"
            self._notify_halt(f"HALTED: {e.reason}")
        finally:
            self.backend.close()
        return self.stats

    def _tap_play(self) -> None:
        """Tap Play once and defer proof to the calibrated named-state boundary.

        A persistent positively named Play screen is the only condition that may
        consume its bounded retry budget on a later dispatch. No per-tap pixel
        verification or short timeout is inserted ahead of the Play-to-mulligan
        cooldown, and the resulting boundary frame is reused by the main loop.
        """
        attempts = max(1, self.cfg.vision.play_tap_attempts)
        attempt = self._fixed_attempt("play", attempts)
        if attempt > 1 and self.debug:
            self.debug.record("play_tap_ignored", attempt=attempt - 1, of=attempts,
                              verification="semantic_source_persisted")
        self._tap_fixed(self.layout.play_button, committing=False, decision_type="commit",
                        novelty=0.1, what="play")
        self._semantic_cooldown(self.cfg.timing.play_to_mulligan_cooldown_s,
                                "play_cooldown", attempt=attempt)
        cls, frame = self._classify(
            {ScreenState.PLAY_SCREEN, ScreenState.QUEUE, ScreenState.VS_SPLASH,
             ScreenState.MULLIGAN, ScreenState.ERROR_DIALOG})
        # This is a phase boundary, not a generic pixel verification.  It is consumed
        # by the next loop iteration so we never recapture merely to dispatch it.
        self._deferred_screen = (cls, frame)

    def _tap_dispatch_button(self, point: Point, *, expected_change: str, what: str,
                             label: str) -> None:
        """Tap a generic, low-stakes dispatch dialog button, retrying a tap this client
        silently dropped -- the same shape as `_tap_play`, generalized for screens plain
        enough (single button, no committing consequence, no special wait-budget needs)
        to share ONE knob (``vision.dispatch_tap_attempts``) instead of a dedicated one
        each. See `_tap_play` for why the failure looks the way it does and why
        ``allow_correction=False`` is right here too.
        """
        attempts = max(1, self.cfg.vision.dispatch_tap_attempts)
        attempt = self._fixed_attempt(what, attempts)
        if attempt > 1 and self.debug:
            self.debug.record(f"{what}_tap_ignored", attempt=attempt - 1, of=attempts,
                              verification="semantic_source_persisted")
        self._tap_fixed(point, committing=False, decision_type="commit", what=what)

    def _dispatch(self, cls: Classification, frame: Frame) -> None:
        st = cls.state
        # The match watch has exactly three benign observations: queue, VS, and an
        # as-yet unreadable mulligan.  Anything else is an interruption or arrival
        # and must resume the ordinary, named-state dispatcher.  Do this before the
        # normal retry/accounting reset below so a modal cannot inherit watch state.
        if self._awaiting_match_class:
            if st in (ScreenState.QUEUE, ScreenState.VS_SPLASH, ScreenState.UNKNOWN):
                return
            # MULLIGAN is handled below by the class gate; every other named state
            # is an interruption and therefore ends the hands-off match watch.
            if st != ScreenState.MULLIGAN:
                self._end_match_watch(outcome="interrupted", state=st.value)
        # A changed, positively classified state is the semantic proof for its fixed
        # predecessor.  Conversely, a source that persists is the only condition that
        # permits the bounded retry below.
        if st != ScreenState.PLAY_SCREEN:
            self._clear_fixed_attempt("play")
        if st != ScreenState.ERROR_DIALOG:
            self._clear_fixed_attempt("error_ok")
        if st != ScreenState.COLLECTION:
            self._clear_fixed_attempt("collection_back")
        if st != ScreenState.CONCEDE_MENU:
            self._clear_fixed_attempt("concede")
        if st != ScreenState.IN_GAME:
            self._in_game_polls = 0
        if st != ScreenState.DECK_SELECT:
            # left the deck list (the user re-selected, or we never landed there): the pause
            # counter resets so the next drop-back starts a fresh wait budget and the hunt
            # resumes with no memory of the interruption.
            self._deck_select_polls = 0
        if st == ScreenState.ERROR_DIALOG:
            # "There was an error starting your game." - a transient network blip.
            # Dismiss and let the loop requeue; its calibrated Play boundary supplies
            # the pacing rather than an extra arbitrary delay here.
            if self.debug:
                self.debug.record("error_dialog_dismissed")
            self._tap_dispatch_button(self.layout.error_ok, expected_change="full_transition",
                                      what="error_ok", label="the error dialog's OK button")
        elif st == ScreenState.RECONNECT_DIALOG:
            self._reconnect()
        elif st == ScreenState.RECONNECTING:
            # Mid-reconnect: the buttons are gone, so any tap hits dead space.
            # Wait it out; _reconnect() owns the bounded polling.
            self._sleep_for(timing.human_delay(self.rng, 1.5, self.cfg.timing),
                            "reconnecting_wait")
        elif st == ScreenState.DECK_SELECT:
            # Dropped back to the deck LIST (Hearthstone lands here after dismissing an
            # "error starting your game", among others). hop still does NOT reopen a deck: it
            # cannot reliably tell which deck was in play (the grid names render too
            # soft/stylised to OCR, and any other pick could open the WRONG deck or an
            # incomplete one), and re-selecting a *different* deck than the user chose is
            # worse than stopping -- so it NEVER taps the list. But rather than hard-HALT the
            # whole hunt on a transient blip (which forced a full restart), it PAUSES: alert
            # once, then wait -- re-looking, never tapping -- and let the loop resume by
            # itself the moment the user is back on a deck's Play screen (the reset above
            # clears this counter and the play branch queues again, stats intact). This
            # separate human-intervention pause remains bounded: if nobody re-selects
            # within the budget, fail closed so a walked-away session stops cleanly.
            self._deck_select_polls += 1
            if self._deck_select_polls > max(1, self.cfg.vision.deck_select_wait_attempts):
                raise Halt("dropped back to the deck list and nobody re-selected a deck; hop "
                           "won't reopen one (it can't tell which you were playing). Re-select "
                           "your deck and restart the hunt from its Play screen.")
            if self._deck_select_polls == 1:
                # Alert ONCE (a silent Mac notification, per the alert design -- only a target
                # match makes a sound); the dashboard phase also reads "re-select your deck".
                if self.debug:
                    self.debug.record("deck_select_pause")
                if self.alerts:
                    self.alerts.info("hop dropped back to the deck list (a game error). "
                                     "Re-select your deck to resume the hunt.")
            self._sleep_for(timing.human_delay(self.rng, 1.5, self.cfg.timing),
                            "deck_select_wait", poll=self._deck_select_polls)
        elif st == ScreenState.INCOMPLETE_DECK:
            # The "Complete deck automatically?" dialog (hop started on it, say). NEVER
            # auto-complete: decline it (No, never Yes). That returns to the deck list,
            # where the branch above pauses.
            self._decline_incomplete_deck()
        elif st == ScreenState.PLAY_SCREEN:
            # the loop's home state: the deck's Play button queues a game
            self._tap_play()
            # ...and from here the game in progress is OURS to abandon. See
            # :meth:`_handle_in_game`.
            self._own_game = True
            # Play was tapped: the next look is the queue (or still the play screen if
            # the transition lags). Scope it; the floor still catches an error dialog.
            self._expected_next = {ScreenState.PLAY_SCREEN, ScreenState.QUEUE}
        elif st == ScreenState.QUEUE:
            # Queue controls can cancel search.  Once it is positively named, capture
            # and scoped classification are the sole cadence until a class resolves;
            # a slow search is allowed to run indefinitely until Stop is requested.
            self._begin_match_watch(source="queue")
        elif st == ScreenState.COLLECTION:
            # hop is never meant to be here; a stray navigation got us in. Back out to
            # the deck list (a home screen) rather than halt. See ScreenState.COLLECTION.
            if self.debug:
                self.debug.record("collection_backout")
            self._tap_dispatch_button(self.layout.collection_back, expected_change="full_transition",
                                      what="collection_back", label="the Collection back arrow")
        elif st == ScreenState.MENU:
            raise Halt("at the Hearthstone main menu; open Play and select a deck first "
                       "(the hunt loop queues from that deck's Play screen)")
        elif st == ScreenState.VS_SPLASH:
            self._begin_match_watch(source="vs_splash")
        elif st == ScreenState.MULLIGAN:
            self._gate_mulligan_class(frame)
        elif st in (ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.REWARDS,
                    ScreenState.RANK_PROGRESS, ScreenState.QUEST_POPUP):
            self._clear_end_screens()
        elif st == ScreenState.CONCEDE_WARNING:
            # A Black Market warning is only safe to tap when `_post_concede_clickthrough`
            # already owns the just-rejected game.  A generic dispatch could be observing a
            # user's in-progress board, so never turn this named modal into a blind action.
            raise Halt("early-concede warning outside the owned post-concede trace")
        elif st == ScreenState.CONCEDE_MENU:
            # commit, not reject: the decision to concede was already made; tapping the
            # Concede button is executing it, not deliberating it again (see _concede).
            attempt = self._fixed_attempt("concede", self.cfg.vision.concede_tap_attempts)
            self._tap_fixed(self.layout.concede_button, committing=True, decision_type="commit",
                            what="concede")
            self._semantic_cooldown(self.cfg.timing.concede_resolve_cooldown_s,
                                    "concede_cooldown", attempt=attempt)
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
            self._sleep_for(timing.read_consider(self.rng, self.cfg.timing, self.state),
                            "in_game_read", poll=self._in_game_polls)
            # still a board next iteration (or the end of a concede animation, which the
            # floor's victory/defeat catches); scope the top-loop look.
            self._expected_next = {ScreenState.IN_GAME}
            return
        if not self._own_game:
            raise Halt("a game is in progress that this hunt did not start; refusing to "
                       "concede it. Finish the game, or stop the hunt and restart it "
                       "from the deck's Play screen.")
        if self.debug:
            self.debug.record("abandoning_unread_game", polls=self._in_game_polls)
        # we never got to read this mulligan, so we never got to hesitate over it
        self._sleep_for(
            timing.think_time(self.rng, "reject", self.cfg.timing, self.state,
                              visual_complexity=0.6),
            "unread_game_concede_think")
        conceded = self._concede(known_state=ScreenState.IN_GAME)
        self._clear_end_screens()
        self._book_game(conceded)

    def _decline_incomplete_deck(self) -> None:
        """Tap 'No' on 'Complete deck automatically?' - never 'Yes'.

        Auto-completing a deck spends the user's dust/cards on cards Hearthstone picks;
        hop must never do that. Tap No and wait for the dialog to leave (an UNKNOWN frame
        is never proof it left - see :meth:`_wait_until_screen_leaves`).

        'No' can be silently dropped like any other button tap. Retried, and ONLY while
        positively still on ``INCOMPLETE_DECK`` -- never a blind tap once the dialog
        might already be gone (that could land on whatever screen follows, including a
        live deck grid). Unlike Concede's board, this dialog sits over a STATIC deck-
        list background with no ambient animation to mask a drop, so a genuinely
        dropped tap here shows literally zero pixel change and `_tap` itself raises
        ``Halt.NO_CHANGE`` immediately -- that is caught below and treated as
        equally-positive proof of "still stuck" as a fresh classify would give (indeed
        stronger: it is a pixel-level fact, not a template match), so a masked drop
        (`_wait_until` classifies afresh) and an unmasked one (`_tap` itself objects)
        both feed the SAME retry decision. This dialog has no roping-opponent-style
        reason to wait longer than the generic ``screen_wait_*`` budget, so it costs
        about the same worst case as Concede's retry.
        """
        arrivals = {ScreenState.DECK_SELECT}
        attempts = max(1, self.cfg.vision.deck_decline_tap_attempts)
        cls = frame = None
        taps_sent = 0
        for attempt in range(attempts):
            taps_sent = attempt + 1
            dropped = False
            try:
                self._tap(self.layout.deck_decline, committing=False, decision_type="commit",
                          expected_change=None, allow_correction=False, what="deck_decline")
            except Halt as e:
                if e.kind != Halt.NO_CHANGE:
                    raise
                dropped = True
            if not dropped:
                ok, cls, frame = self._wait_until(
                    lambda c, _f: c.state not in (ScreenState.INCOMPLETE_DECK, ScreenState.UNKNOWN),
                    what="deck_decline",
                    expected=frozenset({ScreenState.INCOMPLETE_DECK}) | frozenset(arrivals))
                if ok:
                    return
                dropped = cls.state == ScreenState.INCOMPLETE_DECK
            if not dropped or attempt + 1 == attempts:
                break
            if self.debug:
                self.debug.record("deck_decline_tap_ignored", attempt=attempt + 1, of=attempts)
            self._sleep_for(timing.human_delay(self.rng, 0.7, self.cfg.timing),
                            "deck_decline_retry", attempt=attempt + 1)
        # `dropped` here means "the loop stopped because it ran positively-still-stuck taps
        # up to `attempts`" -- a real dropped-committing-tap story, distinct from breaking
        # EARLY (after fewer than `attempts` taps) because the look came back UNKNOWN, which
        # is NOT proof of a drop (it may be a transition we can't yet name, or something
        # else entirely) and must not be reported as one -- see the mis-blamed exhaustion
        # message this replaced, which always claimed "not registering" regardless of why
        # the loop actually stopped.
        if cls is not None and cls.state == ScreenState.UNKNOWN:
            self._record_unknown(frame, where="deck_decline", confidence=round(cls.confidence, 3),
                                 waiting_to_leave=ScreenState.INCOMPLETE_DECK.value)
            raise Halt(
                f"the 'Complete deck automatically?' dialog did not close after {taps_sent} "
                "tap(s) of No, and the screen since could not be identified -- this may "
                "not be a dropped tap; see the kept frame for what followed.")
        raise Halt(
            f"the 'Complete deck automatically?' dialog did not close after {taps_sent} "
            "tap(s) of No; the dialog is still up, so the tap is not registering (a "
            "dropped committing tap, not a wrong coordinate). Only No's own coordinate "
            "was ever tapped.")

    def _book_game(self, conceded: bool) -> None:
        """Count a finished game, then pace the requeue."""
        self.limiter.register_game()
        self.stats.games += 1
        # A game that ended on its own was not conceded. `concedes` is the numerator of
        # the concede/commit ratio surfaced in status - do not inflate it.
        self.stats.concedes += int(conceded)
        self._own_game = False
        self._in_game_polls = 0

        # The direct reject path has already performed its post-concede queue boundary;
        # other paths reuse their next semantic read rather than adding a stale requeue sleep.
        if self.debug:
            self.debug.record("requeue_deferred", paced_by="next_semantic_capture")

    def _reconnect(self) -> None:
        """Tap Reconnect once, then wait out the asynchronous reconnect.

        Hearthstone can shut down idle connections during the loop's calibrated waits,
        so this is expected traffic rather than an anomaly.

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

        try:
            self._tap(self.layout.reconnect_button, committing=False,
                      decision_type="commit", expected_change=None, allow_correction=False,
                      what="reconnect")
        except Halt as e:
            # The Reconnect BUTTON tap itself can be silently dropped, same as any other
            # button (see `_tap_play`) -- a different failure from the async reconnect
            # simply timing out AFTER a tap that took (the poll loop below). `cap`/
            # `_reconnect_attempts` above ALREADY bound how many times this function may
            # be re-entered -- the docstring's "let the outer loop re-enter _reconnect()"
            # is the SAME mechanism the RECONNECT_DIALOG-persists case already relies on
            # -- so a dropped tap just falls into that existing bounded retry instead of
            # needing a second, nested budget of its own.
            if e.kind != Halt.NO_CHANGE:
                raise
            if self.debug:
                self.debug.record("reconnect_tap_ignored", attempt=self._reconnect_attempts)
            return

        for _ in range(max(1, self.cfg.vision.reconnecting_wait_attempts)):
            # still-reconnecting polls resolve on the floor's own dialog anchors; a return
            # to any real screen isn't in scope, so it full-scans and is named correctly.
            cls, _frame = self._classify(
                {ScreenState.RECONNECTING, ScreenState.RECONNECT_DIALOG})
            if cls.state == ScreenState.UNKNOWN:
                # Mid-reconnect the client redraws; an unreadable frame is not
                # evidence the reconnect resolved. Keep waiting (never tapping)
                # rather than book a success we did not observe.
                self._sleep_for(timing.human_delay(self.rng, 1.5, self.cfg.timing),
                                "reconnect_unknown_wait")
                continue
            if cls.state not in (ScreenState.RECONNECTING, ScreenState.RECONNECT_DIALOG):
                # This disconnect episode is over. The cap bounds *consecutive* failed
                # attempts within one episode (that is what "failed to reconnect after N
                # attempts; needs a human" means), so a success must zero the counter -
                # otherwise the lifetime tally of the idle disconnects this loop's own
                # pacing provokes (each individually recovered) trips the cap and falsely
                # halts a long, healthy hunt on its 4th reconnect.
                self._reconnect_attempts = 0
                if self.debug:
                    self.debug.record("reconnect_resolved", state=cls.state.value)
                return
            if cls.state == ScreenState.RECONNECT_DIALOG:
                # The attempt finished and failed: the buttons are back. Let the
                # outer loop re-enter _reconnect(), which re-checks the cap.
                return
            self._sleep_for(timing.human_delay(self.rng, 1.5, self.cfg.timing),
                            "reconnect_poll")
        raise Halt("stuck on 'Reconnecting...'; Hearthstone never came back online")

    def _begin_match_watch(self, *, source: str) -> None:
        """Enter the capture-only search/nameplate observation interval once.

        The paired journal events deliberately bound timing-report attribution: the
        long search's captures are matchmaking observation, never reaction time for
        the eventual action after a class appears.
        """
        if self._awaiting_match_class:
            return
        self._awaiting_match_class = True
        if self.debug:
            self.debug.record("match_watch_start", source=source)

    def _end_match_watch(self, *, outcome: str, state: str | None = None) -> None:
        """Close a started match-watch interval once, for report attribution."""
        if not self._awaiting_match_class:
            return
        self._awaiting_match_class = False
        if self.debug:
            detail = {"outcome": outcome}
            if state is not None:
                detail["state"] = state
            self.debug.record(f"match_watch_{outcome}", **detail)

    def _gate_mulligan_class(self, frame: Frame) -> None:
        """Read the class once from this exact named-mulligan frame.

        The nameplate can arrive after the rest of the mulligan.  Until it does,
        this is still matchmaking observation: do not count, concede, or re-read
        inside the iteration.  The next outer iteration supplies the next capture.
        """
        t0 = self.clock()
        read = self._read_mulligan(frame, self.layout, self.reader, self.cfg.vision)
        ocr_ms = round((self.clock() - t0) * 1000.0)
        self._handle_mulligan(frame, initial_read=read, initial_ocr_ms=ocr_ms)

    def _handle_mulligan(self, frame: Frame, *, initial_read: MulliganRead | None = None,
                         initial_ocr_ms: int | None = None) -> None:
        """Decide a named mulligan only after its opponent class is readable.

        ``initial_read`` lets the match-watch hand the exact frame's one OCR result
        into normal decision logic without taking a second capture or OCR pass.
        """
        if initial_read is None:
            t0 = self.clock()
            read = self._read_mulligan(frame, self.layout, self.reader, self.cfg.vision)
            ocr_ms = round((self.clock() - t0) * 1000.0)
        else:
            read = initial_read
            ocr_ms = 0 if initial_ocr_ms is None else initial_ocr_ms
        observed_name = (DISPLAY_NAMES.get(read.opponent_class, "?")
                         if read.opponent_class else "?")
        # Decide before journalling so a future report can distinguish a true
        # unreadable mulligan from the intentional class-only path.  A 0-card
        # read means ``we_go_second=False`` only as an implementation default;
        # never publish that as a real "going 1st" observation.
        decision = evaluate_matchup(read, self.cfg)
        turn_known = read.num_cards in (3, 4)
        observed_second = read.we_go_second if turn_known else None
        if self.debug:
            # Journal contract: ``pending=True`` means this is an observation-only
            # match-watch frame, not a game decision. It retains OCR timing and
            # diagnostics, but report consumers must not tally or pair it. Old lines
            # without this field predate the watch and retain their old semantics.
            self.debug.record("mulligan_read", opponent=observed_name,
                              second=observed_second, turn_known=turn_known,
                              decision=decision, cards=read.num_cards,
                              conf=round(read.class_confidence, 3), method=read.method,
                              class_raw=read.class_raw, ms=ocr_ms,
                              pending=read.opponent_class is None)

        # A blank/garbled class is not a rejected matchup.  It is a not-yet-ready
        # nameplate: preserve this exact frame's one read, then let the outer watch
        # capture again with no artificial delay.  Crucially, nothing below may book
        # statistics or send a concede until a class exists.
        if read.opponent_class is None:
            self._begin_match_watch(source="blank_mulligan")
            if self.debug:
                self.debug.record("mulligan_class_pending", class_raw=read.class_raw,
                                  cards=read.num_cards, conf=round(read.class_confidence, 3))
            return
        if self._awaiting_match_class:
            self._end_match_watch(outcome="resolved", state="mulligan")
        # A pending blank nameplate is not a perception result we will act on.
        # Only the resolved matchup may affect the next action's humanization.
        self.state.observe_confidence(read.class_confidence)

        if not turn_known:
            self._record_unreadable_mulligan_card_count(frame, read, decision)
        if decision == "unusable":
            # The class is known, but the criteria require a coin/turn result
            # that this frame cannot establish.  Do not concede a possible target.
            raise Halt(f"could not read mulligan (class={read.class_raw!r}, "
                       f"cards={read.num_cards})")

        # session stats (once per game, on the resolved read): the observed class
        # distribution and the coin split the dashboard charts.
        self.stats.last_opponent = observed_name
        self.stats.class_distribution[observed_name] = (
            self.stats.class_distribution.get(observed_name, 0) + 1)
        if turn_known:
            if read.we_go_second:
                self.stats.going_second += 1
            else:
                self.stats.going_first += 1

        # A read the engine ACTS ON but only barely resolved (see _LOW_CONFIDENCE_KEEP): keep its
        # pixels + raw text so a silent misread -- a garbled two-word label snapping to a
        # valid-but-wrong class -- stays diagnosable instead of vanishing with the deleted frame.
        if decision != "unusable" and read.class_confidence < _LOW_CONFIDENCE_KEEP:
            self._record_low_confidence_mulligan(frame, read)

        if decision == "keep":
            self.stats.target_found = True
            # the hunt stops on a target, so `concedes` right now IS "concedes until target".
            self.stats.concedes_until_target = self.stats.concedes
            self._alert_target(read)
            return
        self._execute_reject(read)

    def _record_unreadable_mulligan_card_count(
        self,
        frame: Frame,
        read: MulliganRead,
        decision: str,
    ) -> None:
        """Retain colour evidence when the green-glow card counter cannot read.

        This is intentionally separate from an unknown-screen capture: the
        mulligan anchor *did* identify the screen, while the RGB-only keep-glow
        signal failed.  The screenshot and its structured strip/interior data
        let the bug-report harness distinguish a transient incomplete deal from
        a threshold, geometry, or marked-hand failure.  A class-only decision is
        safe when the criteria do not use the coin *or* the class is already
        deterministically rejected; otherwise the same evidence accompanies the
        retained fail-closed halt.
        """
        if self.debug is None:
            return
        try:
            evidence = mulligan_card_count_diagnostics(frame, self.layout, self.cfg.vision)
        except Exception:
            # Logging must never turn a recoverable observation into a crash.
            evidence = {"rejection": "diagnostics_failed"}
        context = {
            "opponent": DISPLAY_NAMES.get(read.opponent_class, "?"),
            "class_raw": read.class_raw,
            "cards": read.num_cards,
            "require_second": self.cfg.criteria.require_second,
            "decision": decision,
            "mulligan_card_count": evidence,
        }
        self.debug.record("mulligan_card_count_unreadable", **context)
        anomaly = getattr(self.debug, "anomaly", None)
        if callable(anomaly):
            try:
                # The detector is defined by a green hue.  A grayscale anomaly
                # would preserve the wrong evidence, so request the bounded RGB
                # variant introduced specifically for colour-dependent signals.
                anomaly("mulligan card count unreadable", before=frame,
                        colour=True, terminal=decision == "unusable", **context)
            except Exception:
                pass

    def _record_low_confidence_mulligan(self, frame: Frame, read: MulliganRead) -> None:
        """Preserve a RESOLVED-but-shaky opponent-class read (acted on, yet >=2 glyph errors).

        This fires when the class DID read -- to a real class -- but only barely, the
        silent-misread case. It journals the class-region pixel stats and likely
        nearest class, then saves the frame (bounded via anomaly()).
        """
        if self.debug is None:
            return
        stats = _region_gray_stats(frame, self.layout.opponent_class_region)
        near_name, near_dist = "", None
        raw = (read.class_raw or "").strip()
        if raw:
            try:
                near, near_dist = nearest_class(raw)
                near_name = DISPLAY_NAMES.get(near, "") if near else ""
            except Exception:
                near_dist = None
        snapped = DISPLAY_NAMES.get(read.opponent_class, "?") if read.opponent_class else "?"
        self.debug.record("mulligan_low_confidence", class_raw=read.class_raw, snapped=snapped,
                          conf=round(read.class_confidence, 3), cards=read.num_cards,
                          second=read.we_go_second, nearest=near_name, nearest_dist=near_dist,
                          region_gray=stats)
        try:
            self.debug.anomaly("mulligan class read at low confidence", before=frame,
                               class_raw=read.class_raw, snapped=snapped,
                               conf=round(read.class_confidence, 3), region_gray=stats)
        except Exception:
            pass

    # ── rejected-mulligan exit ───────────────────────────────────────────────

    def _post_concede_clickthrough(self, *, source: ScreenState) -> Classification:
        """Run the fixed-phone, open-loop reject exit through the next Play tap.

        This is deliberately a *single bounded action trace*, not a post-game
        screen classifier.  On this calibrated phone the post-game stack accepts
        the deck Play location as its advance control; once matchmaking begins,
        further taps there do not cancel it.  Retained journals put the fastest
        observed Play-to-Mulligan at 30.270 s.  The 20-tap burst has only 19
        0.75--0.90 s gaps and a validated hard deadline for *starting* taps. The
        one semantic capture happens as soon as the boundary is due from the
        *first* Play tap, before a fast successor can reach its mulligan unseen
        and before the main loop is allowed to act again.

        ``source`` is named by the caller, so no speculative source screenshot is
        required.  This is used only for the ordinary rejected-mulligan path;
        exceptional recovery keeps :meth:`_clear_end_screens`' closed loop.
        """
        if source not in (ScreenState.MULLIGAN, ScreenState.IN_GAME):
            raise Halt(f"cannot start post-concede click-through from {source.value!r}")

        # A previous exceptional branch can have left a source-local retry key.
        # This is a new named phase, not evidence that another Concede was ignored.
        self._clear_fixed_attempt("concede")
        self._clear_fixed_attempt("play")
        for key in tuple(self._fixed_tap_attempts):
            if key.startswith("end_dismiss:"):
                self._clear_fixed_attempt(key)

        self._tap_fixed(self.layout.gear_button, committing=False,
                        decision_type="commit", what="gear")
        self._ui_open_delay(self.cfg.timing.gear_menu_cooldown_s,
                            "gear_menu_cooldown", fast_path=True)

        # A dropped Concede must not leave the Game Menu in place.  Send its known
        # top-entry coordinate a bounded number of times before the end-stack burst.
        # Additional taps after a successful concede are harmless upper-screen taps;
        # no point below Concede (Options/Quit) is ever used.
        concede_count = max(1, self.cfg.vision.concede_tap_attempts)
        self._tap_fixed(self.layout.concede_button, committing=True,
                        decision_type="commit", what="concede")
        for tap in range(1, concede_count):
            self._burst_sleep("concede_retry_cadence", tap=tap + 1,
                              of=concede_count)
            self._tap_open_loop(self.layout.concede_button, committing=True,
                                what="concede")

        # Let the game finish rendering the Black Market warning before its one
        # unconditional normal-path tap.  The small fixed opening allowance avoids
        # racing a modal that has not drawn yet, without making the interaction feel
        # artificially slow.
        self._ui_open_delay(self.cfg.timing.post_concede_start_cooldown_s,
                            "post_concede_start_cooldown")
        concede_now_count = 1
        self._tap_open_loop(self.layout.concede_now_button, committing=True,
                            what="concede_now")

        cls, frame, emitted, elapsed = self._post_concede_burst_and_boundary(
            attempt=0, recovery="normal")
        initial_state: ScreenState | None = None
        boundary_attempt, boundary_recovery = 0, "normal"
        if cls.state in (ScreenState.CONCEDE_WARNING, ScreenState.CONCEDE_MENU):
            # The single boundary capture has proved an overlay is still present.
            # A warning has a stable, named Concede Now point; the menu is deliberately
            # ambiguous because an unanchored warning can still expose and match its
            # Game Menu title.  Both recoveries are bounded once and make no attempt
            # to act on an arbitrary normal-dispatch screen.
            initial_state = cls.state
            if cls.state == ScreenState.CONCEDE_WARNING:
                concede_now_count += 1
                self._tap_open_loop(self.layout.concede_now_button, committing=True,
                                    what="concede_now_recovery")
                recovery = "warning"
            else:
                concede_count += 1
                self._tap_open_loop(self.layout.concede_button, committing=True,
                                    what="concede_recovery")
                self._ui_open_delay(self.cfg.timing.post_concede_start_cooldown_s,
                                    "post_concede_start_cooldown", recovery="menu")
                concede_now_count += 1
                self._tap_open_loop(self.layout.concede_now_button, committing=True,
                                    what="concede_now_recovery")
                recovery = "menu"
            cls, frame, emitted, elapsed = self._post_concede_burst_and_boundary(
                attempt=1, recovery=recovery)
            boundary_attempt, boundary_recovery = 1, recovery
            if cls.state in (ScreenState.CONCEDE_WARNING, ScreenState.CONCEDE_MENU):
                self._post_concede_boundary_halt(
                    cls, frame, attempt=1, recovery=recovery, emitted=emitted,
                    elapsed=elapsed, concede_count=concede_count,
                    concede_now_count=concede_now_count, source=source,
                    initial_state=initial_state)
        if cls.state not in self._post_concede_accepted_states():
            self._post_concede_boundary_halt(
                cls, frame, attempt=boundary_attempt, recovery=boundary_recovery, emitted=emitted,
                elapsed=elapsed, concede_count=concede_count,
                concede_now_count=concede_now_count, source=source,
                initial_state=initial_state)
        # UNKNOWN has already armed the capture-only match watch above.  It must
        # never be deferred: a later named interruption ends that watch, and a
        # stale UNKNOWN would otherwise be consumed as if it were a fresh screen
        # read rather than taking the next required capture.
        if cls.state == ScreenState.UNKNOWN:
            self._deferred_screen = None
        else:
            # Reuse the named phase boundary in the top loop: one capture, never a
            # duplicate "what screen are we on?" screenshot immediately after burst.
            self._deferred_screen = (cls, frame)
        self._clear_fixed_attempt("concede")
        return cls

    @staticmethod
    def _post_concede_accepted_states() -> frozenset[ScreenState]:
        return frozenset({
            ScreenState.PLAY_SCREEN, ScreenState.QUEUE, ScreenState.VS_SPLASH,
            ScreenState.MULLIGAN, ScreenState.ERROR_DIALOG, ScreenState.DECK_SELECT,
            # An immediate boundary can still see the named end stack.  Deferring
            # this exact frame lets ordinary dispatch clear it with its own safe,
            # screen-specific controls; do not keep blindly hitting Play until it
            # happens to disappear.
            ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.REWARDS,
            ScreenState.RANK_PROGRESS, ScreenState.QUEST_POPUP,
            ScreenState.UNKNOWN,
        })

    def _post_concede_burst_and_boundary(self, *, attempt: int,
                                          recovery: str) -> tuple[Classification, Frame, int, float]:
        """One bounded no-touch-before-boundary Play burst.

        The only caller is the owned reject exit.  Keeping its burst and boundary in
        one helper makes the warning/menu recovery auditable: each attempt has the
        same deadline, boundary due time, and exactly one capture.
        """
        click_count = self.cfg.timing.post_concede_click_count
        started = self.clock()  # immediately before the first Play-location tap
        deadline = started + self.cfg.timing.post_concede_burst_max_s
        emitted = 0
        while emitted < click_count:
            if self._stop:
                raise StopRequested()
            # Input emission itself can be slow.  Never start a further tap after
            # its calibrated wall-clock budget has expired.
            if deadline - self.clock() <= 1e-9:
                break
            self._tap_open_loop(self.layout.play_button, committing=False,
                                what="post_concede_play")
            emitted += 1
            # There are at most N-1 gaps, and none after the final emitted tap.
            if emitted >= click_count:
                break
            interval = self._burst_interval()
            if self._stop:
                raise StopRequested()
            # Do not start a sleep which cannot lead to another in-budget tap.
            if deadline - self.clock() <= interval + 1e-9:
                break
            self._sleep_for(interval, "post_concede_click_cadence",
                            min_s=self.cfg.timing.post_concede_click_interval_s,
                            max_s=self.cfg.timing.post_concede_click_interval_max_s,
                            tap=emitted, of=click_count - 1)
            if self._stop:
                raise StopRequested()
        if self.debug:
            self.debug.record("post_concede_burst_complete", emitted=emitted,
                              configured=click_count,
                              elapsed_s=round(max(0.0, self.clock() - started), 3),
                              max_s=self.cfg.timing.post_concede_burst_max_s,
                              attempt=attempt, recovery=recovery)

        # The old implementation slept a *second*, full queue cooldown after a
        # nearly-20-second Play burst.  That put its sole capture roughly 53 s
        # after the first Play tap even though this device can reach a successor
        # mulligan in 30.270 s.  The successor's class could therefore pass unseen
        # before the engine stopped tapping.  The cooldown is a due time from the
        # first Play tap, so it overlaps the burst and only its unspent remainder
        # is slept.  Do not humanize this remainder upward: crossing the observed
        # mulligan floor is a safety failure, not a cosmetic pacing choice.
        elapsed_before_boundary = max(0.0, self.clock() - started)
        boundary_due = float(self.cfg.timing.post_concede_queue_cooldown_s)
        remaining = max(0.0, boundary_due - elapsed_before_boundary)
        if remaining > 0:
            self._sleep_for(remaining, "post_concede_queue_cooldown",
                            anchor_s=boundary_due,
                            elapsed_in_burst_s=round(elapsed_before_boundary, 3),
                            overlapped=True)
        accepted = self._post_concede_accepted_states()
        cls, frame = self._classify(accepted)
        elapsed = max(0.0, self.clock() - started)
        # `classify_expected` returns an expected state directly; every other
        # outcome falls back to a full scan of this exact frame.  The distinction
        # matters when a terminal boundary is diagnosed from its saved image.
        scan = "scoped" if cls.state in accepted else "full_fallback"
        floor = float(self.cfg.timing.play_to_mulligan_observed_min_s)
        boundary_evidence = {
            "mulligan_floor_s": round(floor, 3),
            "mulligan_deadline_exceeded": elapsed >= floor,
            "classification_scan": scan,
            "anchor_at": list(cls.at) if cls.at is not None else None,
        }
        if self.debug:
            self.debug.record("post_concede_boundary", state=cls.state.value,
                              confidence=round(cls.confidence, 3), attempt=attempt,
                              recovery=recovery, emitted=emitted,
                              configured=click_count, elapsed_s=round(elapsed, 3),
                              accepted_states=sorted(st.value for st in accepted),
                              **boundary_evidence)
        if cls.state == ScreenState.UNKNOWN:
            # The post-burst boundary can land in an unanchored queue->match transition.
            # This is a hands-off search observation, not evidence of a bad reject
            # journey: arm the raw capture watch and let its next outer iteration
            # continue through QUEUE/VS/UNKNOWN/blank-MULLIGAN until a class resolves.
            # Do not defer UNKNOWN into normal dispatch, which correctly fails closed
            # outside this deliberately no-touch path.
            self._begin_match_watch(source="post_concede_boundary")
            self._clear_fixed_attempt("concede")
        return cls, frame, emitted, elapsed

    def _post_concede_boundary_halt(self, cls: Classification, frame: Frame, *, attempt: int,
                                    recovery: str, emitted: int, elapsed: float,
                                    concede_count: int, concede_now_count: int,
                                    source: ScreenState,
                                    initial_state: ScreenState | None = None) -> None:
        """Persist the already-captured terminal boundary, then halt fail-closed."""
        context = {
            "state": cls.state.value, "confidence": round(cls.confidence, 3),
            "attempt": attempt, "recovery": recovery, "emitted": emitted,
            "configured": self.cfg.timing.post_concede_click_count,
            "elapsed_s": round(elapsed, 3), "concede_taps": concede_count,
            "concede_now_taps": concede_now_count,
            "source": source.value,
            "accepted_states": sorted(st.value for st in self._post_concede_accepted_states()),
            "transport": _transport_snapshot(self.backend),
            # Keep this terminal record self-contained.  The preceding boundary
            # line has the same fields, but a size-capped report may retain only
            # this causal context.
            "mulligan_floor_s": round(float(self.cfg.timing.play_to_mulligan_observed_min_s), 3),
            "mulligan_deadline_exceeded": elapsed >= float(
                self.cfg.timing.play_to_mulligan_observed_min_s),
            "classification_scan": (
                "scoped" if cls.state in self._post_concede_accepted_states()
                else "full_fallback"),
            "anchor_at": list(cls.at) if cls.at is not None else None,
        }
        if initial_state is not None:
            context["initial_state"] = initial_state.value
        if self.debug:
            self.debug.record("post_concede_unexpected_boundary", **context)
            terminal = getattr(self.debug, "terminal_screen", None)
            if callable(terminal):
                terminal("unexpected post-concede boundary", frame, **context)
        raise Halt(f"unexpected {cls.state.value!r} after post-concede click-through")

    def _burst_interval(self) -> float:
        """Draw one deterministically seeded, hard-bounded burst interval."""
        low = self.cfg.timing.post_concede_click_interval_s
        high = self.cfg.timing.post_concede_click_interval_max_s
        return self.rng.uniform(low, high)

    def _burst_sleep(self, reason: str, **detail) -> None:
        """Sleep one deterministic bounded interval outside the Play burst."""
        if self._stop:
            raise StopRequested()
        low = self.cfg.timing.post_concede_click_interval_s
        high = self.cfg.timing.post_concede_click_interval_max_s
        self._sleep_for(self._burst_interval(), reason, min_s=low, max_s=high, **detail)

    def _execute_reject(self, read: MulliganRead) -> None:
        plan = journey.plan_reject(self.rng, read.num_cards)
        if self.debug:
            self.debug.record("reject_plan", concede_point=plan.concede_point,
                              hesitate=plan.hesitate_before_concede,
                              extra_reads=plan.extra_reads,
                              replace_slots=[d.slot for d in plan.mulligan if d.replace])
        # The rejected matchup is already a positively named mulligan.  Do not spend
        # opponent-dependent Confirm/card actions or classify every end screen: the
        # fixed phone's bounded post-concede click-through reaches Play directly.
        boundary = self._post_concede_clickthrough(source=ScreenState.MULLIGAN)
        self._book_game(conceded=True)
        # `_book_game` closes ownership of the just-finished game. Only a named
        # queue/VS/mulligan proves the burst actually started its successor; an
        # UNKNOWN boundary is watched hands-off but leaves ownership false until a
        # successor is positively named. Play and error boundaries likewise remain
        # false until their normal dispatch.
        self._own_game = boundary.state in {
            ScreenState.QUEUE, ScreenState.VS_SPLASH, ScreenState.MULLIGAN,
        }

    def _confirm_mulligan(self) -> Classification:
        """Tap Confirm, then wait for the mulligan to actually go away.

        Confirming is **asynchronous**: the cards fly off and the board draws in over
        a second or more. Within the tap's settle window only the bottom of the
        screen has moved, so demanding ``full_transition`` fails on a confirm that
        worked (measured: ``bottom_sheet``). And `_tap`'s own single-correction retap
        would fire *after* the mulligan is already confirmed - a blind tap into a live
        game, which is precisely what closed-loop navigation exists to prevent -- so
        ``allow_correction=False``, same as `_concede`.

        So: require only that something changed, never correct inside `_tap`, and then
        verify the real semantic end-state - that we are no longer on the mulligan - by
        looking.

        And *looking* means seeing a screen we can name. The old loop accepted
        ``cls.state != MULLIGAN``, which an UNKNOWN frame satisfies - so a single
        unreadable frame (of which a confirm animation produces several) was taken as
        proof the mulligan was gone, and ``_play_beats``/``_concede`` then tapped on
        that premise. UNKNOWN is not an observation; it is the absence of one.

        Confirm can silently drop, so it is retried only after a full calibrated
        semantic wait positively names ``MULLIGAN`` again. UNKNOWN is never a retry
        signal: it may simply be the opponent's unresolved mulligan. The configured
        resolve budget remains long enough for that opponent-side delay, while each
        successful named boundary ends the loop without a duplicate capture.
        """
        # the board draws in (in_game); an opponent who concedes in the window lands us on
        # an end banner. mulligan (the from-state) is kept in scope so a cross-fade where
        # both clear resolves exactly as a full classify would.
        arrivals = {ScreenState.IN_GAME, ScreenState.VICTORY, ScreenState.DEFEAT}
        attempts = max(1, self.cfg.vision.mulligan_confirm_tap_attempts)
        cls = frame = None
        taps_sent = 0
        for attempt in range(attempts):
            taps_sent = attempt + 1
            self._tap_fixed(self.layout.mulligan_confirm, committing=False,
                            decision_type="commit", what="mulligan_confirm")
            ok, cls, frame = self._wait_semantic(
                lambda c, _f: c.state not in (ScreenState.MULLIGAN, ScreenState.UNKNOWN),
                what="mulligan_confirm",
                expected=frozenset({ScreenState.MULLIGAN}) | frozenset(arrivals),
                cooldown_s=self.cfg.timing.mulligan_confirm_cooldown_s,
                attempts=min(self.cfg.vision.mulligan_resolve_attempts,
                             max(1, round(self.cfg.vision.mulligan_resolve_timeout_s /
                                          self.cfg.timing.mulligan_confirm_cooldown_s))))
            if ok:
                return cls
            # A named mulligan persisting through the wait is the only safe retry
            # condition.  UNKNOWN never proves that the button was ignored.
            dropped = cls.state == ScreenState.MULLIGAN
            if not dropped or attempt + 1 == attempts:
                break
            if self.debug:
                self.debug.record("mulligan_confirm_tap_ignored", attempt=attempt + 1, of=attempts)
            self._sleep_for(timing.human_delay(self.rng, 0.7, self.cfg.timing),
                            "mulligan_confirm_retry", attempt=attempt + 1)
        # `dropped` here means the loop stopped because MULLIGAN stayed positively up for
        # `attempts` full-budget tries -- a genuine dropped-committing-tap story. Breaking
        # EARLY (fewer than `attempts` taps) because the look came back UNKNOWN is NOT that
        # story -- it may be the opponent's rope still unresolved, or a frame we simply
        # cannot name -- and must not be reported as a drop (this is the single most
        # safety-critical distinction here: a false "dropped tap" diagnosis on a live
        # roping opponent would point a human at the wrong knob entirely).
        if cls is not None and cls.state == ScreenState.UNKNOWN:
            self._record_unknown(frame, where="mulligan_confirm", confidence=round(cls.confidence, 3),
                                 waiting_to_leave=ScreenState.MULLIGAN.value)
            raise Halt(
                f"mulligan Confirm did not dismiss the mulligan after {taps_sent} tap(s), "
                "and the screen since could not be identified -- this may not be a "
                "dropped tap (a roping opponent reads the same way); see the kept frame "
                "for what followed.")
        raise Halt(
            f"mulligan Confirm did not dismiss the mulligan after {taps_sent} tap(s); the "
            "mulligan is still up, so the tap is not registering (a dropped committing "
            "tap, not a wrong coordinate). Only Confirm's own coordinate was ever tapped.")

    def _replace_card(self, slot: int, center_xf: float, decision_type: str) -> bool:
        """Mark one mulligan card for replacement. Returns whether it took.

        Two things make this unlike every other tap.

        **It changes only that card.** Marking it redraws the card and nothing else:
        measured whole-frame mean-abs-diff 4.80 against a 9.0 threshold, versus 24.24
        inside the card's own rectangle. So verification is scoped to the pixels the
        tap was aimed at - the honest question, and a stricter one, because a tap that
        misses the card leaves the card unchanged and still fails.

        **Hearthstone accepts it only intermittently** - about 1 tap in 3, while
        accepting every button tap. The cause is *unknown*; suspected factors include
        contact scale, position, dwell,
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
            # The mulligan choice was deliberated once (plan_reject); attempt 0 reads and
            # rejects the card at that pace. A retry is NOT a fresh decision - the game
            # dropped the input and we are re-sending the same, already-made choice - so it
            # must not re-pay the slow `reject` deliberation (median ~2.7 s). A human whose
            # card-tap didn't register re-taps quickly; that is a `commit` (fast reaction,
            # median ~1.7 s), not another `reject`. Same attempt count and same ignored-tap
            # accounting. The RNG *draw stream* stays bit-identical (think_time draws one
            # normal and state.tick the same draws regardless of decision_type), but this
            # is a deliberate behavior change, not a no-op: a retry now ticks HumanState
            # with the shorter commit `dt` and records last_action="commit", so the human
            # trajectory diverges from the retry onward -- the intended, more-human shape
            # (see the delay-report analysis of the card-retry loop).
            tap_decision = decision_type if attempt == 0 else "commit"
            try:
                self._tap(point, committing=False, decision_type=tap_decision,
                          visual_complexity=0.5, verify_region=region,
                          allow_correction=False, what=f"mulligan_card[{slot}]")
                return True
            except Halt as e:
                if e.kind != Halt.NO_CHANGE:
                    raise
                if self.debug:
                    self.debug.record("mulligan_card_tap_ignored",
                                      slot=slot, attempt=attempt + 1, of=attempts)
                self._sleep_for(timing.human_delay(self.rng, 0.7, self.cfg.timing),
                                "mulligan_card_retry", slot=slot, attempt=attempt + 1)
        self.stats.ignored_card_taps += 1
        return False

    def _play_beats(self, plan: journey.RejectPlan) -> None:
        """A few plausible 'reading the board / passing turn' beats before bailing.

        Precisely tracking whose turn it is would need more templates; for a game
        we're about to concede, a bounded set of reading pauses plus an occasional
        end-turn tap is a sufficiently human exit. Bounded so it can't spin.
        """
        beats = 1 if plan.concede_point == "turn1" else 2
        for i in range(beats + plan.extra_reads):
            self._sleep_for(timing.read_consider(self.rng, self.cfg.timing, self.state),
                            "play_beat_read", beat=i + 1,
                            total=beats + plan.extra_reads)
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

    def _concede(self, *, known_state: ScreenState | None = None) -> bool:
        """gear -> Concede, then wait for the Game Menu to actually go away.

        **On the Game Menu, no generic/retry fixed point below Concede may be used.**
        That menu reads Concede / Options / Quit top to bottom. The old
        ``concede_confirm`` point at y=0.56 was dead centre on Quit (measured against
        the real concede-menu capture), so treating a still-visible menu as a request
        to tap lower could quit Hearthstone. The recovery is instead to re-tap
        **Concede's own coordinate** (the TOP entry, y=0.196), bounded, and only while
        the menu is positively still up (see below). This does not prohibit the
        separately calibrated Black Market ``concede_now_button``: the direct reject
        trace sends that different-modal coordinate exactly once after its Concede
        group, never as a Game Menu retry.

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
        # The caller commonly just established this state in a semantic wait or
        # top-level dispatch.  Reusing that named fact removes a redundant Pixel
        # screencap; untrusted callers retain the checked fallback.
        if known_state is None:
            cls, _frame = self._classify_settled()
        else:
            cls = Classification(known_state, 1.0)
        if cls.state in self.END_SCREENS:
            if self.debug:
                self.debug.record("concede_skipped", reason="game already over",
                                  state=cls.state.value)
            return False
        if cls.state not in (ScreenState.IN_GAME, ScreenState.MULLIGAN):
            raise Halt(f"asked to concede from {cls.state.value!r}, not a live game or mulligan")

        # The gear opens a menu that dims the whole board, so this is a
        # full_transition, not a bottom_sheet (measured: thirds 21.5/21.1/3.7).
        self._tap_fixed(self.layout.gear_button, committing=False, decision_type="commit",
                        what="gear")
        # ...but `full_transition` does not mean "the Game Menu is up". `_compatible`
        # admits `top_banner` and `partial` too, so that assertion really only says
        # "something other than the bottom third moved" - which an opponent's turn
        # animating behind a *dropped* gear tap also satisfies. Then the loop's single
        # most consequential tap, the concede, would fire on an unclassified screen at
        # a coordinate that on the board is not a button at all. Look first.
        # The known gear->Concede sequence gets a short, fixed menu-render allowance. The next
        # boundary check below remains the only authority to retry or continue.
        self._ui_open_delay(self.cfg.timing.gear_menu_cooldown_s,
                            "gear_menu_cooldown")

        # commit, not reject: opening the Game Menu *was* the deliberation. By the time the
        # Concede button is in front of us the choice is made (reject_plan decided it, and
        # the reluctance already lives in hesitate_before_concede + the concede_point). A
        # human who opened the menu to concede taps it quickly -- a commit reaction, not a
        # fresh slow rejection. `committing=True` already books this as a commit for
        # HumanState, so this only aligns the think to match. RNG-draw-identical (think_time
        # draws one normal either way); only the think DURATION shrinks (~2.7s -> ~1.7s med).
        #
        # A single Concede tap can be silently DROPPED. This client ignores taps under a
        # congested wireless link (this run dropped two mulligan-card taps the same way,
        # retried, and had them honoured), and the concede tap's OWN change-check cannot
        # catch it: the board keeps animating BEHIND the semi-transparent Game Menu (an
        # enemy turn, flames, an attack arrow), so `_tap` sees a `partial` change and its
        # NO_CHANGE correction never fires -- verify passes on ambient motion unrelated to
        # the tap. The only positive proof the Concede took is the Game Menu LEAVING. So
        # retry exactly as the mulligan cards do (`_replace_card`) -- but a concede re-tap
        # is safe ONLY while the menu is POSITIVELY still up:
        #   * The Concede button is the TOP entry of a STATIC menu, so re-tapping its OWN
        #     coordinate (0.5025, 0.196) lands on Concede -- never the Options/Quit entries
        #     below it. There is no generic/retry fixed point below Concede *on this
        #     Game Menu* (the deleted, fatal `concede_confirm`); we re-send the same
        #     button, nothing lower. The separately calibrated Black Market
        #     `concede_now_button` is a different modal point, used once only by the
        #     direct reject trace after its Concede group, never by this retry loop.
        #   * The engine already trusts this exact tap: `_dispatch` re-taps concede every
        #     loop iteration while `concede_menu` persists (an UNBOUNDED implicit retry).
        #     This bounds it and keeps it inside `_concede`.
        #   * The gate can't fire on a live/end board: the concede_menu anchor is the ornate
        #     "Game Menu" title plate, which OFF-PHONE NCC-scores 0.18-0.36 on every other
        #     screen (defeat/victory/in_game/mulligan/rewards/reconnect) vs its 0.72 floor,
        #     and 0.84 only on its own frame -- so a positive concede_menu is the menu, not a
        #     priority-10 false positive masking a dissolving board.
        # An UNKNOWN or any other frame is NOT proof the menu is up, so we never blind-retap
        # on it: we fail closed (keep the pixels + Halt) exactly as before. Each attempt
        # keeps the FULL screen_wait budget before deciding to retap, so a slow-but-HONOURED
        # concede (the menu lingers a poll or two while the board dissolves) leaves within the
        # first wait and never retaps -- the happy path stays RNG/journal byte-identical.
        arrivals = {ScreenState.VICTORY, ScreenState.DEFEAT}
        attempts = max(1, self.cfg.vision.concede_tap_attempts)
        cls = frame = None
        for attempt in range(attempts):
            self._tap_fixed(self.layout.concede_button, committing=True, decision_type="commit",
                            what="concede")
            ok, cls, frame = self._wait_semantic(
                lambda c, _f: c.state in arrivals,
                what="concede",
                # {CONCEDE_MENU} | arrivals, exactly as `_wait_until_screen_leaves` scopes it
                # (+ the interrupt floor); a re-classify here always scans that floor, so a
                # reconnect/error co-drawn over the menu wins and stops the retry.
                expected=frozenset({ScreenState.CONCEDE_MENU}) | frozenset(arrivals),
                cooldown_s=self.cfg.timing.concede_resolve_cooldown_s,
                attempts=1)
            if ok:
                self._deferred_screen = (cls, frame)
                return True
            if cls.state == ScreenState.MULLIGAN:
                # The fixed mulligan gear sequence did not leave its named source.
                # Fall back to the separately bounded Confirm path rather than tapping
                # anywhere unknown or spinning on the same control.
                arrived = self._confirm_mulligan()
                return self._concede(known_state=arrived.state)
            # The menu did not leave. Retry ONLY while it is POSITIVELY still the Game Menu --
            # a re-tap is safe there and nowhere else. UNKNOWN / anything else falls through
            # to the fail-closed halt below.
            if cls.state == ScreenState.IN_GAME and attempt + 1 < attempts:
                # A board immediately after Concede can be the dissolve, not proof
                # that gear failed.  Give it one more complete resolve interval first.
                _, cls, frame = self._wait_semantic(
                    lambda c, _f: c.state in arrivals,
                    what="concede_dissolve", cooldown_s=self.cfg.timing.concede_resolve_cooldown_s,
                    attempts=1,
                    expected=frozenset({ScreenState.IN_GAME}) | frozenset(arrivals))
                if cls.state in arrivals:
                    self._deferred_screen = (cls, frame)
                    return True
                if cls.state != ScreenState.IN_GAME:
                    break
                self._tap_fixed(self.layout.gear_button, committing=False, decision_type="commit",
                                what="gear")
                self._ui_open_delay(self.cfg.timing.gear_menu_cooldown_s,
                                    "gear_menu_cooldown", retry=attempt + 1)
                continue
            if cls.state != ScreenState.CONCEDE_MENU or attempt + 1 == attempts:
                break
            if self.debug:
                self.debug.record("concede_tap_ignored", attempt=attempt + 1, of=attempts)
        if cls is not None and cls.state == ScreenState.UNKNOWN:
            self._record_unknown(frame, where="concede", confidence=round(cls.confidence, 3),
                                 waiting_to_leave=ScreenState.CONCEDE_MENU.value)
        raise Halt(f"Concede did not dismiss the Game Menu after {attempts} tap(s); the menu "
                   "is still up, so the tap is not registering (a dropped committing tap, not "
                   "a wrong coordinate). Only Concede's own coordinate was ever tapped.")

    #: Screens ``_clear_end_screens`` is allowed to tap ``end_dismiss`` on. The tap is
    #: a fixed point, so the set of screens it may land on has to be closed and named.
    END_SCREENS = (ScreenState.VICTORY, ScreenState.DEFEAT, ScreenState.REWARDS,
                   ScreenState.RANK_PROGRESS, ScreenState.QUEST_POPUP)
    #: Screens that mean "the post-game stack is cleared". DECK_SELECT belongs here:
    #: Hearthstone drops back to the deck LIST after a game, and dispatch knows how to
    #: reopen the deck from there. Leaving it out is what let the loop tap end_dismiss
    #: on the deck list - i.e. on "My Collection" and the deck boxes.
    HOME_SCREENS = (ScreenState.PLAY_SCREEN, ScreenState.QUEUE,
                    ScreenState.MENU, ScreenState.DECK_SELECT)

    def _dismiss_end_screen(self, state: ScreenState, *, end_scope, max_attempts: int) -> int:
        """Dismiss one positively identified end screen, returning taps sent.

        ``end_dismiss`` is unusually hazardous as a generic retry target: its fixed point
        overlaps the mulligan's Confirm control and reaches controls on other screens.  A
        dropped tap is nevertheless recoverable -- unlike a slow transition it leaves every
        post-tap look literally static and :meth:`_tap` raises :attr:`Halt.NO_CHANGE`.

        The recovery is therefore deliberately two-stage:

        * re-classify after that specific failure; and
        * re-send the exact same point only if the *same named* end screen is positively
          still present.

        A different named screen means the old tap may have landed late, so the caller returns
        to its closed-loop state machine.  UNKNOWN or an unlisted screen is never evidence for
        a retry and fails closed.  ``max_attempts`` is already clipped to the enclosing stack's
        total-tap cap, so even a succession of dropped screens cannot create extra actions.
        """
        key = f"end_dismiss:{state.value}"
        # ``max_attempts`` is the *remaining* stack budget, whereas ``used`` is
        # already charged to it.  Do not shrink the per-site retry ceiling on the
        # next loop and accidentally reject a still-available final tap.
        used = self._fixed_tap_attempts.get(key, 0)
        attempts = max(1, min(self.cfg.vision.end_dismiss_tap_attempts,
                              used + max_attempts))
        attempt = self._fixed_attempt(key, attempts)
        if attempt > 1 and self.debug:
            self.debug.record("end_dismiss_tap_ignored", screen=state.value,
                              attempt=attempt - 1, of=attempts,
                              verification="semantic_source_persisted")
        # The enclosing loop takes a fresh scoped classification before it ever calls
        # us again.  Therefore a repeat is only possible while this same named end
        # screen is still present; UNKNOWN and every other screen fail closed there.
        self._tap_fixed(self.layout.end_dismiss, committing=False, decision_type="commit",
                        what="end_dismiss")
        self._semantic_cooldown(self.cfg.timing.end_dismiss_cooldown_s,
                                "end_dismiss_cooldown", screen=state.value,
                                attempt=attempt)
        return 1

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
        # Scope must be CLOSED under every branch below -- crucially IN_GAME, so a Defeat
        # banner fading in over a still-drawn board (both anchors clear) resolves exactly
        # as a full classify would and routes to the wait-it-out branch, NOT a blind
        # end_dismiss into the dissolving board. Any un-listed screen just full-scans.
        end_scope = (frozenset(self.END_SCREENS) | frozenset(self.HOME_SCREENS)
                     | {ScreenState.IN_GAME})
        pending = self._deferred_screen
        self._deferred_screen = None
        while True:
            if pending is not None:
                cls, frame = pending
                pending = None
            else:
                cls, frame = self._classify_settled(end_scope)
            if cls.state in self.HOME_SCREENS:
                for key in tuple(self._fixed_tap_attempts):
                    if key.startswith("end_dismiss:"):
                        self._clear_fixed_attempt(key)
                # Main-loop dispatch can consume this named home observation rather
                # than taking an identical screenshot after the post-game stack.
                self._deferred_screen = (cls, frame)
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
                    if self.debug and hasattr(self.debug, "terminal_screen"):
                        context = {"state": cls.state.value,
                                   "confidence": round(cls.confidence, 3),
                                   "waits": waits}
                        if hasattr(self.classifier, "rank"):
                            try:
                                context["near_misses"] = [
                                    {"state": st.value, "score": round(sc, 3), "thr": th}
                                    for st, sc, th in self.classifier.rank(frame)[:6]
                                ]
                            except Exception:
                                pass
                        self.debug.terminal_screen(
                            "board never finished dissolving after the concede", frame, **context)
                    raise Halt("board never finished dissolving after the concede")
                waits += 1
                self._semantic_cooldown(self.cfg.timing.concede_resolve_cooldown_s,
                                        "clear_end_wait", wait=waits)
                continue
            if cls.state not in self.END_SCREENS:
                raise Halt(f"unexpected screen {cls.state.value!r} while clearing end "
                           f"screens; refusing to blind-tap end_dismiss there")
            for key in tuple(self._fixed_tap_attempts):
                if key.startswith("end_dismiss:") and key != f"end_dismiss:{cls.state.value}":
                    self._clear_fixed_attempt(key)
            if taps >= max_taps:
                raise Halt(f"end screens did not clear after {taps} dismiss taps")
            # Count physical gestures (including a dropped retry) against the post-game stack's
            # hard cap.  `_dismiss_end_screen` may return after a late, positively identified
            # transition; the next loop iteration always classifies before deciding what to do.
            taps += self._dismiss_end_screen(cls.state, end_scope=end_scope,
                                              max_attempts=max_taps - taps)

    # ── alerts ───────────────────────────────────────────────────────────────

    def _alert_target(self, read: MulliganRead) -> None:
        name = DISPLAY_NAMES.get(read.opponent_class, "?")
        turn_known = read.num_cards in (3, 4)
        second = read.we_go_second if turn_known else None
        turn = f"going {'2nd' if second else '1st'}" if turn_known else "turn unreadable"
        msg = f"Target matchup: {name} ({turn}) - your turn!"
        if self.debug:
            self.debug.record("target_found", opponent=name, second=second,
                              turn_known=turn_known, cards=read.num_cards)
        if self.alerts:
            self.alerts.target_found(name, second)
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
