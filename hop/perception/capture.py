"""Screencap capture + frame-advance de-duplication.

Standard: *"Dedupe consecutive frames by a downsampled signature (e.g. 24x24
grayscale) to tell 'the screen advanced' from 'frame noise' with no UI
hierarchy. A repeated signature = the screen stopped changing."* And: *"If
frames won't decode or the device wedges (empty/truncated screencaps), refuse to
run rather than continue blind."*
"""

from __future__ import annotations

from .image import Frame, signature_distance


class CaptureError(RuntimeError):
    pass


class Capturer:
    """Captures decoded grayscale frames from the device via ADB."""

    def __init__(self, adb):
        self.adb = adb

    def capture(self) -> Frame:
        png = self.adb.screencap_png()
        if not png or len(png) < 100:
            raise CaptureError("empty/truncated screencap; device may be wedged - refusing to run blind")
        try:
            return Frame.from_png(png)
        except Exception as e:
            raise CaptureError(f"could not decode screencap: {e}") from e


class FrameDeduper:
    """Tells whether the screen advanced vs merely jittered, via NxN signature."""

    def __init__(self, sig_size: int = 24, change_threshold: float = 9.0):
        self.sig_size = sig_size
        self.change_threshold = change_threshold
        self._last_sig: str | None = None

    def advanced(self, frame: Frame) -> bool:
        """True if this frame differs from the last accepted one beyond noise.

        Also updates the stored signature, so calling it each poll turns the
        capture loop into an edge-detector for "the screen changed".
        """
        sig = frame.signature(self.sig_size)
        if self._last_sig is None:
            self._last_sig = sig
            return True
        dist = signature_distance(sig, self._last_sig)
        changed = dist > self.change_threshold
        if changed:
            self._last_sig = sig
        return changed

    def settled(self, frame: Frame) -> bool:
        """True if the screen has STOPPED changing (a repeated signature).

        Used to wait for an animation/transition to finish before acting - the
        inverse of :meth:`advanced` without mutating unless still moving.
        """
        sig = frame.signature(self.sig_size)
        if self._last_sig is None:
            self._last_sig = sig
            return False
        dist = signature_distance(sig, self._last_sig)
        self._last_sig = sig
        return dist <= self.change_threshold

    def reset(self) -> None:
        self._last_sig = None
