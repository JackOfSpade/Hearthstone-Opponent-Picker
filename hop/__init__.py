"""hop - Hearthstone Opponent Picker (v2, Android / wireless-ADB).

Watches Hearthstone's mulligan screen on a physical Android phone over wireless
ADB, reads the opponent's *class* (the game prints the literal class word) and
whether we go second (mulligan card count), then either alerts on a target
matchup or performs a humanized, plausibly-timed concede to re-queue.

This tool applies the Android App Automation Humanization Standard (Rev 2), but
it is **tailored specifically to Hearthstone** - it is not a general framework
for other apps. The module layout borrows that standard's layer numbering only
as an organizing principle, so the humanization reasoning stays legible:

* ``hop.transport``   - Layer 1: persistent virtual HID digitizer (+ adb fallback)
* ``hop.perception``  - Layer 2: screencap + vision (no accessibility tree)
* ``hop.humanize``    - Layers 3-5: motor synthesis, timing/sensorimotor, HumanState
* ``hop.verify``      - Layer 6: verification + coherence gates + fail-closed
* ``hop.hearthstone`` - the Hearthstone-specific screens, coordinates and flow
* ``hop.engine``      - the opponent-picker hunt loop that ties it together

The humanize/verify code is factored into its own modules for testability, not
to be reused across apps; its journey grammar, screens and caps are all
Hearthstone-specific.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
