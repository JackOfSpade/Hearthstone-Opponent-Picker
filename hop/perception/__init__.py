"""Layer 2 - perception without footprint.

Standard: *"Capture, don't inspect."* We read the screen only through
``adb exec-out screencap`` frames and computer vision - never the accessibility
tree, never an on-device inspector (that footprint is what Layer 0 forbids).

* :mod:`hop.perception.image`    - a tiny grayscale/RGB Frame + ops (numpy-fast,
  pure-Python fallback so the logic is testable without native deps)
* :mod:`hop.perception.capture`  - screencap decode + frame-advance dedup
* :mod:`hop.perception.templates`- NCC template match + non-max suppression
* :mod:`hop.perception.diffing`  - region-split change detection
* :mod:`hop.perception.ocr`      - class-label OCR + snap-to-11
* :mod:`hop.perception.screens`  - the Hearthstone screen-state classifier
"""
