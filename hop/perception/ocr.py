"""Reading the opponent's class label off the mulligan screen.

This is the primary class signal (settled in the design discussion): the game
prints the literal class word - "MAGE", "WARLOCK", ... - as plain static text
under the opponent's portrait, immune to hero skins, multi-class skins and
golden-hero animation. Detection is: crop that region, OCR it, and snap the
result to the nearest of the 11 fixed class strings by edit distance
(:func:`hop.hero_classes.snap_ocr_to_class`). Because the vocabulary is 11
words, the snap is effectively a deterministic classifier - no per-run cost and
no cloud API.

Two backends:

* **tesseract** (default) - a real OCR engine (``pytesseract``), constrained to
  uppercase; the 11-word snap corrects its residual glyph errors.
* **templates** - a *zero-ML* alternative: match the crop against 11 stored
  label images by NCC. Offered because it needs no OCR engine at all; slightly
  more calibration. Selected automatically if pytesseract is unavailable and a
  label-template pack exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hero_classes import HeroClass, snap_ocr_to_class
from .image import Frame, ncc

try:  # pragma: no cover
    import pytesseract as _pt
    from PIL import Image as _PILImage
except Exception:
    _pt = None
    _PILImage = None


class OcrUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class ClassRead:
    hero_class: HeroClass | None
    confidence: float          # 0..1, feeds HumanState.observe_confidence
    raw_text: str
    distance: int
    method: str                # "tesseract" | "templates"


def tesseract_available() -> bool:
    return _pt is not None and _PILImage is not None


def _frame_to_pil(frame: Frame):
    if _PILImage is None:
        raise OcrUnavailable("Pillow/pytesseract not installed")
    try:
        import numpy as np
        if not isinstance(frame.data, (bytes, bytearray)):
            return _PILImage.fromarray(frame.data.astype("uint8"), mode="L")
    except Exception:
        pass
    return _PILImage.frombytes("L", (frame.width, frame.height), bytes(frame.data))


def _confidence_from_distance(distance: int, max_edit: int) -> float:
    if distance <= 0:
        return 1.0
    if distance > max_edit:
        return 0.0
    return max(0.0, 1.0 - distance / (max_edit + 1))


class ClassReader:
    """Reads a class label from a pre-cropped label Frame."""

    def __init__(self, max_edit_distance: int = 3, label_templates: dict[HeroClass, Frame] | None = None):
        self.max_edit = max_edit_distance
        self.label_templates = label_templates or {}

    def read(self, label_crop: Frame) -> ClassRead:
        if tesseract_available():
            return self._read_tesseract(label_crop)
        if self.label_templates:
            return self._read_templates(label_crop)
        raise OcrUnavailable(
            "no OCR engine (pytesseract) and no label-template pack; "
            "install 'hop[vision]' or provide templates via `hop calibrate`"
        )

    def _read_tesseract(self, crop: Frame) -> ClassRead:
        pil = _frame_to_pil(crop)
        # PSM 7 = single text line; whitelist uppercase + space for the class word
        cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        raw = _pt.image_to_string(pil, config=cfg).strip()
        cls, dist = snap_ocr_to_class(raw, max_distance=self.max_edit)
        return ClassRead(cls, _confidence_from_distance(dist, self.max_edit), raw, dist, "tesseract")

    def _read_templates(self, crop: Frame) -> ClassRead:
        best_cls: HeroClass | None = None
        best_score = -1.0
        for cls, tmpl in self.label_templates.items():
            # resize-free NCC requires equal dims; templates are captured at the
            # crop size during calibration, so compare directly when they match.
            if tmpl.width == crop.width and tmpl.height == crop.height:
                score = ncc(crop, tmpl)
            else:
                score = ncc(crop.downsample(tmpl.width, tmpl.height), tmpl)
            if score > best_score:
                best_score, best_cls = score, cls
        # map NCC score to a pseudo-distance for a uniform confidence scale
        conf = max(0.0, best_score)
        cls = best_cls if conf >= 0.6 else None
        return ClassRead(cls, conf, f"<template:{best_cls}>", 0 if cls else 99, "templates")
