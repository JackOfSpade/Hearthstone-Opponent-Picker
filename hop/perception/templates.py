"""Template matching by normalized cross-correlation + non-max suppression.

Standard: *"Locate actions by vision, not fixed coordinates. Template-match the
glyph/label with normalized cross-correlation + non-max suppression, filtered by
screen region. Controls move ... a fixed screen-fraction is unreliable. Keep a
calibrated fixed-fraction fallback for when vision misses."*

A :class:`Template` carries its glyph image, the screen *region* to search
(fractions, to cut cost and false positives), an accept threshold, and a
fixed-fraction fallback point. :func:`best_match` returns the top NCC location
(or None), and its score feeds the perception-confidence that drives HumanState
(a weak match -> inspect/hesitate, not instant commit).
"""

from __future__ import annotations

from dataclasses import dataclass

from .image import Frame, ncc, ncc_best_window


@dataclass(frozen=True)
class Region:
    """A search region in screen fractions (0..1)."""

    xf: float = 0.0
    yf: float = 0.0
    wf: float = 1.0
    hf: float = 1.0

    def to_px(self, frame: Frame) -> tuple[int, int, int, int]:
        return (int(self.xf * frame.width), int(self.yf * frame.height),
                max(1, int(self.wf * frame.width)), max(1, int(self.hf * frame.height)))


@dataclass(frozen=True)
class Template:
    name: str
    image: Frame
    region: Region = Region()
    threshold: float = 0.72
    fallback_point: tuple[float, float] | None = None  # screen-fraction fallback


@dataclass(frozen=True)
class Match:
    name: str
    x: int          # center x in device px
    y: int          # center y
    score: float


def _scan_best(frame: Frame, template: Template, stride: int = 2) -> Match | None:
    """Best NCC match of ``template`` in its region, ignoring the threshold gate.

    None only when the template is larger than its region. ``stride`` trades speed
    for precision; the region filter already bounds the cost.

    The whole sliding window is one vectorized pass (:func:`hop.perception.image.ncc_best_window`,
    ~86x faster and result-identical) when numpy is present; the per-window loop below is the
    pure-Python fallback, kept because it is the reference the fast path is validated against
    and the only path when numpy is unavailable. Both score the identical stride grid with the
    identical NCC and take the same first-max, so they return the same :class:`Match`.
    """
    rx, ry, rw, rh = template.region.to_px(frame)
    tw, th = template.image.width, template.image.height
    if tw > rw or th > rh:
        return None
    fast = ncc_best_window(frame, template.image, (rx, ry, rw, rh), stride)
    if fast is not None:
        col, row, score = fast
        return Match(template.name, rx + col + tw // 2, ry + row + th // 2, score)
    best: Match | None = None
    y = ry
    while y + th <= ry + rh:
        x = rx
        while x + tw <= rx + rw:
            patch = frame.crop(x, y, tw, th)
            score = ncc(patch, template.image)
            if best is None or score > best.score:
                best = Match(template.name, x + tw // 2, y + th // 2, score)
            x += stride
        y += stride
    return best


def best_match(frame: Frame, template: Template, stride: int = 2) -> Match | None:
    """Best NCC match of ``template`` within its region, or None below threshold."""
    best = _scan_best(frame, template, stride)
    if best is None or best.score < template.threshold:
        return None
    return best


def best_score(frame: Frame, template: Template, stride: int = 2) -> Match | None:
    """The best match *ignoring* the threshold (None only if the template is bigger
    than its region). For diagnostics: how CLOSE a sub-threshold anchor actually came,
    which ``best_match`` deliberately hides by returning None below threshold."""
    return _scan_best(frame, template, stride)


def match_all(frame: Frame, template: Template, stride: int = 2, min_sep: int | None = None) -> list[Match]:
    """All matches above threshold, non-max suppressed by ``min_sep`` px.

    Used where multiple instances of a glyph appear (e.g. counting repeated UI
    markers). Greedy NMS keeps the highest-scoring peaks that are at least
    ``min_sep`` apart (defaults to the template width).
    """
    rx, ry, rw, rh = template.region.to_px(frame)
    tw, th = template.image.width, template.image.height
    if tw > rw or th > rh:
        return []
    sep = min_sep if min_sep is not None else tw
    raw: list[Match] = []
    y = ry
    while y + th <= ry + rh:
        x = rx
        while x + tw <= rx + rw:
            score = ncc(frame.crop(x, y, tw, th), template.image)
            if score >= template.threshold:
                raw.append(Match(template.name, x + tw // 2, y + th // 2, score))
            x += stride
        y += stride
    raw.sort(key=lambda m: m.score, reverse=True)
    kept: list[Match] = []
    for m in raw:
        if all((abs(m.x - k.x) > sep or abs(m.y - k.y) > sep) for k in kept):
            kept.append(m)
    return kept
