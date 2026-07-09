"""A minimal image type + operations, numpy-accelerated when available.

The perception layer needs only a handful of operations: decode a PNG, convert
to grayscale, crop a region, downsample to a small signature, mean absolute
difference, and normalized cross-correlation. Implementing them behind one
:class:`Frame` type - with a numpy fast path and a pure-Python fallback - keeps
the vision logic unit-testable in environments without numpy/Pillow (the pure
path is correct, just slower; full-frame NCC should use numpy in production).

Grayscale data is stored row-major as a flat ``bytearray`` of length w*h when in
pure-Python mode, or a 2-D ``uint8`` ndarray when numpy is present. Callers use
the methods and never touch the representation directly.
"""

from __future__ import annotations

from dataclasses import dataclass

try:  # pragma: no cover - exercised by availability, not logic
    import numpy as _np
except Exception:  # numpy optional
    _np = None

try:  # pragma: no cover
    from PIL import Image as _PILImage
except Exception:
    _PILImage = None


def numpy_available() -> bool:
    return _np is not None


def pil_available() -> bool:
    return _PILImage is not None


@dataclass
class Frame:
    """A grayscale image. ``data`` is an ndarray (numpy) or flat bytearray.

    ``rgb`` optionally carries the original colour pixels as an ``uint8`` ndarray
    of shape ``[h, w, 3]``. Everything in the vision layer works on grayscale;
    colour is retained only because a few cues are *defined* by hue and are
    destroyed by the grayscale conversion - notably Hearthstone's blue mana gems,
    which are how we count mulligan cards (see :mod:`hop.hearthstone`). It is
    ``None`` when numpy/Pillow are unavailable or the frame came from gray bytes.
    """

    width: int
    height: int
    data: object            # ndarray[h,w] uint8 OR bytearray length w*h
    rgb: object = None      # ndarray[h,w,3] uint8, or None

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_gray_bytes(cls, width: int, height: int, gray: bytes | bytearray) -> "Frame":
        if len(gray) != width * height:
            raise ValueError("gray length != width*height")
        if _np is not None:
            arr = _np.frombuffer(bytes(gray), dtype=_np.uint8).reshape(height, width).copy()
            return cls(width, height, arr)
        return cls(width, height, bytearray(gray))

    @classmethod
    def from_png(cls, png_bytes: bytes, keep_rgb: bool = True) -> "Frame":
        """Decode PNG -> grayscale Frame (requires Pillow in production).

        Also retains the RGB pixels when possible (``keep_rgb``), for the hue-
        dependent cues that grayscale destroys.
        """
        if _PILImage is None:
            raise RuntimeError("Pillow not installed; cannot decode screencap PNG. "
                               "pip install 'hop[vision]'")
        img = _PILImage.open(_bytes_io(png_bytes))
        gray = img.convert("L")
        w, h = gray.size
        if _np is None:
            return cls(w, h, bytearray(gray.tobytes()))
        rgb = _np.asarray(img.convert("RGB"), dtype=_np.uint8).copy() if keep_rgb else None
        return cls(w, h, _np.asarray(gray, dtype=_np.uint8).copy(), rgb)

    # ── access ───────────────────────────────────────────────────────────────

    def get(self, x: int, y: int) -> int:
        if _np is not None:
            return int(self.data[y, x])
        return self.data[y * self.width + x]

    def crop(self, x: int, y: int, w: int, h: int) -> "Frame":
        x = max(0, min(self.width, x)); y = max(0, min(self.height, y))
        w = max(1, min(self.width - x, w)); h = max(1, min(self.height - y, h))
        if _np is not None:
            sub_rgb = self.rgb[y:y + h, x:x + w].copy() if self.rgb is not None else None
            return Frame(w, h, self.data[y:y + h, x:x + w].copy(), sub_rgb)
        out = bytearray(w * h)
        for j in range(h):
            base = (y + j) * self.width + x
            out[j * w:(j + 1) * w] = self.data[base:base + w]
        return Frame(w, h, out)

    def crop_fraction(self, xf: float, yf: float, wf: float, hf: float) -> "Frame":
        return self.crop(int(xf * self.width), int(yf * self.height),
                         int(wf * self.width), int(hf * self.height))

    def downsample(self, nx: int, ny: int) -> "Frame":
        """Box-average downsample to nx*ny - used for frame signatures."""
        if _np is not None:
            ys = (_np.linspace(0, self.height, ny + 1)).astype(int)
            xs = (_np.linspace(0, self.width, nx + 1)).astype(int)
            out = _np.zeros((ny, nx), dtype=_np.uint8)
            for j in range(ny):
                for i in range(nx):
                    block = self.data[ys[j]:max(ys[j] + 1, ys[j + 1]), xs[i]:max(xs[i] + 1, xs[i + 1])]
                    out[j, i] = int(block.mean()) if block.size else 0
            return Frame(nx, ny, out)
        out = bytearray(nx * ny)
        for j in range(ny):
            y0 = j * self.height // ny
            y1 = max(y0 + 1, (j + 1) * self.height // ny)
            for i in range(nx):
                x0 = i * self.width // nx
                x1 = max(x0 + 1, (i + 1) * self.width // nx)
                s = cnt = 0
                for yy in range(y0, y1):
                    base = yy * self.width
                    for xx in range(x0, x1):
                        s += self.data[base + xx]; cnt += 1
                out[j * nx + i] = s // cnt if cnt else 0
        return Frame(nx, ny, out)

    def signature(self, n: int) -> str:
        """Downsample to n*n and return a compact hex signature string."""
        small = self.downsample(n, n)
        if _np is not None:
            return small.data.astype("uint8").tobytes().hex()
        return bytes(small.data).hex()


def mean_abs_diff(a: Frame, b: Frame) -> float:
    """Mean absolute grayscale difference (0..255). Frames must match size."""
    if a.width != b.width or a.height != b.height:
        raise ValueError("frame size mismatch")
    if _np is not None:
        return float(_np.abs(a.data.astype("int16") - b.data.astype("int16")).mean())
    total = 0
    n = a.width * a.height
    for i in range(n):
        total += abs(a.data[i] - b.data[i])
    return total / n if n else 0.0


def signature_distance(sig_a: str, sig_b: str) -> float:
    """Mean absolute byte difference between two equal-length hex signatures."""
    ba, bb = bytes.fromhex(sig_a), bytes.fromhex(sig_b)
    if len(ba) != len(bb) or not ba:
        return 255.0
    return sum(abs(x - y) for x, y in zip(ba, bb)) / len(ba)


def ncc(patch: Frame, template: Frame) -> float:
    """Zero-normalized cross-correlation of two equal-size frames, in [-1, 1].

    Invariant to global brightness/contrast shifts (the property that makes it
    robust for matching UI glyphs against a template regardless of lighting).
    """
    if patch.width != template.width or patch.height != template.height:
        raise ValueError("ncc size mismatch")
    if _np is not None:
        a = patch.data.astype("float64").ravel()
        b = template.data.astype("float64").ravel()
    else:
        a = [float(v) for v in patch.data]
        b = [float(v) for v in template.data]
    return _ncc_vectors(a, b)


def _ncc_vectors(a, b) -> float:
    if _np is not None:
        am = a - a.mean(); bm = b - b.mean()
        denom = (float(_np.sqrt((am * am).sum())) * float(_np.sqrt((bm * bm).sum())))
        if denom == 0:
            return 0.0
        return float((am * bm).sum() / denom)
    n = len(a)
    ma = sum(a) / n; mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _bytes_io(b: bytes):
    import io
    return io.BytesIO(b)
