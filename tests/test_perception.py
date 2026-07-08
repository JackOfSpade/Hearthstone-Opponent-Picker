import random

from hop.perception.capture import FrameDeduper
from hop.perception.diffing import classify_change
from hop.perception.image import Frame, mean_abs_diff, ncc
from hop.perception.templates import Region, Template, best_match, match_all
from hop.transport import hid_descriptor as hd


def test_ncc_brightness_invariant():
    a = Frame.from_gray_bytes(10, 10, bytes(range(100)))
    bright = Frame.from_gray_bytes(10, 10, bytes((v + 40) % 256 for v in range(100)))
    inverted = Frame.from_gray_bytes(10, 10, bytes(255 - v for v in range(100)))
    assert abs(ncc(a, a) - 1.0) < 1e-9
    assert ncc(a, bright) > 0.99   # invariant to a brightness shift
    assert ncc(a, inverted) < -0.99


def test_best_match_locates_glyph():
    rnd = random.Random(1)
    gpix = bytes(rnd.randint(0, 255) for _ in range(64))
    glyph = Frame.from_gray_bytes(8, 8, gpix)
    W = H = 60
    bg = bytearray(rnd.randint(20, 40) for _ in range(W * H))
    gx, gy = 40, 12
    for y in range(8):
        for x in range(8):
            bg[(gy + y) * W + (gx + x)] = gpix[y * 8 + x]
    frame = Frame.from_gray_bytes(W, H, bytes(bg))
    m = best_match(frame, Template("g", glyph, Region(0, 0, 1, 1), 0.7), stride=1)
    assert m is not None and abs(m.x - 44) <= 1 and abs(m.y - 16) <= 1


def test_match_all_counts_copies():
    rnd = random.Random(2)
    gpix = bytes(rnd.randint(0, 255) for _ in range(64))
    glyph = Frame.from_gray_bytes(8, 8, gpix)
    W = H = 60
    bg = bytearray(rnd.randint(20, 40) for _ in range(W * H))
    for ox, oy in [(10, 10), (40, 40)]:
        for y in range(8):
            for x in range(8):
                bg[(oy + y) * W + (ox + x)] = gpix[y * 8 + x]
    frame = Frame.from_gray_bytes(W, H, bytes(bg))
    ms = match_all(frame, Template("g", glyph, Region(0, 0, 1, 1), 0.7), stride=1, min_sep=8)
    assert len(ms) == 2


def _bars_frame(n, W=800, H=400):
    """A frame with ``n`` bright vertical bars: just two distinguishable images.

    (This used to double as a card-count fixture. It doesn't any more -- counting
    mulligan cards by brightness was wrong on real frames. See test_card_count.py.)
    """
    data = bytearray([20]) * (W * H)
    left, right, cw = 0.20, 0.80, int(W * 0.14)
    for slot in range(n):
        xf = left + (right - left) * (slot / (n - 1)) if n > 1 else 0.5
        cx = int(xf * W)
        for y in range(int(H * 0.10), int(H * 0.85)):
            for x in range(max(0, cx - cw // 2), min(W, cx + cw // 2)):
                data[y * W + x] = 200
    return Frame.from_gray_bytes(W, H, bytes(data))


def test_frame_deduper_edge_detects():
    f1 = _bars_frame(3)
    f2 = _bars_frame(4)
    d = FrameDeduper(sig_size=8, change_threshold=9.0)
    assert d.advanced(f1) is True
    assert d.advanced(f1) is False   # same frame -> no advance
    assert d.advanced(f2) is True    # different -> advance


def test_classify_change_kinds():
    W = H = 60
    base = Frame.from_gray_bytes(W, H, bytes([30]) * (W * H))
    # change only the bottom third
    d = bytearray([30]) * (W * H)
    for y in range(40, 60):
        for x in range(W):
            d[y * W + x] = 220
    bottom = Frame.from_gray_bytes(W, H, bytes(d))
    assert classify_change(base, bottom, threshold=9.0) == "bottom_sheet"
    assert classify_change(base, base, threshold=9.0) == "none"


def test_hid_report_layout_roundtrip():
    c = hd.ContactReport(0, 1200, 540, 242, 140, 115, 10, True)
    rep = hd.encode_report([c])
    assert len(rep) == 1 + hd.MAX_CONTACTS * hd.PER_CONTACT_BYTES + 1
    assert rep[0] == hd.REPORT_ID
    assert rep[-1] == 1  # one active contact
    import struct
    flags, cid, x, y, p, maj, mn, orient = struct.unpack_from("<BBHHBBBb", rep, 1)
    assert (x, y, p) == (1200, 540, 242) and (flags & 1) == 1


def test_hid_descriptor_bytes_valid():
    desc = hd.build_digitizer_descriptor(2400, 1080)
    assert len(desc) > 50 and all(0 <= b <= 255 for b in desc)


def test_parse_size_prioritizes_override():
    from hop.adb import _parse_size, AdbError
    import pytest

    # Standard output: override size present
    txt = "Physical size: 1080x2400\nOverride size: 1080x2340\n"
    w, h = _parse_size(txt)
    assert (w, h) == (1080, 2340)

    # Physical size only
    txt2 = "Physical size: 1080x2400\n"
    w2, h2 = _parse_size(txt2)
    assert (w2, h2) == (1080, 2400)

    # Invalid string
    with pytest.raises(AdbError):
        _parse_size("invalid string")


def test_screen_classifier_custom_threshold():
    from unittest.mock import patch
    from hop.perception.screens import ScreenClassifier, ScreenState, Anchor
    from hop.perception.templates import Match

    anchor_tmpl = Template("mulligan", None, Region(0, 0, 1, 1), threshold=0.65)
    anchor = Anchor(ScreenState.MULLIGAN, anchor_tmpl)
    classifier = ScreenClassifier([anchor])

    # Match score is 0.68 (greater than template's 0.65, but less than self.accept 0.72)
    with patch("hop.perception.screens.best_match") as mock_best:
        mock_best.return_value = Match("mulligan", 100, 100, 0.68)
        classification = classifier.classify(None)
        assert classification.state == ScreenState.MULLIGAN
        assert classification.confidence == 0.68
