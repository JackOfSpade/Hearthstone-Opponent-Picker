"""Tests for the display<->native rotation transform.

Anchored on the on-device measurement (Pixel 7a native 1080x2400):
ROTATION_270 sends native (540,1200) to display center (1199,540); the inverse
here must send display center back to native (540,1200). Round-trips and both
landscape orientations are checked so the tool is robust to HS flipping.
"""

from hop.orientation import display_size, display_to_native, ROT_0, ROT_90, ROT_180, ROT_270

NW, NH = 1080, 2400  # native portrait


def test_display_size_swaps_on_landscape():
    assert display_size(NW, NH, ROT_0) == (1080, 2400)
    assert display_size(NW, NH, ROT_180) == (1080, 2400)
    assert display_size(NW, NH, ROT_90) == (2400, 1080)
    assert display_size(NW, NH, ROT_270) == (2400, 1080)


def test_rot0_is_identity():
    assert display_to_native(300, 900, ROT_0, NW, NH) == (300, 900)


def test_rot270_matches_measured_center():
    # display center (1199,540) -> native (540,1200) [measured ground truth]
    assert display_to_native(1199, 540, ROT_270, NW, NH) == (540, 1200)


def test_rot270_corners():
    # display (2399,0) top-right -> native (0,0) top-left
    assert display_to_native(2399, 0, ROT_270, NW, NH) == (0, 0)
    # display (0,1079) bottom-left -> native (1079,2399)
    assert display_to_native(0, 1079, ROT_270, NW, NH) == (1079, 1079 * 0 + 2399)


def test_rot90_is_180_of_rot270():
    # ROTATION_90 is the opposite landscape; a display point maps to the
    # 180-rotation (in native space) of where ROTATION_270 sends it.
    for lx, ly in [(0, 0), (2399, 0), (0, 1079), (2399, 1079), (1200, 540)]:
        nx90, ny90 = display_to_native(lx, ly, ROT_90, NW, NH)
        nx270, ny270 = display_to_native(lx, ly, ROT_270, NW, NH)
        assert (nx90, ny90) == (NW - 1 - nx270, NH - 1 - ny270)


def test_native_stays_in_bounds_all_rotations():
    for rot in (ROT_0, ROT_90, ROT_180, ROT_270):
        dw, dh = display_size(NW, NH, rot)
        for lx, ly in [(0, 0), (dw - 1, 0), (0, dh - 1), (dw - 1, dh - 1)]:
            nx, ny = display_to_native(lx, ly, rot, NW, NH)
            assert 0 <= nx <= NW - 1, (rot, lx, ly, nx)
            assert 0 <= ny <= NH - 1, (rot, lx, ly, ny)
