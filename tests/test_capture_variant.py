"""`hop capture --variant`: one screen state, several visual faces.

Hearthstone's reward popup is a scroll whose header ("Level 12 Reward!") and gold
count change every time, and the ranked medal screen is different again -- but all
dismiss identically, so they are one STATE with several anchors. `load_template_pack`
already accepts any number of anchors per state; this pins that `capture` can WRITE
them without clobbering, while a plain (no-variant) capture stays one-per-state.
"""

from types import SimpleNamespace

import pytest

from hop.cli import cmd_capture
from hop.perception.screens import ScreenState, load_template_pack


@pytest.fixture
def frame_png(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    import numpy as np
    p = tmp_path / "frame.png"
    # deterministic non-uniform pixels (a flat frame gives an undefined NCC)
    arr = ((np.arange(120 * 60) * 37) % 251).astype("uint8").reshape(60, 120)
    Image.fromarray(arr, "L").convert("RGB").save(p)
    return p


def _capture(pack, frame_png, state, variant=None):
    cmd_capture(SimpleNamespace(
        config=None, templates=str(pack), from_file=str(frame_png),
        state=state, variant=variant, glyph="0.1,0.1,0.2,0.2",
        region=None, threshold=0.72, priority=0,
    ))


def test_a_variant_coexists_with_the_plain_anchor(tmp_path, frame_png):
    pack = tmp_path / "pack"
    _capture(pack, frame_png, "rewards")                       # the medal, say
    _capture(pack, frame_png, "rewards", variant="banner")     # the reward scroll

    clf = load_template_pack(pack)
    rewards = [a for a in clf.anchors if a.state == ScreenState.REWARDS]
    assert len(rewards) == 2
    imgs = sorted(p.name for p in pack.glob("rewards*.png") if "_full" not in p.name)
    assert imgs == ["rewards.png", "rewards_banner.png"]
    assert (pack / "rewards_banner_full.png").exists()         # its own reference frame


def test_recapturing_the_same_variant_replaces_only_it(tmp_path, frame_png):
    pack = tmp_path / "pack"
    _capture(pack, frame_png, "rewards")
    _capture(pack, frame_png, "rewards", variant="banner")
    _capture(pack, frame_png, "rewards", variant="banner")     # again

    clf = load_template_pack(pack)
    assert sum(a.state == ScreenState.REWARDS for a in clf.anchors) == 2   # not 3


def test_plain_capture_is_still_one_anchor_per_state(tmp_path, frame_png):
    """Backward compatibility: no --variant behaves exactly as before."""
    pack = tmp_path / "pack"
    _capture(pack, frame_png, "victory")
    _capture(pack, frame_png, "victory")                       # recapture replaces

    clf = load_template_pack(pack)
    assert sum(a.state == ScreenState.VICTORY for a in clf.anchors) == 1


def test_a_plain_recapture_does_not_disturb_a_variant(tmp_path, frame_png):
    pack = tmp_path / "pack"
    _capture(pack, frame_png, "rewards", variant="banner")
    _capture(pack, frame_png, "rewards")                       # add the plain one
    _capture(pack, frame_png, "rewards")                       # replace the plain one

    clf = load_template_pack(pack)
    assert sum(a.state == ScreenState.REWARDS for a in clf.anchors) == 2   # variant survives
