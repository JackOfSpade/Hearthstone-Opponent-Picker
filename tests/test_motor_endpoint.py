"""The tap endpoint must be a distribution, not a ring.

FFitts sigma grows with the target's size, so for a small control it exceeds the
hit radius. The old code scaled every out-of-bounds draw back onto the `0.9 * r`
circle -- which piles the endpoints up on that circle. Measured: **52% of taps at a
30px hit radius landed at exactly 27.0px from centre**, and 21% even at 120px.

No finger produces a ring of identical offsets. Resample instead.
"""

import math
import statistics
from random import Random

import pytest

from hop.config import load_config
from hop.geometry import PanelGeometry
from hop.humanize.motor import _endpoint_inside
from hop.humanize.state import HumanState


@pytest.fixture
def panel():
    return PanelGeometry(2400, 1080, 420.0)


@pytest.mark.parametrize("radius", [30.0, 45.0, 60.0, 120.0])
def test_endpoints_stay_inside_the_hit_area(radius, panel, cfg):
    rng, state = Random(7), HumanState()
    for _ in range(2000):
        ox, oy = _endpoint_inside(rng, radius, panel, cfg.motor, state)
        assert math.hypot(ox, oy) <= radius * 0.9 + 1e-9


@pytest.mark.parametrize("radius", [30.0, 45.0, 120.0])
def test_endpoints_do_not_pile_up_on_the_truncation_ring(radius, panel, cfg):
    """The regression. Clamping made this fraction 0.52 at r=30."""
    rng, state = Random(7), HumanState()
    limit = radius * 0.9
    offsets = [math.hypot(*_endpoint_inside(rng, radius, panel, cfg.motor, state))
               for _ in range(4000)]
    on_ring = sum(1 for d in offsets if d > limit * 0.999) / len(offsets)
    assert on_ring < 0.02, f"{on_ring:.1%} of endpoints sit on the truncation ring"


def test_endpoints_are_still_spread_not_dead_centre(panel, cfg):
    """Landing on the exact pixel every time is its own tell."""
    rng, state = Random(11), HumanState()
    offsets = [math.hypot(*_endpoint_inside(rng, 60.0, panel, cfg.motor, state))
               for _ in range(2000)]
    assert statistics.median(offsets) > 5.0
    assert len(set(round(o, 1) for o in offsets)) > 100
