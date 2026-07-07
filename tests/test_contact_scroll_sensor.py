import math
from dataclasses import replace
from random import Random

from hop.humanize.contact import ContactModel
from hop.humanize.motor import synth_tap
from hop.humanize.scroll import fling_deceleration, synth_fling
from hop.humanize.sensorimotor import (
    CoherenceVerdict,
    ImuSignature,
    Posture,
    SensorimotorModel,
)
from hop.humanize.state import HumanState


def test_pressure_ramp_has_floor_and_peak(cfg):
    cm = ContactModel(cfg.contact)
    assert cm.pressure(0.0) >= cfg.contact.pressure_floor - 1e-9
    peak_u = cfg.contact.pressure_alpha / (cfg.contact.pressure_alpha + cfg.contact.pressure_beta)
    assert cm.pressure(peak_u) <= cfg.contact.pressure_peak + 1e-9
    assert cm.pressure(peak_u) > cm.pressure(0.01)


def test_binary_pressure_semantics_is_flat(cfg):
    binary = replace(cfg.contact, pressure_semantics="binary")
    cm = ContactModel(binary)
    assert cm.pressure(0.1) == 1.0 == cm.pressure(0.9)


def test_orientation_rotates_toward_travel(cfg):
    cm = ContactModel(cfg.contact)
    ch = cm.channels(0.5, travel_dir=0.3)
    assert -math.pi / 2 <= ch.orientation <= math.pi / 2


def test_fling_release_velocity_matches_target(cfg, panel):
    cm = ContactModel(cfg.contact)
    g = synth_fling(Random(7), (1200, 800), (0, -1), 4000, panel, cfg.motor, cfg.scroll, cm, HumanState())
    rv = g.release_velocity
    speed = math.hypot(*rv)
    assert 3000 < speed < 5000  # within tolerance of the requested 4000 px/s


def test_fling_decel_positive(cfg):
    assert fling_deceleration(4000, 400, cfg.scroll.fling_friction) > 0


def test_desk_posture_is_declared_static(cfg, panel):
    cm = ContactModel(cfg.contact)
    g = synth_tap(Random(1), (300, 200), 40, panel, cfg.motor, cm, HumanState())
    sm = SensorimotorModel(cfg.sensor, panel, Posture.DESK_MOUNTED)
    assert sm.validate(g, None) == CoherenceVerdict.DECLARED_STATIC


def test_handheld_flat_imu_is_incoherent(cfg, panel):
    cm = ContactModel(cfg.contact)
    g = synth_tap(Random(1), (300, 200), 40, panel, cfg.motor, cm, HumanState())
    sm = SensorimotorModel(cfg.sensor, panel, Posture.HANDHELD)
    assert sm.validate(g, ImuSignature([])) == CoherenceVerdict.INCOHERENT
    # a matching predicted signature is coherent
    assert sm.validate(g, sm.predict(g)) == CoherenceVerdict.COHERENT


def test_tap_impulse_predicted(cfg, panel):
    cm = ContactModel(cfg.contact)
    g = synth_tap(Random(1), (300, 200), 40, panel, cfg.motor, cm, HumanState())
    sig = SensorimotorModel(cfg.sensor, panel, Posture.HANDHELD).predict(g)
    assert sig.peak_accel_g() > 0
