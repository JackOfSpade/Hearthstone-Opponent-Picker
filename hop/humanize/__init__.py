"""Humanization layers 3-5, tailored to the Hearthstone opponent-picker.

* :mod:`hop.humanize.state`  - Layer 5 latent HumanState (the spine)
* :mod:`hop.humanize.motor`  - Layer 3 motor synthesis (FFitts + submovements)
* :mod:`hop.humanize.contact`- Layer 3 contact-patch evolution
* :mod:`hop.humanize.scroll` - Layer 3 Android fling-physics scrolling
* :mod:`hop.humanize.timing` - Layer 4 stateful timing / think time
* :mod:`hop.humanize.sensorimotor` - Layer 4 IMU coherence (generate + validate)
* :mod:`hop.humanize.journey`- Layer 5 Hearthstone journey grammar
* :mod:`hop.humanize.limiter`- Layer 5 caps + non-repetition

All of these are pure (no I/O) and take an injectable ``rng`` so behavior is
reproducible and unit-testable, per the standard's determinism corollary.
"""
