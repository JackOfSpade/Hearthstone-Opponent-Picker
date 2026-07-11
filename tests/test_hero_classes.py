from hop.hero_classes import HeroClass, nearest_class, parse_class, snap_ocr_to_class


def test_parse_aliases():
    assert parse_class("mage") is HeroClass.MAGE
    assert parse_class("DH") is HeroClass.DEMONHUNTER
    assert parse_class("Death Knight") is HeroClass.DEATHKNIGHT
    assert parse_class("lock") is HeroClass.WARLOCK


def test_snap_exact():
    cls, dist = snap_ocr_to_class("MAGE")
    assert cls is HeroClass.MAGE and dist == 0


def test_snap_ocr_noise():
    # single-glyph OCR errors still resolve
    assert snap_ocr_to_class("MACE")[0] is HeroClass.MAGE
    assert snap_ocr_to_class("WARL0CK")[0] is HeroClass.WARLOCK
    assert snap_ocr_to_class("DEATLI KNIGHT", max_distance=3)[0] is HeroClass.DEATHKNIGHT


def test_snap_two_word_forms():
    assert snap_ocr_to_class("DEMON HUNTER")[0] is HeroClass.DEMONHUNTER
    assert snap_ocr_to_class("DEMONHUNTER")[0] is HeroClass.DEMONHUNTER


def test_snap_partial_read_of_a_two_word_class():
    """Live regression: a Death Knight's long 'DEATH KNIGHT' nameplate OCR'd as 'G DEATH'
    (one word + a stray leading glyph). That sits ~7 edits from the full label, so the snap
    used to reject it and the engine halted 'could not read mulligan (class=G DEATH, cards=4)'.
    The distinctive-fragment fallback now resolves it (and the other partial forms)."""
    assert snap_ocr_to_class("G DEATH", max_distance=3)[0] is HeroClass.DEATHKNIGHT
    assert snap_ocr_to_class("DEATH", max_distance=3)[0] is HeroClass.DEATHKNIGHT
    assert snap_ocr_to_class("KNIGHT", max_distance=3)[0] is HeroClass.DEATHKNIGHT
    assert snap_ocr_to_class("DEMON", max_distance=3)[0] is HeroClass.DEMONHUNTER


def test_snap_lone_hunter_stays_hunter_not_demon_hunter():
    """The fragment fallback must never turn a lone 'HUNTER' (the Hunter class's own word)
    into Demon Hunter -- HUNTER is deliberately excluded from the fragments."""
    assert snap_ocr_to_class("HUNTER", max_distance=3)[0] is HeroClass.HUNTER


def test_fragment_fallback_does_not_admit_junk_or_other_classes():
    """The fragments are >=4 edits from every OTHER class, so fail-closed still holds: a
    non-class read must not be dragged into Death Knight / Demon Hunter."""
    assert snap_ocr_to_class("garbage", max_distance=3)[0] is None
    assert snap_ocr_to_class("DRUID", max_distance=3)[0] is HeroClass.DRUID   # not DK via "DEATH"
    assert snap_ocr_to_class("PRIEST", max_distance=3)[0] is HeroClass.PRIEST # not DK via "KNIGHT"


def test_nearest_class_names_a_fragment_over_the_misleading_full_label():
    """For diagnostics: 'G DEATH' is closest to DRUID(5) among FULL labels, but its fragment
    'DEATH' pins Death Knight(1). nearest_class must report the fragment answer, not Druid."""
    cls, dist = nearest_class("G DEATH")
    assert cls is HeroClass.DEATHKNIGHT and dist == 1


def test_snap_rejects_junk_beyond_threshold():
    cls, dist = snap_ocr_to_class("garbage", max_distance=3)
    assert cls is None and dist > 3


def test_snap_empty():
    cls, _ = snap_ocr_to_class("", max_distance=3)
    assert cls is None
