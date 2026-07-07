from hop.hero_classes import HeroClass, parse_class, snap_ocr_to_class


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


def test_snap_rejects_junk_beyond_threshold():
    cls, dist = snap_ocr_to_class("garbage", max_distance=3)
    assert cls is None and dist > 3


def test_snap_empty():
    cls, _ = snap_ocr_to_class("", max_distance=3)
    assert cls is None
