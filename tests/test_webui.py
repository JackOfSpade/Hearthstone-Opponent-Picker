"""Dashboard server: it persists class picks and shows a single Start/Stop button.

The dashboard used to apply class picks to the run as ephemeral overrides but never
write them, so a restart lost them and a bug report's config showed stale classes ("did
it not log my new selections?"). Start now persists, like the menu bar always has.
"""

from hop.config import load_config
from hop.hero_classes import HeroClass
from hop.webui.server import _INDEX_HTML, DashboardServer


def _server(tmp_path, body="[criteria]\ntarget_classes = []\nrequire_second = false\n"):
    p = tmp_path / "config.toml"
    p.write_text(body)
    cfg = load_config(p)
    return DashboardServer(controller=None, cfg=cfg, config_path=p), p


def test_persist_criteria_writes_the_picks_to_config(tmp_path):
    srv, p = _server(tmp_path)
    srv._persist_criteria({"target_classes": ["Mage", "Priest"], "require_second": True})
    saved = load_config(p).criteria
    assert saved.target_classes == (HeroClass.MAGE, HeroClass.PRIEST)   # display names parsed
    assert saved.require_second is True
    # the server's own cfg (risk meter / criteria endpoint) is updated in step
    assert srv.cfg.criteria.target_classes == (HeroClass.MAGE, HeroClass.PRIEST)


def test_persist_criteria_ignores_a_missing_key(tmp_path):
    srv, p = _server(tmp_path, "[criteria]\ntarget_classes = [\"ROGUE\"]\nrequire_second = false\n")
    srv._persist_criteria({"require_second": True})           # no target_classes key
    assert load_config(p).criteria.target_classes == (HeroClass.ROGUE,)   # untouched


def test_persist_criteria_survives_a_bad_class_name(tmp_path):
    srv, p = _server(tmp_path)
    srv._persist_criteria({"target_classes": ["NotAClass"]})   # must not raise
    assert load_config(p).criteria.target_classes == ()        # nothing written


def test_dashboard_has_one_toggle_button_and_no_silence_button():
    assert 'id="toggleBtn"' in _INDEX_HTML
    assert "ackBtn" not in _INDEX_HTML
    assert "Silence alarm" not in _INDEX_HTML
    assert 'id="stopBtn"' not in _INDEX_HTML and 'id="startBtn"' not in _INDEX_HTML


def test_dashboard_is_branded_with_the_project_name_not_hop():
    assert "Hearthstone Opponent Picker" in _INDEX_HTML
    assert "<h1>hop</h1>" not in _INDEX_HTML
    assert "<title>hop dashboard</title>" not in _INDEX_HTML
