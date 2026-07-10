"""Dashboard server: it persists class picks and shows a single Start/Stop button.

The dashboard used to apply class picks to the run as ephemeral overrides but never
write them, so a restart lost them and a bug report's config showed stale classes ("did
it not log my new selections?"). Start now persists, like the menu bar always has.
"""

import re

from hop.config import load_config
from hop.hero_classes import DISPLAY_NAMES, HeroClass
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


def test_criteria_summary_reflects_config_changed_out_of_band(tmp_path):
    """The criteria endpoint / risk meter must reflect the file, so a class the menu-bar
    twin persisted shows on the dashboard. Both surfaces write the same config; the
    server held its own in-memory copy, so without a reload the pick never appeared here.
    """
    from hop.config import save_criteria

    srv, p = _server(tmp_path)                       # server's cfg starts empty
    save_criteria(target_classes=(HeroClass.MAGE,), require_second=True, path=p)
    summary = srv._criteria_summary()
    assert "Mage" in summary["target_classes"]       # display name, reloaded from disk
    assert summary["require_second"] is True


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


def test_distribution_renders_on_open_not_only_after_a_search_starts():
    """The observed class distribution is day-scoped and persists across runs (status carries
    it even while idle), so the chart must render on every poll -- including on app open,
    before any Search. It regressed once by living inside the `if(s.games!==undefined)` block,
    which only runs after a run begins, so the chart appeared only once searching started."""
    html = _INDEX_HTML
    draw = html.index("drawDistribution(s.class_distribution")
    guard = html.index("if(s.games!==undefined)")
    assert draw < guard, "drawDistribution must be called BEFORE (outside) the per-run games guard"
    assert html.count("drawDistribution(s.class_distribution") == 1   # one unconditional call
    assert "loadCriteria();\npoll();" in html   # first render is immediate, not after the 1 s tick


def test_class_colors_covers_every_hero_class():
    """The chart colours each bar/label by the opponent class (CLASS_COLORS, keyed on the
    DISPLAY_NAMES the engine emits). An unrecognised label falls back to the neutral accent
    with no crash and no failing test -- so if a class is ever added/renamed, the drift is
    silent. Pin the JS map to the Python enum: every DISPLAY_NAME must have a colour."""
    block = re.search(r"const CLASS_COLORS=\{(.*?)\};", _INDEX_HTML, re.S)
    assert block, "CLASS_COLORS map not found in the dashboard HTML"
    keys = set(re.findall(r'"([^"]+)":"#', block.group(1)))
    missing = set(DISPLAY_NAMES.values()) - keys
    assert not missing, f"CLASS_COLORS is missing a colour for: {sorted(missing)}"
