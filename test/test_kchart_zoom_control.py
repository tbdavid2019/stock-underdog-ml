"""
Unit tests for v2.4.46 Stock K-Chart Style Interactive Zoom, Pan & Timeframe Range Control.
"""

from pathlib import Path
from core.version import APP_VERSION

BASE_DIR = Path(__file__).resolve().parent.parent


def test_app_version_bumped():
    """Verify application version is bumped to 2.4.46."""
    assert APP_VERSION == "2.4.46"


def test_index_html_includes_zoom_plugin_and_hammerjs():
    """Verify index.html loads hammerjs and chartjs-plugin-zoom."""
    index_path = BASE_DIR / "api" / "templates" / "index.html"
    content = index_path.read_text(encoding="utf-8")
    assert "hammerjs@2.0.8/hammer.min.js" in content
    assert "chartjs-plugin-zoom@2.0.1/dist/chartjs-plugin-zoom.min.js" in content


def test_tab_history_kchart_zoom_toolbar():
    """Verify tab_history.html contains K-chart timeframe pills and zoom controls."""
    tab_history_path = BASE_DIR / "api" / "templates" / "partials" / "tab_history.html"
    content = tab_history_path.read_text(encoding="utf-8")
    # Timeframe pills
    assert "setPriceChartRange('22')" in content
    assert "setPriceChartRange('45')" in content
    assert "setPriceChartRange('66')" in content
    assert "setPriceChartRange('130')" in content
    assert "setPriceChartRange('ALL')" in content

    # Zoom & Pan action buttons
    assert "zoomPriceChartIn()" in content
    assert "zoomPriceChartOut()" in content
    assert "panPriceChartLeft()" in content
    assert "panPriceChartRight()" in content
    assert "resetPriceChartZoom()" in content

    # Accessibility attributes
    assert ':aria-pressed="priceChartRange ===' in content
    assert '@dblclick="resetPriceChartZoom()"' in content
    assert "cursor-grab" in content
    assert "active:cursor-grabbing" in content
    assert 'role="img"' in content
    assert "touch-action: pan-y" in content


def test_chart_manager_zoom_methods():
    """Verify chart-manager.js implements interactive zoom, pan, autofit, and range controls."""
    cm_path = BASE_DIR / "api" / "static" / "js" / "chart-manager.js"
    content = cm_path.read_text(encoding="utf-8")
    assert "autoFitPriceChartYScale(" in content
    assert "setPriceChartRange(" in content
    assert "zoomPriceChart(" in content
    assert "panPriceChart(" in content
    assert "resetPriceChartZoom(" in content
    assert "zoom:" in content
    assert "onPanComplete:" in content
    assert "onZoomComplete:" in content
    assert "_densePointRadius" in content
    assert "_relaxedPointRadius" in content
    assert "0.012, 0.02" in content
    assert "delete chart.options.scales.y.min" in content


def test_app_js_zoom_state_and_methods():
    """Verify app.js includes reactive priceChartRange state and zoom dispatchers."""
    app_path = BASE_DIR / "api" / "static" / "js" / "app.js"
    content = app_path.read_text(encoding="utf-8")
    assert "priceChartRange: 'ALL'" in content
    assert "setPriceChartRange(range)" in content
    assert "zoomPriceChartIn()" in content
    assert "zoomPriceChartOut()" in content
    assert "panPriceChartLeft()" in content
    assert "panPriceChartRight()" in content
    assert "resetPriceChartZoom()" in content


def test_changelog_documented():
    """Verify both CHANGELOG.md and docs/CHANGELOG.md document v2.4.46."""
    changelog = (BASE_DIR / "CHANGELOG.md").read_text(encoding="utf-8")
    docs_changelog = (BASE_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 2026-10-31 (v2.4.46)" in changelog
    assert "## 2026-10-31 (v2.4.46)" in docs_changelog
