"""
Unit tests for Phase 2B:
Command Palette Component, Workspace Sub-Bar, Sessions Clock, and Watchlist Dual-Track UI.
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app


class TestCommandPaletteAndTabsUI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_index_page_includes_command_palette(self):
        """Verify GET / successfully renders index with native <dialog> command palette modal."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Native dialog modal elements
        self.assertIn('<dialog', html)
        self.assertIn('ref="commandPaletteDialog"', html)
        self.assertIn('aria-label="全域指令與股票搜尋面板"', html)
        self.assertIn('v-if="showCommandPalette"', html)
        self.assertIn('@cancel.prevent="closeCommandPalette()"', html)
        self.assertIn('@close="syncDialogClosed(\'commandPalette\')"', html)
        self.assertIn('placeholder="搜尋股票代號、中文名稱', html)
        self.assertIn('888 StockQuant Engine', html)

    def test_header_has_command_palette_trigger_and_workspace_bar(self):
        """Verify header has ⌘K search trigger button, market sessions clock, and recent tickers workspace."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Trigger button
        self.assertIn('@click="openCommandPalette()"', html)
        self.assertIn('⌘K', html)

        # Workspace bar with sessions clock
        self.assertIn('marketSessions && marketSessions.sessions', html)
        self.assertIn('s.status_label', html)
        self.assertIn('s.countdown_formatted', html)

        # Recent & Pinned tickers workspace
        self.assertIn('recentTickers && recentTickers.length > 0', html)
        self.assertIn('togglePinRecentTicker', html)
        self.assertIn('removeRecentTicker', html)

        # Watchlist share import banner
        self.assertIn('sharedWatchlistTickers && sharedWatchlistTickers.length > 0', html)
        self.assertIn('importSharedWatchlist()', html)
        self.assertIn('dismissSharedWatchlist()', html)

    def test_dashboard_and_history_have_watchlist_elements(self):
        """Verify dashboard has watchlist filter chip and star toggles, and history has star toggle."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Filter chip in dimension 1
        self.assertIn('toggleWatchlistFilter()', html)
        self.assertIn('我的自選', html)
        self.assertIn('watchlistCount', html)

        # Star toggle in dashboard table & cards
        self.assertIn('toggleWatchlist(item.ticker)', html)
        self.assertIn('isWatchlisted(item.ticker)', html)

        # Star toggle in history tab
        self.assertIn('toggleWatchlist(resolvedTicker)', html)
        self.assertIn('isWatchlisted(resolvedTicker)', html)

    def test_app_js_has_workspace_and_command_palette_methods(self):
        """Verify app.js contains all necessary reactive methods for command palette, sessions, and watchlist."""
        app_js_path = Path("api/static/js/app.js")
        self.assertTrue(app_js_path.exists())
        js_code = app_js_path.read_text(encoding="utf-8")

        methods = [
            "openCommandPalette",
            "closeCommandPalette",
            "toggleCommandPalette",
            "onCommandSearch",
            "executeActiveCommand",
            "initWorkspaceAndWatchlist",
            "syncCloudWatchlist",
            "toggleWatchlist",
            "isWatchlisted",
            "importSharedWatchlist",
            "loadMarketSessions",
            "addRecentTicker",
            "togglePinRecentTicker",
            "removeRecentTicker"
        ]
        for m in methods:
            self.assertIn(m, js_code, f"Missing method {m} in app.js")

        # Global keydown listener for ⌘K
        self.assertIn("metaKey || e.ctrlKey", js_code)

    def test_api_client_has_sessions_and_watchlist_methods(self):
        """Verify api-client.js contains getMarketLiveSessions and watchlist methods."""
        client_js_path = Path("api/static/js/api-client.js")
        self.assertTrue(client_js_path.exists())
        js_code = client_js_path.read_text(encoding="utf-8")

        methods = [
            "getMarketLiveSessions",
            "getWatchlist",
            "addWatchlist",
            "removeWatchlist",
            "batchImportWatchlist",
            "resolveTicker"
        ]
        for m in methods:
            self.assertIn(m, js_code, f"Missing method {m} in api-client.js")

    def test_agent_hub_python_snippet_uses_single_source_without_literal_js_origin(self):
        """Regression test: Agent Hub displayed Python snippet must not show literal '{window.location.origin}'."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertNotIn('f"{window.location.origin}/api/v1/predictions/resonance"', html)
        self.assertIn("{{ pythonSnippet }}", html)

    def test_locale_dictionary_and_html_lang_sync_exist(self):
        """Verify app.js provides t() i18n dictionary, syncs <html lang>, and header/dashboard bind t()."""
        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("I18N_MESSAGES", app_js)
        self.assertIn("document.documentElement.lang", app_js)
        self.assertIn("t(key", app_js)

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        self.assertIn("t('nav.dashboard'", html)
        self.assertIn("t('macro.regime')", html)
        self.assertIn("t('screener.title')", html)
        self.assertIn("t('table.ticker')", html)

    def test_all_modals_use_native_dialog_and_coordinators(self):
        """Verify all blocking modals migrate to <dialog> with showModal(), pointer light-dismiss, and scroll lock."""
        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        self.assertIn("openNativeDialog", app_js)
        self.assertIn("closeNativeDialog", app_js)
        self.assertIn("syncDialogClosed", app_js)
        self.assertIn("handleDialogPointerDown", app_js)
        self.assertIn("handleDialogPointerUp", app_js)
        self.assertIn("showModal", app_js)
        self.assertNotIn("document.body.classList.add('overflow-hidden')", app_js)

        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text
        for ref_name in (
            'ref="commandPaletteDialog"',
            'ref="alertsModalDialog"',
            'ref="riskXrayDialog"',
            'ref="earningsCalendarDialog"',
        ):
            self.assertIn(ref_name, html, f"Missing native dialog {ref_name}")
        self.assertNotIn('@keydown.window.esc', html)
        self.assertIn('overscroll-behavior: contain', html)
        self.assertIn('dialog:modal', html)
        self.assertIn('html:has(dialog[open])', html)

    def test_a11y_tables_controls_labels_and_headings(self):
        """Verify keyboard-accessible sortable headers, table scroll region, <button> logo, label-for bindings, and single H1 per tab."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Sortable table headers use <th :aria-sort="..."><button type="button" ...>
        self.assertIn(':aria-sort="tableSortKey === \'ticker\'', html)
        self.assertIn('role="region" tabindex="0"', html)

        # Header logo is a native <button> rather than <div role="button">
        header_html = Path("api/templates/partials/header.html").read_text(encoding="utf-8")
        self.assertNotIn('role="button"', header_html)

        # Alerts modal inputs have explicit label[for] + id pairs
        alerts_html = Path("api/templates/partials/alerts_modal.html").read_text(encoding="utf-8")
        for fid in ("alert-ticker-input", "alert-rule-select", "alert-threshold-input", "alert-api-key-input"):
            self.assertIn(f'for="{fid}"', alerts_html)
            self.assertIn(f'id="{fid}"', alerts_html)

        # No competing global <h1 class="sr-only"> in index.html; each tab partial defines its own visible <h1>
        index_html = Path("api/templates/index.html").read_text(encoding="utf-8")
        self.assertNotIn('<h1 class="sr-only">', index_html)
        for tab_name in (
            "tab_dashboard.html",
            "tab_institutional.html",
            "tab_history.html",
            "tab_industry_chain.html",
            "tab_discovery.html",
            "tab_podcast.html",
            "tab_agent_hub.html",
        ):
            tab_html = Path(f"api/templates/partials/{tab_name}").read_text(encoding="utf-8")
            self.assertEqual(tab_html.count("<h1"), 1, f"{tab_name} should contain exactly one <h1>")

    def test_iconography_and_precision_transitions(self):
        """Verify UI partials avoid transition-all, drop interactive ★/☆/✕ glyph controls, and keep inline SVG only for loading spinners."""
        import re
        for name in ("header.html", "tab_dashboard.html", "tab_history.html", "tab_broker.html"):
            content = Path(f"api/templates/partials/{name}").read_text(encoding="utf-8")
            self.assertNotIn("transition-all", content, f"{name} should use explicit transition properties instead of transition-all")
            svgs = re.findall(r"<svg\b[\s\S]*?</svg>", content)
            non_spinner = [s for s in svgs if "animate-spin" not in s]
            self.assertEqual(non_spinner, [], f"{name} should not contain non-spinner hand-coded <svg> icons")

        header_html = Path("api/templates/partials/header.html").read_text(encoding="utf-8")
        self.assertNotIn("'★'", header_html)
        self.assertNotIn("'☆'", header_html)
        self.assertIn("translateSessionStatus", header_html)
        dash_html = Path("api/templates/partials/tab_dashboard.html").read_text(encoding="utf-8")
        self.assertNotIn("'★'", dash_html)
        self.assertNotIn("'☆'", dash_html)
        self.assertIn('v-for="(item, idx) in displayRecords"', dash_html)

        drawer_tabs_html = Path("api/templates/partials/_stock_drawer_tabs.html").read_text(encoding="utf-8")
        self.assertIn("+ '-' + idx", drawer_tabs_html)
        self.assertNotIn("bg-emerald-600 text-white", drawer_tabs_html)
        self.assertIn("bg-emerald-800 text-white", drawer_tabs_html)

        app_js = Path("api/static/js/app.js").read_text(encoding="utf-8")
        for symbol in ("_isPointOutsideRect", "handleDialogPointerCancel", "_dialogRequestSeq", "_setFallbackModalInert", "translateSessionStatus"):
            self.assertIn(symbol, app_js)
        self.assertNotIn('f"$${window.location.origin}', app_js)


if __name__ == "__main__":
    unittest.main()
