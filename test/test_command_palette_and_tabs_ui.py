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
        """Verify GET / successfully renders index with command_palette modal included."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        html = resp.text

        # Modal elements
        self.assertIn('aria-label="全域指令與股票搜尋面板"', html)
        self.assertIn('v-if="showCommandPalette"', html)
        self.assertIn('@keydown.window.esc="closeCommandPalette"', html)
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


if __name__ == "__main__":
    unittest.main()
