"""
Unit tests for Nikkei 225 (日股 / Japanese Equities) Integration.
Validates constituent resolution, market calendar (TSE holidays), macro regime analysis,
and AI industry value chain graph integration.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from data.fetcher import StockFetcher, FALLBACK_NIKKEI225_STOCKS
from core.market_calendar import MarketCalendar
from data.macro import MacroRegimeAnalyzer
from data.industry_chain_fetcher import IndustryChainFetcher, EMBEDDED_NODES, EMBEDDED_RELATIONS


class TestNikkei225Pipeline(unittest.TestCase):

    def test_nikkei225_fallback_constituents(self):
        """Test that FALLBACK_NIKKEI225_STOCKS contains 225 tickers, all ending with .T."""
        self.assertEqual(len(FALLBACK_NIKKEI225_STOCKS), 225)
        for ticker in FALLBACK_NIKKEI225_STOCKS:
            self.assertTrue(ticker.endswith(".T"), f"Ticker {ticker} should end with .T")

        # Crucial anchor constituents
        key_stocks = ["8035.T", "9984.T", "6857.T", "6146.T", "4063.T", "6920.T", "7203.T", "6758.T", "7974.T"]
        for ks in key_stocks:
            self.assertIn(ks, FALLBACK_NIKKEI225_STOCKS, f"{ks} must be present in Nikkei 225 list")

    @patch("data.fetcher.requests.get")
    @patch("data.fetcher.CacheManager.get_index_cache")
    def test_get_nikkei225_stocks_from_cache(self, mock_get_cache, mock_req_get):
        """Test that get_nikkei225_stocks retrieves from cache if API is unavailable."""
        mock_req_get.side_effect = Exception("API down")
        mock_get_cache.return_value = {"stocks": ["8035.T"] * 225}
        stocks = StockFetcher.get_nikkei225_stocks()
        self.assertEqual(len(stocks), 225)
        self.assertEqual(stocks[0], "8035.T")

    @patch("data.fetcher.requests.get")
    @patch("data.fetcher.CacheManager.get_index_cache", return_value=None)
    @patch("data.fetcher.CacheManager.set_index_cache")
    @patch("urllib.request.urlopen")
    def test_get_nikkei225_stocks_fallback_on_network_error(self, mock_urlopen, mock_set_cache, mock_get_cache, mock_req_get):
        """Test fallback to embedded 225 constituents if API and Wikipedia scrape fails."""
        mock_req_get.side_effect = Exception("API down")
        mock_urlopen.side_effect = Exception("Network timeout")
        stocks = StockFetcher.get_nikkei225_stocks()
        self.assertEqual(len(stocks), 225)
        self.assertIn("8035.T", stocks)
        self.assertIn("9984.T", stocks)

    @patch("data.fetcher.requests.get")
    def test_get_nikkei225_stocks_from_api(self, mock_req_get):
        """Test that get_nikkei225_stocks retrieves from DataAPI when available."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"NIKKEI225": {f"{i:04d}.T": f"Stock {i}" for i in range(225)}}
        mock_req_get.return_value = mock_resp
        stocks = StockFetcher.get_nikkei225_stocks()
        self.assertEqual(len(stocks), 225)
        self.assertEqual(stocks[0], "0000.T")

    def test_get_stocks_by_index_jp(self):
        """Test that get_stocks_by_index resolves '日經225' and synonyms."""
        with patch.object(StockFetcher, "get_nikkei225_stocks", return_value=["8035.T", "9984.T"]):
            for idx_name in ["日經225", "日股225", "NIKKEI225", "N225", "JP225"]:
                res = StockFetcher.get_stocks_by_index(idx_name)
                self.assertEqual(res, ["8035.T", "9984.T"], f"Failed for index name {idx_name}")

    def test_get_index_name_map_jp(self):
        """Test that get_index_name_map resolves Japanese names."""
        name_map = StockFetcher.get_index_name_map("日經225")
        self.assertIn("8035.T", name_map)
        self.assertIn("9984.T", name_map)
        self.assertIn("東京威力科創", name_map["8035.T"])
        self.assertIn("軟銀集團", name_map["9984.T"])

    def test_market_calendar_jp(self):
        """Test MarketCalendar correctly handles TSE holidays and market resolution."""
        cal = MarketCalendar()

        # Resolution
        self.assertEqual(cal.resolve_market("8035.T"), "JP")
        self.assertEqual(cal.resolve_market("9984.T"), "JP")
        self.assertEqual(cal.resolve_market("日經225"), "JP")
        self.assertEqual(cal.resolve_market("JP225"), "JP")
        self.assertEqual(cal.resolve_market("N225"), "JP")

        # TSE Holidays
        # Jan 1 is New Year's Day (TSE closed)
        self.assertFalse(cal.is_trading_day("2026-01-01", market_or_ticker="JP"))
        # Jan 2 is Market Holiday in Japan (TSE closed Jan 1-3)
        self.assertFalse(cal.is_trading_day("2026-01-02", market_or_ticker="JP"))
        # Golden Week (May 4, Greenery Day)
        self.assertFalse(cal.is_trading_day("2026-05-04", market_or_ticker="JP"))
        # Dec 31 is TSE closing day
        self.assertFalse(cal.is_trading_day("2026-12-31", market_or_ticker="JP"))
        # 2026 Autumnal Equinox (Sept 23) & Kokumin no Kyujitsu (Sept 22)
        self.assertFalse(cal.is_trading_day("2026-09-23", market_or_ticker="JP"))
        self.assertFalse(cal.is_trading_day("2026-09-22", market_or_ticker="JP"))

        # Status check
        status = cal.get_trading_day_status("2026-01-01", market_or_ticker="JP")
        self.assertFalse(status["is_trading_day"])
        self.assertEqual(status["market"], "JP")

        # Recent trading days for JP
        recent_jp = cal.get_recent_trading_days(count=5, market_or_ticker="JP")
        self.assertEqual(len(recent_jp), 5)

    @patch("data.macro.MacroRegimeAnalyzer._enrich_with_investing")
    @patch("yfinance.download")
    def test_macro_regime_jp(self, mock_yf_download, mock_enrich):
        """Test MacroRegimeAnalyzer.evaluate_jp_market returns valid macro score and exposure."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq="B")

        # MultiIndex columns as returned by yfinance for multiple tickers
        tickers = ["^N225", "^SOX", "^VIX", "JPY=X"]
        cols = pd.MultiIndex.from_product([tickers, ["Close", "Open", "High", "Low", "Volume"]])
        mock_df = pd.DataFrame(index=dates, columns=cols)
        mock_df[("^N225", "Close")] = np.linspace(35000, 39000, 100)
        mock_df[("^SOX", "Close")] = np.linspace(4800, 5200, 100)
        mock_df[("^VIX", "Close")] = [16.5] * 100
        mock_df[("JPY=X", "Close")] = [152.0] * 100

        mock_yf_download.return_value = mock_df

        result = MacroRegimeAnalyzer.evaluate_jp_market()
        self.assertEqual(result.market, "JP")
        self.assertTrue(result.data_available)
        self.assertIsNotNone(result.regime_name)
        self.assertIsNotNone(result.exposure)
        self.assertGreaterEqual(result.exposure, 0.0)
        self.assertLessEqual(result.exposure, 1.0)
        self.assertGreater(result.twii_price, 30000)

        # Test routing of JP aliases in evaluate_market
        res_alias = MacroRegimeAnalyzer.evaluate_market("jp", force_refresh=True)
        self.assertEqual(res_alias.market, "JP")
        res_alias_upper = MacroRegimeAnalyzer.evaluate_market("JP", force_refresh=True)
        self.assertEqual(res_alias_upper.market, "JP")
        res_alias_cjk = MacroRegimeAnalyzer.evaluate_market("日股", force_refresh=True)
        self.assertEqual(res_alias_cjk.market, "JP")

    def test_industry_chain_jp_integration(self):
        """Test that Japanese anchor nodes and relations are properly registered in IndustryChainFetcher."""
        fetcher = IndustryChainFetcher()
        graph = fetcher.get_theme_graph("ALL")

        # Find JP nodes
        all_nodes = graph["upstream_nodes"] + graph["midstream_nodes"] + graph["downstream_nodes"]
        jp_tickers = {n["ticker"]: n for n in all_nodes if n.get("market") == "JP"}

        self.assertIn("8035.T", jp_tickers)
        self.assertIn("6857.T", jp_tickers)
        self.assertIn("6146.T", jp_tickers)
        self.assertIn("4063.T", jp_tickers)
        self.assertIn("6920.T", jp_tickers)
        self.assertIn("9984.T", jp_tickers)

        # Check node metadata
        self.assertEqual(jp_tickers["8035.T"]["company"], "東京威力科創")
        self.assertEqual(jp_tickers["8035.T"]["layer"], "upstream")
        self.assertEqual(jp_tickers["9984.T"]["company"], "軟銀集團")

        # Check relations
        edges = graph["edges"]
        has_tel_tsmc = any(e["source"] == "8035.T" and e["target"] == "2330" for e in edges)
        self.assertTrue(has_tel_tsmc, "Relation 8035.T -> 2330 should exist in graph edges")

        has_advantest_nvda = any(e["source"] == "6857.T" and e["target"] == "NVDA" for e in edges)
        self.assertTrue(has_advantest_nvda, "Relation 6857.T -> NVDA should exist in graph edges")

    def test_ai_narrative_jp(self):
        """Test AINarrativeEngine correctly labels Japanese equity context."""
        from evaluators.ai_narrative import AINarrativeEngine
        from data.macro import MacroState

        engine = AINarrativeEngine()
        jp_macro = MacroState(
            market="JP",
            regime_name="強勢多頭",
            exposure=0.85,
            vix=16.5,
            data_available=True,
            twii_price=39200.0,
            twii_above_ma60=True,
            sox_above_ma60=True
        )
        report_data = {
            "overlap_results": pd.DataFrame([{
                "ticker": "8035.T",
                "potential": 6.8,
                "resonance_tier": "TIER_1_PINNACLE",
                "tags": ["雙模型共振", "EUV龍頭"]
            }])
        }

        # Test prompt context
        prompt = engine._build_context_prompt("日經225", jp_macro, report_data)
        self.assertIn("日本股市", prompt)
        self.assertIn("日股大盤", prompt)

        # Test template generation
        narrative = engine._generate_template_narrative("日經225", jp_macro, report_data)
        self.assertIn("日股大盤", narrative)
        self.assertIn("8035.T", narrative)


if __name__ == "__main__":
    unittest.main()
