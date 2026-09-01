"""
Unit tests for data.fetcher
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from data.fetcher import StockFetcher, add_base_indicators


class TestStockFetcher(unittest.TestCase):

    def test_add_base_indicators(self):
        dates = pd.date_range("2026-01-01", periods=100)
        df = pd.DataFrame({
            "Open": np.linspace(100, 200, 100),
            "High": np.linspace(105, 205, 100),
            "Low": np.linspace(95, 195, 100),
            "Close": np.linspace(100, 200, 100),
            "Volume": np.full(100, 1000)
        }, index=dates)

        result_df = add_base_indicators(df)
        self.assertIn("ma5", result_df.columns)
        self.assertIn("ma60", result_df.columns)
        self.assertIn("MA60", result_df.columns)
        self.assertIn("rsi14", result_df.columns)
        self.assertFalse(result_df["ma60"].isna().all())

    @patch("requests.get")
    def test_fetch_index_upstream_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "TW0050": {"2330": "台積電", "2317": "鴻海"}
        }
        mock_get.return_value = mock_resp

        stocks, name_map = StockFetcher._fetch_index_with_cache_fallback(
            "TW0050", "http://test", "TW0050", suffix=".TW"
        )
        self.assertEqual(stocks, ["2330.TW", "2317.TW"])
        self.assertEqual(name_map["2330.TW"], "台積電")

    @patch("requests.get", side_effect=Exception("Network error"))
    @patch("data.cache.CacheManager.get_index_cache")
    def test_fetch_index_fallback_to_cache(self, mock_get_cache, mock_get):
        mock_get_cache.return_value = {
            "stocks": ["2330.TW", "2454.TW"],
            "name_map": {"2330.TW": "台積電", "2454.TW": "聯發科"}
        }

        stocks, name_map = StockFetcher._fetch_index_with_cache_fallback(
            "TW0050", "http://test", "TW0050", suffix=".TW"
        )
        self.assertEqual(stocks, ["2330.TW", "2454.TW"])
        self.assertEqual(name_map["2454.TW"], "聯發科")


if __name__ == "__main__":
    unittest.main()
