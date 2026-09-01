"""
Unit tests for data.cache
"""
import unittest
import os
import pandas as pd
from data.cache import CacheManager


class TestCacheManager(unittest.TestCase):

    def setUp(self):
        self.test_ticker = "TEST_CACHE_TICKER"
        self.test_df = pd.DataFrame({
            "Open": [100.0, 101.0],
            "High": [105.0, 106.0],
            "Low": [99.0, 100.0],
            "Close": [103.0, 104.0],
            "Volume": [1000, 2000]
        })

    def tearDown(self):
        # Clean up test files
        p1 = CacheManager._get_stock_data_path(self.test_ticker, "1mo")
        if os.path.exists(p1):
            os.remove(p1)
        p2 = CacheManager._get_fundamentals_path(self.test_ticker)
        if os.path.exists(p2):
            os.remove(p2)

    def test_stock_data_cache_roundtrip(self):
        CacheManager.set_stock_data(self.test_ticker, "1mo", self.test_df)
        cached = CacheManager.get_stock_data(self.test_ticker, "1mo", max_age_hours=1)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached["Close"].iloc[-1], 104.0)

    def test_fundamentals_cache_roundtrip(self):
        fund_data = {"pe": 15.5, "pb": 2.1, "forward_pe": 14.0, "ev_ebitda": 8.5}
        CacheManager.set_fundamentals(self.test_ticker, fund_data)
        cached = CacheManager.get_fundamentals(self.test_ticker, max_age_hours=1)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["pe"], 15.5)
        self.assertEqual(cached["pb"], 2.1)

    def test_index_cache_roundtrip(self):
        test_index = "TEST_INDEX"
        stocks = ["2330.TW", "2317.TW"]
        name_map = {"2330.TW": "台積電", "2317.TW": "鴻海"}
        CacheManager.set_index_cache(test_index, stocks, name_map)
        cached = CacheManager.get_index_cache(test_index, max_age_days=1)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["stocks"], stocks)
        self.assertEqual(cached["name_map"]["2330.TW"], "台積電")


if __name__ == "__main__":
    unittest.main()
