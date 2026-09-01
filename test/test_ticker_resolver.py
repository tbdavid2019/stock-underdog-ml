"""Tests for stock ticker and company-name resolution."""

import os
import unittest

from data.duckdb_manager import DuckDBManager
from data.ticker_resolver import normalize_ticker, resolve_alias


class TestTickerResolver(unittest.TestCase):
    TEST_DB = "test/scratch/test_ticker_resolver.duckdb"

    def setUp(self):
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(self.TEST_DB):
            os.remove(self.TEST_DB)
        self.db = DuckDBManager(db_path=self.TEST_DB)

    def tearDown(self):
        if os.path.exists(self.TEST_DB):
            os.remove(self.TEST_DB)

    def test_normalizes_case_and_numeric_taiwan_symbol(self):
        self.assertEqual(normalize_ticker(" tsla "), "TSLA")
        self.assertEqual(normalize_ticker("2330"), "2330.TW")
        self.assertEqual(normalize_ticker("2330.tw"), "2330.TW")

    def test_resolves_requested_company_aliases(self):
        self.assertEqual(resolve_alias("Tesla"), "TSLA")
        self.assertEqual(resolve_alias("特斯拉"), "TSLA")
        self.assertEqual(resolve_alias("聯電"), "2303.TW")
        self.assertEqual(resolve_alias("聯華電子"), "2303.TW")

    def test_resolves_company_name_from_local_market_metadata(self):
        self.db.save_daily_bars_batch([{
            "date": "2026-09-01",
            "ticker": "2303.TW",
            "raw_code": "2303",
            "name": "聯華電子",
            "close": 52.0,
        }])

        resolved = self.db.resolve_ticker("聯華電子")

        self.assertEqual(resolved["ticker"], "2303.TW")
        self.assertEqual(resolved["name"], "聯華電子")

    def test_unknown_company_is_not_mapped_to_a_stock(self):
        self.assertIsNone(self.db.resolve_ticker("這不是股票"))


if __name__ == "__main__":
    unittest.main()
