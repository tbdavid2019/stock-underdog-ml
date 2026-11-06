"""
test/test_local_bars_loader.py - Tests for local bars loader and global_daily_bars.
Validates:
1. global_daily_bars schema creation and upsert_global_daily_bars.
2. load_local_close_series recent window order (DESC LIMIT reversed to ASC).
3. as_of_date point-in-time filtering (zero lookahead bias).
4. Unavailable tickers properly identified.
5. Zero network I/O verification.
"""

import os
import unittest
from unittest.mock import patch
import pandas as pd

from data.duckdb_manager import DuckDBManager


class TestLocalBarsLoader(unittest.TestCase):
    TEST_DB = "test/scratch/test_local_bars_loader.duckdb"

    @classmethod
    def setUpClass(cls):
        os.makedirs("test/scratch", exist_ok=True)
        for f in [cls.TEST_DB, f"{cls.TEST_DB}.wal", f"{cls.TEST_DB}.write.lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        cls.db = DuckDBManager(db_path=cls.TEST_DB)

        # Seed 10 dates for 2330.TW in tw_daily_bars
        tw_bars = []
        for i in range(1, 11):
            day_str = f"2026-11-{i:02d}"
            tw_bars.append({
                "date": day_str,
                "ticker": "2330.TW",
                "raw_code": "2330",
                "name": "台積電",
                "open": 1000.0 + i,
                "high": 1010.0 + i,
                "low": 995.0 + i,
                "close": 1005.0 + i,
                "volume": 10000 + i * 100,
                "trade_value": 10000000.0,
                "transaction_count": 500,
                "market": "TWSE",
                "created_at": "2026-11-06T00:00:00",
            })
        cls.db.save_daily_bars_batch(tw_bars)

        # Seed 10 dates for NVDA in global_daily_bars
        global_bars = []
        for i in range(1, 11):
            day_str = f"2026-11-{i:02d}"
            global_bars.append({
                "ticker": "NVDA",
                "market": "US",
                "currency": "USD",
                "date": day_str,
                "open": 140.0 + i,
                "high": 145.0 + i,
                "low": 139.0 + i,
                "close": 142.0 + i,
                "volume": 5000000 + i * 1000,
                "is_adjusted": True,
                "updated_at": "2026-11-06 00:00:00",
            })
        cls.db.upsert_global_daily_bars(global_bars)

    @classmethod
    def tearDownClass(cls):
        for f in [cls.TEST_DB, f"{cls.TEST_DB}.wal", f"{cls.TEST_DB}.write.lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_global_daily_bars_upsert_and_query(self):
        with self.db._get_connection(read_only=True) as con:
            cnt = con.execute("SELECT COUNT(*) FROM global_daily_bars WHERE ticker = 'NVDA'").fetchone()[0]
            self.assertEqual(cnt, 10)

    def test_load_recent_window_sorting(self):
        # Request lookback of 5 bars with no as_of_date (most recent)
        res = self.db.load_local_close_series(["2330.TW", "NVDA"], lookback_bars=5)
        self.assertIn("2330.TW", res["series"])
        self.assertIn("NVDA", res["series"])
        
        tw_series = res["series"]["2330.TW"]
        nvda_series = res["series"]["NVDA"]

        self.assertEqual(len(tw_series), 5)
        self.assertEqual(len(nvda_series), 5)

        # The dates should be strictly ASC: 2026-11-06 to 2026-11-10 (the 5 most recent dates!)
        tw_dates = list(tw_series.index)
        self.assertEqual(tw_dates, ["2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10"])
        self.assertEqual(tw_series.iloc[-1], 1015.0)

        nvda_dates = list(nvda_series.index)
        self.assertEqual(nvda_dates, ["2026-11-06", "2026-11-07", "2026-11-08", "2026-11-09", "2026-11-10"])
        self.assertEqual(nvda_series.iloc[-1], 152.0)

        self.assertEqual(res["price_adjustment_modes"]["2330.TW"], "raw_close")
        self.assertEqual(res["price_adjustment_modes"]["NVDA"], "adjusted_close")
        self.assertEqual(res["currencies"]["2330.TW"], "TWD")
        self.assertEqual(res["currencies"]["NVDA"], "USD")

    def test_as_of_date_zero_lookahead_bias(self):
        # If as_of_date is 2026-11-05, it must NOT return dates after 2026-11-05
        res = self.db.load_local_close_series(["2330.TW", "NVDA"], as_of_date="2026-11-05", lookback_bars=10)
        tw_dates = list(res["series"]["2330.TW"].index)
        nvda_dates = list(res["series"]["NVDA"].index)

        self.assertEqual(tw_dates, ["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"])
        self.assertEqual(nvda_dates, ["2026-11-01", "2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05"])
        self.assertNotIn("2026-11-06", tw_dates)

    def test_unavailable_tickers(self):
        res = self.db.load_local_close_series(["2330.TW", "NON_EXISTENT_XYZ_123"], lookback_bars=5)
        self.assertIn("2330.TW", res["available_tickers"])
        self.assertIn("NON_EXISTENT_XYZ_123", res["unavailable_tickers"])
        self.assertNotIn("NON_EXISTENT_XYZ_123", res["series"])

    def test_zero_network_calls(self):
        with patch("urllib.request.urlopen") as mock_url, patch("requests.get") as mock_req:
            mock_url.side_effect = RuntimeError("Network access forbidden!")
            mock_req.side_effect = RuntimeError("Network access forbidden!")
            res = self.db.load_local_close_series(["2330.TW", "NVDA", "MISSING_TICKER"], lookback_bars=5)
            self.assertEqual(len(res["available_tickers"]), 2)
            self.assertEqual(len(res["unavailable_tickers"]), 1)
            mock_url.assert_not_called()
            mock_req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
