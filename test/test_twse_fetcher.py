"""
test/test_twse_fetcher.py - Unit tests for TWSEDailyFetcher and DuckDB market ingestion.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile
import os

from data.twse_daily_fetcher import TWSEDailyFetcher
from data.duckdb_manager import DuckDBManager


class TestTWSEDailyFetcher(unittest.TestCase):

    def test_roc_date_to_iso(self):
        self.assertEqual(TWSEDailyFetcher.roc_date_to_iso("1150831"), "2026-08-31")
        self.assertEqual(TWSEDailyFetcher.roc_date_to_iso("115/08/31"), "2026-08-31")
        self.assertEqual(TWSEDailyFetcher.roc_date_to_iso("1141225"), "2025-12-25")

    @patch("requests.get")
    def test_fetch_twse_quotes_mock(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "Date": "1150831",
                "Code": "2330",
                "Name": "台積電",
                "TradeVolume": "25,000,000",
                "TradeValue": "25,000,000,000",
                "OpeningPrice": "1,000.00",
                "HighestPrice": "1,010.00",
                "LowestPrice": "995.00",
                "ClosingPrice": "1,005.00",
                "Change": "+5.00",
                "Transaction": "12,000"
            }
        ]
        mock_get.return_value = mock_resp

        df = TWSEDailyFetcher.fetch_twse_quotes()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ticker"], "2330.TW")
        self.assertEqual(df.iloc[0]["date"], "2026-08-31")
        self.assertEqual(df.iloc[0]["close"], 1005.0)
        self.assertEqual(df.iloc[0]["volume"], 25000000)

    @patch("requests.get")
    def test_fetch_tpex_quotes_mock(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "Date": "1150831",
                "SecuritiesCompanyCode": "8069",
                "CompanyName": "元太",
                "Close": "250.50",
                "Change": "+2.50",
                "Open": "248.00",
                "High": "252.00",
                "Low": "247.50",
                "TradingShares": "5,000,000",
                "TransactionAmount": "1,250,000,000",
                "TransactionNumber": "3,500"
            }
        ]
        mock_get.return_value = mock_resp

        df = TWSEDailyFetcher.fetch_tpex_quotes()
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ticker"], "8069.TWO")
        self.assertEqual(df.iloc[0]["date"], "2026-08-31")
        self.assertEqual(df.iloc[0]["close"], 250.5)

    def test_duckdb_daily_bars_save_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = os.path.join(tmp_dir, "test_market.duckdb")
            mgr = DuckDBManager(db_path=test_db)
            mgr.enabled = True
            mgr._init_schema()

            sample_records = [
                {
                    "date": "2026-08-30",
                    "ticker": "2330.TW",
                    "raw_code": "2330",
                    "name": "台積電",
                    "open": 990.0,
                    "high": 1000.0,
                    "low": 985.0,
                    "close": 995.0,
                    "volume": 20000000,
                    "trade_value": 20000000000.0,
                    "transaction_count": 10000,
                    "market": "TWSE",
                    "created_at": "2026-08-30 16:00:00"
                },
                {
                    "date": "2026-08-31",
                    "ticker": "2330.TW",
                    "raw_code": "2330",
                    "name": "台積電",
                    "open": 1000.0,
                    "high": 1010.0,
                    "low": 995.0,
                    "close": 1005.0,
                    "volume": 25000000,
                    "trade_value": 25000000000.0,
                    "transaction_count": 12000,
                    "market": "TWSE",
                    "created_at": "2026-08-31 16:00:00"
                }
            ]

            saved = mgr.save_daily_bars_batch(sample_records)
            self.assertEqual(saved, 2)

            # 測試讀取歷史 K 棒
            bars = mgr.get_daily_bars_for_ticker("2330.TW")
            self.assertEqual(len(bars), 2)
            self.assertIn("Close", bars.columns)
            self.assertEqual(bars.iloc[-1]["Close"], 1005.0)

            # 測試重複寫入 (Upsert 覆蓋去重)
            saved_repeat = mgr.save_daily_bars_batch(sample_records)
            self.assertEqual(saved_repeat, 2)
            bars_repeat = mgr.get_daily_bars_for_ticker("2330.TW")
            self.assertEqual(len(bars_repeat), 2)


if __name__ == "__main__":
    unittest.main()
