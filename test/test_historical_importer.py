"""
test/test_historical_importer.py - Unit tests for HistoricalImporter.
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os

from data.historical_importer import HistoricalImporter
from data.duckdb_manager import DuckDBManager


class TestHistoricalImporter(unittest.TestCase):

    @patch("requests.get")
    def test_import_institutional_flows_mock(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            "date,code,name,foreign_net,trust_net,dealer_net,market\n"
            "2025-09-11,0050,元大台灣50,11379253,0,6065139,TWSE\n"
            "2025-09-11,2330,台積電,5000000,1000000,200000,TWSE\n"
        )
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = os.path.join(tmp_dir, "test_hist.duckdb")
            mgr = DuckDBManager(db_path=test_db)
            mgr.enabled = True
            mgr._init_schema()

            imported = HistoricalImporter.import_institutional_flows(db_mgr=mgr)
            self.assertGreater(imported, 0)

            # 驗證查詢
            df_inst = mgr.get_institutional_flow_for_ticker("2330.TW")
            self.assertFalse(df_inst.empty)
            self.assertEqual(df_inst.iloc[0]["foreign_net"], 5000000)

    @patch("requests.get")
    def test_import_benchmark_k_lines_mock(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = (
            "Date,Open,High,Low,Close,Volume\n"
            "2025-09-10,980.0,990.0,975.0,985.0,20000000\n"
            "2025-09-11,990.0,1005.0,985.0,1000.0,25000000\n"
        )
        mock_get.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = os.path.join(tmp_dir, "test_hist_k.duckdb")
            mgr = DuckDBManager(db_path=test_db)
            mgr.enabled = True
            mgr._init_schema()

            imported = HistoricalImporter.import_benchmark_k_lines(tickers=["2330"], db_mgr=mgr)
            self.assertEqual(imported, 2)

            bars = mgr.get_daily_bars_for_ticker("2330.TW")
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars.iloc[-1]["Close"], 1000.0)


if __name__ == "__main__":
    unittest.main()
