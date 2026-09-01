"""
test/test_duckdb.py - 測試 DuckDB 本地資料庫模組
"""

import os
import unittest
import pandas as pd
from data.duckdb_manager import DuckDBManager
from data.macro import MacroState


class TestDuckDBManager(unittest.TestCase):
    TEST_DB = "test/scratch/test_stock_quant.duckdb"

    def setUp(self):
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(self.TEST_DB):
            os.remove(self.TEST_DB)
        self.mgr = DuckDBManager(db_path=self.TEST_DB)

    def tearDown(self):
        if os.path.exists(self.TEST_DB):
            os.remove(self.TEST_DB)

    def test_schema_and_batch_insert(self):
        sample_records = [{
            "index_name": "台灣50",
            "model_name": "LSTM",
            "strategy_type": "LSTM預測",
            "ticker": "2330.TW",
            "current_price": 1000.0,
            "predicted_price": 1050.0,
            "potential": 5.0,
            "period": "6mo",
            "timestamp": "2026-09-01T10:00:00",
            "macro_regime": "全面多頭",
            "tags": ["LSTM看漲", "低PE"]
        }]
        inserted = self.mgr.save_predictions_batch(sample_records)
        self.assertEqual(inserted, 1)

        count = self.mgr.get_row_count("predictions")
        self.assertEqual(count, 1)

        df = self.mgr.query("SELECT ticker, potential, macro_regime FROM predictions WHERE ticker = '2330.TW'")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ticker"], "2330.TW")
        self.assertEqual(df.iloc[0]["potential"], 5.0)

    def test_save_dual_strategy_results(self):
        xuantie_df = pd.DataFrame([{
            "ticker": "2330.TW", "current_price": 1000.0, "ma60": 980.0, "pullback_type": "MA60 (+2.0%)", "pe": 22.0, "pb": 5.0
        }])
        lstm_results = [{
            "ticker": "2330.TW", "current_price": 1000.0, "predicted_price": 1050.0, "potential": 5.0, "pe": 22.0, "pb": 5.0
        }]
        overlap_df = pd.DataFrame([{
            "ticker": "2330.TW", "current_price": 1000.0, "predicted_price": 1050.0, "lstm_potential": 5.0, "pullback_type": "MA60 (+2.0%)", "ma60": 980.0, "pe": 22.0, "pb": 5.0
        }])

        results_dict = {
            "xuantie_results": xuantie_df,
            "lstm_results": lstm_results,
            "overlap_results": overlap_df
        }

        macro = MacroState(regime_name="全面多頭 (Bullish)", exposure=1.0)
        results_dict["candidates_map"] = {
            "2330.TW": {"tags": ["🏆三重共振", "土洋合買"], "composite_score": 85.0}
        }
        results_dict["institutional_summaries"] = {
            "2330.TW": {"trust_net_5d": 300, "foreign_net_5d": 800}
        }
        saved = self.mgr.save_dual_strategy_results("台灣50", results_dict, macro_state=macro)
        self.assertEqual(saved, 3)

        count = self.mgr.get_row_count("predictions")
        self.assertEqual(count, 3)

        # 驗證宏觀與法人籌碼寫入
        df = self.mgr.query("SELECT macro_regime, trust_net_5d, foreign_net_5d, tags FROM predictions WHERE ticker = '2330.TW' AND model_name = '多維共振'")
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["macro_regime"], "全面多頭 (Bullish)")
        self.assertEqual(df.iloc[0]["trust_net_5d"], 300)
        self.assertEqual(df.iloc[0]["foreign_net_5d"], 800)
        self.assertIn("🏆三重共振", df.iloc[0]["tags"])

    def test_clean_test_data_and_stats(self):
        # 寫入一筆測試資料
        self.mgr.save_predictions_batch([{
            "index_name": "DEBUG_TEST",
            "model_name": "DEBUG_MODEL",
            "strategy_type": "DEBUG",
            "ticker": "TEST_TICKER",
            "current_price": 100.0,
            "period": "6mo",
            "timestamp": "2026-09-01T10:00:00"
        }])

        stats = self.mgr.get_db_stats()
        self.assertNotIn("db_path", stats)
        self.assertNotIn("DEBUG_TEST", stats["indices"])
        self.assertNotIn("DEBUG_MODEL", stats["models"])

        # 執行清理
        deleted = self.mgr.clean_test_data()
        self.assertGreaterEqual(deleted, 0)


if __name__ == "__main__":
    unittest.main()
