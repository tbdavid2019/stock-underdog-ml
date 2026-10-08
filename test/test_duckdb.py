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

    def test_get_latest_predictions_timezone_and_midnight(self):
        # 寫入包含 UTC 與 Asia/Taipei 時區的最新批次資料
        sample_records = [
            {
                "index_name": "SP500",
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": "AAPL",
                "current_price": 200.0,
                "predicted_price": 210.0,
                "potential": 5.0,
                "period": "6mo",
                # UTC 時間：2026-09-09 16:30 UTC -> 台北時間 2026-09-10 00:30 (剛跨過台北午夜)
                "timestamp": "2026-09-09T16:30:00Z",
                "macro_regime": "全面多頭",
                "tags": ["LSTM看漲"]
            }
        ]
        self.mgr.save_predictions_batch(sample_records)
        res = self.mgr.get_latest_predictions(index_name="SP500")
        # 台北時間應準確換算為 2026-09-10 (而非 UTC 的 2026-09-09)
        self.assertEqual(res["batch_date"], "2026-09-10")
        self.assertEqual(len(res["records"]), 1)
        self.assertEqual(res["records"][0]["analysis_date"], "2026-09-10")
        self.assertGreaterEqual(res["records"][0]["age_hours"], 0.0)

    def test_get_latest_candidate_snapshot_merges_strategies(self):
        """驗證同一批次中玄鐵、LSTM、TimesFM、多維共振列能完整合併為單一 snapshot"""
        ts = "2026-10-08T10:00:00"
        records = [
            {
                "index_name": "台灣50",
                "model_name": "玄鐵重劍",
                "strategy_type": "玄鐵重劍",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "pullback_type": "MA60 (+2.0%)",
                "ma60": 980.0,
                "pe": 22.0,
                "pb": 5.0,
                "timestamp": ts,
                "tags": "玄鐵買點"
            },
            {
                "index_name": "台灣50",
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "predicted_price": 1055.0,
                "potential": 5.5,
                "timestamp": ts,
                "tags": "LSTM看漲 | 科技龍頭"
            },
            {
                "index_name": "台灣50",
                "model_name": "TimesFM",
                "strategy_type": "TimesFM預測",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "predicted_price": 1048.0,
                "potential": 4.8,
                "timestamp": ts,
                "tags": "TimesFM看漲"
            },
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 5.5,
                "composite_score": 88.5,
                "foreign_net_5d": 1000,
                "trust_net_5d": 500,
                "timestamp": ts,
                "tags": "👑四重共振 | 土洋合買"
            }
        ]
        self.mgr.save_predictions_batch(records)
        snap = self.mgr.get_latest_candidate_snapshot("2330.TW")

        self.assertIsNotNone(snap)
        self.assertEqual(snap["ticker"], "2330.TW")
        self.assertEqual(snap["current_price"], 1000.0)
        self.assertEqual(snap["lstm_potential"], 5.5)
        self.assertEqual(snap["timesfm_potential"], 4.8)
        self.assertEqual(snap["pullback_type"], "MA60 (+2.0%)")
        self.assertEqual(snap["ma60"], 980.0)
        self.assertEqual(snap["pe"], 22.0)
        self.assertEqual(snap["pb"], 5.0)
        self.assertEqual(snap["foreign_net_5d"], 1000)
        self.assertEqual(snap["trust_net_5d"], 500)
        self.assertEqual(snap["composite_score"], 88.5)
        self.assertIn("土洋合買", snap["tags"])
        self.assertIn("👑四重共振", snap["tags"])
        self.assertIn("玄鐵買點", snap["tags"])

    def test_get_latest_candidate_snapshot_preserves_zero_score(self):
        """驗證當儲存的綜合評分為 0.0 (例如宏觀風控曝險歸零) 時，不會被誤當缺失資料覆蓋為合成正分"""
        ts = "2026-10-08T10:00:00"
        records = [
            {
                "index_name": "台灣50",
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 15.0,
                "composite_score": 0.0,  # 宏觀曝險為 0 時的風控歸零結果
                "timestamp": ts,
                "tags": "LSTM看漲"
            }
        ]
        self.mgr.save_predictions_batch(records)
        snap = self.mgr.get_latest_candidate_snapshot("2330.TW")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["composite_score"], 0.0)

    def test_get_resonance_candidates_excludes_defense_and_stale_history(self):
        """驗證 get_resonance_candidates 以最新快照為準，排除轉為防守的標的，且絕不回溯選取舊的看漲記錄"""
        records = [
            # STALE 標的：昨天為共振推薦，今天最新批次轉為防守
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "STALE",
                "current_price": 100.0,
                "potential": 10.0,
                "timestamp": "2026-10-07T10:00:00",
                "tags": "🏆三重共振 | 土洋合買"
            },
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "STALE",
                "current_price": 90.0,
                "potential": -10.0,
                "timestamp": "2026-10-08T10:00:00",
                "tags": "🏆三重共振 | 🔻防守"
            },
            # 正向標的：今日持續共振
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 5.0,
                "timestamp": "2026-10-08T10:00:00",
                "tags": "👑四重共振 | 土洋合買"
            }
        ]
        self.mgr.save_predictions_batch(records)
        cands = self.mgr.get_resonance_candidates()
        tickers = [c["ticker"] for c in cands]
        self.assertIn("2330.TW", tickers)
        self.assertNotIn("STALE", tickers, "轉為防守的 STALE 標的不應入榜，且不應回溯取得前一天的共振記錄")

    def test_get_resonance_candidates_multiple_strategy_rows_in_batch(self):
        """驗證當最新批次中包含同一標的之非共振策略列 (如玄鐵、LSTM) 與共振列時，共振篩選優先於排序，不會漏失標的"""
        ts = "2026-10-08T12:00:00"
        records = [
            {
                "index_name": "台灣50",
                "model_name": "玄鐵重劍",
                "strategy_type": "玄鐵重劍",
                "ticker": "2454.TW",
                "current_price": 1200.0,
                "potential": None,
                "timestamp": ts,
                "tags": "玄鐵買點"
            },
            {
                "index_name": "台灣50",
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": "2454.TW",
                "current_price": 1200.0,
                "potential": 6.0,
                "timestamp": ts,
                "tags": "LSTM看漲"
            },
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2454.TW",
                "current_price": 1200.0,
                "potential": 6.5,
                "timestamp": ts,
                "tags": "🏆三重共振 | 土洋合買"
            }
        ]
        self.mgr.save_predictions_batch(records)
        cands = self.mgr.get_resonance_candidates()
        tickers = [c["ticker"] for c in cands]
        self.assertIn("2454.TW", tickers)
        matched = next(c for c in cands if c["ticker"] == "2454.TW")
        self.assertEqual(matched["potential"], 6.5)

    def test_get_resonance_candidates_scoped_by_index(self):
        """驗證當提供 index_name 時，即使同一時間戳其他指數存在相同標的之記錄，也不會跨指數混雜"""
        ts = "2026-10-08T15:00:00"
        records = [
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 5.0,
                "timestamp": ts,
                "tags": "🏆三重共振"
            },
            {
                "index_name": "台灣中型100",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 9.9,
                "timestamp": ts,
                "tags": "🏆三重共振"
            }
        ]
        self.mgr.save_predictions_batch(records)
        cands_tw50 = self.mgr.get_resonance_candidates(index_name="台灣50")
        self.assertEqual(len(cands_tw50), 1)
        self.assertEqual(cands_tw50[0]["index_name"], "台灣50")
        self.assertEqual(cands_tw50[0]["potential"], 5.0)

        cands_tw100 = self.mgr.get_resonance_candidates(index_name="台灣中型100")
        self.assertEqual(len(cands_tw100), 1)
        self.assertEqual(cands_tw100[0]["index_name"], "台灣中型100")
        self.assertEqual(cands_tw100[0]["potential"], 9.9)

    def test_get_latest_candidate_snapshot_excludes_test_and_debug_indices(self):
        """驗證當同一時間戳存在測試/偵錯指數記錄時，get_latest_candidate_snapshot 不會將其併入快照中"""
        ts = "2026-10-08T18:00:00"
        records = [
            # 生產環境記錄
            {
                "index_name": "台灣50",
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 4.5,
                "timestamp": ts,
                "tags": "PROD_TAG"
            },
            # 測試/偵錯記錄 (同時間戳)
            {
                "index_name": "DEBUG_INDEX_TEST",
                "model_name": "TimesFM",
                "strategy_type": "TimesFM預測",
                "ticker": "2330.TW",
                "current_price": 9999.0,
                "potential": -99.0,
                "timestamp": ts,
                "tags": "DEBUG_TAG"
            }
        ]
        self.mgr.save_predictions_batch(records)
        snap = self.mgr.get_latest_candidate_snapshot("2330.TW")
        self.assertIsNotNone(snap)
        self.assertEqual(snap["current_price"], 1000.0)
        self.assertEqual(snap["lstm_potential"], 4.5)
        self.assertIsNone(snap["timesfm_potential"])
        self.assertIn("PROD_TAG", snap["tags"])
        self.assertNotIn("DEBUG_TAG", snap["tags"])

    def test_get_resonance_candidates_with_null_tags(self):
        """驗證當最新批次中 model_name 為多維共振但 tags 為 NULL 時，不會因 SQL NULL 運算漏失標的"""
        ts = "2026-10-08T20:00:00"
        records = [
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 1000.0,
                "potential": 6.0,
                "timestamp": ts,
                "tags": None  # NULL tags
            }
        ]
        self.mgr.save_predictions_batch(records)
        cands = self.mgr.get_resonance_candidates()
        tickers = [c["ticker"] for c in cands]
        self.assertIn("2330.TW", tickers)


if __name__ == "__main__":
    unittest.main()
