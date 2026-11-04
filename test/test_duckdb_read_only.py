"""
test/test_duckdb_read_only.py - 測試 DuckDBManager 唯讀查詢、鎖衝突重試機制與 ForecastStore 批次寫入
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import duckdb

from data.duckdb_manager import DuckDBManager
from data.forecast_store import ForecastStore


class TestDuckDBReadOnlyAndLockRetry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_read_only.duckdb")
        self.mgr = DuckDBManager(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_query_defaults_to_read_only(self):
        """測試 query() 預設 read_only=True，且能正常查詢空表或已有資料"""
        # 先以寫入模式插入一筆預測記錄
        sample_records = [{
            "index_name": "台灣50",
            "model_name": "TimesFM",
            "strategy_type": "TimesFM看漲",
            "ticker": "2330.TW",
            "current_price": 1000.0,
            "predicted_price": 1080.0,
            "potential": 8.0,
            "period": "6mo",
            "timestamp": "2026-11-04T09:00:00",
            "macro_regime": "多頭",
            "tags": ["TimesFM看漲"]
        }]
        inserted = self.mgr.save_predictions_batch(sample_records)
        self.assertEqual(inserted, 1)

        # 呼叫 query，此時預設 read_only=True
        df = self.mgr.query("SELECT ticker, predicted_price, potential FROM predictions WHERE ticker = ?", ["2330.TW"])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ticker"], "2330.TW")
        self.assertAlmostEqual(df.iloc[0]["predicted_price"], 1080.0)

    def test_get_connection_retry_on_conflicting_lock_read_only(self):
        """測試 _get_connection(read_only=True) 遇到鎖衝突時能自動重試並成功獲取連線"""
        real_connect = duckdb.connect
        call_count = 0

        def flaky_connect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                # 模擬 DuckDB 前兩次連線時拋出衝突鎖例外
                raise duckdb.IOException("Conflicting lock is held to database")
            return real_connect(*args, **kwargs)

        with patch("duckdb.connect", side_effect=flaky_connect):
            con = self.mgr._get_connection(read_only=True, max_retries=4, retry_delay=0.01)
            self.assertIsNotNone(con)
            con.close()

        self.assertEqual(call_count, 3)

    def test_get_connection_retry_on_write_lock_with_allow_downgrade(self):
        """測試 _get_connection 在寫入鎖超時且開啟 allow_downgrade=True 時能優雅降級為 read_only"""
        real_connect = duckdb.connect

        def always_locked(*args, **kwargs):
            if kwargs.get("read_only", False) is False:
                raise duckdb.IOException("could not set lock on file")
            return real_connect(*args, **kwargs)

        with patch("duckdb.connect", side_effect=always_locked):
            con = self.mgr._get_connection(read_only=False, max_retries=2, retry_delay=0.01, allow_downgrade=True)
            self.assertIsNotNone(con)
            con.close()


class TestForecastStoreBatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tmp.name) / "forecast_batch.duckdb"
        self.store = ForecastStore(self.store_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_forecasts_batch_inserts_multiple_records(self):
        """測試 record_forecasts_batch 在單一連線交易中批次新增多筆預測"""
        self.store.record_run({"run_id": "batch_run_1", "model": "TimesFM", "model_revision": "v2"})

        batch_records = [
            {
                "forecast_id": f"fc_{i}",
                "run_id": "batch_run_1",
                "ticker": f"233{i}.TW",
                "market": "TW",
                "model_revision": "v2",
                "issued_at": "2026-11-04T08:00:00Z",
                "last_complete_bar": "2026-11-03",
                "target_date": "2026-11-04",
                "horizon": 1,
                "prediction": 100.0 + i,
                "metadata": {"batch_index": i}
            }
            for i in range(10)
        ]

        inserted_ids = self.store.record_forecasts_batch(batch_records)
        self.assertEqual(len(inserted_ids), 10)

        # 驗證所有 10 筆記錄已成功寫入
        all_forecasts = self.store.list_forecasts()
        self.assertEqual(len(all_forecasts), 10)
        tickers = {f["ticker"] for f in all_forecasts}
        self.assertIn("2330.TW", tickers)
        self.assertIn("2339.TW", tickers)

    def test_record_forecasts_batch_idempotency_and_mismatch(self):
        """測試 record_forecasts_batch 冪等性與衝突阻擋"""
        self.store.record_run({"run_id": "run_idempotent", "model": "MOIRAI-2"})

        record = {
            "forecast_id": "fc_idem_1",
            "run_id": "run_idempotent",
            "ticker": "RDDT",
            "market": "US",
            "model_revision": "v1",
            "issued_at": "2026-11-04T08:00:00Z",
            "last_complete_bar": "2026-11-03",
            "target_date": "2026-11-04",
            "horizon": 1,
            "prediction": 150.0,
            "metadata": {"test": 1}
        }

        # 第一次寫入
        self.store.record_forecasts_batch([record])
        # 第二次寫入完全相同的記錄（應冪等略過）
        self.store.record_forecasts_batch([record])
        self.assertEqual(len(self.store.list_forecasts()), 1)

        # 若 ID 相同但預測內容不同，應拋出 ValueError 阻擋竄改
        corrupted_record = dict(record)
        corrupted_record["prediction"] = 999.0
        with self.assertRaises(ValueError):
            self.store.record_forecasts_batch([corrupted_record])


if __name__ == "__main__":
    unittest.main()
