"""
test/test_duckdb_crossprocess_locking.py - 測試 DuckDB 跨進程鎖與並發安全
"""
import multiprocessing as mp
import os
import tempfile
import time
import unittest
from pathlib import Path

from core.duckdb_lock import DuckDBBusyError, duckdb_write_lock, get_lock_path
from data.duckdb_manager import DuckDBManager


def _hold_lock_worker(db_path: str, hold_duration: float, started_event: mp.Event, error_queue: mp.Queue):
    try:
        with duckdb_write_lock(db_path, timeout=5.0):
            started_event.set()
            time.sleep(hold_duration)
    except Exception as e:
        error_queue.put(f"Worker error: {e}")


def _try_lock_worker(db_path: str, timeout: float, result_queue: mp.Queue):
    try:
        with duckdb_write_lock(db_path, timeout=timeout):
            result_queue.put("ACQUIRED")
    except DuckDBBusyError:
        result_queue.put("BUSY")
    except Exception as e:
        result_queue.put(f"ERROR: {e}")


def _save_batch_worker(db_path: str, ticker_suffix: str, count: int, result_queue: mp.Queue):
    try:
        mgr = DuckDBManager(db_path=db_path)
        records = [{
            "index_name": "台灣50",
            "model_name": "TimesFM",
            "strategy_type": "TimesFM看漲",
            "ticker": f"{i}.{ticker_suffix}",
            "current_price": 100.0 + i,
            "predicted_price": 110.0 + i,
            "potential": 10.0,
            "period": "6mo",
            "timestamp": "2026-11-06T09:00:00",
            "macro_regime": "多頭",
            "tags": ["TimesFM看漲"]
        } for i in range(count)]
        inserted = mgr.save_predictions_batch(records)
        result_queue.put(inserted)
    except Exception as e:
        result_queue.put(f"ERROR: {e}")


class TestDuckDBCrossprocessLocking(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_concurrent.duckdb")

    def tearDown(self):
        self.tmp.cleanup()

    def test_in_memory_bypasses_lock(self):
        """測試 :memory: 資料庫路徑安全跳過檔案鎖"""
        self.assertIsNone(get_lock_path(":memory:"))
        with duckdb_write_lock(":memory:", timeout=1.0):
            pass

    def test_flock_cross_process_mutual_exclusion(self):
        """測試跨進程互斥與逾時 DuckDBBusyError"""
        started = mp.Event()
        err_queue = mp.Queue()
        res_queue = mp.Queue()

        # 進程 1 持鎖 2.0 秒
        p1 = mp.Process(target=_hold_lock_worker, args=(self.db_path, 2.0, started, err_queue))
        p1.start()

        self.assertTrue(started.wait(timeout=5.0))

        # 進程 2 嘗試以 0.05 秒逾時獲取，預期 BUSY
        p2 = mp.Process(target=_try_lock_worker, args=(self.db_path, 0.05, res_queue))
        p2.start()
        p2.join(timeout=5.0)
        self.assertEqual(res_queue.get(timeout=2.0), "BUSY")

        # 進程 3 嘗試以 3.0 秒逾時獲取，預期等 p1 結束後成功 ACQUIRED
        p3 = mp.Process(target=_try_lock_worker, args=(self.db_path, 3.0, res_queue))
        p3.start()
        p3.join(timeout=5.0)
        p1.join(timeout=5.0)
        self.assertEqual(res_queue.get(timeout=2.0), "ACQUIRED")

        self.assertTrue(err_queue.empty())

    def test_concurrent_duckdb_writes(self):
        """測試多進程並發寫入同一 DuckDB 資料庫零死鎖與資料完整性"""
        res_queue = mp.Queue()
        # 先確保 schema 已初始化
        DuckDBManager(db_path=self.db_path)

        p1 = mp.Process(target=_save_batch_worker, args=(self.db_path, "TW", 5, res_queue))
        p2 = mp.Process(target=_save_batch_worker, args=(self.db_path, "US", 5, res_queue))

        p1.start()
        p2.start()

        p1.join(timeout=10.0)
        p2.join(timeout=10.0)

        res1 = res_queue.get(timeout=2.0)
        res2 = res_queue.get(timeout=2.0)

        self.assertEqual(res1, 5)
        self.assertEqual(res2, 5)

        mgr = DuckDBManager(db_path=self.db_path)
        df = mgr.query("SELECT COUNT(*) as cnt FROM predictions")
        self.assertEqual(df.iloc[0]["cnt"], 10)

    def test_reentrant_lock_same_thread(self):
        """測試同執行緒巢狀重入鎖不會引發自我死鎖"""
        start = time.monotonic()
        with duckdb_write_lock(self.db_path, timeout=2.0):
            with duckdb_write_lock(self.db_path, timeout=2.0):
                with duckdb_write_lock(self.db_path, timeout=2.0):
                    pass
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 1.0, "Nested re-entrant lock should acquire instantly without self-deadlock")


if __name__ == "__main__":
    unittest.main()
