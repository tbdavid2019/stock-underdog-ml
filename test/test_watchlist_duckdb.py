"""
test/test_watchlist_duckdb.py - 測試 DuckDB 自選股持久化、代號正規化、容量上限與備註邊界
"""
import os
import tempfile
import unittest
from pathlib import Path

from data.duckdb_manager import DuckDBManager


class TestWatchlistDuckDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_watchlist.duckdb")
        self.mgr = DuckDBManager(db_path=self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_and_get_watchlist_item_with_normalization(self):
        """測試新增自選股、代號正規化與查詢"""
        item = self.mgr.add_watchlist_item(
            profile_id="default",
            ticker="2330",
            note="台積電晶圓代工龍頭",
            sort_order=1
        )
        self.assertEqual(item["canonical_ticker"], "2330.TW")
        self.assertEqual(item["market"], "TW")
        self.assertEqual(item["note"], "台積電晶圓代工龍頭")

        items = self.mgr.get_watchlist_items(profile_id="default")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["canonical_ticker"], "2330.TW")

    def test_watchlist_note_length_boundary_rejection(self):
        """測試備註超過 500 字元被拒絕"""
        long_note = "A" * 501
        with self.assertRaises(ValueError) as ctx:
            self.mgr.add_watchlist_item(ticker="2330.TW", note=long_note)
        self.assertIn("500", str(ctx.exception))

    def test_watchlist_capacity_100_limit_rejection(self):
        """測試單一 Profile 自選股上限 100 檔邊界控制"""
        # 批量填入 100 檔
        tickers = [f"TICKER{i}" for i in range(100)]
        self.mgr.batch_import_watchlist(profile_id="default", tickers=tickers)

        items = self.mgr.get_watchlist_items(profile_id="default")
        self.assertEqual(len(items), 100)

        # 第 101 檔應拋出 ValueError
        with self.assertRaises(ValueError) as ctx:
            self.mgr.add_watchlist_item(profile_id="default", ticker="EXTRA_TICKER")
        self.assertIn("100", str(ctx.exception))

    def test_remove_watchlist_item(self):
        """測試刪除自選股"""
        self.mgr.add_watchlist_item(ticker="NVDA")
        self.assertEqual(len(self.mgr.get_watchlist_items()), 1)

        removed = self.mgr.remove_watchlist_item(ticker="NVDA")
        self.assertTrue(removed)
        self.assertEqual(len(self.mgr.get_watchlist_items()), 0)

    def test_watchlist_enriched_with_predictions(self):
        """測試唯讀連線查詢自選股時附帶最新模型預測摘要"""
        # 先插入預測數據
        self.mgr.save_predictions_batch([{
            "index_name": "台灣50",
            "model_name": "TimesFM",
            "strategy_type": "TimesFM看漲",
            "ticker": "2330.TW",
            "current_price": 1000.0,
            "predicted_price": 1080.0,
            "potential": 8.0,
            "period": "6mo",
            "timestamp": "2026-11-06T09:00:00",
            "macro_regime": "多頭",
            "tags": ["TimesFM看漲"]
        }])

        self.mgr.add_watchlist_item(ticker="2330.TW", note="核心持股")
        items = self.mgr.get_watchlist_items(profile_id="default")
        self.assertEqual(len(items), 1)
        item = items[0]
        self.assertEqual(item["canonical_ticker"], "2330.TW")
        self.assertAlmostEqual(item.get("predicted_price"), 1080.0)
        self.assertAlmostEqual(item.get("potential"), 8.0)


if __name__ == "__main__":
    unittest.main()
