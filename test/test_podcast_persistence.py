"""
test/test_podcast_persistence.py - Test Podcast Decision Persistence to DuckDB
驗證單集決策重要性、評級與個股情緒推論是否正確持久化存入 DuckDB，
杜絕每次開啟網頁或重新查詢時重複推論/重複撈取的嚴重缺陷。
"""

import unittest
import os
import tempfile
import json
from unittest.mock import patch

from data.duckdb_manager import DuckDBManager
from data.podcast_service import PodcastService


class TestPodcastPersistence(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_podcast.duckdb")
        self.db = DuckDBManager(db_path=self.db_path)
        self.service = PodcastService(db_manager=self.db)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_enrich_episodes_persists_decisions_to_duckdb(self):
        """
        驗證：當單集初次進行 enrich 增強時，所計算出的重要性與多空評估必須即時存入 DuckDB，
        再次從 DuckDB 讀取時，importance_score 不得為 None。
        """
        raw_episode = {
            "id": "ep-persist-test-1",
            "channel_id": "all-in",
            "title": "All-In E200: NVIDIA GPU Scaling & AI Infrastructure",
            "published_at": "2026-10-14T10:00:00Z",
            "language": "en",
            "host": "Chamath, Sacks",
            "guest": "Jensen Huang",
            "tldr_zh_tw": "黃仁勳深入探討 Blackwell 晶片放量與 AI 資料中心 Capex 爆發成長，極度看好未來晶片需求。",
            "topic_tags": ["NVIDIA", "AI"],
            "tickers": ["NVDA"],
            "importance_score": None,  # 初次未評估
            "importance_level": None,
            "ticker_sentiments": None,
            "decision_slot": None
        }

        # 1. 寫入 DuckDB
        saved = self.db.save_podcast_episodes_batch([raw_episode])
        self.assertEqual(saved, 1)

        # 驗證剛存入時為 None
        episodes_before = self.db.get_podcast_episodes(limit=10)
        self.assertEqual(len(episodes_before), 1)
        self.assertIsNone(episodes_before[0].get("importance_score"))

        # 2. 執行 enrich_episodes_batch
        enriched = self.service.enrich_episodes_batch(episodes_before)
        self.assertEqual(len(enriched), 1)
        self.assertIsNotNone(enriched[0].get("importance_score"))
        calculated_score = enriched[0]["importance_score"]

        # 3. 🚨 核心斷言：再次從 DuckDB 撈出時，必須已經持久化入庫，不得依然為 None！
        episodes_after = self.db.get_podcast_episodes(limit=10)
        self.assertEqual(len(episodes_after), 1)
        self.assertIsNotNone(
            episodes_after[0].get("importance_score"),
            "DuckDB 中的 importance_score 依然為 None！未正確持久化存入資料庫！"
        )
        self.assertAlmostEqual(episodes_after[0]["importance_score"], calculated_score, places=2)
        self.assertIsNotNone(episodes_after[0].get("importance_level"))
        self.assertIsNotNone(episodes_after[0].get("ticker_sentiments"))

    def test_batch_update_decisions_method(self):
        """
        驗證 DuckDBManager.update_podcast_episodes_decisions_batch 能一口氣批量更新多筆單集
        """
        episodes = [
            {
                "id": f"ep-batch-{i}",
                "channel_id": "test-ch",
                "title": f"Episode {i}",
                "published_at": f"2026-10-14T0{i}:00:00Z",
                "importance_score": None
            }
            for i in range(1, 4)
        ]
        self.db.save_podcast_episodes_batch(episodes)

        updates = [
            {
                "id": f"ep-batch-{i}",
                "importance_score": 4.0 + i * 0.1,
                "importance_level": "important",
                "ticker_sentiments": {"NVDA": {"stance": "bullish"}},
                "decision_slot": "local_rules"
            }
            for i in range(1, 4)
        ]

        count = self.db.update_podcast_episodes_decisions_batch(updates)
        self.assertEqual(count, 3)

        records = self.db.get_podcast_episodes(limit=10)
        self.assertEqual(len(records), 3)
        for r in records:
            self.assertIsNotNone(r.get("importance_score"))
            self.assertEqual(r.get("importance_level"), "important")
            self.assertEqual(r.get("decision_slot"), "local_rules")

    def test_backfill_unevaluated_episodes(self):
        """
        驗證 backfill_unevaluated_episodes 能自動找出所有 NULL 的歷史單集並回填入庫
        """
        episodes = [
            {
                "id": f"ep-backfill-{i}",
                "channel_id": "all-in",
                "title": f"Episode Backfill {i} AI TSMC",
                "published_at": f"2026-10-14T0{i}:00:00Z",
                "tickers": ["2330.TW"],
                "importance_score": None
            }
            for i in range(1, 4)
        ]
        self.db.save_podcast_episodes_batch(episodes)

        backfilled = self.service.backfill_unevaluated_episodes()
        self.assertEqual(backfilled, 3)

        records = self.db.get_podcast_episodes(limit=10)
        for r in records:
            self.assertIsNotNone(r.get("importance_score"))
            self.assertIsNotNone(r.get("importance_level"))
            self.assertEqual(r.get("decision_slot"), "local_rules")


if __name__ == "__main__":
    unittest.main()
