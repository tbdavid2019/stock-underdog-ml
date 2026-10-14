"""
test/test_decision_service.py - Test 5-Tier Resilient Decision Engine with Local Rules Fallback
"""

import unittest
from unittest.mock import patch, MagicMock

from data.decision_service import DecisionService, PodcastDecisionVerdict
from data.clef_client import ClefDecisionVerdict


class TestDecisionService(unittest.TestCase):

    def setUp(self):
        self.service = DecisionService()

    def test_health_check_includes_5_tiers(self):
        """驗證健康檢查涵蓋 5 個 Tier，且本地規則永續在線"""
        health = self.service.check_health()
        self.assertTrue(health["success"])
        self.assertTrue(health["local_rules_ready"])
        endpoints = health["endpoints"]
        self.assertIn("Clef-Create360", endpoints)
        self.assertIn("Clef-Aiurl", endpoints)
        self.assertIn("Jev-SystemOne", endpoints)
        self.assertIn("OpenAI-Decision", endpoints)
        self.assertIn("Local-Rules", endpoints)
        self.assertEqual(endpoints["Local-Rules"]["status"], "healthy")

    @patch("data.decision_service.DecisionService._call_remote_systemone")
    def test_podcast_evaluation_clef_primary_success(self, mock_remote):
        """驗證當 Clef 主力端點正常時，正確解析為遠端答案與徽章"""
        mock_remote.return_value = (
            True,
            {
                "importance_score": {"score": 4.5, "probabilities": {"3": 0.1, "4": 0.9}},
                "market_catalyst": {"choice": "high_impact", "probabilities": {"high_impact": 0.88}},
                "sentiment_nvda": {"choice": "bullish", "confidence": 0.89}
            },
            None
        )

        episode = {
            "id": "ep-101",
            "title": "NVIDIA Blackwell chips demand review",
            "author": "All-In Podcast",
            "tickers": ["NVDA"],
            "tldr": "NVIDIA Blackwell chips are in huge demand, increasing revenue guidance."
        }

        verdict = self.service.evaluate_podcast_episode(episode)
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.importance_level, "heavyweight")
        self.assertEqual(verdict.importance_badge, "⭐ 5星重磅")
        self.assertEqual(verdict.market_catalyst, "high_impact")
        self.assertEqual(verdict.source_slot, "clef_primary")
        self.assertEqual(verdict.ticker_sentiments["NVDA"]["stance"], "bullish")
        self.assertEqual(verdict.ticker_sentiments["NVDA"]["label"], "↑ 看多")

    @patch("data.decision_service.DecisionService._call_remote_systemone")
    @patch("data.decision_service.DecisionService._call_openai_decision")
    def test_podcast_evaluation_complete_remote_failure_graceful_degradation(self, mock_openai, mock_remote):
        """
        🚨 關鍵測試：驗證當「所有遠端決策模型全數崩潰/斷線」時，
        系統必須 100% 成功降級為本地啟發式規則（Local Rules），絕不拋出未捕獲例外，
        且保證輸出合法的 importance_score、level 與 ticker_sentiments。
        """
        # 模擬遠端全部異常
        mock_remote.return_value = (False, {}, "Connection refused / 503 Service Unavailable")
        mock_openai.return_value = (False, {}, "OpenAI timeout")

        episode = {
            "id": "ep-fallback-test",
            "title": "台積電 CoWoS 擴產與大客戶拉貨超預期",
            "author": "股癌 Gooaye",
            "host": "謝孟恭",
            "guest": "行業專家",
            "tickers": ["2330.TW", "NVDA"],
            "tldr": "台積電先進封裝產能持續突破新高，AI 需求強勁增長，展望非常看好。"
        }

        verdict = self.service.evaluate_podcast_episode(episode)

        # 核心驗證：必須成功，且明確標註 source_slot 為 local_rules
        self.assertTrue(verdict.success)
        self.assertEqual(verdict.source_slot, "local_rules")
        self.assertGreaterEqual(verdict.importance_score, 3.5)
        self.assertIn("本地規則", verdict.importance_badge)
        self.assertIn("2330.TW", verdict.ticker_sentiments)
        self.assertIn("NVDA", verdict.ticker_sentiments)
        # 關鍵詞包含「突破新高、看好、增長」，本地情緒詞典應判定為看多
        self.assertEqual(verdict.ticker_sentiments["2330.TW"]["stance"], "bullish")
        self.assertEqual(verdict.ticker_sentiments["2330.TW"]["label"], "↑ 看多")


if __name__ == "__main__":
    unittest.main()
