"""
test/test_notifier.py - 測試 notifier_dual 通知模組
驗證 send_dual_strategy_results 與各管道發送邏輯及異常處理
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from notifier_dual import send_dual_strategy_results, format_dual_strategy_message
from data.macro import MacroState


class TestNotifierDual(unittest.TestCase):
    def setUp(self):
        self.results = {
            "overlap_candidates": [
                {
                    "ticker": "2330.TW",
                    "current_price": 1005.0,
                    "composite_score": 92.0,
                    "resonance_tier": "👑四重共振",
                    "tags": ["👑四重共振", "玄鐵買點", "LSTM看漲", "TimesFM看漲"],
                    "lstm_potential": 5.2,
                    "timesfm_potential": 4.1,
                    "pullback_type": "MA60 (+2.0%)",
                    "pe": 22.0,
                    "pb": 5.0
                }
            ],
            "xuantie_results": pd.DataFrame([
                {"ticker": "2330.TW", "current_price": 1005.0, "pullback_type": "MA60 (+2.0%)", "pe": 22.0, "pb": 5.0}
            ]),
            "lstm_results": [
                {"ticker": "2330.TW", "potential": 5.2, "current_price": 1005.0, "predicted_price": 1057.26}
            ],
            "timesfm_results": [
                {"ticker": "2330.TW", "potential": 4.1, "horizon_predicted_price": 1046.2, "risk_reward_ratio": 2.5}
            ],
            "macro_state": MacroState(regime_name="全面多頭", exposure=1.0, vix=15.0),
            "ai_summary": "今日大盤多頭排列，推薦重點標的。"
        }
        self.name_map = {"2330.TW": "台積電"}

    @patch("notifier_dual.send_to_telegram")
    @patch("notifier_dual.send_to_discord")
    @patch("notifier_dual.send_email")
    def test_send_dual_strategy_results_success(self, mock_email, mock_discord, mock_telegram):
        """驗證正常情況下三管道均被調用"""
        send_dual_strategy_results(
            "台灣50",
            self.results,
            name_map=self.name_map,
            macro_state=self.results["macro_state"],
            ai_summary=self.results["ai_summary"]
        )

        self.assertTrue(mock_telegram.called)
        self.assertTrue(mock_discord.called)
        self.assertTrue(mock_email.called)

        # 檢查傳送至 Telegram 之訊息內容
        tg_args = mock_telegram.call_args[0][0]
        self.assertIn("<b>🚀 多維量化投資日報</b>", tg_args)
        self.assertIn("2330.TW 台積電", tg_args)
        self.assertIn("👑四重共振", tg_args)

    @patch("notifier_dual.send_to_telegram", side_effect=Exception("Telegram API Error"))
    @patch("notifier_dual.send_to_discord")
    @patch("notifier_dual.send_email")
    def test_send_dual_strategy_results_resilience(self, mock_email, mock_discord, mock_telegram):
        """驗證單一管道發送失敗不中斷其他管道與流程"""
        try:
            send_dual_strategy_results(
                "台灣50",
                self.results,
                name_map=self.name_map
            )
        except Exception as e:
            self.fail(f"send_dual_strategy_results 不應拋出未捕獲之異常: {e}")

        self.assertTrue(mock_telegram.called)
        self.assertTrue(mock_discord.called)
        self.assertTrue(mock_email.called)


if __name__ == "__main__":
    unittest.main()
