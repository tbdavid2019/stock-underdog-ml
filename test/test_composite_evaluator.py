"""
test/test_composite_evaluator.py - 測試多策略綜合評估與三重共振標籤
"""

import unittest
import pandas as pd
from evaluators.composite_evaluator import CompositeEvaluator, EvaluationReport
from strategies.base import StrategyResult
from data.macro import MacroState


class TestCompositeEvaluator(unittest.TestCase):
    def test_composite_scoring_and_triple_resonance(self):
        evaluator = CompositeEvaluator()
        macro = MacroState(regime_name="全面多頭 (Bullish)", exposure=1.0, vix=15.0)

        # 構造 4 種策略輸出
        # 1. 2330: XuanTie is_hit + LSTM is_hit + Institutional is_hit (Triple Resonance)
        # 2. 2317: XuanTie is_hit + LSTM not hit
        # 3. 2454: LSTM is_hit only
        strategy_outputs = {
            "xuantie": [
                StrategyResult(ticker="2330.TW", strategy_name="玄鐵重劍", is_hit=True, score=80.0, current_price=1000.0, metrics={"ma60": 980.0, "pe": 22.0, "pb": 5.0}, signals={"pullback_type": "MA60 (+2.0%)"}),
                StrategyResult(ticker="2317.TW", strategy_name="玄鐵重劍", is_hit=True, score=70.0, current_price=200.0, metrics={"ma60": 195.0, "pe": 15.0, "pb": 2.0}, signals={"pullback_type": "MA60 (+2.5%)"})
            ],
            "lstm": [
                StrategyResult(ticker="2330.TW", strategy_name="LSTM", is_hit=True, score=85.0, current_price=1000.0, predicted_price=1050.0, potential=5.0),
                StrategyResult(ticker="2317.TW", strategy_name="LSTM", is_hit=False, score=40.0, current_price=200.0, predicted_price=198.0, potential=-1.0),
                StrategyResult(ticker="2454.TW", strategy_name="LSTM", is_hit=True, score=90.0, current_price=1200.0, predicted_price=1300.0, potential=8.3)
            ],
            "institutional": [
                StrategyResult(ticker="2330.TW", strategy_name="三大法人籌碼", is_hit=True, score=80.0, current_price=1000.0, metadata={"trust_streak": 4, "is_trust_streak": True, "is_sync_buy": True, "trust_net_5d": 3000}),
                StrategyResult(ticker="2317.TW", strategy_name="三大法人籌碼", is_hit=False, score=30.0, current_price=200.0, metadata={"trust_streak": 1, "is_trust_streak": False, "is_sync_buy": False, "trust_net_5d": -100})
            ],
            "sector_rotation": [
                StrategyResult(ticker="2330.TW", strategy_name="板塊輪動", is_hit=True, score=75.0, current_price=1000.0, metadata={"is_top_sector": True, "sector": "半導體與IC設計"})
            ]
        }

        fundamentals = {
            "2330.TW": {"pe": 22.0, "pb": 5.0},
            "2317.TW": {"pe": 15.0, "pb": 2.0}
        }

        report = evaluator.evaluate("台灣50", strategy_outputs, fundamentals, macro_state=macro)
        self.assertIsInstance(report, EvaluationReport)
        self.assertGreater(len(report.overlap_candidates), 0)

        # 驗證 2330.TW 是否獲得 三重共振 標籤
        top_cand = next((c for c in report.overlap_candidates if c["ticker"] == "2330.TW"), None)
        self.assertIsNotNone(top_cand)
        self.assertIn("🏆三重共振", top_cand["tags"])
        self.assertIn("土洋合買", top_cand["tags"])
        self.assertIn("主流板塊(半導體與IC設計)", top_cand["tags"])
        print("2330.TW Tags:", top_cand["tags"])
        print("2330.TW Composite Score:", top_cand["composite_score"])

    def test_macro_discounting(self):
        evaluator = CompositeEvaluator()
        macro_panic = MacroState(regime_name="防禦謹慎 (Defensive)", exposure=0.5, vix=26.0)

        strategy_outputs = {
            "xuantie": [
                StrategyResult(ticker="2330.TW", strategy_name="玄鐵重劍", is_hit=True, score=80.0, current_price=1000.0, metrics={"ma60": 980.0})
            ]
        }
        report = evaluator.evaluate("台灣50", strategy_outputs, {}, macro_state=macro_panic)
        cand = report.ranked_stocks[0]
        # Score is discounted by 50%
        self.assertLess(cand["composite_score"], 80.0 * 0.6)
        print("Panic Score Discounted:", cand["composite_score"])

    def test_timesfm_resonance_and_formatter(self):
        from evaluators.formatter import print_evaluation_report
        import logging

        evaluator = CompositeEvaluator()
        macro = MacroState(regime_name="全面多頭 (Bullish)", exposure=1.0, vix=14.5)

        strategy_outputs = {
            "xuantie": [
                StrategyResult(ticker="2330.TW", strategy_name="玄鐵重劍", is_hit=True, score=85.0, current_price=1000.0, metrics={"ma60": 980.0, "pe": 20.0, "pb": 4.5}, signals={"pullback_type": "MA60 (+2.0%)"})
            ],
            "lstm": [
                StrategyResult(ticker="2330.TW", strategy_name="LSTM", is_hit=True, score=88.0, current_price=1000.0, predicted_price=1050.0, potential=5.0)
            ],
            "timesfm": [
                StrategyResult(ticker="2330.TW", strategy_name="TimesFM", is_hit=True, score=92.0, current_price=1000.0, predicted_price=1060.0, potential=6.0, signals={"risk_reward_ratio": 2.5, "horizon_predicted_price": 1060.0}, metrics={"pe": 20.0, "pb": 4.5}),
                StrategyResult(ticker="2454.TW", strategy_name="TimesFM", is_hit=True, score=80.0, current_price=1200.0, predicted_price=1250.0, potential=4.17, signals={"risk_reward_ratio": 1.8, "horizon_predicted_price": 1250.0})
            ],
            "institutional": [
                StrategyResult(ticker="2330.TW", strategy_name="三大法人籌碼", is_hit=True, score=80.0, current_price=1000.0, metadata={"trust_streak": 3, "is_sync_buy": True})
            ]
        }

        fundamentals = {"2330.TW": {"pe": 20.0, "pb": 4.5}}
        report = evaluator.evaluate("台灣50", strategy_outputs, fundamentals, macro_state=macro)

        # 驗證四重共振與雙ML共振
        cand = next((c for c in report.overlap_candidates if c["ticker"] == "2330.TW"), None)
        self.assertIsNotNone(cand)
        self.assertIn("👑四重共振", cand["tags"])
        self.assertIn("🔮雙ML共振", cand["tags"])
        self.assertIn("高盈虧比", cand["tags"])

        # 驗證 TimesFM 結果列表
        self.assertEqual(len(report.timesfm_results), 2)
        self.assertEqual(report.timesfm_results[0]["ticker"], "2330.TW")
        self.assertEqual(report.timesfm_results[0]["risk_reward_ratio"], 2.5)

        # 驗證 formatter 輸出不報錯
        logger = logging.getLogger("test_formatter")
        logger.setLevel(logging.INFO)
        print_evaluation_report(report, log=logger)


if __name__ == "__main__":
    unittest.main()
