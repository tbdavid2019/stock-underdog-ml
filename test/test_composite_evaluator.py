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

    def test_negative_potential_excluded_from_overlap_candidates(self):
        """
        重現並驗證：雙 ML 大幅看跌或主要模型負報酬之個股（如南亞 1303、南電 8046、中信金 2891），
        即使滿足多個其他策略（如籌碼、板塊或均線），也絕對不得進入 overlap_candidates（優先推薦）。
        """
        evaluator = CompositeEvaluator()
        macro = MacroState(regime_name="全面多頭 (Bullish)", exposure=1.0, vix=15.0)

        strategy_outputs = {
            "xuantie": [
                # 5880: 玄鐵買點
                StrategyResult(ticker="5880.TW", strategy_name="玄鐵重劍", is_hit=True, score=80.0, current_price=26.0, metrics={"ma60": 25.0}, signals={"pullback_type": "MA60 (+4.0%)"}),
                # 2891: 玄鐵買點，但雙 ML 看跌
                StrategyResult(ticker="2891.TW", strategy_name="玄鐵重劍", is_hit=True, score=70.0, current_price=35.0, metrics={"ma60": 34.0}, signals={"pullback_type": "MA60 (+2.9%)"}),
            ],
            "lstm": [
                # 5880: LSTM 看漲 +4.1%
                StrategyResult(ticker="5880.TW", strategy_name="LSTM", is_hit=True, score=85.0, current_price=26.0, predicted_price=27.06, potential=4.1),
                # 1303 (南亞): LSTM 大幅看跌 -34.4%
                StrategyResult(ticker="1303.TW", strategy_name="LSTM", is_hit=False, score=0.0, current_price=50.0, predicted_price=32.8, potential=-34.4),
                # 8046 (南電): LSTM 看跌 -19.9%
                StrategyResult(ticker="8046.TW", strategy_name="LSTM", is_hit=False, score=0.0, current_price=200.0, predicted_price=160.2, potential=-19.9),
                # 2891 (中信金): LSTM 看跌 -6.9%
                StrategyResult(ticker="2891.TW", strategy_name="LSTM", is_hit=False, score=0.0, current_price=35.0, predicted_price=32.58, potential=-6.9),
            ],
            "timesfm": [
                # 5880: TimesFM 看漲 +0.7%
                StrategyResult(ticker="5880.TW", strategy_name="TimesFM", is_hit=False, score=55.0, current_price=26.0, predicted_price=26.18, potential=0.7, signals={"risk_reward_ratio": 1.1}),
                # 1303: TimesFM 看跌 -1.6%
                StrategyResult(ticker="1303.TW", strategy_name="TimesFM", is_hit=False, score=37.0, current_price=50.0, predicted_price=49.2, potential=-1.6, signals={"risk_reward_ratio": 0.5}),
                # 8046: TimesFM 看跌 -3.8%
                StrategyResult(ticker="8046.TW", strategy_name="TimesFM", is_hit=False, score=20.0, current_price=200.0, predicted_price=192.4, potential=-3.8, signals={"risk_reward_ratio": 0.3}),
                # 2891: TimesFM 看跌 -1.4%
                StrategyResult(ticker="2891.TW", strategy_name="TimesFM", is_hit=False, score=38.0, current_price=35.0, predicted_price=34.5, potential=-1.4, signals={"risk_reward_ratio": 0.6}),
            ],
            "institutional": [
                # 5880: 投信買超
                StrategyResult(ticker="5880.TW", strategy_name="三大法人籌碼", is_hit=True, score=70.0, current_price=26.0, metadata={"trust_streak": 3, "is_sync_buy": True}),
                # 1303: 投信買超 (2 hits: institutional + sector)
                StrategyResult(ticker="1303.TW", strategy_name="三大法人籌碼", is_hit=True, score=70.0, current_price=50.0, metadata={"trust_streak": 3, "is_sync_buy": True}),
                # 8046: 投信買超
                StrategyResult(ticker="8046.TW", strategy_name="三大法人籌碼", is_hit=True, score=70.0, current_price=200.0, metadata={"trust_streak": 2, "is_sync_buy": False, "trust_net_5d": 500}),
                # 2891: 投信買超 (2 hits: xuantie + institutional)
                StrategyResult(ticker="2891.TW", strategy_name="三大法人籌碼", is_hit=True, score=70.0, current_price=35.0, metadata={"trust_streak": 3, "is_sync_buy": True}),
            ],
            "sector_rotation": [
                StrategyResult(ticker="1303.TW", strategy_name="板塊輪動", is_hit=True, score=75.0, current_price=50.0, metadata={"is_top_sector": True, "sector": "塑膠"}),
                StrategyResult(ticker="8046.TW", strategy_name="板塊輪動", is_hit=True, score=75.0, current_price=200.0, metadata={"is_top_sector": True, "sector": "半導體"}),
            ]
        }

        report = evaluator.evaluate("台灣50", strategy_outputs, {}, macro_state=macro)
        overlap_tickers = [c["ticker"] for c in report.overlap_candidates]

        # 5880.TW 具備實質正向潛力與多重共振，應在優先推薦中
        self.assertIn("5880.TW", overlap_tickers)

        # 負值/大幅看跌個股絕對不得出現在優先推薦 (多維共振) 中
        self.assertNotIn("1303.TW", overlap_tickers, "1303.TW (南亞) 雙 ML 大幅看跌，不應出現在 overlap_candidates")
        self.assertNotIn("8046.TW", overlap_tickers, "8046.TW (南電) 雙 ML 看跌，不應出現在 overlap_candidates")
        self.assertNotIn("2891.TW", overlap_tickers, "2891.TW (中信金) 雙 ML 看跌，不應出現在 overlap_candidates")

        # 驗證看跌防守標的不會獲得共振階層標記，且標籤清單中已移除共振與符合字樣
        for s in report.ranked_stocks:
            if s["ticker"] in ("1303.TW", "8046.TW", "2891.TW"):
                self.assertEqual(s.get("resonance_tier", ""), "", f"{s['ticker']} 不應具有共振階層")
                for tag in s.get("tags", []):
                    self.assertNotIn("共振", tag, f"{s['ticker']} 標籤 '{tag}' 不應包含共振")
                    self.assertNotIn("符合", tag, f"{s['ticker']} 標籤 '{tag}' 不應包含符合")

    def test_triple_resonance_strict_combinations(self):
        """
        驗證三重共振必須具備技術面 (玄鐵) 或籌碼面 (法人) 的實質支撐：
        - 僅有 板塊 + 雙ML (sector + lstm + timesfm) 不得判定為 🏆三重共振 (應為 🔮雙ML共振)。
        - 具備 玄鐵 + 法人 + LSTM 應判定為 🏆三重共振。
        - 具備 板塊 + 玄鐵 + LSTM 應判定為 🏆三重共振。
        """
        evaluator = CompositeEvaluator()
        macro = MacroState(regime_name="全面多頭 (Bullish)", exposure=1.0, vix=15.0)

        strategy_outputs = {
            "xuantie": [
                # 2330: XuanTie hit
                StrategyResult(ticker="2330.TW", strategy_name="玄鐵重劍", is_hit=True, score=80.0, current_price=1000.0, metrics={"ma60": 980.0}),
                # 2454: XuanTie hit
                StrategyResult(ticker="2454.TW", strategy_name="玄鐵重劍", is_hit=True, score=80.0, current_price=1200.0, metrics={"ma60": 1180.0}),
            ],
            "lstm": [
                # 2330: LSTM hit
                StrategyResult(ticker="2330.TW", strategy_name="LSTM", is_hit=True, score=85.0, current_price=1000.0, predicted_price=1050.0, potential=5.0),
                # 2454: LSTM hit
                StrategyResult(ticker="2454.TW", strategy_name="LSTM", is_hit=True, score=85.0, current_price=1200.0, predicted_price=1260.0, potential=5.0),
                # 3008: LSTM hit
                StrategyResult(ticker="3008.TW", strategy_name="LSTM", is_hit=True, score=85.0, current_price=2500.0, predicted_price=2600.0, potential=4.0),
            ],
            "timesfm": [
                # 3008: TimesFM hit
                StrategyResult(ticker="3008.TW", strategy_name="TimesFM", is_hit=True, score=80.0, current_price=2500.0, predicted_price=2600.0, potential=4.0, signals={"risk_reward_ratio": 2.0}),
            ],
            "institutional": [
                # 2330: Institutional hit
                StrategyResult(ticker="2330.TW", strategy_name="三大法人籌碼", is_hit=True, score=80.0, current_price=1000.0, metadata={"is_sync_buy": True, "trust_streak": 3}),
            ],
            "sector_rotation": [
                # 2454: Sector hit
                StrategyResult(ticker="2454.TW", strategy_name="板塊輪動", is_hit=True, score=75.0, current_price=1200.0, metadata={"is_top_sector": True, "sector": "半導體"}),
                # 3008: Sector hit (3008 only has Sector + LSTM + TimesFM, NO xuantie or inst!)
                StrategyResult(ticker="3008.TW", strategy_name="板塊輪動", is_hit=True, score=75.0, current_price=2500.0, metadata={"is_top_sector": True, "sector": "光電"}),
            ]
        }

        report = evaluator.evaluate("台灣50", strategy_outputs, {}, macro_state=macro)

        cand_2330 = next((c for c in report.overlap_candidates if c["ticker"] == "2330.TW"), None)
        cand_2454 = next((c for c in report.overlap_candidates if c["ticker"] == "2454.TW"), None)
        cand_3008 = next((c for c in report.overlap_candidates if c["ticker"] == "3008.TW"), None)

        # 2330 (XuanTie + Inst + LSTM) -> 🏆三重共振
        self.assertIsNotNone(cand_2330)
        self.assertEqual(cand_2330["resonance_tier"], "🏆三重共振")

        # 2454 (Sector + XuanTie + LSTM) -> 🏆三重共振
        self.assertIsNotNone(cand_2454)
        self.assertEqual(cand_2454["resonance_tier"], "🏆三重共振")

        # 3008 (Sector + LSTM + TimesFM) -> 不得為 🏆三重共振，因缺乏玄鐵買點或法人籌碼支撐，應為 🔮雙ML共振
        self.assertIsNotNone(cand_3008)
        self.assertNotEqual(cand_3008["resonance_tier"], "🏆三重共振")
        self.assertEqual(cand_3008["resonance_tier"], "🔮雙ML共振")

if __name__ == "__main__":
    unittest.main()
