"""
test/test_timesfm_strategy.py - Unit tests for TimesFM Strategy.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from strategies.base import StockContext
from strategies.timesfm import TimesFMStrategy
from models.timesfm_model import TimesFMWrapper


class TestTimesFMStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = TimesFMStrategy(
            horizon=5,
            context_len=128,
            min_potential_hit=1.0,
            min_risk_reward_hit=1.2
        )

    def test_insufficient_data(self):
        df = pd.DataFrame({"Close": [100.0] * 10})
        ctx = StockContext(ticker="TEST", df=df)
        res = self.strategy.evaluate(ctx)
        self.assertFalse(res.is_hit)
        self.assertEqual(res.signals.get("reason"), "數據不足")

    @patch("strategies.timesfm.get_timesfm_model")
    def test_successful_prediction_hit(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.forecast_batch.return_value = {
            "2330.TW": {
                "current_price": 1000.0,
                "predicted_price": 1040.0,
                "horizon_predicted_price": 1060.0,
                "potential": 6.0,
                "horizon_potential": 6.0,
                "day1_potential": 4.0,
                "trajectory": [1010.0, 1020.0, 1030.0, 1040.0, 1060.0],
                "quantiles": {"p10": 980.0, "p50": 1040.0, "p90": 1080.0},
                "downside_risk": -2.0,
                "upside_potential": 8.0,
                "risk_reward_ratio": 4.0,
                "horizon": 5
            }
        }
        mock_get_model.return_value = mock_model

        df = pd.DataFrame({
            "Open": [990.0] * 60,
            "High": [1010.0] * 60,
            "Low": [980.0] * 60,
            "Close": [1000.0] * 60,
            "Volume": [10000] * 60
        })
        ctx = StockContext(ticker="2330.TW", df=df, fundamentals={"pe": 22.0, "pb": 5.0})

        res = self.strategy.evaluate(ctx)

        self.assertTrue(res.is_hit)
        self.assertEqual(res.predicted_price, 1040.0)
        self.assertEqual(res.potential, 6.0)
        self.assertIn("TimesFM強", res.tags)
        self.assertIn("高盈虧比", res.tags)
        self.assertEqual(res.signals["risk_reward_ratio"], 4.0)

    @patch("strategies.timesfm.get_timesfm_model")
    def test_batch_evaluation_multiple_stocks(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.forecast_batch.return_value = {
            "2330.TW": {
                "current_price": 1000.0,
                "predicted_price": 1030.0,
                "potential": 3.0,
                "risk_reward_ratio": 2.5,
                "quantiles": {"p10": 990.0, "p50": 1030.0, "p90": 1050.0},
            },
            "2454.TW": {
                "current_price": 1200.0,
                "predicted_price": 1180.0,
                "potential": -1.67,
                "risk_reward_ratio": 0.5,
                "quantiles": {"p10": 1150.0, "p50": 1180.0, "p90": 1210.0},
            }
        }
        mock_get_model.return_value = mock_model

        df1 = pd.DataFrame({"Close": [1000.0] * 50})
        df2 = pd.DataFrame({"Close": [1200.0] * 50})

        contexts = {
            "2330.TW": StockContext(ticker="2330.TW", df=df1),
            "2454.TW": StockContext(ticker="2454.TW", df=df2)
        }

        results = self.strategy.evaluate_batch(contexts)

        self.assertEqual(len(results), 2)
        res_map = {r.ticker: r for r in results}
        self.assertTrue(res_map["2330.TW"].is_hit)
        self.assertFalse(res_map["2454.TW"].is_hit)
        self.assertIn("TimesFM強", res_map["2330.TW"].tags)

    @patch("strategies.timesfm.get_timesfm_model")
    def test_hit_uses_day1_potential(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.forecast_batch.return_value = {
            "TEST_MISS": {
                "current_price": 100.0,
                "predicted_price": 100.5,
                "horizon_predicted_price": 105.0,
                "potential": 0.5,
                "horizon_potential": 5.0,
                "risk_reward_ratio": 2.0,
                "quantiles": {"p10": 99.0, "p50": 100.5, "p90": 103.0},
            },
            "TEST_HIT": {
                "current_price": 100.0,
                "predicted_price": 102.5,
                "horizon_predicted_price": 106.0,
                "potential": 2.5,
                "horizon_potential": 6.0,
                "risk_reward_ratio": 2.5,
                "quantiles": {"p10": 99.0, "p50": 102.5, "p90": 105.0},
            }
        }
        mock_get_model.return_value = mock_model

        ctx_miss = StockContext(ticker="TEST_MISS", df=pd.DataFrame({"Close": [100.0] * 60}))
        res_miss = self.strategy.evaluate(ctx_miss)
        # 0.5% < min_potential_hit (1.0%), so it should NOT be hit even though horizon is 5.0%
        self.assertFalse(res_miss.is_hit)

        ctx_hit = StockContext(ticker="TEST_HIT", df=pd.DataFrame({"Close": [100.0] * 60}))
        res_hit = self.strategy.evaluate(ctx_hit)
        self.assertTrue(res_hit.is_hit)
        self.assertEqual(res_hit.potential, 2.5)

    def test_wrapper_uses_day1_for_potential_and_risk_reward(self):
        wrapper = TimesFMWrapper()
        wrapper._is_initialized = True
        wrapper.model = MagicMock()
        wrapper.model.forecast.return_value = (
            np.array([[101.0, 102.0, 103.0, 104.0, 110.0]]),
            np.array([
                [
                    [98.0, 99.0, 100.0, 100.5, 101.0, 101.0, 102.0, 103.0, 105.0, 120.0],
                    [98.0, 99.0, 100.0, 101.0, 102.0, 102.0, 103.0, 104.0, 106.0, 130.0],
                    [98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 108.0, 140.0],
                    [98.0, 99.0, 100.0, 101.0, 103.0, 104.0, 105.0, 106.0, 110.0, 145.0],
                    [98.0, 98.0, 99.0, 100.0, 106.0, 108.0, 110.0, 112.0, 120.0, 150.0],
                ]
            ])
        )

        result = wrapper.forecast_batch({"TEST": np.array([100.0] * 60)}, horizon=5)["TEST"]

        # T+1 day1_pred is 101.0, curr is 100.0 -> potential = 1.0
        self.assertEqual(result["predicted_price"], 101.0)
        self.assertEqual(result["potential"], 1.0)
        self.assertEqual(result["day1_potential"], 1.0)
        self.assertEqual(result["horizon_potential"], 10.0)
        # T+1: p50_day1 = 101.0, p10_day1 = 99.0, curr = 100.0 -> reward 1.0, risk 1.0 -> rr 1.0
        self.assertEqual(result["risk_reward_ratio"], 1.0)

    def test_default_initialization_from_config(self):
        """驗證 TimesFMStrategy 預設值正確自 config 讀取 T+1 適配之 0.5% 與 0.35x 門檻"""
        strat = TimesFMStrategy()
        self.assertEqual(strat.min_potential_hit, 0.5)
        self.assertEqual(strat.min_risk_reward_hit, 0.35)

    @patch("strategies.timesfm.get_timesfm_model")
    def test_t1_moderate_bullish_hit_with_default_thresholds(self, mock_get_model):
        """驗證單日 T+1 溫和但顯著看漲標的 (+0.8%，盈虧比 0.45x) 可在預設配置下成功命中"""
        strat = TimesFMStrategy()
        mock_model = MagicMock()
        mock_model.forecast_batch.return_value = {
            "LRCX": {
                "current_price": 250.0,
                "predicted_price": 252.0,
                "horizon_predicted_price": 255.0,
                "potential": 0.8,
                "horizon_potential": 2.0,
                "day1_potential": 0.8,
                "risk_reward_ratio": 0.45,
                "quantiles": {"p10": 245.5, "p50": 252.0, "p90": 260.0},
            }
        }
        mock_get_model.return_value = mock_model
        ctx = StockContext(ticker="LRCX", df=pd.DataFrame({"Close": [250.0] * 60}))
        res = strat.evaluate(ctx)
        self.assertTrue(res.is_hit)
        self.assertIn("TimesFM看漲", res.tags)


if __name__ == "__main__":
    unittest.main()
