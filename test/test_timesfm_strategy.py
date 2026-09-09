"""
test/test_timesfm_strategy.py - Unit tests for TimesFM Strategy.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from strategies.base import StockContext
from strategies.timesfm import TimesFMStrategy


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
                "potential": 4.0,
                "horizon_potential": 6.0,
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
        self.assertEqual(res.potential, 4.0)
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


if __name__ == "__main__":
    unittest.main()
