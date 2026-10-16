"""
test/test_timestral_strategy.py - Comprehensive Unit Tests for Mistral Timestral Strategy & Wrapper.
"""

import unittest
from unittest.mock import patch, MagicMock
import threading
import pandas as pd
import numpy as np

from strategies.base import StockContext
from strategies.timestral import TimestralStrategy
from models.timestral_model import TimestralWrapper, get_timestral_model


class TestTimestralStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = TimestralStrategy(
            horizon=5,
            context_len=128,
            min_potential_hit=1.0,
            min_risk_reward_hit=1.2
        )

    def test_insufficient_data_less_than_60_bars(self):
        df = pd.DataFrame({"Close": [100.0] * 30})
        ctx = StockContext(ticker="TEST", df=df)
        res = self.strategy.evaluate(ctx)
        self.assertFalse(res.is_hit)
        self.assertIn("數據不足", res.signals.get("reason", ""))

    @patch("strategies.timestral.get_timestral_model")
    def test_successful_prediction_hit(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.forecast_batch.return_value = {
            "2330.TW": {
                "current_price": 1000.0,
                "predicted_price": 1020.0,
                "horizon_predicted_price": 1050.0,
                "potential": 5.0,
                "horizon_potential": 5.0,
                "day1_potential": 2.0,
                "trajectory": [1010.0, 1020.0, 1030.0, 1040.0, 1050.0],
                "quantiles": {"p10": 980.0, "p50": 1020.0, "p90": 1080.0, "p10_horizon": 980.0, "p50_horizon": 1050.0, "p90_horizon": 1080.0},
                "downside_risk": -2.0,
                "upside_potential": 8.0,
                "risk_reward_ratio": 4.0,
                "horizon": 5
            }
        }
        mock_get_model.return_value = mock_model

        df = pd.DataFrame({
            "Open": [990.0] * 65,
            "High": [1010.0] * 65,
            "Low": [980.0] * 65,
            "Close": [1000.0 + i * 0.1 for i in range(65)],
            "Volume": [10000] * 65
        })
        ctx = StockContext(ticker="2330.TW", df=df, fundamentals={"pe": 22.0, "pb": 5.0})

        res = self.strategy.evaluate(ctx)

        self.assertTrue(res.is_hit)
        self.assertEqual(res.predicted_price, 1020.0)
        self.assertEqual(res.potential, 5.0)
        self.assertIn("Timestral強", res.tags)
        self.assertIn("高風報比", res.tags)
        self.assertEqual(res.signals["risk_reward_ratio"], 4.0)

    def test_wrapper_flat_zero_volatility_skipped(self):
        wrapper = TimestralWrapper()
        wrapper._is_initialized = True
        wrapper.model = MagicMock()

        # Input with completely flat prices (zero std)
        flat_series = np.array([100.0] * 70)
        res = wrapper.forecast_batch({"FLAT": flat_series}, horizon=5)
        self.assertEqual(res, {})  # Correctly filtered out

    def test_wrapper_quantile_crossing_monotonic_clamping(self):
        wrapper = TimestralWrapper()
        wrapper._is_initialized = True

        mock_forecast = MagicMock()
        # Inverted quantiles from raw model: p10=110, p50=100, p90=95
        mock_forecast.quantile.side_effect = lambda q: [110.0] * 5 if abs(q - 0.1) < 1e-3 else ([95.0] * 5 if abs(q - 0.9) < 1e-3 else [100.0] * 5)
        mock_forecast.median = [100.0] * 5

        mock_model = MagicMock()
        mock_model.forecast.return_value = mock_forecast
        wrapper.model = mock_model

        # Moving prices with volatility
        prices = np.array([100.0 + (i % 5) for i in range(70)])
        res = wrapper.forecast_batch({"TEST": prices}, horizon=5)["TEST"]

        # Ensure monotonic clamp: p10 <= p50 <= p90
        q = res["quantiles"]
        self.assertLessEqual(q["p10_horizon"], q["p50_horizon"])
        self.assertLessEqual(q["p50_horizon"], q["p90_horizon"])

    def test_wrapper_bounded_risk_reward_when_p10_above_current(self):
        wrapper = TimestralWrapper()
        wrapper._is_initialized = True

        # Extremely bullish: current_price=100, p10=102, p50=108, p90=120
        mock_forecast = MagicMock()
        mock_forecast.quantile.side_effect = lambda q: [102.0] * 5 if abs(q - 0.1) < 1e-3 else ([120.0] * 5 if abs(q - 0.9) < 1e-3 else [108.0] * 5)
        mock_forecast.median = [108.0] * 5

        mock_model = MagicMock()
        mock_model.forecast.return_value = mock_forecast
        wrapper.model = mock_model

        prices = np.array([90.0 + i * 0.15 for i in range(70)])
        curr_price = float(prices[-1])  # ~100.35
        res = wrapper.forecast_batch({"BULL": prices}, horizon=5)["BULL"]

        # R/R must be strictly positive and bounded, never negative or inf
        rr = res["risk_reward_ratio"]
        self.assertGreater(rr, 0.0)
        self.assertLessEqual(rr, 50.0)

    def test_singleton_thread_safety(self):
        instances = []

        def worker():
            inst = get_timestral_model()
            instances.append(inst)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(instances), 10)
        # All instances must be the exact same singleton
        self.assertTrue(all(inst is instances[0] for inst in instances))


if __name__ == "__main__":
    unittest.main()
