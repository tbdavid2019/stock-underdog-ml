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
        self.assertEqual(res.metrics["p10"], 980.0)
        self.assertEqual(res.metrics["p50"], 1020.0)
        self.assertEqual(res.metrics["p90"], 1080.0)

    @patch("strategies.timestral.get_timestral_model")
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

        ctx_miss = StockContext(ticker="TEST_MISS", df=pd.DataFrame({"Close": [100.0] * 65}))
        res_miss = self.strategy.evaluate(ctx_miss)
        # 0.5% < min_potential_hit (1.0%), so it should NOT be hit even though horizon is 5.0%
        self.assertFalse(res_miss.is_hit)

        ctx_hit = StockContext(ticker="TEST_HIT", df=pd.DataFrame({"Close": [100.0] * 65}))
        res_hit = self.strategy.evaluate(ctx_hit)
        self.assertTrue(res_hit.is_hit)
        self.assertEqual(res_hit.potential, 2.5)

    def test_wrapper_uses_day1_for_potential_and_risk_reward(self):
        wrapper = TimestralWrapper()
        wrapper._is_initialized = True

        mock_forecast = MagicMock()
        mock_forecast.quantile.side_effect = lambda q: [99.0, 98.0, 97.0, 96.0, 95.0] if abs(q - 0.1) < 1e-3 else ([105.0, 106.0, 107.0, 108.0, 115.0] if abs(q - 0.9) < 1e-3 else [102.0, 104.0, 106.0, 108.0, 110.0])
        mock_forecast.median = [102.0, 104.0, 106.0, 108.0, 110.0]

        mock_model = MagicMock()
        mock_model.forecast.return_value = mock_forecast
        wrapper.model = mock_model

        prices = np.array([90.0 + i * 0.15 for i in range(70)])
        prices[-1] = 100.0  # curr_price = 100.0
        res = wrapper.forecast_batch({"BULL": prices}, horizon=5)["BULL"]

        self.assertEqual(res["predicted_price"], 102.0)
        self.assertEqual(res["potential"], 2.0)
        self.assertEqual(res["day1_potential"], 2.0)
        self.assertEqual(res["horizon_potential"], 10.0)
        # T+1 quantiles: p10=99.0, p90=105.0 -> risk_span=1.0, reward_span=5.0 -> rr=5.0
        self.assertEqual(res["risk_reward_ratio"], 5.0)

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

    def test_default_initialization_from_config(self):
        """驗證 TimestralStrategy 預設值正確自 config 讀取 T+1 適配之 0.5% 與 1.2x 門檻"""
        strat = TimestralStrategy()
        self.assertEqual(strat.min_potential_hit, 0.5)
        self.assertEqual(strat.min_risk_reward_hit, 1.2)

    @patch("strategies.timestral.get_timestral_model")
    def test_t1_moderate_bullish_hit_with_default_thresholds(self, mock_get_model):
        """驗證單日 T+1 預期漲幅 +0.8% 與盈虧比 1.5x 標的在預設配置下成功命中"""
        strat = TimestralStrategy()
        mock_model = MagicMock()
        mock_model.forecast_batch.return_value = {
            "KLAC": {
                "current_price": 150.0,
                "predicted_price": 151.2,
                "horizon_predicted_price": 153.0,
                "potential": 0.8,
                "horizon_potential": 2.0,
                "day1_potential": 0.8,
                "risk_reward_ratio": 1.5,
                "quantiles": {"p10": 147.0, "p50": 151.2, "p90": 155.0},
            }
        }
        mock_get_model.return_value = mock_model
        ctx = StockContext(ticker="KLAC", df=pd.DataFrame({"Close": [150.0] * 70}))
        res = strat.evaluate(ctx)
        self.assertTrue(res.is_hit)
        self.assertIn("Timestral看漲", res.tags)


if __name__ == "__main__":
    unittest.main()
