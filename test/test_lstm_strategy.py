"""
Unit tests for strategies.lstm
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from strategies.base import StockContext
from strategies.lstm import LSTMStrategy


class TestLSTMStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = LSTMStrategy(time_step=10, min_potential_hit=1.0)

    def test_insufficient_data(self):
        df = pd.DataFrame({"Close": [100.0] * 10})
        ctx = StockContext(ticker="TEST", df=df)
        res = self.strategy.evaluate(ctx)
        self.assertFalse(res.is_hit)
        self.assertEqual(res.signals.get("reason"), "數據不足")

    @patch("strategies.lstm.prepare_data")
    @patch("strategies.lstm.train_lstm_model")
    @patch("strategies.lstm.predict_next_day")
    def test_successful_prediction_hit(self, mock_predict, mock_train, mock_prepare):
        # 50 days of dummy data
        df = pd.DataFrame({
            "Open": [100.0] * 50,
            "High": [105.0] * 50,
            "Low": [95.0] * 50,
            "Close": [100.0] * 50,
            "Volume": [1000] * 50
        })
        mock_prepare.return_value = (np.zeros((10, 10, 5)), np.zeros((10, 1)), MagicMock())
        mock_train.return_value = MagicMock()
        # Predict 105 (+5%)
        mock_predict.return_value = 105.0

        ctx = StockContext(ticker="2330.TW", df=df, fundamentals={"pe": 20.0})
        res = self.strategy.evaluate(ctx)

        self.assertTrue(res.is_hit)
        self.assertEqual(res.predicted_price, 105.0)
        self.assertAlmostEqual(res.potential, 5.0)
        self.assertIn("LSTM強", res.tags)


if __name__ == "__main__":
    unittest.main()
