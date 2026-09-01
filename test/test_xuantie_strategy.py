"""
Unit tests for strategies.xuantie
"""
import unittest
import pandas as pd
import numpy as np
from strategies.base import StockContext
from strategies.xuantie import XuanTieStrategy


class TestXuanTieStrategy(unittest.TestCase):

    def setUp(self):
        self.strategy = XuanTieStrategy(lookback=10, tolerance=0.05)

    def test_insufficient_data(self):
        df = pd.DataFrame({"Close": [100.0] * 20})
        ctx = StockContext(ticker="TEST", df=df)
        res = self.strategy.evaluate(ctx)
        self.assertFalse(res.is_hit)
        self.assertEqual(res.signals.get("reason"), "數據不足")

    def test_major_trend_up_and_pullback_hit(self):
        # 100 days of data where MA60 slopes upwards
        closes = np.linspace(100, 200, 100)
        df = pd.DataFrame({"Close": closes})
        df["MA60"] = df["Close"].rolling(window=60).mean()
        df["MA120"] = df["Close"].rolling(window=60).mean() * 0.95
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA250"] = df["Close"].rolling(window=60).mean()

        # Set last price near MA60 (+1%)
        ma60_last = df["MA60"].iloc[-1]
        df.iloc[-1, df.columns.get_loc("Close")] = ma60_last * 1.01

        ctx = StockContext(ticker="2330.TW", df=df, fundamentals={"pe": 18.0, "pb": 2.2})
        res = self.strategy.evaluate(ctx)

        self.assertTrue(res.is_hit)
        self.assertIn("玄鐵買點", res.tags)
        self.assertIn("MA60", res.signals["pullback_type"])
        self.assertEqual(res.metrics["pe"], 18.0)


if __name__ == "__main__":
    unittest.main()
