"""
Unit tests for strategies.registry
"""
import unittest
import pandas as pd
from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import StrategyRegistry, register_strategy


@register_strategy("test_mock_strat")
class MockStrategy(BaseStrategy):
    name = "Mock Strategy"
    category = "test"
    required_lookback = 10

    def evaluate(self, context: StockContext) -> StrategyResult:
        is_hit = context.df["Close"].iloc[-1] > 100
        return StrategyResult(
            ticker=context.ticker,
            strategy_name=self.name,
            is_hit=is_hit,
            score=85.0 if is_hit else 0.0,
            current_price=float(context.df["Close"].iloc[-1]),
            tags=["測試命中"] if is_hit else []
        )


class TestStrategyRegistry(unittest.TestCase):

    def test_registration_and_instantiation(self):
        self.assertIn("test_mock_strat", StrategyRegistry.list_strategies())
        strat = StrategyRegistry.get_strategy("test_mock_strat")
        self.assertIsInstance(strat, MockStrategy)
        self.assertEqual(strat.name, "Mock Strategy")

    def test_evaluate_execution(self):
        strat = StrategyRegistry.get_strategy("test_mock_strat")
        df = pd.DataFrame({"Close": [90, 110]})
        ctx = StockContext(ticker="2330.TW", df=df)
        result = strat.evaluate(ctx)

        self.assertTrue(result.is_hit)
        self.assertEqual(result.score, 85.0)
        self.assertEqual(result.tags, ["測試命中"])

    def test_unknown_strategy_error(self):
        with self.assertRaises(ValueError):
            StrategyRegistry.get_strategy("non_existent_strategy_xyz")


if __name__ == "__main__":
    unittest.main()
