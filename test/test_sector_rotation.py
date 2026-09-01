"""
test/test_sector_rotation.py - 測試板塊輪動策略模組
"""

import unittest
import pandas as pd
import numpy as np
from core.config import config
from strategies.base import StockContext
from strategies.sector_rotation import SectorRotationStrategy
from strategies.registry import StrategyRegistry


class TestSectorRotationStrategy(unittest.TestCase):
    def test_sector_config(self):
        self.assertEqual(config.sector.get_sector("2330.TW"), "半導體與IC設計")
        self.assertEqual(config.sector.get_sector("2882.TW"), "金融保險")
        self.assertEqual(config.sector.get_sector("NVDA"), "半導體與IC設計")
        self.assertEqual(config.sector.get_sector("SPCX"), "太空與航太防衛")

    def test_strategy_batch_evaluation(self):
        # 構造模擬數據
        dates = pd.date_range("2026-01-01", periods=100)
        # 上升趨勢
        df_semi = pd.DataFrame({"Close": np.linspace(500, 700, 100)}, index=dates)
        # 下降趨勢
        df_fin = pd.DataFrame({"Close": np.linspace(50, 40, 100)}, index=dates)

        contexts = {
            "2330.TW": StockContext(ticker="2330.TW", df=df_semi),
            "2882.TW": StockContext(ticker="2882.TW", df=df_fin)
        }

        strat = SectorRotationStrategy(top_sectors=1)
        results = strat.evaluate_batch(contexts)
        self.assertEqual(len(results), 2)

        semi_res = next(r for r in results if r.ticker == "2330.TW")
        fin_res = next(r for r in results if r.ticker == "2882.TW")

        self.assertTrue(semi_res.is_hit)
        self.assertTrue(semi_res.metadata.get("is_top_sector"))
        self.assertFalse(fin_res.is_hit)
        print(f"Sector Rotation Test Passed: 2330.TW is_hit={semi_res.is_hit}, signals={semi_res.signals}")

    def test_registry_discovery(self):
        self.assertIn("sector_rotation", StrategyRegistry.list_strategies())
        self.assertIn("institutional", StrategyRegistry.list_strategies())


if __name__ == "__main__":
    unittest.main()
