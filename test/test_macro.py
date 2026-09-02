"""
test/test_macro.py - 測試美股宏觀風控分析模組
"""

import unittest
from data.macro import MacroRegimeAnalyzer, MacroState


class TestMacroRegimeAnalyzer(unittest.TestCase):
    def test_macro_state_defaults(self):
        state = MacroState()
        self.assertEqual(state.exposure, 1.0)
        self.assertTrue(state.spy_above_ma60)
        d = state.to_dict()
        self.assertIn("regime_name", d)
        self.assertIn("exposure", d)
        self.assertIn("vix", d)

    def test_evaluate_us_market(self):
        state = MacroRegimeAnalyzer.evaluate_us_market(period="3mo")
        self.assertIsInstance(state, MacroState)
        self.assertEqual(state.market, "US")
        self.assertGreater(state.vix, 0)
        self.assertIn(state.exposure, [0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.85, 1.0])
        print("Test US Macro State:", state.summary)
        print("Warnings:", state.warnings)

    def test_evaluate_tw_market(self):
        state = MacroRegimeAnalyzer.evaluate_tw_market(period="3mo")
        self.assertIsInstance(state, MacroState)
        self.assertEqual(state.market, "TW")
        self.assertGreater(state.vix, 0)
        self.assertIn(state.exposure, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 1.0])
        print("Test TW Market State:", state.summary)

    def test_evaluate_market_routing(self):
        tw_state = MacroRegimeAnalyzer.evaluate_market("台灣50", period="3mo")
        self.assertEqual(tw_state.market, "TW")
        us_state = MacroRegimeAnalyzer.evaluate_market("SP500", period="3mo")
        self.assertEqual(us_state.market, "US")


if __name__ == "__main__":
    unittest.main()
