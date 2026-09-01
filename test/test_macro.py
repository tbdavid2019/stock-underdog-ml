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
        self.assertGreater(state.vix, 0)
        self.assertIn(state.exposure, [0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.85, 1.0])
        print("Test Macro State:", state.summary)
        print("Warnings:", state.warnings)


if __name__ == "__main__":
    unittest.main()
