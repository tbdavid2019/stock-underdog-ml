"""
test/test_ai_narrative.py - 測試 3-Tier Fallback LLM 研報解讀引擎
"""

import unittest
from evaluators.ai_narrative import AINarrativeEngine
from data.macro import MacroState


class TestAINarrativeEngine(unittest.TestCase):
    def test_template_fallback(self):
        engine = AINarrativeEngine()
        macro = MacroState(regime_name="全面多頭 (Strong Bull)", exposure=1.0, vix=15.2)
        sample_report = {
            "xuantie_results": [{"ticker": "2330.TW"}],
            "lstm_results": [{"ticker": "2330.TW"}],
            "overlap_results": [{
                "ticker": "2330.TW",
                "potential": 5.5,
                "tags": ["三重共振", "投信連買", "MA60回調"],
                "pullback_type": "MA60 (+2.1%)",
                "pe": 24.5,
                "pb": 6.2
            }]
        }

        # 測試規則模板生成 (無依賴)
        template_text = engine._generate_template_narrative("台灣50", macro, sample_report)
        self.assertIn("全面多頭", template_text)
        self.assertIn("2330.TW", template_text)
        print("Generated Template Narrative:\n", template_text)

    def test_engine_generate_narrative(self):
        engine = AINarrativeEngine()
        macro = MacroState(regime_name="多頭回調 (Bull Pullback)", exposure=0.85, vix=17.5)
        sample_report = {
            "xuantie_results": [],
            "lstm_results": [],
            "overlap_results": []
        }
        res = engine.generate_narrative("台灣中型100", macro, sample_report)
        self.assertTrue(len(res) > 20)
        print("Engine Generated Output:\n", res)


if __name__ == "__main__":
    unittest.main()
