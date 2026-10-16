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
        # 使用 mock 確保單元測試在 0.001 秒內極速完成且不受外網波動影響
        from unittest.mock import patch
        with patch.object(engine, '_call_openai_compatible', return_value="台灣中型100多頭回調格局，建議維持85%曝險，採取防禦觀望策略。"):
            res = engine.generate_narrative("台灣中型100", macro, sample_report)
            self.assertTrue(len(res) > 20)
            self.assertIn("台灣中型100", res)

    def test_timesfm_in_prompt_and_narrative(self):
        engine = AINarrativeEngine()
        macro = MacroState(regime_name="全面多頭 (Strong Bull)", exposure=1.0, vix=14.0)
        sample_report = {
            "xuantie_results": [{"ticker": "2330.TW"}],
            "lstm_results": [{"ticker": "2330.TW", "potential": 4.5}],
            "timesfm_results": [
                {"ticker": "2330.TW", "potential": 5.8, "risk_reward_ratio": 2.6},
                {"ticker": "2454.TW", "potential": 3.9, "risk_reward_ratio": 1.9}
            ],
            "overlap_results": [{
                "ticker": "2330.TW",
                "lstm_potential": 4.5,
                "timesfm_potential": 5.8,
                "risk_reward_ratio": 2.6,
                "tags": ["👑四重共振", "🔮雙ML共振", "高盈虧比"],
                "pullback_type": "MA60 (+1.8%)",
                "pe": 22.0,
                "pb": 5.0
            }]
        }

        # 1. 驗證 prompt 中是否注入 TimesFM 指標與雙ML共振提示
        prompt = engine._build_context_prompt("台灣50", macro, sample_report)
        self.assertIn("Google TimesFM (200M) 預測完成: 2 支", prompt)
        self.assertIn("TimesFM +5.80%(盈虧比2.60x)", prompt)
        self.assertIn("🔮雙ML共振", prompt)

        # 2. 驗證純程式規則模板是否包含雙 ML 預期
        template = engine._generate_template_narrative("台灣50", macro, sample_report)
        self.assertIn("2330.TW", template)
        self.assertIn("TimesFM +5.80% / LSTM +4.50%", template)
        self.assertIn("👑四重共振", template)

    def test_timestral_in_prompt_and_narrative(self):
        engine = AINarrativeEngine()
        macro = MacroState(regime_name="全面多頭 (Strong Bull)", exposure=1.0, vix=14.0)
        sample_report = {
            "xuantie_results": [{"ticker": "2330.TW"}],
            "timesfm_results": [
                {"ticker": "2330.TW", "potential": 5.8, "risk_reward_ratio": 2.6}
            ],
            "timestral_results": [
                {"ticker": "2330.TW", "potential": 6.2, "risk_reward_ratio": 2.8}
            ],
            "overlap_results": [{
                "ticker": "2330.TW",
                "timesfm_potential": 5.8,
                "timestral_potential": 6.2,
                "timestral_p10": 980.0,
                "timestral_p90": 1050.0,
                "risk_reward_ratio": 2.8,
                "tags": ["🏆雙模型共振", "高盈虧比"],
                "pullback_type": "MA60 (+1.8%)",
                "pe": 22.0,
                "pb": 5.0
            }]
        }

        # 1. 驗證 prompt 中是否注入 TimesFM 與 Timestral 雙模型共振提示
        prompt = engine._build_context_prompt("台灣50", macro, sample_report)
        self.assertIn("Google TimesFM (200M) 預測完成: 1 支", prompt)
        self.assertIn("Mistral Timestral (209M) 預測完成: 1 支", prompt)
        self.assertIn("Timestral +6.20%[980.0~1050.0]", prompt)
        self.assertIn("🏆雙模型共振", prompt)

        # 2. 驗證純程式規則模板是否包含雙 Foundation Model 預期
        template = engine._generate_template_narrative("台灣50", macro, sample_report)
        self.assertIn("2330.TW", template)
        self.assertIn("TimesFM +5.80% / Timestral +6.20%", template)
        self.assertIn("🏆雙模型共振", template)


if __name__ == "__main__":
    unittest.main()
