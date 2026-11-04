"""
Unit tests for core.config
"""
import unittest
from core.config import config, Config, DatabaseConfig, DataAPIConfig, PipelineConfig


class TestConfig(unittest.TestCase):

    def test_config_singleton(self):
        self.assertIsInstance(config, Config)
        self.assertEqual(config.api.BASE_URL, "https://answerbook.david888.com")
        self.assertTrue(config.api.TW0050_URL.endswith("/TW0050"))

    def test_pipeline_config_defaults(self):
        self.assertEqual(config.pipeline.DEFAULT_PERIOD, "6mo")
        self.assertIn("xuantie", config.pipeline.ENABLED_STRATEGIES)
        self.assertIn("timestral", config.pipeline.ENABLED_STRATEGIES)
        self.assertIn("timesfm", config.pipeline.ENABLED_STRATEGIES)
        self.assertIn("moirai", config.pipeline.ENABLED_STRATEGIES)
        self.assertAlmostEqual(config.pipeline.STRATEGY_WEIGHTS["xuantie"], 0.30)
        self.assertAlmostEqual(config.pipeline.STRATEGY_WEIGHTS["timesfm"], 0.20)
        self.assertAlmostEqual(config.pipeline.STRATEGY_WEIGHTS["timestral"], 0.20)
        self.assertAlmostEqual(config.pipeline.STRATEGY_WEIGHTS["moirai"], 0.15)
        self.assertAlmostEqual(config.pipeline.STRATEGY_WEIGHTS["institutional"], 0.15)

    def test_model_version_and_initial_roles(self):
        self.assertEqual(config.pipeline.TIMESFM_MODEL_ID, "google/timesfm-3.0-pytorch")
        self.assertEqual(config.pipeline.MOIRAI_MODEL_ID, "Salesforce/moirai-2.0-R-small")
        self.assertEqual(config.pipeline.MODEL_ROLES["moirai"], "active")
        self.assertEqual(config.pipeline.MODEL_ROLES["timesfm"], "active")


if __name__ == "__main__":
    unittest.main()
