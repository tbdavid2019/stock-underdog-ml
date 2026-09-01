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
        self.assertIn("lstm", config.pipeline.ENABLED_STRATEGIES)
        self.assertAlmostEqual(config.pipeline.STRATEGY_WEIGHTS["xuantie"], 0.4)

    def test_legacy_import_compatibility(self):
        import config as legacy_config
        self.assertEqual(legacy_config.config.api.BASE_URL, "https://answerbook.david888.com")


if __name__ == "__main__":
    unittest.main()
