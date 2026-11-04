"""
test/test_model_plugin_protocol.py - Comprehensive Unit Tests for Elephant & Goldfish ModelPlugin Protocol.

Validates:
1. Protocol conformity via runtime_checkable for all foundation model plugins.
2. Incomplete plugin rejection.
3. ModelPluginRegistry registration, lazy instantiation, unregistration, and teardown.
4. Lifecycle hooks (initialize, health_check, predict_batch, teardown).
5. Cross-market parameter matrix constraints (e.g. Taiwan ±10% circuit limit clipping).
"""
import unittest
from unittest.mock import MagicMock
import numpy as np

from models.protocol import BaseModelPlugin, ModelPlugin, ModelPluginRegistry
from models.plugins import TimesFMPlugin, TimestralPlugin, Moirai2Plugin


class DummyIncompletePlugin:
    model_id = "dummy/incomplete"
    # Intentionally missing display_name, version, etc.


class TestModelPluginProtocol(unittest.TestCase):

    def test_standard_plugins_conform_to_protocol(self):
        """驗證 TimesFM, Timestral, MOIRAI-2 插件實例均符合 ModelPlugin 協議"""
        tfm = TimesFMPlugin()
        tst = TimestralPlugin()
        moi = Moirai2Plugin()

        self.assertTrue(isinstance(tfm, ModelPlugin))
        self.assertTrue(isinstance(tst, ModelPlugin))
        self.assertTrue(isinstance(moi, ModelPlugin))

        # Check required fields
        for plugin in (tfm, tst, moi):
            self.assertTrue(hasattr(plugin, "model_id") and bool(plugin.model_id))
            self.assertTrue(hasattr(plugin, "display_name") and bool(plugin.display_name))
            self.assertTrue(hasattr(plugin, "version") and bool(plugin.version))
            self.assertTrue(hasattr(plugin, "author") and bool(plugin.author))
            self.assertIsInstance(plugin.supported_markets, list)
            self.assertIn("tw", plugin.supported_markets)
            self.assertIn("us", plugin.supported_markets)
            self.assertIn("jp", plugin.supported_markets)
            self.assertIn(plugin.default_role, ["active", "shadow", "candidate"])
            self.assertIsInstance(plugin.supports_covariates, bool)
            self.assertIsInstance(plugin.required_covariates, list)

    def test_incomplete_plugin_rejected(self):
        """驗證缺漏生命週期欄位或方法時無法通過 ModelPlugin 檢查"""
        dummy = DummyIncompletePlugin()
        self.assertFalse(isinstance(dummy, ModelPlugin))

    def test_model_plugin_registry_operations(self):
        """驗證 ModelPluginRegistry 增刪查改、類別延遲實例化與 Teardown"""
        reg = ModelPluginRegistry()

        # 註冊類別
        reg.register(TimesFMPlugin)
        self.assertIn("google/timesfm-3.0-pytorch", reg.list_models())

        # 延遲實例化
        instance = reg.get("google/timesfm-3.0-pytorch")
        self.assertIsNotNone(instance)
        self.assertIsInstance(instance, TimesFMPlugin)

        # 註冊實例
        mock_custom = TimesFMPlugin()
        mock_custom.model_id = "custom/test-model"
        reg.register(mock_custom)
        self.assertIn("custom/test-model", reg.list_models())
        self.assertEqual(reg.get("custom/test-model"), mock_custom)

        # 重複註冊報錯 (override=False) - 實例重複
        with self.assertRaises(KeyError):
            reg.register(mock_custom, override=False)

        # 重複註冊報錯 (override=False) - 類別重複
        class DummyPluginClass:
            model_id = "custom/test-model"
        with self.assertRaises(KeyError):
            reg.register(DummyPluginClass, override=False)

        # 卸載
        self.assertTrue(reg.unregister("custom/test-model"))
        self.assertNotIn("custom/test-model", reg.list_models())
        self.assertFalse(reg.unregister("non_existent"))

        # 清空註冊表
        reg.clear()
        self.assertEqual(len(reg.list_models()), 0)

    def test_timesfm_plugin_lifecycle_and_market_clipping(self):
        """驗證 TimesFM 插件初始化、健康檢查、推論輸出與台股 ±10% 剛性截斷"""
        mock_wrapper = MagicMock()
        mock_wrapper.last_status = "ready"
        mock_wrapper.device = "cpu"
        mock_wrapper.load_model.return_value = True

        # Mock wrapper forecast output
        # Current price = 100.0, raw forecast = 120.0 (would be +20%, exceeding TW 10% limit)
        mock_wrapper.forecast_batch.return_value = {
            "2330.TW": {
                "current_price": 100.0,
                "predicted_price": 120.0,
                "potential": 20.0,
                "quantiles": {"p10": 95.0, "p50": 120.0, "p90": 130.0},
                "trajectory": [105.0, 110.0, 115.0, 118.0, 120.0],
                "risk_reward_ratio": 2.5,
            }
        }

        plugin = TimesFMPlugin(wrapper=mock_wrapper)

        # 1. Initialize
        init_res = plugin.initialize({"wrapper": mock_wrapper})
        self.assertTrue(init_res)

        # 2. Health check
        health = plugin.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["device"], "cpu")
        self.assertEqual(health["model_id"], plugin.model_id)

        # 3. Predict batch with market_params limit (TW 10% max daily change)
        series_dict = {"2330.TW": np.linspace(90, 100, 60)}
        covariates_dict = {
            "2330.TW": {
                "vix": np.ones(60, dtype=np.float32) * 15.0,
                "usdtwd": np.ones(60, dtype=np.float32) * 32.0,
            }
        }
        market_params = {
            "max_daily_change_pct": 10.0,
            "min_potential_threshold": 0.5,
        }

        results = plugin.predict_batch(
            series_dict=series_dict,
            covariates_dict=covariates_dict,
            horizon=5,
            market="tw",
            market_params=market_params,
        )

        self.assertIn("2330.TW", results)
        res = results["2330.TW"]

        # 原預測 120.0 在台股應被嚴格防呆截斷在 110.0 (現價 100 + 10%)
        self.assertAlmostEqual(res["predicted_price"], 110.0, places=2)
        self.assertAlmostEqual(res["potential"], 10.0, places=2)
        self.assertLessEqual(res["quantiles"]["p50"], 110.0)
        self.assertLessEqual(res["quantiles"]["p90"], 110.0)
        self.assertEqual(len(res["trajectory"]), 5)
        self.assertEqual(res["risk_reward_ratio"], 2.5)

        # 4. Teardown
        plugin.teardown()
        self.assertIsNone(plugin.wrapper)

    def test_timestral_plugin_lifecycle_and_prediction(self):
        """驗證 Timestral 插件生命週期與批次推論運算"""
        mock_wrapper = MagicMock()
        mock_wrapper.last_status = "ready"
        mock_wrapper.device = "mps"
        mock_wrapper.forecast_batch.return_value = {
            "NVDA": {
                "current_price": 120.0,
                "predicted_price": 126.0,
                "potential": 5.0,
                "quantiles": {"p10": 115.0, "p50": 126.0, "p90": 132.0},
                "trajectory": [121.0, 122.5, 124.0, 125.0, 126.0],
                "risk_reward_ratio": 3.0,
            }
        }

        plugin = TimestralPlugin(wrapper=mock_wrapper)
        health = plugin.health_check()
        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["device"], "mps")

        # US market without limit collar
        market_params = {"max_daily_change_pct": None, "min_potential_threshold": 1.5}
        results = plugin.predict_batch(
            series_dict={"NVDA": np.linspace(110, 120, 128)},
            market="us",
            market_params=market_params,
        )

        self.assertIn("NVDA", results)
        res = results["NVDA"]
        self.assertAlmostEqual(res["predicted_price"], 126.0)
        self.assertAlmostEqual(res["potential"], 5.0)

    def test_moirai2_plugin_lifecycle_and_covariates(self):
        """驗證 MOIRAI-2 多變量側參數矩陣傳遞與推論"""
        mock_wrapper = MagicMock()
        mock_wrapper.last_status = "ready"
        mock_wrapper.device = "cuda:0"
        mock_wrapper.forecast_batch.return_value = {
            "7203.T": {
                "current_price": 2500.0,
                "predicted_price": 2550.0,
                "potential": 2.0,
                "quantiles": {"p10": 2400.0, "p50": 2550.0, "p90": 2650.0},
                "trajectory": [2510.0, 2520.0, 2530.0, 2540.0, 2550.0],
                "risk_reward_ratio": 1.8,
            }
        }

        plugin = Moirai2Plugin(wrapper=mock_wrapper)
        self.assertTrue(plugin.supports_covariates)
        self.assertIn("vix", plugin.required_covariates)

        cov_dict = {
            "7203.T": {
                "vix": np.ones(90, dtype=np.float32) * 18.0,
                "usdjpy": np.ones(90, dtype=np.float32) * 155.0,
            }
        }

        results = plugin.predict_batch(
            series_dict={"7203.T": np.linspace(2400, 2500, 90)},
            covariates_dict=cov_dict,
            horizon=5,
            market="jp",
        )

        self.assertIn("7203.T", results)
        self.assertAlmostEqual(results["7203.T"]["predicted_price"], 2550.0)
        # Verify wrapper received 2D array past_covariates
        call_kwargs = mock_wrapper.forecast_batch.call_args.kwargs
        past_covs = call_kwargs["past_covariates"]["7203.T"]
        self.assertEqual(past_covs.shape, (2, 90))

    def test_dynamic_tag_generation_and_active_plugins(self):
        """驗證純粹基於模型實體名稱的動態標籤生成與活躍插件清單"""
        tfm = TimesFMPlugin()
        tst = TimestralPlugin()
        moi = Moirai2Plugin()

        # 看多標籤
        self.assertEqual(tfm.generate_tags(6.2), ["TimesFM看多", "TimesFM強"])
        self.assertEqual(tst.generate_tags(3.5), ["Timestral看多"])
        self.assertEqual(moi.generate_tags(8.0), ["MOIRAI-2看多", "MOIRAI-2強"])

        # 看空標籤
        self.assertEqual(tfm.generate_tags(-4.5), ["TimesFM看空"])
        self.assertEqual(tst.generate_tags(-6.0), ["Timestral看空"])

        # 活躍插件清單
        reg = ModelPluginRegistry()
        reg.register(tfm)
        reg.register(tst)
        active = reg.list_active_plugins()
        active_ids = [getattr(p, "plugin_id", "") for p in active]
        self.assertIn("timesfm", active_ids)
        self.assertIn("timestral", active_ids)


if __name__ == "__main__":
    unittest.main()
