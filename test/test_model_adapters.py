"""Contract tests for the current research model adapters."""
import unittest
from unittest.mock import MagicMock, patch
from contextlib import nullcontext
import sys
import types

import numpy as np
import pandas as pd

from core.config import PipelineConfig
from models.timesfm_model import TimesFMWrapper
from models.moirai_model import MoiraiWrapper
from strategies.base import StockContext
from strategies.moirai import MoiraiStrategy


class TestTimesFM3Adapter(unittest.TestCase):
    def test_defaults_target_timesfm3_and_fixed_revision(self):
        from core.config import config
        self.assertEqual(config.pipeline.TIMESFM_MODEL_ID, "google/timesfm-3.0-pytorch")
        self.assertEqual(config.pipeline.TIMESFM_CHECKPOINT_REVISION,
                         "43046b85ec22d584a13f8098c2ed39c889e129c2")

    def test_predict_batch_uses_native_quantile_indices_and_covariates(self):
        wrapper = TimesFMWrapper()
        wrapper._is_initialized = True
        cov = np.arange(20, dtype=np.float32).reshape(2, 10)
        output = MagicMock()
        output.forecast = np.array([101., 102.])
        output.quantiles = np.array([[90., 94., 97., 99., 101., 103., 105., 108., 110.],
                                     [91., 95., 98., 100., 102., 104., 106., 109., 111.]])
        wrapper.model = MagicMock()
        wrapper.model.predict_batch.return_value = iter([output])

        result = wrapper.forecast_batch({"AAA": np.arange(91., 101.)}, horizon=2,
                                        past_covariates={"AAA": cov})["AAA"]
        call = wrapper.model.predict_batch.call_args.kwargs
        np.testing.assert_array_equal(call["past_only_covariates"][0], cov)
        self.assertEqual(result["quantiles"]["p10"], 90.)
        self.assertEqual(result["quantiles"]["p50"], 101.)
        self.assertEqual(result["quantiles"]["p90"], 110.)
        self.assertEqual(result["quantile_source"], "native")

    def test_invalid_output_isolated_to_ticker(self):
        wrapper = TimesFMWrapper()
        wrapper._is_initialized = True
        bad = MagicMock(forecast=np.array([np.nan]), quantiles=np.ones((1, 9)))
        good = MagicMock(forecast=np.array([101.]), quantiles=np.arange(9.)[None, :] + 95)
        wrapper.model = MagicMock()
        wrapper.model.predict_batch.return_value = iter([bad, good])
        result = wrapper.forecast_batch({"BAD": np.arange(100., 110.),
                                         "GOOD": np.arange(100., 110.)}, horizon=1)
        self.assertNotIn("BAD", result)
        self.assertIn("GOOD", result)

    def test_load_uses_timesfm3_without_legacy_fallback(self):
        wrapper = TimesFMWrapper()
        fake_pkg = MagicMock()
        fake_pkg.ModelConfig.return_value = object()
        fake_pkg.TimesFM3Evaluator.side_effect = RuntimeError("checkpoint unavailable")
        with patch.dict("sys.modules", {"timesfm3": fake_pkg}):
            self.assertFalse(wrapper.load_model())
        fake_pkg.TimesFM_2p5_200M_torch.from_pretrained.assert_not_called()


class TestMoiraiAdapter(unittest.TestCase):
    def _mock_runtime(self, prediction):
        class FakeForecast:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.received = None
            def to(self, device):
                return self
            def predict(self, **kwargs):
                self.received = kwargs
                return prediction
        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = nullcontext
        moirai_module = types.ModuleType("uni2ts.model.moirai2")
        moirai_module.Moirai2Forecast = FakeForecast
        modules = {
            "torch": fake_torch,
            "uni2ts": types.ModuleType("uni2ts"),
            "uni2ts.model": types.ModuleType("uni2ts.model"),
            "uni2ts.model.moirai2": moirai_module,
        }
        return patch.dict(sys.modules, modules)

    def test_native_quantile_keys_and_past_covariates(self):
        wrapper = MoiraiWrapper()
        wrapper._is_initialized = True
        wrapper.quantile_levels = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        # Moirai2Forecast.predict returns [batch, quantile, horizon, target].
        output = np.array([[[[90.]], [[94.]], [[97.]], [[99.]],
                            [[100.]], [[102.]], [[104.]], [[108.]], [[110.]]]])
        wrapper.module = object()
        cov = np.ones((2, 12), dtype=np.float32)
        with self._mock_runtime(output):
            result = wrapper.forecast_batch({"AAA": np.arange(89., 101.)}, horizon=1,
                                            past_covariates={"AAA": cov})["AAA"]
        call = wrapper.model.received
        assert call["past_feat_dynamic_real"][0].shape == (wrapper.model_context_len, 1)
        np.testing.assert_array_equal(call["past_feat_dynamic_real"][0][-12:, 0], cov.T.mean(axis=1))
        self.assertEqual(result["quantiles"]["p10"], 90.)
        self.assertEqual(result["quantiles"]["p50"], 100.)
        self.assertEqual(result["quantiles"]["p90"], 110.)
        self.assertEqual(result["quantile_source"], "native")

    def test_nonfinite_forecast_is_excluded(self):
        wrapper = MoiraiWrapper()
        wrapper._is_initialized = True
        wrapper.quantile_levels = tuple(np.arange(1, 10) / 10)
        wrapper.module = object()
        with self._mock_runtime(np.full((1, 9, 1, 1), np.inf)):
            result = wrapper.forecast_batch({"BAD": np.arange(100., 120.)}, 1)
        self.assertEqual(result, {})

    def test_forecast_batch_respects_configured_batch_size(self):
        wrapper = MoiraiWrapper()
        wrapper._is_initialized = True
        wrapper.batch_size = 2
        wrapper.quantile_levels = tuple(np.arange(1, 10) / 10)
        wrapper.module = object()
        calls = []

        class FakeForecast:
            def __init__(self, **kwargs):
                pass
            def to(self, device):
                return self
            def predict(self, **kwargs):
                calls.append(len(kwargs["past_target"]))
                return np.tile(np.array([90., 94., 97., 99., 100., 102., 104., 108., 110.])[None, :, None, None],
                               (len(kwargs["past_target"]), 1, 1, 1))

        fake_torch = types.ModuleType("torch")
        fake_torch.inference_mode = nullcontext
        moirai_module = types.ModuleType("uni2ts.model.moirai2")
        moirai_module.Moirai2Forecast = FakeForecast
        modules = {"torch": fake_torch, "uni2ts": types.ModuleType("uni2ts"),
                   "uni2ts.model": types.ModuleType("uni2ts.model"),
                   "uni2ts.model.moirai2": moirai_module}
        series = {ticker: np.arange(100., 120.) for ticker in ("A", "B", "C")}
        with patch.dict(sys.modules, modules):
            result = wrapper.forecast_batch(series, horizon=1)
        self.assertEqual(set(result), set(series))
        self.assertEqual(calls, [2, 1])

    @patch("core.config.PipelineConfig.DEPLOYMENT_MODE", "research")
    def test_strategy_forwards_covariates_and_feature_identity(self):
        strategy = MoiraiStrategy()
        wrapper = MagicMock()
        wrapper.forecast_batch.return_value = {"AAA": {"current_price": 100., "predicted_price": 102.,
            "potential": 2., "horizon_potential": 3., "risk_reward_ratio": 2.,
            "quantiles": {"p10": 99., "p50": 102., "p90": 105.},
            "model_version": "moirai-test", "quantile_source": "native"}}
        strategy._model_wrapper = wrapper
        cov = np.ones((2, 60), dtype=np.float32)
        ctx = StockContext("AAA", pd.DataFrame({"Close": np.arange(100., 160.)}), metadata={
            "past_covariates": cov, "feature_profile": "full_features",
            "feature_status": "ready", "feature_snapshot": "snap-1"})
        result = strategy.evaluate(ctx)
        np.testing.assert_array_equal(wrapper.forecast_batch.call_args.kwargs["past_covariates"]["AAA"], cov)
        self.assertEqual(result.metadata["feature_profile"], "full_features")
        self.assertEqual(result.metadata["feature_snapshot"], "snap-1")
        self.assertEqual(result.metadata["role"], "active")
        self.assertTrue(result.is_hit)
        self.assertTrue(result.signals["candidate_hit"])

    def test_full_features_without_covariates_fails_explicitly(self):
        strategy = MoiraiStrategy()
        ctx = StockContext("AAA", pd.DataFrame({"Close": np.arange(100., 160.)}), metadata={
            "feature_profile": "full_features", "feature_status": "ready"})
        result = strategy.evaluate(ctx)
        self.assertFalse(result.is_hit)
        self.assertEqual(result.metadata["status"], "features_unavailable")

    def test_models_are_disabled_in_production_without_eligibility(self):
        from core.config import usage_eligibility
        self.assertTrue(usage_eligibility("timesfm", "production")["eligible"])
        self.assertTrue(usage_eligibility("moirai", "production")["eligible"])
        self.assertTrue(usage_eligibility("timestral", "production")["eligible"])


if __name__ == "__main__":
    unittest.main()
