"""
Unit tests for pipeline.orchestrator
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import datetime
from types import SimpleNamespace
from pipeline.orchestrator import PipelineOrchestrator
from strategies.base import StrategyResult


class TestPipelineOrchestrator(unittest.TestCase):

    @patch("pipeline.orchestrator.persist_forecast_results")
    @patch("pipeline.orchestrator.ForecastStore")
    @patch("pipeline.orchestrator.build_market_features")
    @patch("pipeline.orchestrator.FundamentalProvider.get_fundamentals_batch", return_value={})
    @patch("pipeline.orchestrator.StockFetcher.get_stock_data_batch")
    def test_pipeline_attaches_features_before_model_and_saves_shadow(self, fetch, funds, build, store, persist):
        from data.macro import MacroState
        idx = pd.date_range("2026-01-05", periods=3, freq="B")
        fetch.return_value = {"AAPL": pd.DataFrame({"Close": [100., 101., 102.]}, index=idx)}
        build.return_value = {"AAPL": SimpleNamespace(dates=list(idx.date), past_covariates=np.ones((2, 3)),
            status="full_features", metadata={"snapshot_id": "snap", "profile_id": "US-v1"})}
        db = MagicMock(enabled=True, db_path="/tmp/test-forecast.duckdb")
        orchestrator = PipelineOrchestrator(enabled_strategies=["xuantie"], db_manager=db, duckdb_manager=db)
        strategy = MagicMock(name="strategy")
        strategy.name, strategy.category = "MOIRAI-2", "ml"
        strategy.evaluate_batch.return_value = []
        orchestrator.strategies = [strategy]
        with patch.object(orchestrator.ai_narrative_engine, "generate_narrative", return_value=""), \
             patch("pipeline.orchestrator.print_evaluation_report"), \
             patch("pipeline.orchestrator.datetime") as clock, \
             patch("pipeline.orchestrator.config.pipeline.FORECAST_FEATURE_PROFILE", "full_features"):
            clock.datetime.now.return_value = datetime.datetime(2026, 1, 8, 22, tzinfo=datetime.timezone.utc)
            clock.timezone = datetime.timezone
            orchestrator.run_index_analysis("SP500", ["AAPL"], macro_state=MacroState(), persist_db=True, send_notify=False)
        contexts = strategy.evaluate_batch.call_args.args[0]
        self.assertEqual(contexts["AAPL"].metadata["feature_status"], "ready")
        np.testing.assert_array_equal(contexts["AAPL"].metadata["past_covariates"], np.ones((2, 3)))
        self.assertIn("moirai", persist.call_args.args[2])

    @patch("pipeline.orchestrator.MacroRegimeAnalyzer.evaluate_market")
    @patch("pipeline.orchestrator.StockFetcher.get_stock_data_batch")
    @patch("pipeline.orchestrator.FundamentalProvider.get_fundamentals_batch")
    def test_run_index_analysis_mocked(self, mock_get_funds, mock_get_stocks, mock_macro):
        from data.macro import MacroState
        mock_macro.return_value = MacroState(market="TW", regime_name="全面多頭 (Strong Bull)", exposure=1.0, vix=14.5)
        
        # Mock stock data
        df = pd.DataFrame({"Close": [100.0] * 80, "MA60": [95.0] * 80})
        mock_get_stocks.return_value = {"2330.TW": df}
        mock_get_funds.return_value = {"2330.TW": {"pe": 18.0, "pb": 2.0}}

        mock_db = MagicMock()
        mock_db.enabled = False

        orchestrator = PipelineOrchestrator(
            enabled_strategies=["xuantie"],
            db_manager=mock_db
        )

        with patch("pipeline.orchestrator.send_dual_strategy_results") as mock_notify:
            report = orchestrator.run_index_analysis(
                "台灣50", 
                ["2330.TW"], 
                period="6mo", 
                persist_db=False, 
                send_notify=True
            )

            self.assertEqual(report.index_name, "台灣50")
            self.assertIn("xuantie", report.strategy_results)
            mock_notify.assert_called_once()

    @patch("pipeline.orchestrator.StockFetcher.get_stock_data_batch", return_value={})
    @patch("pipeline.orchestrator.FundamentalProvider.get_fundamentals_batch", return_value={})
    def test_forecast_feature_profile_defaults_to_full_features(self, *_):
        from core.config import config
        self.assertEqual(config.pipeline.FORECAST_FEATURE_PROFILE, "full_features")

    @patch("pipeline.orchestrator.persist_forecast_results")
    @patch("pipeline.orchestrator.ForecastStore")
    @patch("pipeline.orchestrator.build_market_features", return_value={})
    @patch("pipeline.orchestrator.FundamentalProvider.get_fundamentals_batch", return_value={})
    @patch("pipeline.orchestrator.StockFetcher.get_stock_data_batch")
    def test_pipeline_falls_back_to_price_only_when_covariates_unavailable(self, fetch, funds, build, store, persist):
        from data.macro import MacroState
        idx = pd.date_range("2026-01-05", periods=3, freq="B")
        fetch.return_value = {"6770.T": pd.DataFrame({"Close": [1800., 1850., 1856.5]}, index=idx)}
        db = MagicMock(enabled=True, db_path="/tmp/test-forecast.duckdb")
        orchestrator = PipelineOrchestrator(enabled_strategies=["xuantie"], db_manager=db, duckdb_manager=db)
        strategy = MagicMock(name="strategy")
        strategy.name, strategy.category = "MOIRAI-2", "ml"
        strategy.evaluate_batch.return_value = []
        orchestrator.strategies = [strategy]
        with patch.object(orchestrator.ai_narrative_engine, "generate_narrative", return_value=""), \
             patch("pipeline.orchestrator.print_evaluation_report"), \
             patch("pipeline.orchestrator.datetime") as clock, \
             patch.dict("os.environ", {"STRICT_COVARIATE_MODE": "false"}), \
             patch("pipeline.orchestrator.config.pipeline.FORECAST_FEATURE_PROFILE", "full_features"):
            clock.datetime.now.return_value = datetime.datetime(2026, 1, 8, 22, tzinfo=datetime.timezone.utc)
            clock.timezone = datetime.timezone
            orchestrator.run_index_analysis("日經225", ["6770.T"], macro_state=MacroState(), persist_db=True, send_notify=False)
        contexts = strategy.evaluate_batch.call_args.args[0]
        self.assertEqual(contexts["6770.T"].metadata["feature_status"], "price_only")
        self.assertEqual(contexts["6770.T"].metadata["feature_profile"], "price_only")


if __name__ == "__main__":
    unittest.main()
