"""
Unit tests for pipeline.orchestrator
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pipeline.orchestrator import PipelineOrchestrator
from strategies.base import StrategyResult


class TestPipelineOrchestrator(unittest.TestCase):

    @patch("pipeline.orchestrator.StockFetcher.get_stock_data_batch")
    @patch("pipeline.orchestrator.FundamentalProvider.get_fundamentals_batch")
    def test_run_index_analysis_mocked(self, mock_get_funds, mock_get_stocks):
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


if __name__ == "__main__":
    unittest.main()
