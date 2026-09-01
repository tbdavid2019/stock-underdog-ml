import unittest
from unittest.mock import patch, MagicMock
from main import parse_args
from pipeline.orchestrator import PipelineOrchestrator


class TestCLIMarketFiltering(unittest.TestCase):
    """Test CLI argument parsing and market filtering logic"""

    def test_parse_args_default(self):
        with patch("sys.argv", ["main.py"]):
            args = parse_args()
            self.assertEqual(args.market, "all")
            self.assertIsNone(args.index)
            self.assertFalse(args.no_db)
            self.assertFalse(args.no_notify)
            self.assertFalse(args.dry_run)

    def test_parse_args_tw_market(self):
        with patch("sys.argv", ["main.py", "--market", "tw", "--dry-run"]):
            args = parse_args()
            self.assertEqual(args.market, "tw")
            self.assertTrue(args.dry_run)

    def test_parse_args_us_market(self):
        with patch("sys.argv", ["main.py", "-m", "us", "--no-db"]):
            args = parse_args()
            self.assertEqual(args.market, "us")
            self.assertTrue(args.no_db)

    def test_parse_args_custom_index(self):
        with patch("sys.argv", ["main.py", "-i", "台灣50", "SP500"]):
            args = parse_args()
            self.assertEqual(args.index, ["台灣50", "SP500"])

    @patch("data.fetcher.StockFetcher.get_tw0050_stocks", return_value=["2330.TW"])
    @patch("data.fetcher.StockFetcher.get_tw0051_stocks", return_value=["2603.TW"])
    @patch("data.fetcher.StockFetcher.get_sp500_stocks", return_value=["AAPL"])
    @patch("data.macro.MacroRegimeAnalyzer.evaluate_us_market")
    def test_orchestrator_market_segmentation(self, mock_macro, mock_sp500, mock_tw51, mock_tw50):
        mock_macro_state = MagicMock()
        mock_macro_state.regime_name = "全面多頭 (Strong Bull)"
        mock_macro_state.exposure = 1.0
        mock_macro.return_value = mock_macro_state

        orchestrator = PipelineOrchestrator(enabled_strategies=["xuantie"])
        orchestrator.run_index_analysis = MagicMock(return_value=MagicMock())

        # Test TW market
        orchestrator.run_all_indices(market="tw", persist_db=False, send_notify=False)
        called_indices_tw = [call[0][0] for call in orchestrator.run_index_analysis.call_args_list]
        self.assertIn("台灣50", called_indices_tw)
        self.assertIn("台灣中型100", called_indices_tw)
        self.assertNotIn("SP500", called_indices_tw)

        # Test US market
        orchestrator.run_index_analysis.reset_mock()
        orchestrator.run_all_indices(market="us", persist_db=False, send_notify=False)
        called_indices_us = [call[0][0] for call in orchestrator.run_index_analysis.call_args_list]
        self.assertNotIn("台灣50", called_indices_us)
        self.assertIn("SP500", called_indices_us)

        # Test specific index names
        orchestrator.run_index_analysis.reset_mock()
        orchestrator.run_all_indices(index_names=["台灣50"], persist_db=False, send_notify=False)
        called_indices_custom = [call[0][0] for call in orchestrator.run_index_analysis.call_args_list]
        self.assertEqual(called_indices_custom, ["台灣50"])


if __name__ == "__main__":
    unittest.main()
