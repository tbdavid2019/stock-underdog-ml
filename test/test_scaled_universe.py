"""
Unit tests for Scaled 1,500 Stock Quant Universe (1,000 TW + 500 US)
Validates dynamic universe selection, delisting filtering, S&P 500 expansion, and chunked batch fetching.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from data.fetcher import StockFetcher


class TestScaledUniverse(unittest.TestCase):

    @patch("data.fetcher.CacheManager.set_index_cache")
    @patch("data.fetcher.requests.get")
    def test_get_sp500_stocks_full_scale(self, mock_get, mock_set_cache):
        """Test that get_sp500_stocks defaults to fetching the full 500+ constituent universe without artificial limit."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # Simulate 503 S&P 500 constituents
        mock_sp500_dict = {f"TICKER_{i}": f"Company {i}" for i in range(503)}
        mock_resp.json.return_value = {"SP500": mock_sp500_dict}
        mock_get.return_value = mock_resp

        stocks = StockFetcher.get_sp500_stocks()
        # Should contain all 503 + extra space/tech stocks
        self.assertGreaterEqual(len(stocks), 503)
        self.assertIn("TICKER_0", stocks)
        self.assertIn("TICKER_502", stocks)
        self.assertIn("ASTS", stocks)
        self.assertIn("RKLB", stocks)

    @patch("data.duckdb_manager.DuckDBManager")
    @patch.object(StockFetcher, "get_tw0050_stocks", return_value=[f"{2300+i}.TW" for i in range(50)])
    @patch.object(StockFetcher, "get_tw0051_stocks", return_value=[f"{3000+i}.TW" for i in range(100)])
    def test_get_tw_universe_dynamic_selection(self, mock_tw51, mock_tw50, mock_db_cls):
        """Test dynamic selection of 1,000 TW stocks from DuckDB with active/delisting filtering."""
        mock_db = MagicMock()
        mock_db.enabled = True
        mock_db_cls.return_value = mock_db

        # Build simulated market_universe and turnover rows
        # 50 TW50 + 100 TW51 + 88 supply chain nodes + 900 active liquid + 50 delisted (turnover 0)
        simulated_active_rows = []
        for i in range(1200):
            ticker = f"{5000+i}.TW"
            # Some tickers have 0 turnover (delisted or inactive)
            turnover = 10000000.0 if i < 900 else 0.0
            simulated_active_rows.append({
                "ticker": ticker,
                "local_symbol": str(5000+i),
                "name_local": f"Active {i}",
                "total_turnover": turnover,
                "listing_status": "ACTIVE" if i < 1100 else "DELISTED"
            })

        active_filtered = [
            r for r in simulated_active_rows
            if r["total_turnover"] > 0 and r["listing_status"] == "ACTIVE"
        ]
        mock_db.get_tw_ranked_active_stocks.return_value = active_filtered

        stocks = StockFetcher.get_tw_universe(limit=1000)
        self.assertEqual(len(stocks), 1000)
        # All TW0050 stocks must be present
        for st in mock_tw50.return_value:
            self.assertIn(st, stocks)
        # All TW0051 stocks must be present
        for st in mock_tw51.return_value:
            self.assertIn(st, stocks)
        # Delisted or 0 turnover stocks must NOT be present
        self.assertNotIn("5950.TW", stocks)

    @patch.object(StockFetcher, "get_tw_universe")
    @patch.object(StockFetcher, "get_tw0050_stocks", return_value=[f"{2300+i}.TW" for i in range(50)])
    @patch.object(StockFetcher, "get_tw0051_stocks", return_value=[f"{3000+i}.TW" for i in range(100)])
    def test_get_tw_active_stocks_disjoint(self, mock_tw51, mock_tw50, mock_tw_universe):
        """Test that get_tw_active_stocks returns stocks disjoint from TW0050 and TW0051."""
        all_1000 = mock_tw50.return_value + mock_tw51.return_value + [f"{4000+i}.TW" for i in range(850)]
        mock_tw_universe.return_value = all_1000

        active_stocks = StockFetcher.get_tw_active_stocks(limit=850, exclude_indices=True)
        self.assertEqual(len(active_stocks), 850)
        tw50_set = set(mock_tw50.return_value)
        tw51_set = set(mock_tw51.return_value)
        for st in active_stocks:
            self.assertNotIn(st, tw50_set)
            self.assertNotIn(st, tw51_set)

    @patch("data.fetcher.CacheManager.get_stock_data", return_value=None)
    @patch("data.fetcher.CacheManager.set_stock_data")
    @patch("yfinance.download")
    def test_get_stock_data_batch_chunking(self, mock_yf_download, mock_set_cache, mock_get_cache):
        """Test that get_stock_data_batch splits large lists (>200) into safe chunks of at most 200 tickers."""
        tickers = [f"TICKER_{i}" for i in range(450)]
        
        # Mock yfinance return
        def side_effect(ticker_chunk, **kwargs):
            cols = pd.MultiIndex.from_product([ticker_chunk, ["Open", "High", "Low", "Close", "Volume"]])
            df = pd.DataFrame(100.0, index=pd.date_range("2026-01-01", periods=10), columns=cols)
            return df

        mock_yf_download.side_effect = side_effect

        results = StockFetcher.get_stock_data_batch(tickers, period="6mo", use_cache=False, chunk_size=200)
        self.assertEqual(len(results), 450)
        # Should have called yfinance 3 times (200 + 200 + 50)
        self.assertEqual(mock_yf_download.call_count, 3)

    @patch("pipeline.orchestrator.MacroRegimeAnalyzer.evaluate_tw_market")
    @patch("pipeline.orchestrator.StockFetcher.get_tw0050_stocks", return_value=["2330.TW"])
    @patch("pipeline.orchestrator.StockFetcher.get_tw0051_stocks", return_value=["3017.TW"])
    @patch("pipeline.orchestrator.StockFetcher.get_tw_active_stocks", return_value=["6805.TW"])
    def test_pipeline_orchestrator_market_tw_routing(self, mock_active, mock_tw51, mock_tw50, mock_macro):
        """Test that PipelineOrchestrator routes --market tw to TW50, TW100, and TW-Active."""
        from pipeline.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(enabled_strategies=[])
        
        with patch.object(orchestrator, "run_index_analysis") as mock_run_index:
            orchestrator.run_all_indices(market="tw", send_notify=False, persist_db=False)
            called_indices = [call.args[0] for call in mock_run_index.call_args_list]
            self.assertIn("台灣50", called_indices)
            self.assertIn("台灣中型100", called_indices)
            self.assertIn("台股活躍精選", called_indices)
            self.assertEqual(len(called_indices), 3)


if __name__ == "__main__":
    unittest.main()
