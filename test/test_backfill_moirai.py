"""
Unit tests for scripts.backfill_moirai_predictions
"""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from scripts.backfill_moirai_predictions import backfill_moirai


class TestBackfillMoirai(unittest.TestCase):

    @patch("scripts.backfill_moirai_predictions.StockFetcher.get_stock_data")
    @patch("scripts.backfill_moirai_predictions.get_moirai_model")
    @patch("scripts.backfill_moirai_predictions.DuckDBManager")
    def test_backfill_moirai_execution(self, mock_db_cls, mock_get_model, mock_get_stock):
        mock_db = MagicMock()
        mock_db.enabled = True
        mock_db_cls.return_value = mock_db

        # Mock query return
        mock_conn = MagicMock()
        mock_db._get_connection.return_value.__enter__.return_value = mock_conn

        df_pending = pd.DataFrame([
            {
                "ticker": "6770.T",
                "dt": "2026-11-03",
                "index_name": "日經225",
                "record_ts": "2026-11-03T07:16:12",
                "stored_curr_price": 1856.5
            }
        ])
        mock_conn.execute.return_value.df.return_value = df_pending
        mock_conn.execute.return_value.fetchone.return_value = (0,) # moi_row_count = 0

        # Mock model
        mock_model = MagicMock()
        mock_model.load_model.return_value = True
        mock_model.forecast_batch.return_value = {
            "6770.T": {
                "predicted_price": 1951.24,
                "potential": 5.10,
                "quantiles": {"p10": 1836.8, "p90": 2090.1}
            }
        }
        mock_get_model.return_value = mock_model

        # Mock price data
        idx = pd.date_range("2026-01-01", periods=100, freq="B")
        prices = [1800.0] * 99 + [1856.5]
        mock_get_stock.return_value = pd.DataFrame({"Close": prices}, index=idx)

        backfill_moirai(ticker_filter="6770.T", days=7)

        # Verify UPDATE was executed
        execute_calls = mock_conn.execute.call_args_list
        update_called = any("UPDATE predictions" in str(c) for c in execute_calls)
        insert_called = any("INSERT INTO predictions" in str(c) for c in execute_calls)

        self.assertTrue(update_called)
        self.assertTrue(insert_called)


if __name__ == "__main__":
    unittest.main()
