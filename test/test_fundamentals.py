"""
Unit tests for data.fundamentals
"""
import unittest
from unittest.mock import patch, MagicMock
from data.fundamentals import sanitize_metric, FundamentalProvider


class TestFundamentals(unittest.TestCase):

    def test_sanitize_metric(self):
        self.assertEqual(sanitize_metric(15.5), 15.5)
        self.assertEqual(sanitize_metric("20.1"), 20.1)
        self.assertIsNone(sanitize_metric(None))
        self.assertIsNone(sanitize_metric(float("nan")))
        self.assertIsNone(sanitize_metric(float("inf")))
        self.assertIsNone(sanitize_metric("invalid_string"))

    @patch("data.cache.CacheManager.get_fundamentals", return_value=None)
    @patch("data.cache.CacheManager.set_fundamentals")
    def test_get_fundamentals_mocked(self, mock_set_cache, mock_get_cache):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.info = {
                "trailingPE": 18.2,
                "forwardPE": 16.0,
                "priceToBook": 2.5,
                "enterpriseToEbitda": 9.1
            }
            mock_ticker.return_value = mock_instance

            res = FundamentalProvider.get_fundamentals("2330.TW", use_cache=True)
            self.assertEqual(res["pe"], 18.2)
            self.assertEqual(res["pb"], 2.5)
            self.assertEqual(res["forward_pe"], 16.0)
            self.assertEqual(res["ev_ebitda"], 9.1)
            mock_set_cache.assert_called_once()

    def test_get_fundamentals_error_handling(self):
        with patch("data.cache.CacheManager.get_fundamentals", return_value=None):
            with patch("yfinance.Ticker", side_effect=Exception("API Error")):
                res = FundamentalProvider.get_fundamentals("INVALID.TICKER")
                self.assertIsNone(res["pe"])
                self.assertIsNone(res["pb"])


if __name__ == "__main__":
    unittest.main()
