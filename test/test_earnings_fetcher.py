import unittest
from unittest.mock import MagicMock, patch

from data.earnings_fetcher import EarningsCallFetcher


class TestEarningsCallFetcher(unittest.TestCase):
    @patch("data.earnings_fetcher.os.path.exists", return_value=False)
    def test_missing_us_cache_returns_no_fabricated_schedule(self, _exists):
        fetcher = EarningsCallFetcher(db_manager=MagicMock())

        records = fetcher.fetch_us_calls()

        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
