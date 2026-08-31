import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from stock_cache import StockListCache


class TestStockListCache(unittest.TestCase):
    def write_cache(self, path: Path, timestamp: str, stocks):
        path.write_text(
            json.dumps(
                {
                    "TW0050": {
                        "stocks": stocks,
                        "timestamp": timestamp,
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_fetches_latest_data_even_when_cache_is_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "stock_lists.json"
            self.write_cache(cache_path, datetime.now().isoformat(), ["OLD.TW"])
            fetch_calls = []

            def fetch_latest():
                fetch_calls.append(True)
                return {"stocks": ["NEW.TW"], "name_map": {"NEW.TW": "New"}}

            with patch("stock_cache.CACHE_FILE", str(cache_path)):
                result = StockListCache().get("TW0050", fetch_latest)

            self.assertEqual(result, ["NEW.TW"])
            self.assertEqual(fetch_calls, [True])

    def test_uses_cache_only_when_fetch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "stock_lists.json"
            self.write_cache(cache_path, datetime.now().isoformat(), ["CACHED.TW"])

            def fetch_fails():
                raise RuntimeError("temporary API failure")

            with patch("stock_cache.CACHE_FILE", str(cache_path)):
                result = StockListCache().get("TW0050", fetch_fails)

            self.assertEqual(result, ["CACHED.TW"])


if __name__ == "__main__":
    unittest.main()
