import unittest
from unittest.mock import MagicMock, patch
import json
import datetime

from data.earnings_fetcher import (
    EarningsCallFetcher,
    normalize_tw_ticker,
    is_tw_weight_stock,
    is_us_weight_stock,
    generate_date_windows,
    parse_roc_date
)


class TestEarningsCallFetcher(unittest.TestCase):
    @patch("data.earnings_fetcher.os.path.exists", return_value=False)
    @patch.object(EarningsCallFetcher, "fetch_biggo_calls", return_value=[])
    def test_missing_us_cache_returns_no_fabricated_schedule(self, _mock_biggo, _exists):
        fetcher = EarningsCallFetcher(db_manager=MagicMock())
        records = fetcher.fetch_us_calls()
        self.assertEqual(records, [])

    def test_normalize_tw_ticker(self):
        self.assertEqual(normalize_tw_ticker("2330"), "2330.TW")
        self.assertEqual(normalize_tw_ticker("2330.TW"), "2330.TW")
        self.assertEqual(normalize_tw_ticker("6126.TWO"), "6126.TWO")
        self.assertEqual(normalize_tw_ticker("AAPL"), "AAPL")
        self.assertEqual(normalize_tw_ticker(""), "")

    def test_weight_stock_detection(self):
        self.assertTrue(is_tw_weight_stock("2330.TW"))
        self.assertTrue(is_tw_weight_stock("2330"))
        self.assertTrue(is_tw_weight_stock("2603.TW"))
        self.assertFalse(is_tw_weight_stock("1319.TW"))

        self.assertTrue(is_us_weight_stock("NVDA"))
        self.assertTrue(is_us_weight_stock("AAPL"))
        self.assertFalse(is_us_weight_stock("UNKNOWN_PENNY_STOCK"))

    def test_generate_date_windows(self):
        base = datetime.date(2026, 10, 14)
        windows = generate_date_windows(as_of=base, past_windows=2, future_windows=2, window_days=6)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0], ("2026-10-02", "2026-10-08"))
        self.assertEqual(windows[1], ("2026-10-08", "2026-10-14"))
        self.assertEqual(windows[2], ("2026-10-14", "2026-10-20"))
        self.assertEqual(windows[3], ("2026-10-20", "2026-10-26"))

    def test_parse_roc_date(self):
        self.assertEqual(parse_roc_date("113/10/17"), "2024-10-17")
        self.assertEqual(parse_roc_date("1151014"), "2026-10-14")
        self.assertEqual(parse_roc_date("2026-10-14"), "2026-10-14")

    @patch("urllib.request.urlopen")
    def test_fetch_biggo_calls_tw_mock(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "result": True,
            "data": {
                "list": [
                    {
                        "call_id": "TW_2330.TW_2026-10-14",
                        "stock_id": "2330.TW",
                        "name": "台積電",
                        "title": "台積電 2026-10-14 法說會",
                        "date": "2026-10-14",
                        "has_video": True,
                        "has_slides": False,
                        "has_transcript": True
                    }
                ]
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        fetcher = EarningsCallFetcher(db_manager=MagicMock())
        records = fetcher.fetch_biggo_calls(market="tw", past_windows=0, future_windows=1, enrich_slides_limit=0)

        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["id"], "TW_2330.TW_2026-10-14")
        self.assertEqual(r["market"], "TW")
        self.assertEqual(r["ticker"], "2330.TW")
        self.assertEqual(r["name"], "台積電")
        self.assertEqual(r["call_date"], "2026-10-14")
        self.assertEqual(r["timezone"], "Asia/Taipei")
        self.assertEqual(r["meeting_type"], "online")
        self.assertTrue(r["is_weight_stock"])
        self.assertEqual(r["source"], "BigGo / MOPS")

    @patch("urllib.request.urlopen")
    def test_fetch_biggo_calls_us_mock(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "result": True,
            "data": {
                "list": [
                    {
                        "call_id": "US_NVDA_2026-11-18",
                        "stock_id": "NVDA",
                        "name": "NVIDIA",
                        "title": "NVIDIA Q3 FY2027 Earnings Call",
                        "date": "2026-11-18",
                        "has_video": False,
                        "has_slides": True,
                        "has_transcript": True
                    }
                ]
            }
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        fetcher = EarningsCallFetcher(db_manager=MagicMock())
        records = fetcher.fetch_biggo_calls(market="us", past_windows=0, future_windows=1, enrich_slides_limit=0)

        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["market"], "US")
        self.assertEqual(r["ticker"], "NVDA")
        self.assertEqual(r["timezone"], "America/New_York")
        self.assertTrue(r["is_weight_stock"])
        self.assertEqual(r["source"], "BigGo / US IR")


if __name__ == "__main__":
    unittest.main()
