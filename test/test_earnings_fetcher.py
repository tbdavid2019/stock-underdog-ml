import unittest
from unittest.mock import MagicMock, patch
import json
import datetime

from data.earnings_fetcher import (
    EarningsCallFetcher,
    normalize_tw_ticker,
    is_tw_weight_stock,
    is_us_weight_stock,
    is_weight_stock,
    parse_biggo_stock_metadata,
    parse_biggo_guidance,
    extract_financial_period,
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

    def test_parse_biggo_stock_metadata_jp_and_kr(self):
        # 日本股市
        jp_item1 = {"stock_id": "7203.T", "region": "jp"}
        meta_jp1 = parse_biggo_stock_metadata(jp_item1)
        self.assertEqual(meta_jp1["market"], "JP")
        self.assertEqual(meta_jp1["symbol"], "7203.T")
        self.assertEqual(meta_jp1["timezone"], "Asia/Tokyo")

        jp_item2 = {"stock_id": "9984", "region": "jp"}
        meta_jp2 = parse_biggo_stock_metadata(jp_item2)
        self.assertEqual(meta_jp2["market"], "JP")
        self.assertEqual(meta_jp2["symbol"], "9984.T")

        # 韓國股市
        kr_item = {"stock_id": "005930.KS", "region": "kr"}
        meta_kr = parse_biggo_stock_metadata(kr_item)
        self.assertEqual(meta_kr["market"], "KR")
        self.assertEqual(meta_kr["symbol"], "005930.KS")
        self.assertEqual(meta_kr["timezone"], "Asia/Seoul")

    def test_extract_financial_period(self):
        self.assertEqual(extract_financial_period("台積電 2026年第三季法說會", "2026-10-14"), "FY2026 Q3")
        self.assertEqual(extract_financial_period("NVIDIA Q3 FY2027 Earnings Call", "2026-11-18"), "FY2027 Q3")
        self.assertEqual(extract_financial_period("聯發科 法人說明會", "2026-10-25"), "2026 Q4")

    def test_parse_biggo_guidance(self):
        detail = {
            "outlook": "營收：446億至458億美元\n毛利率：65%至67%\n資本支出：600至640億美元",
            "summary": "台積電公佈強勁財報，各項財務數據符合預期。",
            "qa_highlights": "Q: N2 量產進度？\nA: 按計畫於 2026 年底量產。"
        }
        g = parse_biggo_guidance(detail)
        self.assertIn("446億", g["outlook"])
        self.assertIn("65%", g["margin"])
        self.assertIn("600至640億", g["capex"])
        self.assertIn("強勁財報", g["summary"])
        self.assertIn("N2", g["qa"])

    def test_duckdb_earnings_calls_persistence(self):
        from data.duckdb_manager import DuckDBManager
        db = DuckDBManager()
        test_records = [
            {
                "id": "TEST_JP_7203_20261014",
                "market": "JP",
                "ticker": "7203.T",
                "name": "Toyota Motor",
                "call_date": "2026-10-14",
                "call_time": "15:00",
                "timezone": "Asia/Tokyo",
                "meeting_type": "online",
                "location": "Online",
                "webcast_url": "https://toyota.jp",
                "presentation_url": "https://toyota.jp/slides.pdf",
                "transcript_url": "",
                "language": "ja",
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": True,
                "remarks": "FY2027 Q2 Earnings Call",
                "financial_period": "FY2027 Q2",
                "guidance_summary": "Toyota reports steady Q2",
                "guidance_outlook": "Revenue +8%",
                "guidance_margin": "Operating margin 10%",
                "guidance_capex": "CapEx 1.2T JPY",
                "guidance_qa": "EV investments update",
                "source": "BigGo / JP IR"
            },
            {
                "id": "TEST_KR_005930_20261014",
                "market": "KR",
                "ticker": "005930.KS",
                "name": "Samsung Electronics",
                "call_date": "2026-10-14",
                "call_time": "14:00",
                "timezone": "Asia/Seoul",
                "meeting_type": "online",
                "location": "Online",
                "webcast_url": "https://samsung.com",
                "presentation_url": "https://samsung.com/deck.pdf",
                "transcript_url": "",
                "language": "ko",
                "eps_estimate": None,
                "eps_actual": None,
                "revenue_estimate": None,
                "revenue_actual": None,
                "is_weight_stock": True,
                "remarks": "Samsung 2026 Q3 Earnings Call",
                "financial_period": "2026 Q3",
                "guidance_summary": "HBM3E memory ramping up",
                "guidance_outlook": "Memory shipments up 15%",
                "guidance_margin": "Gross margin 42%",
                "guidance_capex": "CapEx 50T KRW",
                "guidance_qa": "HBM roadmap Q&A",
                "source": "BigGo / KR IR"
            }
        ]

        # 寫入
        saved = db.save_earnings_calls_batch(test_records)
        self.assertEqual(saved, 2)

        # 查詢 JP
        jp_calls = db.get_earnings_calls(market="JP", start_date="2026-10-14", end_date="2026-10-14")
        self.assertTrue(any(c["ticker"] == "7203.T" for c in jp_calls))

        # 查詢 KR
        kr_calls = db.get_earnings_calls(market="KR", start_date="2026-10-14", end_date="2026-10-14")
        self.assertTrue(any(c["ticker"] == "005930.KS" for c in kr_calls))

        # 查詢 Calendar (結構化輸出測試)
        cal = db.get_earnings_calendar(year=2026, month=10, market="ALL")
        self.assertTrue(len(cal) > 0)
        day_14 = next((d for d in cal if d["call_date"] == "2026-10-14"), None)
        self.assertEqual(day_14["weekday"], "星期三")
        self.assertGreaterEqual(day_14["total_calls"], 2)
        self.assertIn("calls", day_14)
        self.assertGreaterEqual(len(day_14["calls"]), 2)
        self.assertTrue(any(c["market"] in ("JP", "KR") for c in day_14["calls"]))

    def test_upcoming_endpoint_time_filtering(self):
        from fastapi.testclient import TestClient
        from api.main import app

        client = TestClient(app)

        # 1. 查詢 ALL
        resp = client.get("/api/v1/earnings-call/upcoming?market=ALL&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("today_count", data)
        self.assertIn("upcoming_count", data)
        self.assertIn("past_count", data)

        # 2. 查詢 PAST
        resp_past = client.get("/api/v1/earnings-call/upcoming?market=ALL&time_filter=PAST&limit=10")
        self.assertEqual(resp_past.status_code, 200)
        data_past = resp_past.json()
        self.assertTrue(data_past["success"])

        # 3. 查詢 Calendar API
        resp_cal = client.get("/api/v1/earnings-call/calendar?year=2026&month=10&market=ALL")
        self.assertEqual(resp_cal.status_code, 200)
        data_cal = resp_cal.json()
        self.assertTrue(data_cal["success"])
        if data_cal["data"]:
            self.assertIn("calls", data_cal["data"][0])

        # 4. 查詢指定明確時間區間 (Explicit bounded date range)
        resp_bounded = client.get("/api/v1/earnings-call/upcoming?start_date=2026-10-01&end_date=2026-10-14&market=ALL")
        self.assertEqual(resp_bounded.status_code, 200)
        data_bounded = resp_bounded.json()
        self.assertTrue(data_bounded["success"])
        for item in data_bounded["data"]:
            self.assertGreaterEqual(item["call_date"], "2026-10-01")
            self.assertLessEqual(item["call_date"], "2026-10-14")

        # 5. 查詢今天 (TODAY) 驗證台積電 2330.TW 官方法說會 (指定 ref_date=2026-10-15 避免隨執行日期漂移)
        resp_today = client.get("/api/v1/earnings-call/upcoming?time_filter=TODAY&market=ALL&ref_date=2026-10-15")
        self.assertEqual(resp_today.status_code, 200)
        data_today = resp_today.json()
        self.assertTrue(data_today["success"])
        self.assertGreaterEqual(data_today["today_count"], 1)
        self.assertTrue(any(c["ticker"] == "2330.TW" for c in data_today["data"]))
        tsmc = next(c for c in data_today["data"] if c["ticker"] == "2330.TW")
        self.assertEqual(tsmc["call_date"], "2026-10-15")
        self.assertEqual(tsmc["call_time"], "14:00")
        self.assertTrue(tsmc["is_weight_stock"])
        self.assertIsNotNone(tsmc["webcast_url"])

    def test_verified_corporate_earnings_fallback(self):
        """測試當 BigGo 無回應或未收錄時，自動退回官方基準庫 (如台積電 2330.TW)"""
        from data.earnings_fetcher import EarningsCallFetcher, load_verified_corporate_earnings
        benchmarks = load_verified_corporate_earnings()
        self.assertGreaterEqual(len(benchmarks), 20)
        self.assertTrue(any(b["ticker"] == "2330.TW" for b in benchmarks))

        # 模擬 BigGo API 回傳空字典
        fetcher = EarningsCallFetcher(db=None)
        with unittest.mock.patch.object(fetcher, "_fetch_biggo_all_windows", return_value={}):
            results = fetcher.fetch_ticker_calls("2330.TW")
            self.assertGreaterEqual(len(results), 1)
            tsmc = next(r for r in results if r["ticker"] == "2330.TW")
            self.assertEqual(tsmc["call_date"], "2026-10-15")
            self.assertEqual(tsmc["call_time"], "14:00")
            self.assertIn("investor.tsmc.com", tsmc["webcast_url"])


if __name__ == "__main__":
    unittest.main()
