"""
test/test_mops_integration.py - Unit & integration tests for Taiwan MOPS material events and monthly revenue.
"""
import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from data.duckdb_manager import DuckDBManager
from data.mops_fetcher import (
    MOPSFetcher,
    classify_material_subject,
    parse_roc_date,
    parse_roc_time,
    parse_roc_ym,
)
from evaluators.composite_evaluator import CompositeEvaluator
from strategies.base import StrategyResult


class TestMOPSIntegration(unittest.TestCase):
    def test_roc_parsers(self):
        from data.mops_fetcher import _build_available_at

        self.assertEqual(parse_roc_date("1151119"), "2026-11-19")
        self.assertEqual(parse_roc_date("115/11/20"), "2026-11-20")
        self.assertEqual(parse_roc_date("115/1/5"), "2026-01-05")
        self.assertEqual(parse_roc_date("1150230"), "")
        self.assertEqual(parse_roc_time("64821"), "06:48:21")
        self.assertEqual(parse_roc_time("185232"), "18:52:32")
        self.assertEqual(parse_roc_time("18:5:32"), "18:05:32")
        self.assertEqual(parse_roc_time("99:99:99"), "")
        self.assertEqual(parse_roc_ym("11510"), "2026-10")
        self.assertEqual(parse_roc_ym("115/1"), "2026-01")
        self.assertEqual(parse_roc_ym("115/13"), "")
        self.assertEqual(_build_available_at("", ""), "")
        self.assertEqual(_build_available_at("2026-11-19", ""), "2026-11-19T23:59:59+08:00")

    def test_classify_material_subject(self):
        st_pos, kws_pos = classify_material_subject("公告本公司董事會決議買回公司股份(庫藏股)")
        self.assertEqual(st_pos, "positive")
        self.assertIn("買回公司股份", kws_pos)

        st_risk, kws_risk = classify_material_subject("公告本公司辦理現金增資發行新股")
        self.assertEqual(st_risk, "risk")
        self.assertIn("現金增資", kws_risk)

        st_neu, kws_neu = classify_material_subject("公告本公司召開法人說明會")
        self.assertEqual(st_neu, "neutral")
        self.assertEqual(kws_neu, [])

    def test_duckdb_persistence_and_factor_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DuckDBManager(db_path=str(Path(tmp) / "mops_test.duckdb"))
            ev_twse = MOPSFetcher._parse_material_row({
                "出表日期": "1151120",
                "發言日期": "1151119",
                "發言時間": "64821",
                "公司代號": "2330",
                "公司名稱": "台積電",
                "主旨 ": "公告本公司取得機器設備及廠務設備資本支出",
                "符合條款": "第20款",
                "事實發生日": "1151119",
                "說明": "擴大先進封裝產能",
            }, "TW", "TWSE")
            self.assertIsNotNone(ev_twse)
            self.assertEqual(ev_twse["ticker"], "2330.TW")
            self.assertEqual(ev_twse["sentiment_tag"], "positive")
            self.assertEqual(ev_twse["available_at"], "2026-11-19T06:48:21+08:00")

            rev_twse = MOPSFetcher._parse_revenue_row({
                "出表日期": "1151110",
                "資料年月": "11510",
                "公司代號": "2330",
                "公司名稱": "台積電",
                "產業別": "半導體業",
                "營業收入-當月營收": "314240000",
                "營業收入-上月營收": "290000000",
                "營業收入-去年當月營收": "240000000",
                "營業收入-上月比較增減(%)": "8.36",
                "營業收入-去年同月增減(%)": "30.93",
                "累計營業收入-當月累計營收": "2500000000",
                "累計營業收入-去年累計營收": "1900000000",
                "累計營業收入-前期比較增減(%)": "31.58",
                "備註": "-",
            }, "TW", "TWSE")
            self.assertIsNotNone(rev_twse)
            self.assertEqual(rev_twse["revenue_ym"], "2026-10")
            self.assertEqual(rev_twse["revenue_tag"], "營收高成長")

            # Use today's date so get_mops_factor_map window matches regardless of wall clock
            ev_twse["speech_date"] = datetime.date.today().isoformat()
            self.assertEqual(db.save_mops_events_batch([ev_twse]), 1)
            self.assertEqual(db.save_monthly_revenue_batch([rev_twse]), 1)

            events = db.get_mops_events("2330.TW", limit=10)
            self.assertEqual(len(events), 1)
            self.assertIn("資本支出", events[0]["matched_keywords"])

            revs = db.get_monthly_revenue("2330", limit=5)
            self.assertEqual(len(revs), 1)
            self.assertAlmostEqual(revs[0]["yoy_pct"], 30.93, places=2)

            fmap = db.get_mops_factor_map(["2330.TW", "NVDA"])
            self.assertIn("2330.TW", fmap)
            self.assertNotIn("NVDA", fmap)
            self.assertGreater(fmap["2330.TW"]["score_delta"], 0.0)
            self.assertIn("營收YoY+31%", fmap["2330.TW"]["tags"])
            self.assertIn("擴產CapEx", fmap["2330.TW"]["tags"])

    def test_composite_evaluator_applies_mops_map(self):
        ev = CompositeEvaluator()
        res = StrategyResult(
            ticker="2330.TW",
            strategy_name="玄鐵重劍",
            is_hit=True,
            score=80.0,
            current_price=1000.0,
            tags=["玄鐵買點"],
        )
        rep_base = ev.evaluate("台灣50", {"xuantie": [res]})
        rep_mops = ev.evaluate(
            "台灣50",
            {"xuantie": [res]},
            mops_map={"2330.TW": {"score_delta": 10.0, "tags": ["營收YoY+31%", "擴產CapEx"]}},
        )
        base_score = rep_base.ranked_stocks[0]["composite_score"]
        mops_score = rep_mops.ranked_stocks[0]["composite_score"]
        self.assertGreater(mops_score, base_score)
        self.assertIn("擴產CapEx", rep_mops.ranked_stocks[0]["tags"])

    def test_api_endpoints_and_drawer_ui(self):
        from api.main import app
        from api.routes.market import get_duckdb

        with tempfile.TemporaryDirectory() as tmp:
            db = DuckDBManager(db_path=str(Path(tmp) / "mops_api.duckdb"))
            ev = MOPSFetcher._parse_material_row({
                "發言日期": "1151120",
                "發言時間": "091500",
                "公司代號": "2330",
                "公司名稱": "台積電",
                "主旨 ": "公告買回公司股份",
                "說明": "庫藏股買回",
            }, "TW", "TWSE")
            rev = MOPSFetcher._parse_revenue_row({
                "出表日期": "1151110",
                "資料年月": "11510",
                "公司代號": "2330",
                "公司名稱": "台積電",
                "產業別": "半導體業",
                "營業收入-當月營收": "300000000",
                "營業收入-去年同月增減(%)": "25.5",
                "營業收入-上月比較增減(%)": "5.2",
                "累計營業收入-前期比較增減(%)": "28.0",
            }, "TW", "TWSE")
            db.save_mops_events_batch([ev])
            db.save_monthly_revenue_batch([rev])

            app.dependency_overrides[get_duckdb] = lambda: db
            try:
                client = TestClient(app)
                r_hi = client.get("/api/v1/market/mops/highlights?limit=5")
                self.assertEqual(r_hi.status_code, 200)
                self.assertEqual(len(r_hi.json()["catalysts"]), 1)
                self.assertEqual(len(r_hi.json()["top_revenue_growth"]), 1)

                r_tw = client.get("/api/v1/market/mops/2330.TW")
                self.assertEqual(r_tw.status_code, 200)
                self.assertTrue(r_tw.json()["applicable"])
                self.assertEqual(r_tw.json()["latest_revenue"]["revenue_ym"], "2026-10")

                r_us = client.get("/api/v1/market/mops/NVDA")
                self.assertEqual(r_us.status_code, 200)
                self.assertFalse(r_us.json()["applicable"])

                r_ui = client.get("/")
                self.assertEqual(r_ui.status_code, 200)
                self.assertIn("drawerActiveSubTab === 'mops'", r_ui.text)
                self.assertIn("historyDetailSubTab === 'mops'", r_ui.text)
                self.assertIn("ensureMopsIntelLoaded", r_ui.text)

                r_get_sync = client.get("/api/v1/market/mops/sync")
                self.assertEqual(r_get_sync.status_code, 405)

                with patch("data.mops_fetcher.MOPSFetcher.sync_to_duckdb", return_value={"events": 0, "revenues": 0}):
                    r_post_sync = client.post("/api/v1/market/mops/sync")
                    self.assertEqual(r_post_sync.status_code, 200)
            finally:
                app.dependency_overrides.pop(get_duckdb, None)

    def test_nan_revenue_highlights_pit_cutoff_and_macro_exposure_zero(self):
        from data.macro import MacroState
        from data.mops_fetcher import TZ_TAIPEI
        from api.main import app
        from api.routes.market import get_duckdb

        with tempfile.TemporaryDirectory() as tmp:
            db = DuckDBManager(db_path=str(Path(tmp) / "mops_pit.duckdb"))
            rev_valid = MOPSFetcher._parse_revenue_row({
                "出表日期": "1151110",
                "資料年月": "11510",
                "公司代號": "2330",
                "公司名稱": "台積電",
                "產業別": "半導體業",
                "營業收入-當月營收": "300000",
                "營業收入-去年同月增減(%)": "25.0",
                "營業收入-上月比較增減(%)": "5.0",
                "累計營業收入-前期比較增減(%)": "20.0",
            }, "TW", "TWSE")
            rev_nan = MOPSFetcher._parse_revenue_row({
                "出表日期": "1151110",
                "資料年月": "11510",
                "公司代號": "2317",
                "公司名稱": "鴻海",
                "產業別": "其他電子",
                "營業收入-當月營收": "400000",
                "營業收入-去年同月增減(%)": "-",
                "營業收入-上月比較增減(%)": "-",
                "累計營業收入-前期比較增減(%)": "-",
            }, "TW", "TWSE")
            ev_morning = MOPSFetcher._parse_material_row({
                "發言日期": "1151119",
                "發言時間": "083000",
                "公司代號": "2330",
                "公司名稱": "台積電",
                "主旨 ": "公告庫藏股買回",
            }, "TW", "TWSE")
            ev_evening = MOPSFetcher._parse_material_row({
                "發言日期": "1151119",
                "發言時間": "203000",
                "公司代號": "2317",
                "公司名稱": "鴻海",
                "主旨 ": "公告辦理現金增資",
            }, "TW", "TWSE")
            self.assertEqual(db.save_monthly_revenue_batch([rev_valid, rev_nan]), 2)
            self.assertEqual(db.save_mops_events_batch([ev_morning, ev_evening]), 2)

            cutoff = datetime.datetime(2026, 11, 19, 12, 0, 0, tzinfo=TZ_TAIPEI)
            fmap = db.get_mops_factor_map(["2330.TW", "2317.TW"], days=14, as_of=cutoff)
            self.assertIn("2330.TW", fmap)
            self.assertIn("庫藏股催化", fmap["2330.TW"]["tags"])
            self.assertNotIn("重訊稀釋/風險警示", (fmap.get("2317.TW") or {}).get("tags", []))

            app.dependency_overrides[get_duckdb] = lambda: db
            try:
                client = TestClient(app)
                r_hi = client.get("/api/v1/market/mops/highlights?limit=5")
                self.assertEqual(r_hi.status_code, 200)
                tickers = [x["ticker"] for x in r_hi.json()["top_revenue_growth"]]
                self.assertEqual(tickers, ["2330.TW"])
            finally:
                app.dependency_overrides.pop(get_duckdb, None)

        ev = CompositeEvaluator()
        res = StrategyResult(ticker="2330.TW", strategy_name="玄鐵重劍", is_hit=True, score=80.0, current_price=1000.0)
        panic_macro = MacroState(regime_name="極度恐慌", exposure=0.0)
        rep = ev.evaluate(
            "台灣50",
            {"xuantie": [res]},
            macro_state=panic_macro,
            mops_map={"2330.TW": {"score_delta": 10.0, "tags": ["庫藏股催化"]}},
        )
        self.assertEqual(rep.ranked_stocks[0]["composite_score"], 0.0)

    def test_mops_intraday_filtering_and_per_feed_fallback(self):
        fake_t05st02 = MagicMock(status_code=200)
        fake_t05st02.json.return_value = {
            "result": {
                "data": [
                    ["115/11/20", "10:15:00", "2330", "台積電", "公告買回庫藏股", {"parameters": {"marketKind": "sii"}}],
                    ["115/11/20", "10:20:00", "6488", "環球晶", "公告資本支出", {"parameters": {"marketKind": "otc"}}],
                    ["115/11/20", "10:25:00", "9999", "興櫃測試", "興櫃公告", {"parameters": {"marketKind": "rotc"}}],
                ]
            }
        }
        with patch("data.mops_fetcher.requests.post", return_value=fake_t05st02):
            rows = MOPSFetcher.fetch_intraday_mops_t05st02()
        self.assertEqual([r["ticker"] for r in rows], ["2330.TW", "6488.TWO"])

        with tempfile.TemporaryDirectory() as tmp:
            twse_row = [{
                "發言日期": "1151119", "發言時間": "083000",
                "公司代號": "2330", "公司名稱": "台積電", "主旨 ": "公告庫藏股"
            }]
            tpex_row = [{
                "Date": "1151119", "發言時間": "093000",
                "公司代號": "6488", "公司名稱": "環球晶", "主旨": "公告擴建廠房"
            }]

            def fake_get_ok(url, **kwargs):
                if "twse.com.tw" in url:
                    r = MagicMock(status_code=200)
                    r.json.return_value = twse_row
                    return r
                r = MagicMock(status_code=200)
                r.json.return_value = tpex_row
                return r

            with patch("data.mops_fetcher.MOPS_CACHE_DIR", tmp), \
                 patch("data.mops_fetcher.requests.get", side_effect=fake_get_ok):
                ok_events = MOPSFetcher.fetch_material_events(include_intraday_mops=False)
            self.assertEqual({e["ticker"] for e in ok_events}, {"2330.TW", "6488.TWO"})

            def fake_get_twse_down(url, **kwargs):
                if "twse.com.tw" in url:
                    return MagicMock(status_code=503)
                r = MagicMock(status_code=200)
                r.json.return_value = tpex_row
                return r

            with patch("data.mops_fetcher.MOPS_CACHE_DIR", tmp), \
                 patch("data.mops_fetcher.requests.get", side_effect=fake_get_twse_down):
                fallback_events = MOPSFetcher.fetch_material_events(include_intraday_mops=False)
            self.assertEqual({e["ticker"] for e in fallback_events}, {"2330.TW", "6488.TWO"})

    def test_canary_mops_probe(self):
        from data.loader_canary import PROBE_REGISTRY
        self.assertIn("mops", PROBE_REGISTRY)
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = [{
            "公司代號": "2330",
            "資料年月": "11510",
            "營業收入-去年同月增減(%)": "30.0",
        }]
        with patch("data.loader_canary.requests.get", return_value=mock_resp):
            code, missing, err = PROBE_REGISTRY["mops"](5.0)
            self.assertEqual(code, 200)
            self.assertEqual(missing, [])
            self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
