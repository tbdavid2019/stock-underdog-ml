"""
test/test_market_calendar.py - 跨市場交易行事曆與休市守衛單元測試
"""

import unittest
import datetime
from fastapi.testclient import TestClient
from core.market_calendar import MarketCalendar
from api.main import app


class TestMarketCalendar(unittest.TestCase):

    def test_ticker_and_market_resolution(self):
        """測試台股與美股代號及市場字串自動解析"""
        self.assertEqual(MarketCalendar.resolve_market("2330.TW"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("2317.tw"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("6488.TWO"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("8069.two"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("2330"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("0050"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("00929"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("006208"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("TW"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("twse"), "TW")
        self.assertEqual(MarketCalendar.resolve_market("TPEX"), "TW")

        # 美股
        self.assertEqual(MarketCalendar.resolve_market("AAPL"), "US")
        self.assertEqual(MarketCalendar.resolve_market("CAT"), "US")
        self.assertEqual(MarketCalendar.resolve_market("MSFT"), "US")
        self.assertEqual(MarketCalendar.resolve_market("NVDA"), "US")
        self.assertEqual(MarketCalendar.resolve_market("SPY"), "US")
        self.assertEqual(MarketCalendar.resolve_market("US"), "US")

        # 預設
        self.assertEqual(MarketCalendar.resolve_market(None), "TW")
        self.assertEqual(MarketCalendar.resolve_market(""), "TW")

    def test_weekend_detection(self):
        """測試台美股週末休市偵測"""
        # 2026-10-31 為週六，2026-11-01 為週日
        self.assertFalse(MarketCalendar.is_trading_day("2026-10-31", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-11-01", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-10-31", "US"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-11-01", "US"))

        st_sat = MarketCalendar.get_trading_day_status("2026-10-31", "TW")
        self.assertFalse(st_sat["is_trading_day"])
        self.assertIn("週六", st_sat["reason"])

        st_sun = MarketCalendar.get_trading_day_status("2026-11-01", "US")
        self.assertFalse(st_sun["is_trading_day"])
        self.assertIn("週日", st_sun["reason"])

        # 2026-11-02 為週一（非假日）
        self.assertTrue(MarketCalendar.is_trading_day("2026-11-02", "TW"))
        self.assertTrue(MarketCalendar.is_trading_day("2026-11-02", "US"))

    def test_twse_holidays(self):
        """測試台股重要休市日判定 (元旦、春節結算交割、清明、端午、中秋、國慶等)"""
        # 1. 中華民國開國紀念日
        self.assertFalse(MarketCalendar.is_trading_day("2026-01-01", "TW"))
        self.assertTrue(MarketCalendar.is_trading_day("2026-01-02", "TW"))

        # 2. 農曆春節前最後交易日 (2026-02-11 為開市交易)
        self.assertTrue(MarketCalendar.is_trading_day("2026-02-11", "TW"))

        # 3. 農曆春節前市場無交易僅辦理結算交割 (2026-02-12, 2026-02-13)
        self.assertFalse(MarketCalendar.is_trading_day("2026-02-12", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-02-13", "TW"))
        st_cny_settle = MarketCalendar.get_trading_day_status("2026-02-12", "TW")
        self.assertIn("結算交割", st_cny_settle["reason"])

        # 4. 春節連假與補假 (2026-02-16 ~ 2026-02-20)
        for d in ["2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20"]:
            self.assertFalse(MarketCalendar.is_trading_day(d, "TW"))

        # 5. 春節後開市日 (2026-02-23)
        self.assertTrue(MarketCalendar.is_trading_day("2026-02-23", "TW"))

        # 6. 和平紀念日補假 (2026-02-27 週五)
        self.assertFalse(MarketCalendar.is_trading_day("2026-02-27", "TW"))

        # 7. 兒童節與清明掃墓補假 (2026-04-03 週五, 2026-04-06 週一)
        self.assertFalse(MarketCalendar.is_trading_day("2026-04-03", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-04-06", "TW"))

        # 8. 勞動節 (2026-05-01)
        self.assertFalse(MarketCalendar.is_trading_day("2026-05-01", "TW"))

        # 9. 端午節 (2026-06-19)
        self.assertFalse(MarketCalendar.is_trading_day("2026-06-19", "TW"))

        # 10. 中秋節與教師節 (2026-09-25, 2026-09-28)
        self.assertFalse(MarketCalendar.is_trading_day("2026-09-25", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-09-28", "TW"))

        # 11. 國慶日補假 (2026-10-09 週五)
        self.assertFalse(MarketCalendar.is_trading_day("2026-10-09", "TW"))

        # 12. 臺灣光復紀念補假 (2026-10-26) 與行憲紀念日 (2025-12-25, 2026-12-25)
        self.assertFalse(MarketCalendar.is_trading_day("2025-12-25", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-10-26", "TW"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-12-25", "TW"))

    def test_us_holidays(self):
        """測試美股 10 大法定假日與復活節好禮拜五及補假規則"""
        # 1. New Year's Day (2026-01-01)
        self.assertFalse(MarketCalendar.is_trading_day("2026-01-01", "US"))

        # 2. Martin Luther King Jr. Day (2026-01-19)
        self.assertFalse(MarketCalendar.is_trading_day("2026-01-19", "US"))

        # 3. Washington's Birthday (2026-02-16)
        self.assertFalse(MarketCalendar.is_trading_day("2026-02-16", "US"))

        # 4. Good Friday (Meeus/Butcher 復活節算法：2024-03-29, 2025-04-18, 2026-04-03)
        self.assertFalse(MarketCalendar.is_trading_day("2024-03-29", "US"))
        self.assertFalse(MarketCalendar.is_trading_day("2025-04-18", "US"))
        self.assertFalse(MarketCalendar.is_trading_day("2026-04-03", "US"))
        st_gf = MarketCalendar.get_trading_day_status("2026-04-03", "US")
        self.assertEqual(st_gf["reason"], "Good Friday")

        # 5. Memorial Day (2026-05-25)
        self.assertFalse(MarketCalendar.is_trading_day("2026-05-25", "US"))

        # 6. Juneteenth (2026-06-19)
        self.assertFalse(MarketCalendar.is_trading_day("2026-06-19", "US"))

        # 7. Independence Day (2026-07-04 適逢週六，提前於 2026-07-03 週五休市)
        self.assertFalse(MarketCalendar.is_trading_day("2026-07-03", "US"))

        # 8. Labor Day (2026-09-07)
        self.assertFalse(MarketCalendar.is_trading_day("2026-09-07", "US"))

        # 9. Thanksgiving Day (2026-11-26)
        self.assertFalse(MarketCalendar.is_trading_day("2026-11-26", "US"))

        # 10. Christmas Day (2026-12-25)
        self.assertFalse(MarketCalendar.is_trading_day("2026-12-25", "US"))

    def test_t_plus_and_t_minus_calculations(self):
        """測試跨週末與跨長假前後交易日 (T+1 / T-1) 計算"""
        # 一般週末跳轉 (週五 10-30 -> 週一 11-02)
        next_d = MarketCalendar.get_next_trading_day("2026-10-30", "TW")
        self.assertEqual(next_d.strftime("%Y-%m-%d"), "2026-11-02")
        prev_d = MarketCalendar.get_previous_trading_day("2026-11-02", "TW")
        self.assertEqual(prev_d.strftime("%Y-%m-%d"), "2026-10-30")

        # 農曆春節前最後交易日跳轉至開市日 (2026-02-11 -> 2026-02-23)
        cny_next = MarketCalendar.get_next_trading_day("2026-02-11", "TW")
        self.assertEqual(cny_next.strftime("%Y-%m-%d"), "2026-02-23")
        cny_prev = MarketCalendar.get_previous_trading_day("2026-02-23", "TW")
        self.assertEqual(cny_prev.strftime("%Y-%m-%d"), "2026-02-11")

        # 中秋與教師節四連假跳轉 (2026-09-24 週四 -> 2026-09-29 週二)
        # 9/25(五)中秋, 9/26(六), 9/27(日), 9/28(一)教師節
        mid_autumn_next = MarketCalendar.get_next_trading_day("2026-09-24", "TW")
        self.assertEqual(mid_autumn_next.strftime("%Y-%m-%d"), "2026-09-29")
        mid_autumn_prev = MarketCalendar.get_previous_trading_day("2026-09-29", "TW")
        self.assertEqual(mid_autumn_prev.strftime("%Y-%m-%d"), "2026-09-24")

    def test_get_recent_trading_days(self):
        """測試最近開盤交易日推算 (跳過休市與 15:30 截止門檻)"""
        # 測試在中秋/教師節連續休市期間 (2026-09-29 16:00 開市後) 索取最近 3 個交易日
        # 應為 20260929, 20260924, 20260923 (跳過 9/28, 9/27, 9/26, 9/25)
        dt_post_holiday = datetime.datetime(2026, 9, 29, 16, 0, 0)
        days = MarketCalendar.get_recent_trading_days(count=3, reference_dt=dt_post_holiday, market_or_ticker="TW")
        self.assertEqual(days, ["20260929", "20260924", "20260923"])

        # 測試台股 15:30 門檻：在盤中 15:00 索取，起算基準往前推算一日
        dt_pre_close = datetime.datetime(2026, 9, 29, 15, 0, 0)
        days_pre = MarketCalendar.get_recent_trading_days(count=2, reference_dt=dt_pre_close, market_or_ticker="TW")
        self.assertEqual(days_pre, ["20260924", "20260923"])

    def test_get_holiday_list(self):
        """測試全年度休市清冊查詢"""
        tw_holidays = MarketCalendar.get_holiday_list(year=2026, market="TW")
        self.assertGreater(len(tw_holidays), 15)
        for h in tw_holidays:
            self.assertEqual(h["market"], "TW")
            self.assertTrue(h["date"].startswith("2026-"))
            self.assertTrue(len(h["name"]) > 0)

        us_holidays = MarketCalendar.get_holiday_list(year=2026, market="US")
        self.assertEqual(len(us_holidays), 10)
        for h in us_holidays:
            self.assertEqual(h["market"], "US")
            self.assertTrue(h["date"].startswith("2026-"))

    def test_fastapi_calendar_endpoints(self):
        """測試 FastAPI 交易日行事曆 API 端點與 Pydantic Schema"""
        client = TestClient(app)

        # 1. 查詢台股休市日狀態
        resp = client.get("/api/v1/market/calendar/status?market=TW&date=2026-02-12")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["is_trading_day"])
        self.assertEqual(data["market"], "TW")
        self.assertEqual(data["date"], "2026-02-12")
        self.assertEqual(data["previous_trading_day"], "2026-02-11")
        self.assertEqual(data["next_trading_day"], "2026-02-23")

        # 2. 透過個股代號查詢美股狀態 (AAPL on Good Friday)
        resp_us = client.get("/api/v1/market/calendar/status?ticker=AAPL&date=2026-04-03")
        self.assertEqual(resp_us.status_code, 200)
        data_us = resp_us.json()
        self.assertTrue(data_us["success"])
        self.assertFalse(data_us["is_trading_day"])
        self.assertEqual(data_us["market"], "US")
        self.assertEqual(data_us["ticker"], "AAPL")
        self.assertEqual(data_us["reason"], "Good Friday")

        # 3. 查詢全年度休市清冊
        resp_h = client.get("/api/v1/market/calendar/holidays?market=US&year=2026")
        self.assertEqual(resp_h.status_code, 200)
        h_list = resp_h.json()
        self.assertEqual(len(h_list), 10)
        self.assertEqual(h_list[0]["name"], "New Year's Day")


if __name__ == "__main__":
    unittest.main()
