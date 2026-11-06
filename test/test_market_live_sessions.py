"""
test/test_market_live_sessions.py - 測試跨市場交易時段（含東京 11:30–12:30 午休）與交易日感知新鮮度
"""
import datetime
import unittest
import zoneinfo

from core.market_calendar import MarketCalendar


class TestMarketLiveSessions(unittest.TestCase):
    def test_tokyo_lunch_break_exact_session(self):
        """測試東京時間交易日 12:00 精確判斷為 lunch_break 且倒數至 12:30 後場開盤"""
        # 構造一個週三 (2026-11-04) 12:00:00 JST (03:00:00 UTC)
        tz_tokyo = zoneinfo.ZoneInfo("Asia/Tokyo")
        dt_tokyo = datetime.datetime(2026, 11, 4, 12, 0, 0, tzinfo=tz_tokyo)
        dt_utc = dt_tokyo.astimezone(datetime.timezone.utc)

        summary = MarketCalendar.get_live_sessions_summary(dt_utc)
        self.assertIn("JP", summary)
        jp_session = summary["JP"]
        self.assertEqual(jp_session["session"], "lunch_break")
        self.assertIn("午休", jp_session["session_name"])
        # 距離 12:30 開盤還有 30 分鐘 = 1800 秒
        self.assertAlmostEqual(jp_session["seconds_to_transition"], 1800, delta=5)

    def test_tokyo_regular_morning_and_afternoon(self):
        """測試日股前場 10:00 與後場 13:00 為 regular"""
        tz_tokyo = zoneinfo.ZoneInfo("Asia/Tokyo")
        # 10:00 JST 前場
        dt_10 = datetime.datetime(2026, 11, 4, 10, 0, 0, tzinfo=tz_tokyo).astimezone(datetime.timezone.utc)
        s_10 = MarketCalendar.get_live_sessions_summary(dt_10)["JP"]
        self.assertEqual(s_10["session"], "regular")

        # 13:00 JST 後場
        dt_13 = datetime.datetime(2026, 11, 4, 13, 0, 0, tzinfo=tz_tokyo).astimezone(datetime.timezone.utc)
        s_13 = MarketCalendar.get_live_sessions_summary(dt_13)["JP"]
        self.assertEqual(s_13["session"], "regular")

    def test_taiwan_pre_market_and_regular(self):
        """測試台股 08:45 為 pre_market，09:30 為 regular，14:00 為 after_hours"""
        tz_tw = zoneinfo.ZoneInfo("Asia/Taipei")
        # 08:45 TW 盤前
        dt_pre = datetime.datetime(2026, 11, 4, 8, 45, 0, tzinfo=tz_tw).astimezone(datetime.timezone.utc)
        s_pre = MarketCalendar.get_live_sessions_summary(dt_pre)["TW"]
        self.assertEqual(s_pre["session"], "pre_market")

        # 09:30 TW 盤中
        dt_reg = datetime.datetime(2026, 11, 4, 9, 30, 0, tzinfo=tz_tw).astimezone(datetime.timezone.utc)
        s_reg = MarketCalendar.get_live_sessions_summary(dt_reg)["TW"]
        self.assertEqual(s_reg["session"], "regular")

        # 14:00 TW 盤後定價
        dt_post = datetime.datetime(2026, 11, 4, 14, 0, 0, tzinfo=tz_tw).astimezone(datetime.timezone.utc)
        s_post = MarketCalendar.get_live_sessions_summary(dt_post)["TW"]
        self.assertEqual(s_post["session"], "after_hours")

    def test_us_dst_aware_sessions_and_early_close(self):
        """測試美股時段感知夏冬令時區與提前收盤日"""
        tz_ny = zoneinfo.ZoneInfo("America/New_York")
        # 2026-11-04 10:00 ET (美股冬令開盤中)
        dt_us = datetime.datetime(2026, 11, 4, 10, 0, 0, tzinfo=tz_ny).astimezone(datetime.timezone.utc)
        s_us = MarketCalendar.get_live_sessions_summary(dt_us)["US"]
        self.assertEqual(s_us["session"], "regular")

        # 感恩節隔日 (Black Friday) 提前 13:00 收盤
        # 2026 年感恩節為 11 月第 4 個週四 (11-26)，隔日為 11-27
        dt_bf = datetime.datetime(2026, 11, 27, 13, 15, 0, tzinfo=tz_ny).astimezone(datetime.timezone.utc)
        s_bf = MarketCalendar.get_live_sessions_summary(dt_bf)["US"]
        self.assertEqual(s_bf["session"], "after_hours")

    def test_weekend_and_holiday_closed(self):
        """測試週末與國定假日顯示 closed 且提供休市原因"""
        # 週日
        dt_sun = datetime.datetime(2026, 11, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
        s_sun = MarketCalendar.get_live_sessions_summary(dt_sun)
        self.assertEqual(s_sun["TW"]["session"], "closed")
        self.assertEqual(s_sun["US"]["session"], "closed")
        self.assertEqual(s_sun["JP"]["session"], "closed")

    def test_trading_day_aware_freshness(self):
        """測試連假或週末期間不誤報 stale，對照最近已收盤交易日判定"""
        tz_tw = zoneinfo.ZoneInfo("Asia/Taipei")
        # 假設當前為農曆春節期間的週日 (2025-01-26)，前一封關交易日為 2025-01-21
        dt_cny = datetime.datetime(2025, 1, 26, 10, 0, 0, tzinfo=tz_tw)
        freshness = MarketCalendar.check_market_freshness(
            market="TW",
            last_data_date="2025-01-21",
            now_dt=dt_cny
        )
        self.assertFalse(freshness["stale"])
        self.assertEqual(freshness["expected_date"], "2025-01-21")

        # 若資料日期比預期交易日更舊，則標記為 stale
        stale_check = MarketCalendar.check_market_freshness(
            market="TW",
            last_data_date="2025-01-10",
            now_dt=dt_cny
        )
        self.assertTrue(stale_check["stale"])


if __name__ == "__main__":
    unittest.main()
