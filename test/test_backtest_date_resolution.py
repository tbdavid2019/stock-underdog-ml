"""
test/test_backtest_date_resolution.py - 測試回測模組午夜邊界與盤前/盤後目標交易日判定
"""

import unittest
import datetime
import zoneinfo
import pandas as pd
from backtest.resolver import (
    parse_prediction_timestamp,
    get_target_trading_date,
    is_market_closed,
    find_actual_close_price
)


class TestBacktestDateResolution(unittest.TestCase):
    def setUp(self):
        self.tz_tw = zoneinfo.ZoneInfo("Asia/Taipei")
        self.tz_ny = zoneinfo.ZoneInfo("America/New_York")

    def test_parse_prediction_timestamp_utc(self):
        ts_str = "2026-09-10T00:00:00Z"
        dt = parse_prediction_timestamp(ts_str)
        self.assertIsNotNone(dt.tzinfo)
        dt_tw = dt.astimezone(self.tz_tw)
        self.assertEqual(dt_tw.hour, 8)

    def test_parse_prediction_timestamp_naive(self):
        ts_str = "2026-09-10T08:00:00"
        dt = parse_prediction_timestamp(ts_str)
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.hour, 8)

    def test_tw_premarket_targets_today(self):
        # 1. 台股 08:00 盤前預測，目標日應為當日 (2026-09-10)
        dt = datetime.datetime(2026, 9, 10, 8, 0, 0, tzinfo=self.tz_tw)
        target = get_target_trading_date(dt, "2330.TW")
        self.assertEqual(target, "2026-09-10")

    def test_tw_postmarket_targets_next_day(self):
        # 2. 台股 14:00 盤後預測，目標日應為隔日 (2026-09-11)
        dt = datetime.datetime(2026, 9, 10, 14, 0, 0, tzinfo=self.tz_tw)
        target = get_target_trading_date(dt, "2330.TW")
        self.assertEqual(target, "2026-09-11")

    def test_us_premarket_taipei_evening_targets_today(self):
        # 3. 美股 20:30 台北時間盤前預測 (紐約 08:30 EDT)，目標日應為當日美股交易日 (2026-09-10)
        dt = datetime.datetime(2026, 9, 10, 20, 30, 0, tzinfo=self.tz_tw)
        target = get_target_trading_date(dt, "AAPL")
        self.assertEqual(target, "2026-09-10")

    def test_us_midnight_boundary_in_taipei(self):
        # 4. 台北時間午夜跨日 01:00 (2026-09-11 01:00 台北時間，紐約仍為 2026-09-10 13:00 EDT)
        # 美股仍在盤中交易！目標交易日應為紐約當日 (2026-09-10)，絕不可跳到 2026-09-11
        dt = datetime.datetime(2026, 9, 11, 1, 0, 0, tzinfo=self.tz_tw)
        target = get_target_trading_date(dt, "AAPL")
        self.assertEqual(target, "2026-09-10")

    def test_us_postmarket_taipei_morning(self):
        # 5. 台北時間早上 06:00 (美股已收盤，紐約 18:00 EDT)，目標日應為下個交易日 (2026-09-11)
        dt = datetime.datetime(2026, 9, 11, 6, 0, 0, tzinfo=self.tz_tw)
        target = get_target_trading_date(dt, "AAPL")
        self.assertEqual(target, "2026-09-11")

    def test_find_actual_close_price_with_tz_datetimeindex(self):
        # 6. 模擬 yfinance 歷史數據 (包含 America/New_York 時區索引)
        idx = pd.date_range("2026-09-08", periods=3, tz="America/New_York", name="Date")
        hist = pd.DataFrame({
            "Close": [140.0, 145.0, 150.0]
        }, index=idx)

        # 模擬現在時間是 2026-09-10 20:00 (當日美股已收盤)
        now_dt = datetime.datetime(2026, 9, 10, 20, 0, 0, tzinfo=self.tz_ny)
        price, actual_date = find_actual_close_price(hist, "2026-09-10", "AAPL", now_dt=now_dt)
        self.assertEqual(price, 150.0)
        self.assertEqual(actual_date, "2026-09-10")

    def test_market_closed_check(self):
        # 7. 當日盤前不可視為已收盤
        now_premarket = datetime.datetime(2026, 9, 10, 8, 30, 0, tzinfo=self.tz_tw)
        self.assertFalse(is_market_closed("2026-09-10", "2330.TW", now_dt=now_premarket))

        # 當日盤後視為已收盤
        now_postmarket = datetime.datetime(2026, 9, 10, 14, 0, 0, tzinfo=self.tz_tw)
        self.assertTrue(is_market_closed("2026-09-10", "2330.TW", now_dt=now_postmarket))


if __name__ == "__main__":
    unittest.main()
