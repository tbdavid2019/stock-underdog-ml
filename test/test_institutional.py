"""
test/test_institutional.py - 測試三大法人籌碼數據模組
"""

import unittest
from data.institutional import InstitutionalProvider, InstitutionalSummary


class TestInstitutionalProvider(unittest.TestCase):
    def test_summary_dataclass(self):
        s = InstitutionalSummary(ticker="2330.TW", trust_net_5d=500, foreign_net_5d=1000, trust_streak=4)
        s.is_trust_streak = s.trust_streak >= 3
        s.is_sync_buy = s.foreign_net_5d > 0 and s.trust_net_5d > 0
        self.assertTrue(s.is_trust_streak)
        self.assertTrue(s.is_sync_buy)
        d = s.to_dict()
        self.assertEqual(d["trust_streak"], 4)
        self.assertEqual(d["trust_net_5d"], 500)

    def test_recent_trading_days(self):
        days = InstitutionalProvider.get_recent_trading_days(count=5)
        self.assertEqual(len(days), 5)
        for d in days:
            self.assertEqual(len(d), 8)
            self.assertTrue(d.isdigit())

    def test_recent_trading_days_midnight_boundary(self):
        import datetime
        # 1. 週三 08:00 (台股開盤前): 應推算至週二 (2026-09-08)
        dt_premarket = datetime.datetime(2026, 9, 9, 8, 0, 0)
        days = InstitutionalProvider.get_recent_trading_days(count=3, reference_dt=dt_premarket)
        self.assertEqual(days[0], "20260908")

        # 2. 週三 00:05 (剛過午夜): 應推算至週二 (2026-09-08)
        dt_midnight = datetime.datetime(2026, 9, 9, 0, 5, 0)
        days = InstitutionalProvider.get_recent_trading_days(count=3, reference_dt=dt_midnight)
        self.assertEqual(days[0], "20260908")

        # 3. 週三 16:00 (盤後籌碼已結算): 應包含週三 (2026-09-09)
        dt_afternoon = datetime.datetime(2026, 9, 9, 16, 0, 0)
        days = InstitutionalProvider.get_recent_trading_days(count=3, reference_dt=dt_afternoon)
        self.assertEqual(days[0], "20260909")

        # 4. 週一 08:00 (週一開盤前): 應跳過週末推算至上週五 (2026-09-04)
        dt_monday_pre = datetime.datetime(2026, 9, 7, 8, 0, 0)
        days = InstitutionalProvider.get_recent_trading_days(count=3, reference_dt=dt_monday_pre)
        self.assertEqual(days[0], "20260904")

    def test_batch_summary_tw50(self):
        tickers = ["2330.TW", "2317.TW", "2454.TW"]
        summaries = InstitutionalProvider.get_institutional_summary_batch(tickers, lookback_days=5)
        self.assertEqual(len(summaries), 3)
        for t in tickers:
            self.assertIn(t, summaries)
            s = summaries[t]
            self.assertIsInstance(s, InstitutionalSummary)
            print(f"Institutional Summary for {t}: 投信5D={s.trust_net_5d}張, 外資5D={s.foreign_net_5d}張, 連買={s.trust_streak}天")


if __name__ == "__main__":
    unittest.main()
