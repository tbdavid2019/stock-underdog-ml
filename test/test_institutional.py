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
