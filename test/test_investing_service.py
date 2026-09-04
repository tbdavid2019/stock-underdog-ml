"""
test/test_investing_service.py - Unit tests for InvestingService (Fed rate, earnings calendar, economic calendar, commodities)
"""

import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from data.investing_service import InvestingService


class TestInvestingService(unittest.TestCase):
    def setUp(self):
        InvestingService._MEM_CACHE.clear()
        if os.path.exists(InvestingService.CACHE_DIR):
            shutil.rmtree(InvestingService.CACHE_DIR, ignore_errors=True)

    def tearDown(self):
        InvestingService._MEM_CACHE.clear()
        if os.path.exists(InvestingService.CACHE_DIR):
            shutil.rmtree(InvestingService.CACHE_DIR, ignore_errors=True)

    @patch("data.investing_service.InvestingService._fetch_url_markdown")
    def test_get_fed_rate_monitor(self, mock_fetch):
        mock_fetch.return_value = """
Title: Fed Rate Monitor Tool - Investing.com
Fed Interest Rate Decision Sep 16, 2026 02:00PM ET
Meeting Time:_Sep 16, 2026 02:00PM ET_
Future Price:_96.313_

| Target Rate | Current Probability% | Previous Day Probability% | Previous Week Probability% |
| --- | --- | --- | --- |
| 3.50 - 3.75 | 49.6% | 39.9% | 65.6% |
| 3.75 - 4.00 | 50.4% | 60.1% | 34.4% |
"""
        res = InvestingService.get_fed_rate_monitor()
        self.assertEqual(res["meeting_date"], "Sep 16, 2026 02:00PM ET")
        self.assertEqual(res["future_price"], 96.313)
        self.assertEqual(len(res["target_rates"]), 2)
        self.assertEqual(res["highest_prob_rate"], "3.75 - 4.00")
        self.assertEqual(res["highest_probability"], "50.4%")
        self.assertIn("3.75 - 4.00", res["summary"])

    @patch("data.investing_service.InvestingService._fetch_url_markdown")
    def test_get_earnings_calendar(self, mock_fetch):
        mock_fetch.return_value = """
|  | Friday, September 4, 2026 |  |  |  |  |  |  |  |  |
|  | #### KNOT Offshore Partners LP ( [KNOP](https://www.investing.com/equities/knot-offshore-partners-lp-earnings "KNOT Offshore Partners LP") ) KNOT Offshore Partners LP | - / | 0.1526 | - / | 91.7M | 388.03M |  | 11.23-0.53% | 11.55 +2.81% | [aa.aa](https://www.investing.com/pro/pricing) | - / | - |
|  | Tuesday, September 8, 2026 |  |  |  |  |  |  |  |  |
|  | #### GameStop Corp ( [GME](https://www.investing.com/equities/gamestop-corp-earnings "GameStop Corp") ) GameStop Corp | - / | 0.19 | - / | 1.07B | 8.63B |  | 19.23+1.37% | 19.31 +0.42% | [aa.aa](https://www.investing.com/pro/pricing) | - / | - |
"""
        items = InvestingService.get_earnings_calendar()
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["ticker"], "KNOP")
        self.assertEqual(items[0]["eps_forecast"], "0.1526")
        self.assertEqual(items[1]["ticker"], "GME")
        self.assertEqual(items[1]["revenue_forecast"], "1.07B")
        self.assertEqual(items[1]["market_cap"], "8.63B")

    @patch("data.investing_service.InvestingService._fetch_url_markdown")
    def test_get_commodities_summary(self, mock_fetch):
        mock_fetch.return_value = """
| - [x] | #### [Gold](https://www.investing.com/commodities/gold "Gold - (CFD)")derived | -0.06% | -0.06% | -0.46% | -0.16% | +5.05% | +3.11% | +133.81% |
| - [x] | #### [Copper](https://www.investing.com/commodities/copper "Copper - (CFD)")derived | -0.13% | -0.20% | +0.11% | +0.30% | -0.72% | +17.55% | +74.03% |
| - [x] | #### [Crude Oil WTI](https://www.investing.com/commodities/crude-oil "Crude Oil WTI - (CFD)")derived | -0.17% | -0.34% | +0.47% | +9.94% | +21.90% | +59.68% | +6.60% |
"""
        comms = InvestingService.get_commodities_summary()
        self.assertIn("gold", comms)
        self.assertIn("copper", comms)
        self.assertIn("crude_oil", comms)
        self.assertEqual(comms["gold"]["daily_change"], "-0.06%")
        self.assertEqual(comms["copper"]["weekly_change"], "+0.11%")
        self.assertEqual(comms["crude_oil"]["daily_change"], "-0.17%")

    @patch("data.investing_service.InvestingService._fetch_url_markdown")
    def test_get_economic_calendar(self, mock_fetch):
        mock_fetch.return_value = """
| Friday, September 4, 2026 |
| 08:30 US | 08:30 | US | [Nonfarm Payrolls (Aug)](https://www.investing.com/economic-calendar/nonfarm-payrolls-227) Act: - Cons: 160K Prev.: 114K |
| 08:30 US | 08:30 | US | [Unemployment Rate (Aug)](https://www.investing.com/economic-calendar/unemployment-rate-300) Act: - Cons: 4.2% Prev.: 4.3% |
"""
        events = InvestingService.get_economic_calendar()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["country"], "US")
        self.assertIn("Nonfarm Payrolls", events[0]["event"])
        self.assertEqual(events[0]["forecast"], "160K")
        self.assertEqual(events[0]["previous"], "114K")
        self.assertTrue(events[0]["is_high_impact"])

    @patch("data.investing_service.InvestingService._fetch_url_markdown")
    def test_persistent_disk_caching(self, mock_fetch):
        mock_fetch.return_value = """
Meeting Time:_Sep 16, 2026 02:00PM ET_
| 3.50 - 3.75 | 49.6% | 39.9% | 65.6% |
"""
        # First call fetches and writes to disk
        res1 = InvestingService.get_fed_rate_monitor()
        self.assertEqual(mock_fetch.call_count, 1)

        # Clear L1 RAM cache (simulating container/server restart)
        InvestingService._MEM_CACHE.clear()

        # Second call should read from L2 disk cache without network fetch!
        res2 = InvestingService.get_fed_rate_monitor()
        self.assertEqual(mock_fetch.call_count, 1)  # Still 1!
        self.assertEqual(res2["meeting_date"], "Sep 16, 2026 02:00PM ET")


if __name__ == "__main__":
    unittest.main()
