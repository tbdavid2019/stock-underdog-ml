"""
test/test_polymarket_service.py - Unit test suite for Polymarket Service
"""

import unittest
from unittest.mock import patch
from data.polymarket_service import PolymarketService


class TestPolymarketService(unittest.TestCase):
    def setUp(self):
        PolymarketService._MEM_CACHE.clear()

    def test_exclude_patterns(self):
        noisy_titles = [
            "Will Kansas City Chiefs win the Super Bowl?",
            "Lakers vs Celtics winner tonight",
            "UEFA Champions League final winner",
            "Premier League Arsenal vs Chelsea",
        ]
        for title in noisy_titles:
            self.assertTrue(
                bool(PolymarketService.EXCLUDE_PATTERNS.search(title)),
                f"Expected '{title}' to match noise filter"
            )

        legit_titles = [
            "Fed decreases interest rates by 25 bps at next FOMC?",
            "Will the US enter a recession in 2026?",
            "US tariff on Chinese semiconductors above 50%?",
            "Nvidia Q3 revenue exceeds $40B?",
            "OpenAI releases GPT-5 before July 2026?"
        ]
        for title in legit_titles:
            self.assertFalse(
                bool(PolymarketService.EXCLUDE_PATTERNS.search(title)),
                f"Expected '{title}' NOT to match noise filter"
            )

    def test_classify_categories(self):
        q1 = "Fed decreases interest rates by 25 bps at next FOMC?"
        cats1 = [cat for cat, pat in PolymarketService.PATTERNS.items() if pat.search(q1)]
        self.assertIn("fed_rates", cats1)

        q2 = "Will US enter an official recession in 2026?"
        cats2 = [cat for cat, pat in PolymarketService.PATTERNS.items() if pat.search(q2)]
        self.assertIn("macro_recession", cats2)

        q3 = "US imposes new tariff on European auto imports"
        cats3 = [cat for cat, pat in PolymarketService.PATTERNS.items() if pat.search(q3)]
        self.assertIn("geopolitics", cats3)

        q4 = "Nvidia hits $4 Trillion market cap in 2026"
        cats4 = [cat for cat, pat in PolymarketService.PATTERNS.items() if pat.search(q4)]
        self.assertIn("tech_giants", cats4)

    @patch("data.polymarket_service.PolymarketService._fetch_raw_markets")
    def test_get_macro_sentiment_processing(self, mock_fetch):
        mock_fetch.return_value = [
            {
                "id": "m1",
                "question": "Fed decrease interest rates 25 bps?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.85", "0.15"]',
                "volume24hr": 600000.0,
                "volume": 2000000.0,
                "liquidity": 1200000.0,
                "category": "Economics",
                "slug": "fed-rates-sep"
            },
            {
                "id": "m2",
                "question": "Lakers vs Celtics winner",
                "outcomes": '["Lakers", "Celtics"]',
                "outcomePrices": '["0.5", "0.5"]',
                "volume24hr": 99999.0,
            }
        ]

        result = PolymarketService.get_macro_sentiment(force_refresh=True)
        self.assertTrue(result["success"])
        self.assertIn("fed_real_money_odds", result)
        self.assertEqual(result["fed_real_money_odds"]["cut_25bps"], 85.0)
        # Verify noise was excluded (only 1 market kept)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["markets"][0]["category"], "fed_rates")
        self.assertEqual(result["markets"][0]["yes_prob"], 85.0)
        self.assertEqual(result["markets"][0]["probability"], 85.0)
        self.assertEqual(result["markets"][0]["top_outcome"], "Yes")

        all_result = PolymarketService.get_macro_sentiment(
            force_refresh=True, category="all"
        )
        self.assertEqual(all_result["count"], 1)

    @patch("data.polymarket_service.PolymarketService._fetch_raw_markets")
    @patch("data.polymarket_service.PolymarketService._read_stale_cache")
    def test_upstream_failure_is_not_reported_as_fresh_success(
        self, mock_stale_cache, mock_fetch
    ):
        mock_fetch.return_value = []
        mock_stale_cache.return_value = None

        result = PolymarketService.get_macro_sentiment(force_refresh=True)

        self.assertFalse(result["success"])
        self.assertFalse(result["stale"])
        self.assertEqual(result["markets"], [])
        self.assertIn("error", result)

    @patch("data.polymarket_service.PolymarketService._fetch_raw_markets")
    @patch("data.polymarket_service.PolymarketService._read_stale_cache")
    def test_upstream_failure_returns_last_known_good_as_stale(
        self, mock_stale_cache, mock_fetch
    ):
        mock_fetch.return_value = []
        mock_stale_cache.return_value = {
            "success": True,
            "stale": False,
            "source": "2md_reader",
            "count": 1,
            "fed_real_money_odds": {"cut_25bps": 72.0},
            "markets": [{"category": "fed_rates", "probability": 72.0}],
        }

        result = PolymarketService.get_macro_sentiment(
            force_refresh=True, category="fed_rates"
        )

        self.assertFalse(result["success"])
        self.assertTrue(result["stale"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["markets"][0]["probability"], 72.0)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
