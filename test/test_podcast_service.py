import unittest

from data.podcast_service import PodcastService


class TestPodcastService(unittest.TestCase):
    def test_missing_quote_is_explicitly_unavailable(self):
        service = PodcastService.__new__(PodcastService)

        quote = service.get_ticker_quote_badge("NVDA", cached_quotes={})

        self.assertEqual(quote["ticker"], "NVDA")
        self.assertIsNone(quote["price"])
        self.assertIsNone(quote["change_pct"])
        self.assertEqual(quote["direction"], "flat")
        self.assertEqual(quote["source"], "unavailable")

    def test_prediction_snapshot_does_not_become_quote_change(self):
        service = PodcastService.__new__(PodcastService)

        quote = service.get_ticker_quote_badge(
            "NVDA",
            cached_quotes={
                "NVDA": {
                    "ticker": "NVDA",
                    "name": "NVDA",
                    "price": 100.0,
                    "change_pct": None,
                    "direction": "flat",
                    "source": "quant_prediction_snapshot",
                    "as_of": "2026-10-14T09:00:00+08:00",
                }
            },
        )

        self.assertEqual(quote["source"], "quant_prediction_snapshot")
        self.assertIsNone(quote["change_pct"])
        self.assertEqual(quote["direction"], "flat")


if __name__ == "__main__":
    unittest.main()
