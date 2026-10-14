import unittest
from unittest.mock import MagicMock, patch

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

    @patch("data.podcast_service.urllib.request.urlopen")
    def test_rss_missing_metadata_is_not_fabricated_or_marked_extracted(self, mock_urlopen):
        response = MagicMock()
        response.read.return_value = (
            b"<rss><channel><title>Example</title><item>"
            b"<title>Raw episode</title><link>https://example.com/episode</link>"
            b"</item></channel></rss>"
        )
        mock_urlopen.return_value.__enter__.return_value = response
        service = PodcastService.__new__(PodcastService)

        episodes = service.fetch_rss_episodes("https://example.com/feed", "example", limit=1)

        self.assertEqual(len(episodes), 1)
        self.assertIsNone(episodes[0]["duration_seconds"])
        self.assertEqual(episodes[0]["language"], "unknown")
        self.assertIsNone(episodes[0]["tldr_zh_tw"])
        self.assertEqual(episodes[0]["key_takeaways"], [])
        self.assertEqual(episodes[0]["status"], "raw")


if __name__ == "__main__":
    unittest.main()
