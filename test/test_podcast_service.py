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

    @patch("data.podcast_service.urllib.request.urlopen")
    def test_fetch_biggo_live_episodes_success(self, mock_urlopen):
        import json
        payload = {
            "result": True,
            "data": {
                "total": 1,
                "episodes": [
                    {
                        "id": "ep_biggo_001",
                        "title": "台積電 CoWoS 擴產與先進封裝",
                        "summary": "深入探討台積電先進封裝產能進展與 NVIDIA 需求。",
                        "source_id": "c2c3e3fcee7e",
                        "source_name": "Founders",
                        "source_image_url": "https://img.example.com/ch.webp",
                        "source_description": "Founders podcast",
                        "source_episode_count": 50,
                        "source_follower_count": 100000,
                        "published_at": 1792042440,
                        "duration": 3600.0,
                        "language": "zh",
                        "topics": ["台積電", "CoWoS"],
                        "key_entities": ["TSMC"],
                        "business_highlights": [{"insight": "CoWoS 先進封裝產能持續倍增"}],
                        "speakers": [{"role": "Host", "name": "David Senra"}],
                        "companies": [{"symbol": "2330.TW", "name": "TSMC", "sentiment": "bullish", "mentions": 5}],
                        "url": "https://example.com/audio.mp3"
                    }
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        service = PodcastService.__new__(PodcastService)
        service.db = MagicMock()
        service.db.save_podcast_episodes_batch.return_value = 1
        service.db.save_podcast_channels_batch.return_value = 0
        from data.decision_service import DecisionService
        service.decision_service = DecisionService()
        service.last_live_sync_at = 0.0
        service.get_seed_channels = MagicMock(return_value=[{"source_id": "c2c3e3fcee7e", "channel_id": "founders"}])
        service.extract_entities_and_tickers = MagicMock(return_value=(["台積電"], ["2330.TW"]))

        saved = service.fetch_biggo_live_episodes(max_pages=1)
        self.assertEqual(saved, 1)
        self.assertGreater(service.last_live_sync_at, 0)

        # 檢驗傳入 DuckDB 的欄位轉換
        service.db.save_podcast_episodes_batch.assert_called_once()
        saved_batch = service.db.save_podcast_episodes_batch.call_args[0][0]
        self.assertEqual(len(saved_batch), 1)
        item = saved_batch[0]
        self.assertEqual(item["id"], "ep_biggo_001")
        self.assertEqual(item["channel_id"], "founders")
        self.assertEqual(item["published_at"], "2026-10-15T05:34:00Z")
        self.assertEqual(item["host"], "David Senra")
        self.assertIn("2330.TW", item["tickers"])
        self.assertEqual(item["ticker_sentiments"]["2330.TW"]["stance"], "bullish")
        self.assertEqual(item["key_takeaways"], ["CoWoS 先進封裝產能持續倍增"])
        self.assertIsNotNone(item["importance_score"])

    @patch("data.podcast_service.urllib.request.urlopen")
    def test_fetch_biggo_live_episodes_network_error_gracefully_handled(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection timeout")

        service = PodcastService.__new__(PodcastService)
        service.db = MagicMock()
        service.decision_service = MagicMock()
        service.last_live_sync_at = 0.0
        service.get_seed_channels = MagicMock(return_value=[])

        saved = service.fetch_biggo_live_episodes(max_pages=1)
        self.assertEqual(saved, 0)


if __name__ == "__main__":
    unittest.main()
