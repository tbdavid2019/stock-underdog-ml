"""
test/test_egm_slice1_fixes.py - Regression tests for Slice 1 EGM fixes.
"""
import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestEGMSlice1Fixes(unittest.TestCase):
    def test_mcp_canary_loader_health_with_populated_file(self):
        """Regression: get_canary_loader_health must not fail with NameError when cache file exists."""
        import mcp_server

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "data" / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "canary_health_latest.json"
            payload = {"status": "healthy", "sources": {"twse": {"status": "healthy"}}}
            cache_file.write_text(json.dumps(payload), encoding="utf-8")

            fake_file = str(Path(tmp) / "mcp_server.py")
            with patch.object(mcp_server, "__file__", fake_file):
                res = mcp_server.get_canary_loader_health()

            self.assertTrue(res["success"])
            self.assertEqual(res["data"]["status"], "healthy")
            self.assertIn("twse", res["data"]["sources"])

    def test_cache_get_fundamentals_none_max_age_allows_stale(self):
        """Regression: CacheManager.get_fundamentals(max_age_hours=None) returns stale cache instead of TypeError."""
        from data.cache import CacheManager

        with tempfile.TemporaryDirectory() as tmp:
            with patch("data.cache.FUNDAMENTALS_CACHE_DIR", tmp):
                CacheManager.set_fundamentals("2330.TW", {"pe": 20.5, "pb": 5.1})
                path = CacheManager._get_fundamentals_path("2330.TW")
                old_ts = 1_000_000_000
                os.utime(path, (old_ts, old_ts))

                # With normal TTL, stale file expires
                self.assertIsNone(CacheManager.get_fundamentals("2330.TW", max_age_hours=24))
                # With max_age_hours=None, stale file is returned
                stale = CacheManager.get_fundamentals("2330.TW", max_age_hours=None)
                self.assertIsNotNone(stale)
                self.assertEqual(stale["pe"], 20.5)

    def test_notifier_timeouts_and_failure_propagation(self):
        """Regression: notifier passes explicit timeout and raises on HTTP failures."""
        import notifier
        from core.config import TelegramConfig, DiscordConfig

        with patch.object(TelegramConfig, "BOT_TOKEN", "tok"), \
             patch.object(TelegramConfig, "CHANNEL_ID", "chan"):
            bad_resp = MagicMock(status_code=500, text="upstream error")
            with patch("notifier.requests.post", return_value=bad_resp) as mock_post:
                with self.assertRaises(RuntimeError):
                    notifier.send_to_telegram("hello")
                self.assertIn("timeout", mock_post.call_args.kwargs)

        with patch.object(DiscordConfig, "WEBHOOK_URL", "https://discord.example/webhook"):
            bad_resp = MagicMock(status_code=400, text="bad webhook")
            with patch("notifier.requests.post", return_value=bad_resp) as mock_post:
                with self.assertRaises(RuntimeError):
                    notifier.send_to_discord("hello")
                self.assertIn("timeout", mock_post.call_args.kwargs)

    def test_telegram_config_accepts_chat_id_fallback(self):
        """Regression: TelegramConfig falls back to TELEGRAM_CHAT_ID when TELEGRAM_CHANNEL_ID is unset."""
        import subprocess
        import sys

        env = dict(os.environ)
        env["TELEGRAM_CHANNEL_ID"] = ""
        env["TELEGRAM_CHAT_ID"] = "-100999"
        proc = subprocess.run(
            [sys.executable, "-c", "from core.config import TelegramConfig; print(TelegramConfig.CHANNEL_ID)"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.stdout.strip(), "-100999")

    def test_asset_fingerprint_uses_cache_when_default_dirs(self):
        """Regression: get_asset_fingerprint caches result across consecutive default calls."""
        import api.main as api_main

        with api_main._ASSET_FINGERPRINT_LOCK:
            api_main._ASSET_FINGERPRINT_CACHE["val"] = None
            api_main._ASSET_FINGERPRINT_CACHE["expiry"] = 0.0

        with patch("api.main.os.walk", wraps=os.walk) as mock_walk:
            fp1 = api_main.get_asset_fingerprint()
            calls_after_first = mock_walk.call_count
            fp2 = api_main.get_asset_fingerprint()
            self.assertEqual(fp1, fp2)
            self.assertEqual(mock_walk.call_count, calls_after_first)

    def test_ticker_canonicalizer_uses_config_sector_map_and_duckdb_path(self):
        """Regression: TickerUniverseValidator includes config.sector.TICKER_SECTOR_MAP and config.db.DUCKDB_PATH."""
        from core.ticker_canonicalizer import TickerUniverseValidator
        from core.config import config

        with patch.object(config.db, "DUCKDB_PATH", "/nonexistent/test.duckdb"):
            v = TickerUniverseValidator()
            wl = v.get_whitelist(force_refresh=True)
            self.assertIn("2330", wl)
            self.assertIn("NVDA", wl)


if __name__ == "__main__":
    unittest.main()
