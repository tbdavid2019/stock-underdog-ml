"""
test/test_egm_slice2_fixes.py - Regression tests for Slice 2 EGM security, DB, cache & governance fixes.
"""
import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi.testclient import TestClient


class TestEGMSlice2Fixes(unittest.TestCase):
    def test_podcast_evaluate_requires_auth_in_production(self):
        """Regression: POST /podcast/batch-evaluate and /episodes/{id}/evaluate must fail closed in production."""
        from api.main import app

        client = TestClient(app)
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "production", "WATCHLIST_API_KEY": ""}, clear=False):
            r1 = client.post("/api/v1/podcast/batch-evaluate?limit=1")
            self.assertEqual(r1.status_code, 401)
            r2 = client.post("/api/v1/podcast/episodes/ep1/evaluate")
            self.assertEqual(r2.status_code, 401)

    def test_sync_endpoints_enforce_rate_limit_and_optional_ops_key(self):
        """Regression: /sync endpoints enforce OPS_API_KEY when set and rate-limit floods."""
        from api.main import app
        from core.netlimiter import sync_limiter

        client = TestClient(app)
        with patch.dict(os.environ, {"OPS_API_KEY": "ops_secret_123"}, clear=False):
            r_unauth = client.post("/api/v1/earnings-call/sync")
            self.assertEqual(r_unauth.status_code, 401)

        with sync_limiter._lock:
            sync_limiter._store.clear()

        with patch("api.routes.earnings_call.EarningsCallFetcher.fetch_all_and_sync", return_value=0), \
             patch.dict(os.environ, {"OPS_API_KEY": ""}, clear=False):
            codes = [client.post("/api/v1/earnings-call/sync").status_code for _ in range(7)]
            self.assertEqual(codes[:6], [200] * 6)
            self.assertEqual(codes[6], 429)

        with sync_limiter._lock:
            sync_limiter._store.clear()

    def test_decision_get_stock_force_refresh_blocks_unauthenticated_when_key_set(self):
        """Regression: GET /decision/stock/{ticker}?force_refresh=true must require key if DECISION_API_KEY is set."""
        from api.main import app

        client = TestClient(app)
        with patch.dict(os.environ, {"DECISION_API_KEY": "dec_secret_xyz"}, clear=False):
            r = client.get("/api/v1/decision/stock/2330.TW?force_refresh=true")
            self.assertEqual(r.status_code, 401)

    def test_earnings_call_invalid_ref_date_returns_422(self):
        """Regression: Invalid ref_date returns HTTP 422 instead of silently falling back to today."""
        from api.main import app

        client = TestClient(app)
        r = client.get("/api/v1/earnings-call/upcoming?ref_date=not-a-date")
        self.assertEqual(r.status_code, 422)

    def test_duckdb_table_name_validation_blocks_injection(self):
        """Regression: get_row_count / export_to_parquet reject unallowlisted or injected table names."""
        from data.duckdb_manager import DuckDBManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = DuckDBManager(db_path=str(Path(tmp) / "test_sec.duckdb"))
            self.assertEqual(mgr.get_row_count("predictions; DROP TABLE predictions;--"), 0)
            self.assertEqual(mgr.get_row_count("watchlist_items"), 0)
            with self.assertRaises(ValueError):
                mgr.export_to_parquet(str(Path(tmp) / "out.parquet"), table_name="predictions; DROP TABLE predictions;--")
            with self.assertRaises(ValueError):
                mgr.export_to_parquet(str(Path(tmp) / "wl.parquet"), table_name="watchlist_items")
            with self.assertRaises(ValueError):
                mgr.export_to_parquet(str(Path(tmp) / "bad;path.parquet"), table_name="predictions")

    def test_duckdb_decision_verdict_respects_macro_regime(self):
        """Regression: get_decision_verdict separates bull vs panic macro_regime verdicts & rejects legacy untagged rows."""
        from data.duckdb_manager import DuckDBManager

        with tempfile.TemporaryDirectory() as tmp:
            mgr = DuckDBManager(db_path=str(Path(tmp) / "test_regime.duckdb"))
            mgr.save_decision_verdict(
                "2330.TW",
                {"action_decision": "strong_buy", "action_confidence": 0.9, "conviction_score": 4.2, "is_favorable_entry": 0.8},
                macro_regime="bull",
            )
            mgr.save_decision_verdict(
                "2330.TW",
                {"action_decision": "avoid", "action_confidence": 0.85, "conviction_score": 1.2, "is_favorable_entry": 0.1},
                macro_regime="panic",
            )
            mgr.save_decision_verdict(
                "AAPL",
                {"action_decision": "buy", "action_confidence": 0.7, "conviction_score": 3.0, "is_favorable_entry": 0.6},
                macro_regime="",
            )
            bull_hit = mgr.get_decision_verdict("2330.TW", macro_regime="bull")
            panic_hit = mgr.get_decision_verdict("2330.TW", macro_regime="panic")
            legacy_miss = mgr.get_decision_verdict("AAPL", macro_regime="panic")
            self.assertIsNotNone(bull_hit)
            self.assertEqual(bull_hit["action_decision"], "strong_buy")
            self.assertIsNotNone(panic_hit)
            self.assertEqual(panic_hit["action_decision"], "avoid")
            self.assertIsNone(legacy_miss)

    def test_fetcher_batch_trips_and_respects_circuit_breaker(self):
        """Regression: get_stock_data_batch checks and trips _stock_fetch_circuit_open_until on 429."""
        from data.fetcher import StockFetcher

        orig = StockFetcher._stock_fetch_circuit_open_until
        try:
            StockFetcher._stock_fetch_circuit_open_until = 0.0
            with patch("data.cache.CacheManager.get_stock_data", return_value=None), \
                 patch("yfinance.download", side_effect=RuntimeError("HTTP 429 Too Many Requests")), \
                 patch("data.fetcher.StockFetcher.get_stock_data", return_value=pd.DataFrame()) as mock_single:
                res = StockFetcher.get_stock_data_batch(["2330.TW"], period="6mo", use_cache=True)
                self.assertEqual(res, {})
                self.assertGreater(StockFetcher._stock_fetch_circuit_open_until, 0.0)
                mock_single.assert_called_once()
        finally:
            StockFetcher._stock_fetch_circuit_open_until = orig

    def test_supabase_save_dual_strategy_includes_timestral(self):
        """Regression: SupabaseManager.save_dual_strategy_results persists timestral_results."""
        from database import SupabaseManager

        mgr = SupabaseManager()
        mgr.enabled = True
        mock_table = MagicMock()
        mock_client = MagicMock()
        mock_client.table.return_value = mock_table
        mgr.client = mock_client

        results = {
            "xuantie_results": pd.DataFrame(),
            "timestral_results": [{"ticker": "2330.TW", "current_price": 1000.0, "predicted_price": 1020.0, "potential": 2.0}],
            "timesfm_results": [],
            "moirai_results": [],
            "overlap_results": pd.DataFrame(),
        }
        mgr.save_dual_strategy_results("台灣50", results)
        inserted = mock_table.insert.call_args[0][0]
        self.assertEqual(len(inserted), 1)
        self.assertEqual(inserted[0]["model_name"], "Timestral")

    def test_moirai_plugin_default_revision_matches_config(self):
        """Regression: Moirai2Plugin fallback checkpoint_revision matches core.config."""
        import inspect
        from models.plugins import Moirai2Plugin
        from core.config import PipelineConfig

        src = inspect.getsource(Moirai2Plugin.initialize)
        self.assertIn(PipelineConfig.MOIRAI_CHECKPOINT_REVISION, src)

    def test_fundamentals_skips_caching_all_none(self):
        """Regression: all-None yfinance info does not poison L1/L2/L3 fundamentals caches."""
        from data.fundamentals import FundamentalProvider

        FundamentalProvider._MEM_CACHE.clear()
        fake_ticker = "MOCKNONE.US"
        fake_stock = MagicMock()
        fake_stock.info = {}
        fake_yf = MagicMock()
        fake_yf.Ticker.return_value = fake_stock

        with patch("data.fundamentals.yf", fake_yf), \
             patch("data.fundamentals.CacheManager.get_fundamentals", return_value=None), \
             patch("data.fundamentals.CacheManager.set_fundamentals") as mock_set_disk, \
             patch("data.fundamentals.DuckDBManager", None):
            res = FundamentalProvider.get_fundamentals(fake_ticker, use_cache=True)

        self.assertEqual(res, {"pe": None, "forward_pe": None, "pb": None, "ev_ebitda": None})
        mock_set_disk.assert_not_called()
        self.assertNotIn(fake_ticker, FundamentalProvider._MEM_CACHE)

    def test_cache_manager_atomic_write_leaves_no_tmp(self):
        """Regression: CacheManager.set_fundamentals writes atomically via temp + os.replace."""
        import tempfile
        from data.cache import CacheManager

        with tempfile.TemporaryDirectory() as tmpdir:
            target = os.path.join(tmpdir, "2330.TW_fund.json")
            with patch.object(CacheManager, "_get_fundamentals_path", return_value=target):
                CacheManager.set_fundamentals("2330.TW", {"pe": 20.5})
            self.assertTrue(os.path.exists(target))
            leftovers = [f for f in os.listdir(tmpdir) if ".tmp." in f]
            self.assertEqual(leftovers, [])

    def test_company_profile_rejects_unsafe_ticker_url_and_caps_cache(self):
        """Regression: _fetch_profile_from_2md blocks unsafe ticker characters and caps _CACHE."""
        from data.company_profile import CompanyProfileService

        CompanyProfileService._CACHE.clear()
        with patch("data.company_profile.requests.get") as mock_get:
            CompanyProfileService._fetch_profile_from_2md({"ticker": "2330.TW/../../evil"}, is_tw=True)
            mock_get.assert_not_called()

        for i in range(CompanyProfileService._CACHE_MAX_SIZE + 5):
            CompanyProfileService._set_mem_cache(f"T{i}.TW", 9999999999.0, {"ticker": f"T{i}.TW"})
        self.assertLessEqual(len(CompanyProfileService._CACHE), CompanyProfileService._CACHE_MAX_SIZE)
        CompanyProfileService._CACHE.clear()

    def test_duckdb_get_quant_alerts_all_profiles_and_model_ranking_helper(self):
        """Regression: get_quant_alerts(profile_id=None) spans profiles & _get_model_top_ranked works."""
        import tempfile
        from data.duckdb_manager import DuckDBManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db = DuckDBManager(db_path=os.path.join(tmpdir, "alerts_rank.duckdb"))
            db.add_quant_alert("2330.TW", "above_price", 1200.0, profile_id="default")
            db.add_quant_alert("AAPL", "above_price", 250.0, profile_id="alpha_desk")

            default_only = db.get_quant_alerts(profile_id="default", enabled_only=True)
            all_profiles = db.get_quant_alerts(profile_id=None, enabled_only=True)
            self.assertEqual(len(default_only), 1)
            self.assertEqual(len(all_profiles), 2)

            with db._write_lock(), db._get_connection(read_only=False) as con:
                con.execute("""
                    INSERT INTO predictions (ticker, model_name, strategy_type, index_name, timestamp, potential, composite_score)
                    VALUES
                        ('2330.TW', 'Timestral', 'Timestral', '台灣50', '2026-11-20 08:00:00', 8.5, 90.0),
                        ('2317.TW', 'TimesFM', 'TimesFM', '台灣50', '2026-11-20 08:00:00', 6.2, 85.0),
                        ('2454.TW', 'MOIRAI-2', 'MOIRAI-2', '台灣50', '2026-11-20 08:00:00', 7.1, 88.0)
                """)

            self.assertEqual(db.get_timestral_top_bullish(limit=5)[0]["ticker"], "2330.TW")
            self.assertEqual(db.get_top_bullish(limit=5)[0]["ticker"], "2330.TW")
            self.assertEqual(db.get_timesfm_top_bullish(limit=5)[0]["ticker"], "2317.TW")
            self.assertEqual(db.get_moirai_top_bullish(limit=5)[0]["ticker"], "2454.TW")

    def test_predictions_timestral_and_timesfm_routes_use_cache(self):
        """Regression: timestral & timesfm top-bullish/bearish endpoints populate _PRED_CACHE."""
        from api.routes import predictions as pred_routes

        with pred_routes._CACHE_LOCK:
            pred_routes._PRED_CACHE.clear()
        mock_db = MagicMock()
        mock_db.get_timestral_top_bullish.return_value = [{"ticker": "2330.TW", "potential": 5.0}]
        mock_db.get_latest_macro_state.return_value = None

        with patch("api.routes.predictions._build_prediction_response", return_value={"status": "success", "count": 1, "data": []}):
            pred_routes.get_timestral_top_bullish(index_name=None, limit=10, db=mock_db)
            pred_routes.get_timestral_top_bullish(index_name=None, limit=10, db=mock_db)

        self.assertEqual(mock_db.get_timestral_top_bullish.call_count, 1)

    def test_cross_profile_evaluate_and_dispatch_alerts(self):
        """Regression: evaluate_and_dispatch_alerts(profile_id=None) dispatches non-default profile alerts end-to-end."""
        from data.alert_evaluator import evaluate_and_dispatch_alerts
        from data.duckdb_manager import DuckDBManager

        with tempfile.TemporaryDirectory() as tmp:
            db = DuckDBManager(db_path=os.path.join(tmp, "alerts_e2e.duckdb"))
            created = db.add_quant_alert("AAPL", "above_price", 150.0, profile_id="alpha_desk")
            with db._write_lock(), db._get_connection(read_only=False) as con:
                con.execute("""
                    INSERT INTO predictions (ticker, model_name, strategy_type, index_name, timestamp, current_price, potential, composite_score)
                    VALUES ('AAPL', 'Timestral', 'Timestral', 'NASDAQ100', '2026-11-20 08:00:00', 210.0, 5.0, 82.0)
                """)

            with patch("data.alert_evaluator.dispatch_alert_notification", return_value=True) as mock_dispatch:
                res = evaluate_and_dispatch_alerts(profile_id=None, db=db)

            self.assertEqual(res["evaluated"], 1)
            self.assertEqual(res["triggered"], 1)
            mock_dispatch.assert_called_once()
            updated = [a for a in db.get_quant_alerts(profile_id="alpha_desk") if a["alert_id"] == created["alert_id"]]
            self.assertIsNotNone(updated[0]["last_triggered_at"])

    def test_notifier_redacts_token_url_and_podcast_distinct_keys(self):
        """Regression: notifier redacts secret tokens on transport errors & podcast evaluate avoids OPS_API_KEY conflict."""
        import requests as req_mod
        import notifier
        from core.config import TelegramConfig, DiscordConfig
        from api.main import app

        secret_tok = "123456:SECRET_TG_TOKEN"
        with patch.object(TelegramConfig, "BOT_TOKEN", secret_tok), \
             patch.object(TelegramConfig, "CHANNEL_ID", "-1001"), \
             patch("notifier.requests.post", side_effect=req_mod.ConnectionError(f"https://api.telegram.org/bot{secret_tok}/sendMessage down")):
            with self.assertRaises(RuntimeError) as ctx:
                notifier.send_to_telegram("hello")
            self.assertNotIn(secret_tok, str(ctx.exception))

        secret_hook = "https://discord.com/api/webhooks/999/SECRET_WEBHOOK_TOK"
        with patch.object(DiscordConfig, "WEBHOOK_URL", secret_hook), \
             patch("notifier.requests.post", side_effect=req_mod.Timeout(f"{secret_hook} timed out")):
            with self.assertRaises(RuntimeError) as ctx2:
                notifier.send_to_discord("hello")
            self.assertNotIn("SECRET_WEBHOOK_TOK", str(ctx2.exception))

        client = TestClient(app)
        with patch.dict(os.environ, {"WATCHLIST_API_KEY": "wl_only_key", "OPS_API_KEY": "ops_other_key"}, clear=False), \
             patch("api.routes.podcast.podcast_service.evaluate_episodes_batch", return_value={"success": True, "evaluated": 0}):
            resp = client.post("/api/v1/podcast/batch-evaluate", headers={"X-API-Key": "wl_only_key"})
            self.assertEqual(resp.status_code, 200)

    def test_decision_get_rejects_invalid_macro_regime(self):
        """Regression: arbitrary macro_regime values return HTTP 422 instead of poisoning caches."""
        from api.main import app
        client = TestClient(app)
        resp = client.get("/api/v1/decision/stock/2330.TW?macro_regime=evil_cache_buster")
        self.assertEqual(resp.status_code, 422)


if __name__ == "__main__":
    unittest.main()
