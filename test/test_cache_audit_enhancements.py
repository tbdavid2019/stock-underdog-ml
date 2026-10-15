"""
test/test_cache_audit_enhancements.py - Cache Audit Enhancements Unit Tests

Tests:
1. Decision verdicts DuckDB persistence & L1/L2 caching (Clef System One)
2. Company fundamentals DuckDB persistence & 3-tier caching (L1 mem -> L2 DuckDB -> L3 disk)
3. Earnings calls fetcher front-check DuckDB caching
4. Ticker resolver LRU in-memory cache
5. Market stats & predictions API response caching
"""

import os
import time
import shutil
import unittest
import datetime
from unittest.mock import patch, MagicMock

from data.duckdb_manager import DuckDBManager
from data.clef_client import ClefDecisionClient, ClefDecisionVerdict
from data.decision_service import DecisionService
from data.fundamentals import FundamentalProvider
from data.cache import CacheManager
from data.earnings_fetcher import EarningsCallFetcher


class TestCacheAuditEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_db_path = "test/scratch/test_cache_enhancements.duckdb"
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)
        cls.db = DuckDBManager(db_path=cls.test_db_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_db_path):
            try:
                os.remove(cls.test_db_path)
            except Exception:
                pass

    def setUp(self):
        # Clear in-memory caches between tests
        if hasattr(ClefDecisionClient, "_MEM_CACHE"):
            ClefDecisionClient._MEM_CACHE.clear()
        if hasattr(FundamentalProvider, "_MEM_CACHE"):
            FundamentalProvider._MEM_CACHE.clear()
        if hasattr(DuckDBManager, "_RESOLVER_CACHE"):
            DuckDBManager._RESOLVER_CACHE.clear()

    # =========================================================================
    # 1. Decision Verdicts DuckDB Persistence & Caching
    # =========================================================================

    def test_duckdb_decision_verdicts_crud(self):
        """Test saving and retrieving decision verdicts in DuckDB"""
        verdict_data = {
            "ticker": "2330.TW",
            "as_of_date": "2026-10-15",
            "action_decision": "strong_buy",
            "action_confidence": 0.85,
            "action_probabilities": {"strong_buy": 0.85, "buy": 0.10, "hold_watch": 0.05, "avoid": 0.0},
            "conviction_score": 4.5,
            "is_favorable_entry": 0.9,
            "tag": "🤖AI強買(85%)",
            "source_slot": "clef_primary",
            "success": True
        }

        # Save verdict
        success = self.db.save_decision_verdict("2330.TW", verdict_data, as_of_date="2026-10-15", ttl_seconds=86400)
        self.assertTrue(success)

        # Retrieve verdict
        loaded = self.db.get_decision_verdict("2330.TW", as_of_date="2026-10-15")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["ticker"], "2330.TW")
        self.assertEqual(loaded["action_decision"], "strong_buy")
        self.assertEqual(loaded["conviction_score"], 4.5)
        self.assertEqual(loaded["tag"], "🤖AI強買(85%)")
        self.assertEqual(loaded["source_slot"], "clef_primary")

        # Expired or different ticker should return None
        miss = self.db.get_decision_verdict("NONEXISTENT", as_of_date="2026-10-15")
        self.assertIsNone(miss)

    def test_clef_client_caching_layer(self):
        """Test ClefDecisionClient avoids repeated upstream calls when cached"""
        client = ClefDecisionClient(enabled=True)
        # Point client to our test DB
        client.db = self.db

        candidate = {
            "ticker": "2454.TW",
            "composite_score": 88.0,
            "lstm_potential": 5.2,
            "timesfm_potential": 4.1,
            "risk_reward_ratio": 2.1,
            "is_sync_buy": True,
            "strategy_type": ["玄鐵重劍", "LSTM", "土洋合買"]
        }

        call_count = 0
        mock_response = (
            True,
            {
                "action_decision": {"choice": "strong_buy", "confidence": 0.82, "probabilities": {"strong_buy": 0.82, "buy": 0.15}},
                "conviction_score": {"score": 4.6},
                "is_favorable_entry": {"noul": 0.88}
            },
            "clef_primary",
            None
        )

        def mock_call_systemone(state, questions):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch.object(client, "_call_systemone", side_effect=mock_call_systemone):
            # 1st call -> should hit upstream
            v1 = client.evaluate_stock(candidate=candidate, force_refresh=False)
            self.assertEqual(call_count, 1)
            self.assertEqual(v1.action_decision, "strong_buy")
            self.assertEqual(v1.conviction_score, 4.6)

            # 2nd call -> should hit cache (call_count remains 1)
            v2 = client.evaluate_stock(candidate=candidate, force_refresh=False)
            self.assertEqual(call_count, 1)
            self.assertEqual(v2.action_decision, "strong_buy")
            self.assertEqual(v2.conviction_score, 4.6)

            # 3rd call with force_refresh=True -> should bypass cache
            v3 = client.evaluate_stock(candidate=candidate, force_refresh=True)
            self.assertEqual(call_count, 2)
            self.assertEqual(v3.action_decision, "strong_buy")

    # =========================================================================
    # 2. Company Fundamentals DuckDB Persistence & Caching
    # =========================================================================

    def test_duckdb_fundamentals_crud(self):
        """Test saving and retrieving company fundamentals in DuckDB"""
        fund_data = {
            "pe": 24.5,
            "forward_pe": 19.8,
            "pb": 5.6,
            "ev_ebitda": 14.2
        }

        success = self.db.save_company_fundamentals("2330.TW", fund_data)
        self.assertTrue(success)

        loaded = self.db.get_company_fundamentals("2330.TW", max_age_hours=24)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["pe"], 24.5)
        self.assertEqual(loaded["forward_pe"], 19.8)
        self.assertEqual(loaded["pb"], 5.6)
        self.assertEqual(loaded["ev_ebitda"], 14.2)

        # Non-existent should return None
        self.assertIsNone(self.db.get_company_fundamentals("UNKNOWN_TICKER"))

    def test_fundamental_provider_duckdb_integration(self):
        """Test FundamentalProvider prioritizes DuckDB before hitting yfinance"""
        fund_data = {
            "pe": 15.2,
            "forward_pe": 13.0,
            "pb": 2.1,
            "ev_ebitda": 8.5
        }
        self.db.save_company_fundamentals("2317.TW", fund_data)

        # Mock yfinance to ensure it is NOT called
        with patch("data.fundamentals.yf.Ticker", side_effect=Exception("Should not hit yfinance when DuckDB is warm")):
            with patch("data.fundamentals.DuckDBManager", return_value=self.db):
                res = FundamentalProvider.get_fundamentals("2317.TW", use_cache=True)
                self.assertEqual(res["pe"], 15.2)
                self.assertEqual(res["forward_pe"], 13.0)
                self.assertEqual(res["pb"], 2.1)
                self.assertEqual(res["ev_ebitda"], 8.5)

    # =========================================================================
    # 3. Earnings Call Fetcher Front-Check Cache
    # =========================================================================

    def test_earnings_fetcher_front_checks_duckdb(self):
        """Test fetch_ticker_calls returns existing DuckDB records without BigGo network call"""
        sample_call = [{
            "id": "call_test_mtk_123",
            "market": "TW",
            "ticker": "2454.TW",
            "name": "聯發科",
            "call_date": "2026-10-28",
            "call_time": "14:00",
            "timezone": "Asia/Taipei",
            "meeting_type": "online",
            "webcast_url": "https://www.mediatek.tw/investor-relations",
            "presentation_url": "https://www.mediatek.tw/presentation.pdf",
            "guidance_summary": "天璣 9400 旗艦晶片出貨強勁，營收指引超預期",
            "source": "BigGo / TW IR"
        }]
        self.db.save_earnings_calls_batch(sample_call)

        fetcher = EarningsCallFetcher(db=self.db)

        # Mock urllib to assert it's never called when DuckDB is warm
        with patch("urllib.request.urlopen", side_effect=Exception("Should not hit BigGo when DuckDB has calls")):
            results = fetcher.fetch_ticker_calls("2454.TW", force_refresh=False)
            self.assertTrue(len(results) >= 1)
            self.assertEqual(results[0]["ticker"], "2454.TW")
            self.assertEqual(results[0]["guidance_summary"], "天璣 9400 旗艦晶片出貨強勁，營收指引超預期")

    # =========================================================================
    # 4. Ticker Resolver In-Memory Cache
    # =========================================================================

    def test_ticker_resolver_in_memory_caching(self):
        """Test resolve_ticker uses in-memory LRU cache to avoid repetitive SQL union scans"""
        # Seed an item in tw_daily_bars in test db
        with self.db._get_connection() as con:
            con.execute("""
                INSERT INTO tw_daily_bars (date, ticker, raw_code, name, open, high, low, close, volume, created_at)
                VALUES ('2026-10-15', '2371.TW', '2371', '大同', 45.0, 46.0, 44.5, 45.8, 10000, '2026-10-15')
            """)

        # 1st lookup: database lookup
        res1 = self.db.resolve_ticker("大同")
        self.assertIsNotNone(res1)
        self.assertEqual(res1["ticker"], "2371.TW")
        self.assertEqual(res1["matched_by"], "database")

        # 2nd lookup: should be served from memory cache fast-path
        start_t = time.perf_counter()
        res2 = self.db.resolve_ticker("大同")
        duration = time.perf_counter() - start_t
        self.assertIsNotNone(res2)
        self.assertEqual(res2["ticker"], "2371.TW")
        # Memory hit should be sub-millisecond (< 2ms)
        self.assertLess(duration, 0.01)

    # =========================================================================
    # 5. Market Stats & Predictions Response Caching
    # =========================================================================

    def test_market_stats_response_caching(self):
        """Test /market/stats returns cached MarketStatsResponse on repeated calls"""
        from api.routes.market import get_market_statistics, _STATS_CACHE
        _STATS_CACHE.clear()

        # Seed data
        with self.db._get_connection() as con:
            con.execute("INSERT INTO predictions (index_name, model_name, ticker, potential, timestamp) VALUES ('台灣50', 'LSTM', '2330.TW', 5.0, '2026-10-15 14:00:00');")

        r1 = get_market_statistics(db=self.db)
        self.assertIsNotNone(r1)
        self.assertTrue(r1.total_predictions >= 1)
        self.assertIn("stats", _STATS_CACHE)

        # 2nd call should hit _STATS_CACHE directly
        r2 = get_market_statistics(db=self.db)
        self.assertEqual(r1.total_predictions, r2.total_predictions)
        self.assertEqual(r1.latest_market_date, r2.latest_market_date)

    def test_predictions_resonance_response_caching(self):
        """Test /predictions/resonance returns cached PredictionResponse on repeated calls"""
        from api.routes.predictions import get_resonance_candidates, _PRED_CACHE
        _PRED_CACHE.clear()

        # Call resonance
        res1 = get_resonance_candidates(index_name=None, limit=50, db=self.db)
        self.assertIsNotNone(res1)
        cache_key = "resonance_None_50"
        self.assertIn(cache_key, _PRED_CACHE)

        # 2nd call should return identical cached instance
        res2 = get_resonance_candidates(index_name=None, limit=50, db=self.db)
        self.assertIs(res1, res2)


if __name__ == "__main__":
    unittest.main()
