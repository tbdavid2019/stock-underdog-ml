"""
test/test_risk_and_canary_api.py - Unit & Integration tests for Risk X-Ray and Canary API endpoints.
Validates:
1. POST /api/v1/risk/xray bounds, 422 on oversized basket (> 100) or negative weights, 200 on valid.
2. GET /api/v1/watchlist/risk-xray returns 200 with risk xray or graceful fallback status.
3. POST /api/v1/risk/optimize-weights bounds, 422 on > 50 tickers, 200 on valid.
4. GET /api/v1/paper/accounts/{account_id}/risk-xray: regex 422, fail-closed 401 without key, 200 with key.
5. GET /api/v1/stats/canary-health: reads cached snapshot with ZERO external requests.
"""

import json
import os
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app
from backtest.paper import PaperLedger
from core.config import config
from data.duckdb_manager import DuckDBManager


class TestRiskAndCanaryApi(unittest.TestCase):
    TEST_DB = "test/scratch/test_risk_canary_api.duckdb"

    @classmethod
    def setUpClass(cls):
        os.makedirs("test/scratch", exist_ok=True)
        for f in [cls.TEST_DB, f"{cls.TEST_DB}.wal", f"{cls.TEST_DB}.write.lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        cls.db = DuckDBManager(db_path=cls.TEST_DB)

        # Seed sample daily bars for 2330.TW and NVDA
        bars_2330 = []
        for i in range(1, 70):
            bars_2330.append({
                "date": f"2026-05-{i:02d}" if i <= 31 else f"2026-06-{i-31:02d}",
                "ticker": "2330.TW",
                "raw_code": "2330",
                "name": "台積電",
                "open": 900.0 + i,
                "high": 910.0 + i,
                "low": 895.0 + i,
                "close": 905.0 + i,
                "volume": 20000 + i * 100,
                "trade_value": 10000000.0,
                "transaction_count": 500,
                "market": "TWSE",
                "created_at": "2026-06-01T00:00:00",
            })
        cls.db.save_daily_bars_batch(bars_2330)

        bars_nvda = []
        for i in range(1, 70):
            bars_nvda.append({
                "ticker": "NVDA",
                "market": "US",
                "currency": "USD",
                "date": f"2026-05-{i:02d}" if i <= 31 else f"2026-06-{i-31:02d}",
                "open": 130.0 + i,
                "high": 135.0 + i,
                "low": 128.0 + i,
                "close": 132.0 + i,
                "volume": 1000000 + i * 500,
                "is_adjusted": True,
                "updated_at": "2026-06-01 00:00:00",
            })
        cls.db.upsert_global_daily_bars(bars_nvda)

        # Create paper account in test database
        cls.ledger = PaperLedger(cls.TEST_DB, read_only=False)
        cls.ledger.create_account("acc_test_risk", market="TW", currency="TWD", cash=1000000.0)

        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        for f in [cls.TEST_DB, f"{cls.TEST_DB}.wal", f"{cls.TEST_DB}.write.lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_post_risk_xray_oversized_basket_rejected(self):
        # 101 tickers -> 422
        oversized = [f"T_{i}" for i in range(101)]
        resp = self.client.post("/api/v1/risk/xray", json={"tickers": oversized, "window": 60})
        self.assertEqual(resp.status_code, 422)

    def test_post_risk_xray_negative_weights_rejected(self):
        resp = self.client.post("/api/v1/risk/xray", json={
            "tickers": ["2330.TW", "NVDA"],
            "weights": {"2330.TW": 0.8, "NVDA": -0.2},
            "window": 60
        })
        self.assertEqual(resp.status_code, 422)

    def test_post_risk_xray_valid(self):
        with patch("api.routes.risk.DuckDBManager", return_value=self.db):
            resp = self.client.post("/api/v1/risk/xray", json={
                "tickers": ["2330.TW", "NVDA"],
                "window": 60
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertIn("annualized_volatility_total", data)
            self.assertIn("euler_contributions", data)
            self.assertEqual(data["currency_mode"], "local_currency_return_proxy")

    def test_get_watchlist_risk_xray(self):
        with patch("api.routes.risk.DuckDBManager", return_value=self.db):
            resp = self.client.get("/api/v1/watchlist/risk-xray")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("status", data)

    def test_post_optimize_weights_bounds(self):
        # 51 tickers -> 422
        oversized = [f"T_{i}" for i in range(51)]
        resp = self.client.post("/api/v1/risk/optimize-weights", json={"tickers": oversized})
        self.assertEqual(resp.status_code, 422)

    def test_post_optimize_weights_valid(self):
        with patch("api.routes.risk.DuckDBManager", return_value=self.db):
            resp = self.client.post("/api/v1/risk/optimize-weights", json={
                "tickers": ["2330.TW", "NVDA"],
                "gamma_buy": 0.0015,
                "gamma_sell": 0.0045,
                "window": 60
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("weights", data)
            self.assertIn("one_way_turnover", data)

    def test_paper_account_risk_xray_auth_and_regex(self):
        # 1. Invalid account format -> 422
        resp_bad_id = self.client.get("/api/v1/paper/accounts/bad!account@id/risk-xray")
        self.assertEqual(resp_bad_id.status_code, 422)

        # 2. In production mode with unauthenticated request -> 401 Fail-Closed
        with patch.dict(os.environ, {"DEPLOYMENT_MODE": "production", "WATCHLIST_API_KEY": "super_secret_test_key_xyz"}):
            resp_no_auth = self.client.get("/api/v1/paper/accounts/acc_test_risk/risk-xray")
            self.assertEqual(resp_no_auth.status_code, 401)

            # 3. Valid key -> 200
            headers = {"X-API-Key": "super_secret_test_key_xyz"}
            with patch("core.config.config.db.DUCKDB_PATH", self.TEST_DB):
                resp_auth = self.client.get("/api/v1/paper/accounts/acc_test_risk/risk-xray", headers=headers)
                self.assertEqual(resp_auth.status_code, 200)
                data = resp_auth.json()
                self.assertEqual(data["account_id"], "acc_test_risk")

                # 4. Nonexistent account with valid key -> 404
                resp_not_found = self.client.get("/api/v1/paper/accounts/nonexistent_acc/risk-xray", headers=headers)
                self.assertEqual(resp_not_found.status_code, 404)

    def test_post_optimize_weights_duplicate_tickers_deduplicated(self):
        with patch("api.routes.risk.DuckDBManager", return_value=self.db):
            resp = self.client.post("/api/v1/risk/optimize-weights", json={
                "tickers": ["2330.TW", "2330.TW", "NVDA", "2330.TW"],
                "gamma_buy": 0.0015,
                "gamma_sell": 0.0045,
                "window": 60
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            weights = data["weights"]
            self.assertEqual(len(weights), 2)
            self.assertIn("2330.TW", weights)
            self.assertIn("NVDA", weights)
            self.assertTrue(sum(weights.values()) <= 1.0 + 1e-6)

    def test_stats_canary_health_cached_zero_network(self):
        with patch("requests.get") as mock_req:
            resp = self.client.get("/api/v1/stats/canary-health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("status", data)
            # Must NOT trigger live network request!
            mock_req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
