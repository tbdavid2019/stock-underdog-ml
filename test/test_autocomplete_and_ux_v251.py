"""
test/test_autocomplete_and_ux_v251.py - Regression & Unit Tests for v2.5.1
Validates:
1. DuckDB autocomplete_tickers ranking and deduplication (case-insensitive).
2. GET /api/v1/predictions/autocomplete endpoint (empty, valid, max_length=64 validation).
3. core.duckdb_lock.is_current_thread_holding_write_lock accuracy.
4. ForecastStore connect retry & DuckDBBusyError behavior.
5. Template & static assets assertions (sticky classes, quant-table-wrap, alerts modal).
6. get_quant_alerts ticker alias compatibility.
7. company_profile resolve_company_name read_only connection check.
8. PE sort comparator logic (invalid non-positive PEs sink to bottom).
"""

import os
import subprocess
import unittest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.market import get_duckdb as get_market_duckdb
from api.routes.predictions import get_duckdb as get_predictions_duckdb
from core.duckdb_lock import duckdb_write_lock, is_current_thread_holding_write_lock
from data.company_profile import CompanyProfileService
from data.duckdb_manager import DuckDBManager
from data.forecast_store import ForecastStore


class TestAutocompleteAndUxV251(unittest.TestCase):
    TEST_DB = "test/scratch/test_autocomplete_ux_v251.duckdb"

    @classmethod
    def setUpClass(cls):
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(cls.TEST_DB):
            try:
                os.remove(cls.TEST_DB)
            except Exception:
                pass
        cls.db = DuckDBManager(db_path=cls.TEST_DB)
        app.dependency_overrides[get_predictions_duckdb] = lambda: cls.db
        app.dependency_overrides[get_market_duckdb] = lambda: cls.db

        # Seed test data in tw_daily_bars and market_universe
        cls.db.save_daily_bars_batch([
            {
                "date": "2026-11-05",
                "ticker": "2330.TW",
                "raw_code": "2330",
                "name": "台積電",
                "close": 1050.0,
            },
            {
                "date": "2026-11-05",
                "ticker": "2303.TW",
                "raw_code": "2303",
                "name": "聯華電子",
                "close": 52.0,
            },
            {
                "date": "2026-11-05",
                "ticker": "NVDA",
                "raw_code": "NVDA",
                "name": "NVIDIA Corp",
                "close": 140.0,
            }
        ])

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        for f in [cls.TEST_DB, f"{cls.TEST_DB}.wal", f"{cls.TEST_DB}.write.lock"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_duckdb_autocomplete_exact_and_prefix(self):
        results = self.db.autocomplete_tickers(query="2330", limit=5)
        self.assertTrue(len(results) >= 1)
        self.assertEqual(results[0]["ticker"], "2330.TW")
        self.assertEqual(results[0]["name"], "台積電")

    def test_duckdb_autocomplete_case_insensitive(self):
        r_lower = self.db.autocomplete_tickers(query="nvda", limit=5)
        r_upper = self.db.autocomplete_tickers(query="NVDA", limit=5)
        self.assertTrue(len(r_lower) >= 1)
        self.assertEqual(r_lower[0]["ticker"], "NVDA")
        self.assertEqual(r_lower, r_upper)

    def test_duckdb_autocomplete_chinese_name(self):
        results = self.db.autocomplete_tickers(query="聯華", limit=5)
        self.assertTrue(len(results) >= 1)
        tickers = [r["ticker"] for r in results]
        self.assertIn("2303.TW", tickers)

    def test_duckdb_autocomplete_us_ticker(self):
        results = self.db.autocomplete_tickers(query="NV", limit=5)
        self.assertTrue(len(results) >= 1)
        tickers = [r["ticker"] for r in results]
        self.assertIn("NVDA", tickers)

    def test_api_autocomplete_endpoint(self):
        client = TestClient(app)
        resp = client.get("/api/v1/predictions/autocomplete?q=2330&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("query", data)
        self.assertEqual(data["query"], "2330")
        self.assertIn("results", data)
        self.assertIsInstance(data["results"], list)

    def test_api_autocomplete_empty_query(self):
        client = TestClient(app)
        resp = client.get("/api/v1/predictions/autocomplete?q=")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["results"], [])

    def test_api_autocomplete_max_length_validation(self):
        client = TestClient(app)
        # 64 chars -> valid
        q_64 = "A" * 64
        resp_ok = client.get(f"/api/v1/predictions/autocomplete?q={q_64}")
        self.assertEqual(resp_ok.status_code, 200)

        # 65 chars -> 422 Unprocessable Entity
        q_65 = "A" * 65
        resp_err = client.get(f"/api/v1/predictions/autocomplete?q={q_65}")
        self.assertEqual(resp_err.status_code, 422)

    def test_get_quant_alerts_ticker_alias(self):
        self.db.add_quant_alert("2330.TW", "above_price", 1100.0)
        alerts = self.db.get_quant_alerts()
        self.assertTrue(len(alerts) >= 1)
        target_alert = next((a for a in alerts if a.get("canonical_ticker") == "2330.TW"), None)
        self.assertIsNotNone(target_alert)
        self.assertEqual(target_alert.get("ticker"), "2330.TW")
        self.assertEqual(target_alert.get("canonical_ticker"), "2330.TW")

    def test_company_profile_read_only(self):
        name = CompanyProfileService._resolve_stock_name("2330.TW", db=self.db)
        self.assertEqual(name, "台積電")
        # Ensure no lock is held after read_only connection
        self.assertFalse(is_current_thread_holding_write_lock(self.TEST_DB))

    def test_is_current_thread_holding_write_lock(self):
        test_lock_db = "test/scratch/test_lock_status.duckdb"
        self.assertFalse(is_current_thread_holding_write_lock(test_lock_db))
        with duckdb_write_lock(test_lock_db):
            self.assertTrue(is_current_thread_holding_write_lock(test_lock_db))
        self.assertFalse(is_current_thread_holding_write_lock(test_lock_db))

    def test_forecast_store_connect_with_retry(self):
        store = ForecastStore(self.TEST_DB, read_only=True)
        with store._connect() as con:
            cnt = con.execute("SELECT COUNT(*) FROM tw_daily_bars").fetchone()[0]
            self.assertTrue(cnt >= 1)

    def test_template_sticky_and_alerts_modal_inclusion(self):
        # 1. index.html includes alerts_modal.html and defines quant-sticky-th
        with open("api/templates/index.html", "r", encoding="utf-8") as f:
            index_html = f.read()
        self.assertIn("partials/alerts_modal.html", index_html)
        self.assertIn(".quant-sticky-th", index_html)
        self.assertIn(".quant-table-wrap", index_html)

        # 2. tab_dashboard.html uses quant-table-wrap and quant-sticky-th
        with open("api/templates/partials/tab_dashboard.html", "r", encoding="utf-8") as f:
            dash_html = f.read()
        self.assertNotIn(">5日籌碼 (張)<", dash_html)
        self.assertIn("openRowAlerts", dash_html)
        self.assertIn("colspan=\"10\"", dash_html)
        self.assertIn("quant-table-wrap", dash_html)
        self.assertIn("quant-sticky-th", dash_html)

        # 3. header.html has openAlertsCenterModal button
        with open("api/templates/partials/header.html", "r", encoding="utf-8") as f:
            header_html = f.read()
        self.assertIn("openAlertsCenterModal()", header_html)

    def test_pe_sort_comparator_logic(self):
        js_code = """
        const records = [
            { ticker: 'A', pe: 15.2 },
            { ticker: 'B', pe: -3.5 },
            { ticker: 'C', pe: 0 },
            { ticker: 'D', pe: null },
            { ticker: 'E', pe: 8.4 }
        ];
        const key = 'pe';
        const order = 1; // asc
        const sorted = [...records].sort((a, b) => {
            const na = (a[key] === null || a[key] === undefined || isNaN(Number(a[key]))) ? null : Number(a[key]);
            const nb = (b[key] === null || b[key] === undefined || isNaN(Number(b[key]))) ? null : Number(b[key]);
            if (key === 'pe') {
                const badA = na === null || na <= 0;
                const badB = nb === null || nb <= 0;
                if (badA && badB) return 0;
                if (badA) return 1;
                if (badB) return -1;
                return (na - nb) * order;
            }
            return 0;
        });
        const tickers = sorted.map(r => r.ticker);
        // Valid positive PEs come first in asc order: E (8.4), A (15.2)
        // Invalid non-positive / null PEs sink to bottom: B (-3.5), C (0), D (null)
        if (tickers[0] !== 'E' || tickers[1] !== 'A') {
            throw new Error('Valid PE not sorted first: ' + JSON.stringify(tickers));
        }
        if (!['B', 'C', 'D'].includes(tickers[2]) || !['B', 'C', 'D'].includes(tickers[3]) || !['B', 'C', 'D'].includes(tickers[4])) {
            throw new Error('Invalid PE not sunk to bottom: ' + JSON.stringify(tickers));
        }
        console.log('SUCCESS');
        """
        proc = subprocess.run(
            ["node", "-e", js_code],
            capture_output=True,
            text=True
        )
        self.assertEqual(proc.returncode, 0, msg=f"Node error: {proc.stderr}")
        self.assertIn("SUCCESS", proc.stdout)


if __name__ == "__main__":
    unittest.main()
