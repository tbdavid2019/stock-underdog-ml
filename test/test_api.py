"""
test/test_api.py - FastAPI REST & DuckDB Endpoints Unit Tests
"""

import os
import unittest
from fastapi.testclient import TestClient
from api.main import app
from data.duckdb_manager import DuckDBManager


class TestFastAPIService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 使用測試 DuckDB 資料庫
        cls.test_db_path = "test/scratch/test_api_quant.duckdb"
        os.makedirs("test/scratch", exist_ok=True)
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

        cls.db = DuckDBManager(db_path=cls.test_db_path)
        
        import datetime
        now_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # 寫入測試模擬數據
        sample_records = [
            {
                "index_name": "台灣50",
                "model_name": "多維共振",
                "strategy_type": "多維共振",
                "ticker": "2330.TW",
                "current_price": 2400.0,
                "predicted_price": 2500.0,
                "potential": 4.17,
                "ma60": 2350.0,
                "pullback_type": "MA60 (+2.1%)",
                "pe": 28.0,
                "pb": 9.5,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "trust_net_5d": 500,
                "foreign_net_5d": 1200,
                "tags": "🏆三重共振 | 土洋合買 | 投信連買4天"
            },
            {
                "index_name": "台灣50",
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": "2454.TW",
                "current_price": 3900.0,
                "predicted_price": 3600.0,
                "potential": -7.69,
                "pe": 60.0,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "tags": "LSTM看跌"
            },
            {
                "index_name": "台灣50",
                "model_name": "玄鐵重劍",
                "strategy_type": "玄鐵重劍",
                "ticker": "1216.TW",
                "current_price": 76.0,
                "ma60": 74.0,
                "pullback_type": "MA60 (+2.7%)",
                "pe": 18.5,
                "period": "6mo",
                "timestamp": now_ts,
                "macro_regime": "全面多頭",
                "tags": "玄鐵買點"
            }
        ]
        cls.db.save_predictions_batch(sample_records)
        
        from api.routes.predictions import get_duckdb as get_duckdb_pred
        from api.routes.stats import get_duckdb as get_duckdb_stats
        app.dependency_overrides[get_duckdb_pred] = lambda: cls.db
        app.dependency_overrides[get_duckdb_stats] = lambda: cls.db
        
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()
        if os.path.exists(cls.test_db_path):
            os.remove(cls.test_db_path)

    def test_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("duckdb_records", data)

    def test_latest_predictions(self):
        resp = self.client.get("/api/v1/predictions/latest?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["count"], 1)
        self.assertIn("batch_date", data)
        self.assertIn("is_stale", data)
        item = data["data"][0]
        self.assertIn("analysis_date", item)
        self.assertIn("data_as_of", item)
        self.assertIn("age_hours", item)

    def test_resonance_endpoint(self):
        resp = self.client.get("/api/v1/predictions/resonance")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_xuantie_endpoint(self):
        resp = self.client.get("/api/v1/predictions/xuantie")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_lstm_top_bullish_and_bearish(self):
        bull_resp = self.client.get("/api/v1/predictions/lstm/top-bullish?limit=5")
        self.assertEqual(bull_resp.status_code, 200)
        
        bear_resp = self.client.get("/api/v1/predictions/lstm/top-bearish?limit=5")
        self.assertEqual(bear_resp.status_code, 200)

    def test_stats_summary(self):
        resp = self.client.get("/api/v1/stats/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("total_records", data)
        self.assertNotIn("db_path", data)

    def test_root_homepage(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("888 Stock Quant", resp.text)

    def test_homepage_has_complete_social_and_search_metadata(self):
        html = self.client.get("/").text
        expected_fragments = [
            '<meta name="description"',
            '<meta property="og:description"',
            '<meta property="og:image"',
            '<meta property="og:url" content="https://stockdata.david888.com/"',
            '<meta name="twitter:card" content="summary_large_image"',
            '<meta name="twitter:description"',
            '<meta name="twitter:image"',
            '<link rel="canonical" href="https://stockdata.david888.com/"',
            '<link rel="apple-touch-icon"',
            '<link rel="icon" type="image/png" sizes="32x32"',
            '<link rel="icon" type="image/svg+xml"',
            '<link rel="manifest" href="/site.webmanifest"',
            '<script type="application/ld+json">',
            '<meta name="theme-color"',
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, html)

    def test_homepage_social_assets_are_served(self):
        assets = {
            "/favicon.svg": "image/svg+xml",
            "/favicon-32x32.png": "image/png",
            "/apple-touch-icon.png": "image/png",
            "/og-image.png": "image/png",
            "/site.webmanifest": "application/manifest+json",
        }
        for path, content_type in assets.items():
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, path)
            self.assertTrue(resp.headers["content-type"].startswith(content_type), path)
            self.assertGreater(len(resp.content), 0, path)

    def test_resolves_company_name_before_history_query(self):
        resp = self.client.get("/api/v1/predictions/resolve/特斯拉")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["ticker"], "TSLA")

    def test_unknown_history_query_does_not_return_records(self):
        resp = self.client.get("/api/v1/predictions/history/這不是股票")
        self.assertEqual(resp.status_code, 404)
        self.assertIn("查無法辨識", resp.json()["detail"])

    def test_ai_plugin_manifest(self):
        resp = self.client.get("/.well-known/ai-plugin.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name_for_model"], "stock_quant_engine")
        self.assertIn("openapi", data["api"]["type"])

    def test_skill_endpoint(self):
        resp = self.client.get("/skill")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("name: stock-quant", resp.text)

    def test_webmcp_manifest(self):
        resp = self.client.get("/.well-known/mcp.json")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["name"], "stock-quant-engine")
        self.assertEqual(data["transport"], "sse")
        self.assertIn("get_market_macro_regime", data["tools"])

        # Test alias /mcp.json
        resp_alias = self.client.get("/mcp.json")
        self.assertEqual(resp_alias.status_code, 200)
        self.assertEqual(resp_alias.json()["name"], "stock-quant-engine")

    def test_llms_txt_endpoints(self):
        resp1 = self.client.get("/llms.txt")
        self.assertEqual(resp1.status_code, 200)
        self.assertIn("888 Stock Quant", resp1.text)
        self.assertIn("/mcp/sse", resp1.text)

        resp2 = self.client.get("/llms-full.txt")
        self.assertEqual(resp2.status_code, 200)
        self.assertIn("888 Stock Quant", resp2.text)

    def test_root_head_method(self):
        resp = self.client.head("/")
        self.assertEqual(resp.status_code, 200)

    def test_cloudflare_webmcp_bridge(self):
        resp = self.client.get("/.webmcp/bridge.js")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Cloudflare & Chrome WebMCP Bridge", resp.text)
        self.assertIn("registerTool", resp.text)


if __name__ == "__main__":
    unittest.main()
