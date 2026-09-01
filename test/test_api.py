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
                "timestamp": "2026-09-01T10:00:00",
                "macro_regime": "全面多頭",
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
                "timestamp": "2026-09-01T10:00:00",
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
                "timestamp": "2026-09-01T10:00:00",
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


if __name__ == "__main__":
    unittest.main()
