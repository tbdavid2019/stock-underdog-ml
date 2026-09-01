"""
test/test_api_market.py - Unit tests for market & institutional flow API routes.
"""

import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestMarketAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_market_stats(self):
        resp = self.client.get("/api/v1/market/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_bars", data)
        self.assertIn("total_institutional_records", data)
        self.assertIn("total_predictions", data)

    def test_get_top_institutional_flows(self):
        resp = self.client.get("/api/v1/market/institutional/top?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

    def test_get_ticker_institutional_history(self):
        resp = self.client.get("/api/v1/market/institutional/2330.TW?limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
