"""
test/test_api_decision.py - Test FastAPI Decision API routes
"""

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import app


class TestDecisionAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    @patch("data.clef_client.ClefDecisionClient.check_health")
    def test_decision_health(self, mock_health):
        mock_health.return_value = {
            "Clef-Create360": {"status": "healthy", "url": "https://clef.create360.ai/health"},
            "Clef-Aiurl": {"status": "healthy", "url": "https://clef.aiurl.tw/health"},
            "Clef-Jev": {"status": "healthy", "url": "https://clef.create360.ai/health"}
        }

        resp = self.client.get("/api/v1/decision/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertTrue(data["any_available"])
        self.assertIn("Clef-Create360", data["endpoints"])

    @patch("data.clef_client.ClefDecisionClient.evaluate_stock")
    def test_decision_evaluate_stock(self, mock_eval):
        mock_verdict = MagicMock()
        mock_verdict.success = True
        mock_verdict.to_dict.return_value = {
            "ticker": "2330.TW",
            "action_decision": "strong_buy",
            "action_confidence": 0.88,
            "conviction_score": 4.1,
            "is_favorable_entry": 0.75,
            "tag": "🤖AI強買(88%)",
            "source_slot": "Clef-Create360",
            "success": True,
            "error": None
        }
        mock_eval.return_value = mock_verdict

        payload = {
            "ticker": "2330.TW",
            "candidate": {"current_price": 1000.0, "composite_score": 85.0},
            "macro_regime": "bull"
        }
        resp = self.client.post("/api/v1/decision/evaluate", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["ticker"], "2330.TW")
        self.assertEqual(data["verdict"]["action_decision"], "strong_buy")
        # 驗證 candidate_data 確實同步了頂層 ticker
        call_kwargs = mock_eval.call_args[1]
        self.assertEqual(call_kwargs["candidate"]["ticker"], "2330.TW")

    @patch("data.clef_client.ClefDecisionClient.evaluate_stock")
    def test_decision_evaluate_custom_state(self, mock_eval):
        mock_verdict = MagicMock()
        mock_verdict.success = True
        mock_verdict.ticker = "PORTFOLIO_STATE"
        mock_verdict.to_dict.return_value = {
            "ticker": "PORTFOLIO_STATE",
            "action_decision": "buy",
            "action_confidence": 0.75,
            "conviction_score": 3.8,
            "is_favorable_entry": 0.65,
            "tag": "🤖AI看好(75%)",
            "source_slot": "Clef-Aiurl",
            "success": True,
            "error": None
        }
        mock_eval.return_value = mock_verdict

        custom_state = {"arbitrary_key": "val", "vix": 14.5}
        resp = self.client.post("/api/v1/decision/evaluate", json={"state": custom_state, "macro_regime": "bull"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        mock_eval.assert_called_once_with(raw_state=custom_state, macro_regime="bull")

    @patch.dict("os.environ", {"DECISION_API_KEY": "secret_test_key_123"})
    def test_decision_evaluate_api_key_auth(self):
        """驗證當設定 API Key 時，未授權請求回傳 401，正確 Header 則通過"""
        payload = {"ticker": "2330.TW", "candidate": {"current_price": 1000.0, "score": 85.0}}

        # 無 Key -> 401
        resp = self.client.post("/api/v1/decision/evaluate", json=payload)
        self.assertEqual(resp.status_code, 401)

        # 錯誤 Key -> 401
        resp = self.client.post(
            "/api/v1/decision/evaluate",
            json=payload,
            headers={"X-API-Key": "wrong_key"}
        )
        self.assertEqual(resp.status_code, 401)

        # 正確 Key (Header X-API-Key) -> 通過 (200)
        resp = self.client.post(
            "/api/v1/decision/evaluate",
            json=payload,
            headers={"X-API-Key": "secret_test_key_123"}
        )
        self.assertEqual(resp.status_code, 200)

        # 正確 Key (Bearer Token) -> 通過 (200)
        resp = self.client.post(
            "/api/v1/decision/evaluate",
            json=payload,
            headers={"Authorization": "Bearer secret_test_key_123"}
        )
        self.assertEqual(resp.status_code, 200)

    @patch("api.routes.decision.DEFAULT_RATE_LIMIT_PER_MIN", 2)
    def test_decision_evaluate_rate_limit(self):
        """驗證超出滑動視窗速率限制時回傳 429"""
        from api.routes.decision import _RATE_LIMIT_STORE
        _RATE_LIMIT_STORE.clear()

        payload = {"ticker": "2330.TW", "candidate": {"current_price": 1000.0, "score": 85.0}}
        resp1 = self.client.post("/api/v1/decision/evaluate", json=payload)
        self.assertEqual(resp1.status_code, 200)

        resp2 = self.client.post("/api/v1/decision/evaluate", json=payload)
        self.assertEqual(resp2.status_code, 200)

        # 第三次超過每分鐘 2 次限制 -> 429
        resp3 = self.client.post("/api/v1/decision/evaluate", json=payload)
        self.assertEqual(resp3.status_code, 429)
        self.assertIn("Rate limit exceeded", resp3.json()["detail"])
        _RATE_LIMIT_STORE.clear()

    def test_prune_rate_limit_store_bounds_memory(self):
        """驗證當儲存過多 IP 紀錄時，過期 IP 會被自動清除，確保記憶體受控"""
        import time
        from api.routes.decision import _RATE_LIMIT_STORE, _prune_rate_limit_store
        _RATE_LIMIT_STORE.clear()

        now = time.time()
        # 模擬 2500 個過期 IP 條目
        for i in range(2500):
            _RATE_LIMIT_STORE[f"10.0.{i // 256}.{i % 256}"] = [now - 120]  # 2 分鐘前，已過期

        # 模擬 5 個活躍 IP
        for i in range(5):
            _RATE_LIMIT_STORE[f"192.168.1.{i}"] = [now - 10]

        _prune_rate_limit_store(now)
        # 過期的 2500 個條目應全數被修剪，僅保留活躍 IP
        self.assertEqual(len(_RATE_LIMIT_STORE), 5)
        _RATE_LIMIT_STORE.clear()

    @patch("api.routes.decision.db.get_latest_candidate_snapshot", return_value=None)
    @patch("api.routes.decision.db.get_resonance_candidates", return_value=[])
    def test_decision_evaluate_ticker_not_found_returns_404(self, mock_res, mock_snap):
        """驗證當資料庫完全沒有該 ticker 且未提供候選字典時，回傳 404 而非使用假資料"""
        payload = {"ticker": "NON_EXISTENT_9999.TW"}
        resp = self.client.post("/api/v1/decision/evaluate", json=payload)
        self.assertEqual(resp.status_code, 404)
        self.assertIn("No quantitative records found", resp.json()["detail"])

    def test_mcp_tool_decision_auth_and_lookup(self):
        """驗證 MCP 工具 get_clef_stock_verdict 具備相同授權與依 ticker 查詢之安全機制"""
        import os
        from mcp_server import get_clef_stock_verdict

        # 1. 測試當設定 API Key 時，未授權 MCP 呼叫會被阻擋
        with patch.dict(os.environ, {"DECISION_API_KEY": "mcp_secret_key"}):
            res_no_key = get_clef_stock_verdict(ticker="2330.TW")
            self.assertFalse(res_no_key["success"])
            self.assertIn("Unauthorized", res_no_key["error"])

            res_wrong_key = get_clef_stock_verdict(ticker="2330.TW", api_key="bad_key")
            self.assertFalse(res_wrong_key["success"])
            self.assertIn("Unauthorized", res_wrong_key["error"])

        # 2. 測試當 ticker 在 DuckDB 查無資料時，回傳明確錯誤而非虛構假資料
        with patch("mcp_server.db.get_latest_candidate_snapshot", return_value=None), \
             patch("mcp_server.db.get_resonance_candidates", return_value=[]):
            res_not_found = get_clef_stock_verdict(ticker="UNKNOWN_TICKER.TW")
            self.assertFalse(res_not_found["success"])
            self.assertIn("未在量化資料庫中找到標的", res_not_found["error"])


if __name__ == "__main__":
    unittest.main()
