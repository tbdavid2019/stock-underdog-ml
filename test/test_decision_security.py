"""
test/test_decision_security.py - 測試決策模型路由之 IP 解析、授權認證、健康度快取與資訊遮罩
"""
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.decision import router as decision_router
from core.netlimiter import resolve_client_ip, sanitize_health_data, verify_watchlist_write_access


class TestDecisionSecurity(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(decision_router, prefix="/api/v1")
        self.client = TestClient(self.app)

    def test_resolve_client_ip_priority(self):
        """測試真實 IP 依序解析：CF-Connecting-IP -> X-Real-IP -> client.host"""
        mock_req1 = MagicMock()
        mock_req1.headers = {"CF-Connecting-IP": "203.0.113.195, 198.51.100.1"}
        self.assertEqual(resolve_client_ip(mock_req1), "203.0.113.195")

        mock_req2 = MagicMock()
        mock_req2.headers = {"X-Real-IP": "198.51.100.42"}
        self.assertEqual(resolve_client_ip(mock_req2), "198.51.100.42")

        mock_req3 = MagicMock()
        mock_req3.headers = {}
        mock_req3.client.host = "192.168.1.10"
        self.assertEqual(resolve_client_ip(mock_req3), "192.168.1.10")

    def test_sanitize_health_data_masks_urls_and_errors(self):
        """測試健康度輸出遮罩敏感 URL 參數與堆疊錯誤，僅保留 host 與錯誤型別"""
        raw_report = {
            "clef_primary": {
                "status": "unreachable",
                "url": "https://internal-api.example.com:8443/v1/systemone?token=secret123",
                "error": "ConnectionError: HTTPConnectionPool(host='internal-api.example.com', port=8443): Max retries exceeded"
            },
            "clef_aiurl": {
                "status": "healthy",
                "url": "https://aiurl.david888.com/v1/systemone",
                "data": {"status": "ok"}
            }
        }
        sanitized = sanitize_health_data(raw_report)

        self.assertNotIn("url", sanitized["clef_primary"])
        self.assertEqual(sanitized["clef_primary"]["host"], "internal-api.example.com:8443")
        self.assertNotIn("token=secret123", str(sanitized))
        self.assertEqual(sanitized["clef_primary"]["error_type"], "ConnectionError")

        self.assertNotIn("url", sanitized["clef_aiurl"])
        self.assertEqual(sanitized["clef_aiurl"]["host"], "aiurl.david888.com")

    @patch.dict("os.environ", {"DECISION_API_KEY": "secret_decision_key"})
    def test_decision_evaluate_auth_enforcement(self):
        """測試未帶金鑰或金鑰錯誤時回傳 401 Unauthorized"""
        # 未帶金鑰
        resp = self.client.post("/api/v1/decision/evaluate", json={"ticker": "2330.TW"})
        self.assertEqual(resp.status_code, 401)

        # 錯誤金鑰
        resp = self.client.post(
            "/api/v1/decision/evaluate",
            json={"ticker": "2330.TW"},
            headers={"X-API-Key": "wrong_key"}
        )
        self.assertEqual(resp.status_code, 401)

    @patch("api.routes.decision.decision_service.check_health")
    def test_decision_health_caching_and_sanitization(self, mock_check_health):
        """測試 /health 端點 60 秒記憶體快取與輸出脫敏"""
        import api.routes.decision as decision_mod
        decision_mod._HEALTH_CACHE = {}
        decision_mod._HEALTH_CACHE_TIME = 0.0

        mock_check_health.return_value = {
            "tier1": {
                "status": "healthy",
                "url": "https://api.internal.network/v1/systemone",
                "status_code": 200
            }
        }

        # 第一次請求，應呼叫 check_health
        resp1 = self.client.get("/api/v1/decision/health")
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertEqual(data1["tier1"]["host"], "api.internal.network")
        self.assertNotIn("url", data1["tier1"])
        self.assertEqual(mock_check_health.call_count, 1)

        # 第二次立即請求，應由快取回傳，不重複呼叫上游 check_health
        resp2 = self.client.get("/api/v1/decision/health")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(mock_check_health.call_count, 1)


if __name__ == "__main__":
    unittest.main()
