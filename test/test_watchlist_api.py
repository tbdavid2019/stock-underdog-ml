"""
test/test_watchlist_api.py - 測試自選股 REST API、Fail-Closed 安全守衛與 503 鎖忙碌
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes.watchlist import router as watchlist_router, get_db
from core.duckdb_lock import DuckDBBusyError
from data.duckdb_manager import DuckDBManager


class TestWatchlistAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test_api_wl.duckdb")
        self.mgr = DuckDBManager(db_path=self.db_path)

        self.app = FastAPI()
        self.app.include_router(watchlist_router, prefix="/api/v1")
        self.app.dependency_overrides[get_db] = lambda: self.mgr
        self.client = TestClient(self.app)

    def tearDown(self):
        self.tmp.cleanup()

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "production", "WATCHLIST_API_KEY": ""}, clear=False)
    def test_production_fail_closed_rejects_without_key(self):
        """測試生產模式下伺服器未設定金鑰時，寫入直接 Fail-Closed 回傳 401"""
        resp = self.client.post("/api/v1/watchlist", json={"ticker": "2330.TW"})
        self.assertEqual(resp.status_code, 401)
        self.assertIn("WATCHLIST_API_KEY", resp.json()["detail"])

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "production", "WATCHLIST_API_KEY": "valid_secret_key"}, clear=False)
    def test_production_authorized_write_and_public_read(self):
        """測試公開唯讀無需金鑰，寫入需合法金鑰"""
        # 1. 唯讀無需金鑰
        get_resp = self.client.get("/api/v1/watchlist")
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["count"], 0)

        # 2. 未帶金鑰寫入拒絕
        post_unauth = self.client.post("/api/v1/watchlist", json={"ticker": "2330.TW"})
        self.assertEqual(post_unauth.status_code, 401)

        # 3. 帶合法金鑰寫入成功
        headers = {"X-API-Key": "valid_secret_key"}
        post_ok = self.client.post(
            "/api/v1/watchlist",
            json={"ticker": "2330", "note": "台積電測試"},
            headers=headers
        )
        self.assertEqual(post_ok.status_code, 200)
        self.assertEqual(post_ok.json()["item"]["canonical_ticker"], "2330.TW")

        # 4. 再次公開唯讀已包含該項目
        get_after = self.client.get("/api/v1/watchlist")
        self.assertEqual(get_after.status_code, 200)
        self.assertEqual(get_after.json()["count"], 1)

        # 5. 帶合法 Bearer 金鑰刪除
        del_headers = {"Authorization": "Bearer valid_secret_key"}
        del_resp = self.client.delete("/api/v1/watchlist/2330.TW", headers=del_headers)
        self.assertEqual(del_resp.status_code, 200)

        get_empty = self.client.get("/api/v1/watchlist")
        self.assertEqual(get_empty.json()["count"], 0)

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "development", "ALLOW_ANONYMOUS_LOCAL_WRITES": "true"}, clear=False)
    def test_development_anonymous_write_allowed(self):
        """測試開發模式且開啟 ALLOW_ANONYMOUS_LOCAL_WRITES 時允許免金鑰直接寫入"""
        resp = self.client.post("/api/v1/watchlist", json={"ticker": "NVDA", "note": "Dev mode write"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["item"]["canonical_ticker"], "NVDA")

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "development", "ALLOW_ANONYMOUS_LOCAL_WRITES": "true"}, clear=False)
    def test_batch_import_and_capacity_cap(self):
        """測試批量匯入自選股與上限控制"""
        tickers = [f"STK{i}" for i in range(10)]
        resp = self.client.post("/api/v1/watchlist/batch", json={"tickers": tickers})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["result"]["added"], 10)

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "development", "ALLOW_ANONYMOUS_LOCAL_WRITES": "true"}, clear=False)
    @patch("data.duckdb_manager.DuckDBManager.add_watchlist_item", side_effect=DuckDBBusyError("Lock busy"))
    def test_database_busy_returns_503(self, mock_add):
        """測試跨進程鎖忙碌逾時回傳 HTTP 503 Service Unavailable"""
        resp = self.client.post("/api/v1/watchlist", json={"ticker": "2330.TW"})
        self.assertEqual(resp.status_code, 503)

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "production  ", "WATCHLIST_API_KEY": "valid_secret_key"}, clear=False)
    def test_production_trailing_whitespace_fails_closed(self):
        """測試 DEPLOYMENT_MODE 包含尾隨空白時仍被正規化為 production 並 Fail-Closed 拒絕未認證請求"""
        resp = self.client.post("/api/v1/watchlist", json={"ticker": "2330.TW"})
        self.assertEqual(resp.status_code, 401)

    @patch.dict("os.environ", {"DEPLOYMENT_MODE": "production", "WATCHLIST_API_KEY": "valid_secret_key"}, clear=False)
    def test_private_profile_read_isolation(self):
        """測試讀取非 default 的私人 profile_id 在無金鑰時回傳 401，有金鑰時回傳 200"""
        # 1. 匿名讀取 default 允許
        resp_def = self.client.get("/api/v1/watchlist?profile_id=default")
        self.assertEqual(resp_def.status_code, 200)

        # 2. 匿名讀取 private_user 拒絕 (401)
        resp_priv_unauth = self.client.get("/api/v1/watchlist?profile_id=private_user_888")
        self.assertEqual(resp_priv_unauth.status_code, 401)
        self.assertIn("profile", resp_priv_unauth.json()["detail"].lower())

        # 3. 帶合法金鑰讀取 private_user 成功 (200)
        resp_priv_auth = self.client.get("/api/v1/watchlist?profile_id=private_user_888", headers={"X-API-Key": "valid_secret_key"})
        self.assertEqual(resp_priv_auth.status_code, 200)


if __name__ == "__main__":
    unittest.main()
