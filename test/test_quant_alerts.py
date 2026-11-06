"""
test/test_quant_alerts.py
Unit tests for quant price alerts CRUD, dynamic P10 validation,
Fail-Closed write protection, and atomic evaluator triggering.
"""

import unittest
import tempfile
import os
import shutil
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from fastapi.testclient import TestClient

from data.duckdb_manager import DuckDBManager
from data.alert_evaluator import evaluate_and_dispatch_alerts, dispatch_alert_notification
from core.duckdb_lock import DuckDBBusyError
from api.main import app
from api.routes.watchlist import get_db


class TestQuantAlerts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_alerts.duckdb")
        self.db = DuckDBManager(db_path=self.db_path)
        app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(app)

        self.api_key = "test_watchlist_key_abc123"
        self.orig_env_key = os.environ.get("WATCHLIST_API_KEY")
        self.orig_env_mode = os.environ.get("DEPLOYMENT_MODE")
        os.environ["WATCHLIST_API_KEY"] = self.api_key
        os.environ["DEPLOYMENT_MODE"] = "production"

        # Seed sample predictions for 2330.TW (with P10 and P50) and 9999.TW (without P10)
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            con.execute("""
                INSERT INTO predictions (
                    index_name, model_name, strategy_type, ticker, current_price, predicted_price,
                    potential, timestral_p10, moirai_p10, ma60, ma120, timestamp
                ) VALUES
                ('TW0050', 'ensemble', 'resonance', '2330.TW', 1000.0, 1080.0, 8.0, 950.0, 960.0, 980.0, 920.0, ?),
                ('TW0050', 'ensemble', 'resonance', '9999.TW', 50.0, 52.0, 4.0, NULL, NULL, NULL, NULL, ?)
            """, [now_iso, now_iso])

    def tearDown(self):
        app.dependency_overrides.pop(get_db, None)
        if self.orig_env_key is not None:
            os.environ["WATCHLIST_API_KEY"] = self.orig_env_key
        else:
            os.environ.pop("WATCHLIST_API_KEY", None)

        if self.orig_env_mode is not None:
            os.environ["DEPLOYMENT_MODE"] = self.orig_env_mode
        else:
            os.environ.pop("DEPLOYMENT_MODE", None)

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_static_alert_validation(self):
        """Static alert requires positive threshold_value."""
        # Missing threshold
        resp = self.client.post("/api/v1/alerts", json={
            "ticker": "2330.TW",
            "rule_type": "above_price"
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 422)

        # Valid threshold
        resp = self.client.post("/api/v1/alerts", json={
            "ticker": "2330.TW",
            "rule_type": "above_price",
            "threshold_value": 1050.0
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["alert"]["canonical_ticker"], "2330.TW")
        self.assertEqual(data["alert"]["rule_type"], "above_price")
        self.assertEqual(data["alert"]["threshold_value"], 1050.0)

    def test_create_dynamic_p10_alert_rejection(self):
        """Creating below_p10 alert for ticker without P10 in predictions must fail with 422."""
        resp = self.client.post("/api/v1/alerts", json={
            "ticker": "9999.TW",
            "rule_type": "below_p10"
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(resp.status_code, 422)
        self.assertIn("P10", resp.json()["detail"])

        # For 2330.TW which has P10, it must succeed
        resp_ok = self.client.post("/api/v1/alerts", json={
            "ticker": "2330.TW",
            "rule_type": "below_p10"
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(resp_ok.status_code, 200)

    def test_production_write_auth_fail_closed(self):
        """In production mode, write and check endpoints reject requests without valid WATCHLIST_API_KEY."""
        # 1. No key -> 401
        resp = self.client.post("/api/v1/alerts", json={
            "ticker": "2330.TW",
            "rule_type": "above_price",
            "threshold_value": 1100.0
        })
        self.assertEqual(resp.status_code, 401)

        # 2. Wrong key -> 401
        resp_wrong = self.client.post("/api/v1/alerts", json={
            "ticker": "2330.TW",
            "rule_type": "above_price",
            "threshold_value": 1100.0
        }, headers={"X-API-Key": "wrong_key"})
        self.assertEqual(resp_wrong.status_code, 401)

        # 3. Correct key -> 200
        resp_ok = self.client.post("/api/v1/alerts", json={
            "ticker": "2330.TW",
            "rule_type": "above_price",
            "threshold_value": 1100.0
        }, headers={"X-API-Key": self.api_key})
        self.assertEqual(resp_ok.status_code, 200)

        # 4. Check endpoint with correct key
        resp_check = self.client.post("/api/v1/alerts/check", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp_check.status_code, 200)

    def test_evaluator_triggers_and_respects_cooldown(self):
        """Evaluator detects condition, updates DB atomically, and respects cooldown window."""
        # 2330.TW current price is 1000.0. Create alert above_price with threshold 980.0 (triggered)
        alert = self.db.add_quant_alert(
            canonical_ticker="2330.TW",
            rule_type="above_price",
            threshold_value=980.0,
            cooldown_minutes=60
        )
        alert_id = alert["alert_id"]

        with patch("data.alert_evaluator.dispatch_alert_notification") as mock_dispatch:
            mock_dispatch.return_value = True

            # First run: should trigger
            res1 = evaluate_and_dispatch_alerts(db=self.db)
            self.assertEqual(res1["status"], "success")
            self.assertEqual(res1["triggered"], 1)
            self.assertEqual(len(res1["alerts"]), 1)
            self.assertEqual(res1["alerts"][0]["alert_id"], alert_id)
            mock_dispatch.assert_called_once()

            # Verify DB was updated
            alerts_in_db = self.db.get_quant_alerts()
            self.assertIsNotNone(alerts_in_db[0]["last_triggered_at"])
            self.assertEqual(alerts_in_db[0]["last_triggered_price"], 1000.0)

            # Second run immediately: should be in cooldown, not triggered
            mock_dispatch.reset_mock()
            res2 = evaluate_and_dispatch_alerts(db=self.db)
            self.assertEqual(res2["status"], "success")
            self.assertEqual(res2["triggered"], 0)
            mock_dispatch.assert_not_called()

    def test_delete_alert(self):
        """Deleting an alert succeeds and subsequent queries don't return it."""
        alert = self.db.add_quant_alert(
            canonical_ticker="2330.TW",
            rule_type="above_price",
            threshold_value=1200.0
        )
        alert_id = alert["alert_id"]

        del_resp = self.client.delete(f"/api/v1/alerts/{alert_id}", headers={"X-API-Key": self.api_key})
        self.assertEqual(del_resp.status_code, 200)

        # Deleting again returns 404
        del_again = self.client.delete(f"/api/v1/alerts/{alert_id}", headers={"X-API-Key": self.api_key})
        self.assertEqual(del_again.status_code, 404)

    def test_multi_model_above_p50_alert(self):
        """測試跨模型 above_p50 告警觸發與 reference_model 映射"""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.db._write_lock(), self.db._get_connection(read_only=False) as con:
            con.execute("""
                INSERT INTO predictions (
                    index_name, model_name, strategy_type, ticker, current_price, predicted_price,
                    potential, timestamp
                ) VALUES
                ('TW0050', 'Moirai', 'moirai_pred', '2330.TW', 1080.0, 1070.0, 7.0, ?)
            """, [now_iso])

        # Create alert for above_p50 with reference_model=moirai
        alert = self.db.add_quant_alert(
            canonical_ticker="2330.TW",
            rule_type="above_p50",
            reference_model="moirai",
            cooldown_minutes=60
        )
        self.assertEqual(alert["reference_model"], "moirai")

        with patch("data.alert_evaluator.dispatch_alert_notification") as mock_dispatch:
            mock_dispatch.return_value = True
            res = evaluate_and_dispatch_alerts(db=self.db)
            self.assertEqual(res["triggered"], 1)
            self.assertEqual(res["alerts"][0]["rule_type"], "above_p50")
            self.assertEqual(res["alerts"][0]["reference_model"], "moirai")

    def test_html_escaping_in_notification(self):
        """測試通知發送前對 ticker 與 note 進行嚴格 HTML escaping 杜絕 XSS"""
        from core.config import TelegramConfig, DiscordConfig, EmailConfig
        with patch.object(TelegramConfig, "BOT_TOKEN", "mock_bot_token"), \
             patch.object(TelegramConfig, "CHANNEL_ID", "mock_channel_id"), \
             patch.object(DiscordConfig, "WEBHOOK_URL", None), \
             patch.object(EmailConfig, "SENDER_EMAIL", None), \
             patch("notifier.send_to_telegram") as mock_tg:
            mock_tg.return_value = True
            alert_payload = {
                "alert_id": "test_xss_alert",
                "canonical_ticker": "<script>alert('xss')</script>",
                "rule_type": "above_price",
                "threshold_value": 100.0,
                "current_price": 105.0,
                "trigger_description": "<b>malicious bold</b> & 'test'"
            }
            ok = dispatch_alert_notification(alert_payload)
            self.assertTrue(ok)
            mock_tg.assert_called_once()
            sent_msg = mock_tg.call_args[0][0]
            self.assertNotIn("<script>", sent_msg)
            self.assertIn("&lt;script&gt;", sent_msg)
            self.assertNotIn("<b>malicious", sent_msg)
            self.assertIn("&lt;b&gt;malicious", sent_msg)
            self.assertIn("&amp;", sent_msg)

    def test_atomic_cas_cooldown(self):
        """測試 record_alert_trigger 之原子 CAS 機制：同冷卻期內第二次呼叫回傳 False"""
        alert = self.db.add_quant_alert(
            canonical_ticker="2330.TW",
            rule_type="above_price",
            threshold_value=990.0,
            cooldown_minutes=60
        )
        alert_id = alert["alert_id"]

        # First trigger succeeds (atomic update returns True)
        ok1 = self.db.record_alert_trigger(alert_id, triggered_price=1000.0, cooldown_minutes=60)
        self.assertTrue(ok1)

        # Immediate second trigger fails because cooldown has not elapsed (atomic update returns False)
        ok2 = self.db.record_alert_trigger(alert_id, triggered_price=1005.0, cooldown_minutes=60)
        self.assertFalse(ok2)

    def test_database_busy_returns_503_on_check(self):
        """測試 /api/v1/alerts/check 在底層鎖忙碌時回傳 503"""
        with patch("data.alert_evaluator.evaluate_and_dispatch_alerts", side_effect=DuckDBBusyError("DB lock conflict")):
            resp = self.client.post("/api/v1/alerts/check", headers={"X-API-Key": self.api_key})
            self.assertEqual(resp.status_code, 503)
            self.assertIn("busy", resp.json()["detail"].lower())

    def test_private_profile_alerts_requires_key(self):
        """測試 GET /api/v1/alerts 在生產環境無金鑰時回傳 401，有金鑰時回傳 200"""
        # 無金鑰
        resp_unauth = self.client.get("/api/v1/alerts")
        self.assertEqual(resp_unauth.status_code, 401)

        # 有金鑰
        resp_auth = self.client.get("/api/v1/alerts", headers={"X-API-Key": self.api_key})
        self.assertEqual(resp_auth.status_code, 200)


if __name__ == "__main__":
    unittest.main()
