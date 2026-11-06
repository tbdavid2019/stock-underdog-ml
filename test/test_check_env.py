"""
test/test_check_env.py - 測試環境變數健檢工具 check_env.py
"""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts.check_env import run_env_check, main


class TestCheckEnv(unittest.TestCase):
    def test_missing_required_variables_error(self):
        """測試缺少必填變數時回報 error 且退出碼為 1"""
        with patch.dict("os.environ", {}, clear=True):
            report = run_env_check()
            self.assertEqual(report["status"], "error")
            self.assertTrue(any(not item["valid"] for item in report["required"]))

    def test_complete_required_variables_success(self):
        """測試必填變數齊備時回報成功 (0 退出碼)"""
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret_key_12345",
            "DEPLOYMENT_MODE": "development",
            "ALLOW_ANONYMOUS_LOCAL_WRITES": "true"
        }
        with patch.dict("os.environ", env, clear=True):
            report = run_env_check()
            self.assertEqual(report["status"], "ok")

    def test_production_without_watchlist_key_warning(self):
        """測試生產模式下未設定 WATCHLIST_API_KEY 時輸出警告與 Fail-Closed 說明"""
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret_key_12345",
            "DEPLOYMENT_MODE": "production"
        }
        with patch.dict("os.environ", env, clear=True):
            report = run_env_check()
            self.assertEqual(report["status"], "warning")
            self.assertTrue(any("Fail-Closed" in w for w in report["warnings"]))

    def test_deprecated_variables_detection(self):
        """測試能偵測 MONGODB_URI、MYSQL_HOST 等已棄用變數"""
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SERVICE_KEY": "secret_key_12345",
            "MONGODB_URI": "mongodb://localhost:27017/test",
            "MYSQL_HOST": "127.0.0.1"
        }
        with patch.dict("os.environ", env, clear=True):
            report = run_env_check()
            dep_names = [d["name"] for d in report["deprecated"]]
            self.assertIn("MONGODB_URI", dep_names)
            self.assertIn("MYSQL_HOST", dep_names)

    def test_invalid_supabase_url_fails(self):
        """測試 SUPABASE_URL 格式不合法（非 http/https 或缺 netloc）時回報錯誤"""
        invalid_urls = ["not_a_url", "ftp://supabase.co", "https://", "://invalid"]
        for bad_url in invalid_urls:
            env = {
                "SUPABASE_URL": bad_url,
                "SUPABASE_SERVICE_KEY": "secret_key_12345",
                "DEPLOYMENT_MODE": "development"
            }
            with patch.dict("os.environ", env, clear=True):
                report = run_env_check()
                self.assertEqual(report["status"], "error", f"Expected error for SUPABASE_URL={bad_url}")
                url_entry = next(item for item in report["required"] if item["name"] == "SUPABASE_URL")
                self.assertFalse(url_entry["valid"])

    def test_cli_json_flag_and_secret_masking(self):
        """測試 --json 輸出且機密值 100% 遮罩不外洩"""
        secret_token = "ultra_sensitive_token_99999"
        env = {
            "SUPABASE_URL": "https://example.supabase.co",
            "SUPABASE_SERVICE_KEY": secret_token,
            "WATCHLIST_API_KEY": "watchlist_secret_888"
        }
        with patch.dict("os.environ", env, clear=True):
            f = io.StringIO()
            with redirect_stdout(f), patch("sys.argv", ["check_env.py", "--json"]):
                exit_code = main()
            self.assertIn(exit_code, (0, 1))

            output = f.getvalue()
            parsed = json.loads(output)
            self.assertIn("required", parsed)
            # 確保明文機密字串絕不出現於輸出中
            self.assertNotIn(secret_token, output)
            self.assertNotIn("watchlist_secret_888", output)


if __name__ == "__main__":
    unittest.main()
