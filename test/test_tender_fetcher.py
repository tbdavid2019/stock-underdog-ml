"""
test/test_tender_fetcher.py - Unit tests for ToBid Taiwan Government Tender Service & Market API
"""

import os
import shutil
import unittest
import requests
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from data.tender_fetcher import TenderService, format_currency_ntd
from api.main import app


class TestCurrencyFormatting(unittest.TestCase):
    def test_format_currency_ntd(self):
        self.assertEqual(format_currency_ntd(0), "0 元")
        self.assertEqual(format_currency_ntd(-100), "0 元")
        self.assertEqual(format_currency_ntd(None), "0 元")
        self.assertEqual(format_currency_ntd(5000), "5,000 元")
        self.assertEqual(format_currency_ntd(50000), "5 萬元")
        self.assertEqual(format_currency_ntd(5500000), "550 萬元")
        self.assertEqual(format_currency_ntd(8642000000), "86.42 億元")
        self.assertEqual(format_currency_ntd(123456789000), "1,234.6 億元")


class TestTenderService(unittest.TestCase):
    def setUp(self):
        TenderService.clear_in_memory_cache()
        if os.path.exists(TenderService.CACHE_DIR):
            shutil.rmtree(TenderService.CACHE_DIR, ignore_errors=True)

    def tearDown(self):
        TenderService.clear_in_memory_cache()
        if os.path.exists(TenderService.CACHE_DIR):
            shutil.rmtree(TenderService.CACHE_DIR, ignore_errors=True)

    def test_non_tw_stock(self):
        res = TenderService.get_company_tender("AAPL")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("has_data"))
        self.assertIn("ToBid", res.get("message", ""))

    def test_empty_ticker(self):
        self.assertIsNone(TenderService.get_company_tender(""))
        self.assertIsNone(TenderService.get_company_tender(None))

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_get_company_tender_success(self, mock_post, mock_resolve):
        # 1. Mock entity resolution
        mock_resolve.return_value = {
            "name": "漢翔航空工業股份有限公司",
            "code": "2634",
            "market": "TWSE"
        }

        # 2. Mock ToBid MCP tool responses
        report_payload = {
            "tax_id": "89564808",
            "known_award_total": 86420000000,
            "participations": 450,
            "wins": 380,
            "win_rate_pct": 84.4,
            "sole_bid_rate_pct": 52.1,
            "top_clients": [
                {"unit": "國防部空軍司令部", "wins": 180, "amount": 62000000000},
                {"unit": "國家中山科學研究院", "wins": 60, "amount": 14000000000}
            ],
            "top_rivals": [
                {"name": "亞航股份有限公司", "met": 40, "this_vendor_wins": 30, "rival_wins": 10}
            ],
            "main_products": {
                "terms": [
                    {"term": "發動機維護", "cases": 45},
                    {"term": "飛機零附件", "cases": 38}
                ]
            },
            "recent_losses": [
                {
                    "date": "20240901",
                    "title": "無人機載具採購",
                    "lost_to": "雷虎科技股份有限公司",
                    "award_amount": 150000000,
                    "lost_reason": "價格評選序位第二"
                }
            ],
            "web": "https://tobid.tw/?vendor=89564808"
        }

        awards_payload = {
            "items": [
                {
                    "date": "2024-10-01",
                    "unit_name": "空軍第一後勤指揮部",
                    "title": "F-16型機零附件補給專案",
                    "job_number": "HA-113001",
                    "award_amount": 250000000,
                    "bidders": 1,
                    "sole_bid": True
                }
            ],
            "yearly": [
                {"year": 2024, "cases": 35, "amount": 12000000000},
                {"year": 2023, "cases": 42, "amount": 15000000000}
            ]
        }

        def mock_post_side_effect(url, json=None, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            tool_name = json.get("params", {}).get("name") if json else ""
            if tool_name == "vendor_report":
                mock_resp.json.return_value = {
                    "result": {
                        "content": [{"type": "text", "text": __import__("json").dumps(report_payload)}]
                    }
                }
            elif tool_name == "vendor_awards":
                mock_resp.json.return_value = {
                    "result": {
                        "content": [{"type": "text", "text": __import__("json").dumps(awards_payload)}]
                    }
                }
            elif tool_name == "vendor_compliance":
                mock_resp.json.return_value = {
                    "result": {
                        "content": [{"type": "text", "text": __import__("json").dumps({
                            "currently_debarred": False,
                            "debarments": [],
                            "revoked": [],
                            "article_101_cases": []
                        })}]
                    }
                }
            else:
                mock_resp.json.return_value = {"error": {"message": "Unknown tool"}}
            return mock_resp

        mock_post.side_effect = mock_post_side_effect

        data = TenderService.get_company_tender("2634.TW")
        self.assertIsNotNone(data)
        self.assertTrue(data.get("has_data"))
        self.assertEqual(data.get("ticker"), "2634.TW")
        self.assertEqual(data.get("name"), "漢翔航空工業股份有限公司")
        self.assertEqual(data.get("wins"), 380)
        self.assertEqual(data.get("participations"), 450)
        self.assertEqual(data.get("win_rate_pct"), 84.4)
        self.assertEqual(data.get("sole_bid_rate_pct"), 52.1)
        self.assertEqual(data.get("known_award_total_formatted"), "864.20 億元")

        # Verify compliance check (normal)
        self.assertIn("compliance", data)
        self.assertFalse(data["compliance"]["currently_debarred"])
        self.assertEqual(data["compliance"]["status"], "PASSED")
        self.assertIn("正常", data["compliance"]["label"])

        # Verify top clients
        clients = data.get("top_clients", [])
        self.assertEqual(len(clients), 2)
        self.assertEqual(clients[0]["unit"], "國防部空軍司令部")
        self.assertEqual(clients[0]["amount_formatted"], "620.00 億元")

        # Verify top rivals
        rivals = data.get("top_rivals", [])
        self.assertEqual(len(rivals), 1)
        self.assertEqual(rivals[0]["name"], "亞航股份有限公司")
        self.assertEqual(rivals[0]["win_rate"], 75.0)

        # Verify awards & yearly
        self.assertEqual(len(data.get("recent_awards", [])), 1)
        self.assertEqual(data["recent_awards"][0]["title"], "F-16型機零附件補給專案")
        self.assertEqual(len(data.get("yearly_awards", [])), 2)

        # Verify L1 Memory Cache: Next call should not hit mock_post again
        call_count_before = mock_post.call_count
        cached_data = TenderService.get_company_tender("2634.TW")
        self.assertEqual(mock_post.call_count, call_count_before)
        self.assertEqual(cached_data.get("wins"), 380)

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_get_company_tender_not_found_negative_cache(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "瑞昱半導體股份有限公司",
            "code": "2379",
            "market": "TWSE"
        }

        # Mock ToBid returning "查無此廠商" error
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "Error executing tool vendor_report: 查無此廠商：'瑞昱半導體股份有限公司'"}]
            }
        }
        mock_post.return_value = mock_resp

        data = TenderService.get_company_tender("2379.TW")
        self.assertIsNotNone(data)
        self.assertFalse(data.get("has_data"))
        self.assertIn("查無政府採購標案", data.get("message", ""))

        # Check negative cache in L1
        call_count_before = mock_post.call_count
        cached_data = TenderService.get_company_tender("2379.TW")
        self.assertEqual(mock_post.call_count, call_count_before)
        self.assertFalse(cached_data.get("has_data"))

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_get_company_tender_compliance_debarred_black_swan(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "某營造工程股份有限公司",
            "code": "9999",
            "market": "TWSE"
        }
        report_payload = {
            "tax_id": "12345678",
            "known_award_total": 500000000,
            "wins": 10,
            "participations": 20,
            "win_rate_pct": 50.0
        }
        compliance_payload = {
            "currently_debarred": True,
            "debarments": [
                {
                    "unit_name": "交通部鐵道局",
                    "title": "邊坡改善工程",
                    "reason": "政府採購法第101條第1項第10款",
                    "from": "2026-01-01",
                    "to": "2027-01-01"
                }
            ],
            "article_101_cases": [{"case_no": "113001"}]
        }

        def mock_post_side_effect(url, json=None, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            tool_name = json.get("params", {}).get("name") if json else ""
            if tool_name == "vendor_report":
                mock_resp.json.return_value = {"result": {"content": [{"type": "text", "text": __import__("json").dumps(report_payload)}]}}
            elif tool_name == "vendor_awards":
                mock_resp.json.return_value = {"result": {"content": [{"type": "text", "text": __import__("json").dumps({"items": [], "yearly": []})}]}}
            elif tool_name == "vendor_compliance":
                mock_resp.json.return_value = {"result": {"content": [{"type": "text", "text": __import__("json").dumps(compliance_payload)}]}}
            return mock_resp

        mock_post.side_effect = mock_post_side_effect
        data = TenderService.get_company_tender("9999.TW")
        self.assertIsNotNone(data)
        self.assertTrue(data.get("has_data"))
        self.assertIn("compliance", data)
        self.assertTrue(data["compliance"]["currently_debarred"])
        self.assertEqual(data["compliance"]["status"], "DEBARRED")
        self.assertIn("停權生效中", data["compliance"]["label"])
        self.assertEqual(data["compliance"]["debarment_count"], 1)

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_get_company_tender_compliance_failure_reports_unknown(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "漢翔航空工業股份有限公司",
            "code": "2634",
            "market": "TWSE"
        }
        report_payload = {"known_award_total": 1000000, "wins": 5, "participations": 10}

        def mock_post_side_effect(url, json=None, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            tool_name = json.get("params", {}).get("name") if json else ""
            if tool_name == "vendor_report":
                mock_resp.json.return_value = {"result": {"content": [{"type": "text", "text": __import__("json").dumps(report_payload)}]}}
            elif tool_name == "vendor_awards":
                mock_resp.json.return_value = {"result": {"content": [{"type": "text", "text": __import__("json").dumps({"items": [], "yearly": []})}]}}
            elif tool_name == "vendor_compliance":
                # Mock compliance timeout / failure
                raise requests.Timeout("Connection timed out")
            return mock_resp

        mock_post.side_effect = mock_post_side_effect
        data = TenderService.get_company_tender("2634.TW")
        self.assertIsNotNone(data)
        self.assertTrue(data.get("has_data"))
        self.assertIn("compliance", data)
        # Verify status is UNKNOWN instead of blindly PASSED (Codex P1)
        self.assertEqual(data["compliance"]["status"], "UNKNOWN")
        self.assertIn("暫不可用", data["compliance"]["label"])

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_get_company_tender_transient_failure_not_cached(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "漢翔航空工業股份有限公司",
            "code": "2634",
            "market": "TWSE"
        }
        # Mock transient timeout on vendor_report
        mock_post.side_effect = requests.Timeout("Gateway Timeout")

        data = TenderService.get_company_tender("2634.TW")
        self.assertIsNotNone(data)
        self.assertFalse(data.get("has_data"))
        self.assertIn("連線逾時", data.get("message", ""))

        # Verify it was NOT persisted to L1 cache (Codex P2)
        self.assertNotIn("2634", TenderService._MEM_CACHE)

    def test_duckdb_cache_integration(self):
        mock_db = MagicMock()
        fake_db_data = {
            "ticker": "1519.TW",
            "raw_code": "1519",
            "name": "華城電機股份有限公司",
            "has_data": True,
            "wins": 150,
            "known_award_total_formatted": "85.3 億元"
        }
        mock_db.get_company_tender.return_value = fake_db_data

        data = TenderService.get_company_tender("1519.TW", db=mock_db)
        self.assertIsNotNone(data)
        self.assertEqual(data.get("wins"), 150)
        mock_db.get_company_tender.assert_called_once()

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_bearer_token_and_api_key_sent_to_mcp(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "漢翔航空工業股份有限公司",
            "code": "2634",
            "market": "TWSE",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "content": [{"type": "text", "text": '{"known_award_total": 1000000, "participations": 2, "wins": 1, "win_rate_pct": 50.0, "sole_bid_rate_pct": 0.0}'}]
            }
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"TOBID_MCP_TOKEN": "test_bearer_xyz", "TOBID_API_KEY": "test_key_123"}):
            TenderService.clear_in_memory_cache()
            data = TenderService.get_company_tender("2634.TW")

        self.assertTrue(data.get("has_data"))
        self.assertGreaterEqual(mock_post.call_count, 1)
        called_url = mock_post.call_args_list[0].args[0]
        called_headers = mock_post.call_args_list[0].kwargs.get("headers", {})
        self.assertIn("key=test_key_123", called_url)
        self.assertEqual(called_headers.get("Authorization"), "Bearer test_bearer_xyz")

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_auth_required_error_not_cached(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "天虹科技股份有限公司",
            "code": "6937",
            "market": "TWSE",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "【需要登入】ToBid MCP 自 2026-10-29 起需要登入 ToBid 帳號才能使用"}],
            }
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {}, clear=False):
            for k in ("TOBID_MCP_TOKEN", "TOBID_API_TOKEN", "TOBID_ACCESS_TOKEN", "TOBID_REFRESH_TOKEN", "TOBID_CLIENT_ID"):
                os.environ.pop(k, None)
            TenderService.clear_in_memory_cache()
            data = TenderService.get_company_tender("6937.TW")

        self.assertIsNotNone(data)
        self.assertFalse(data.get("has_data"))
        self.assertTrue(data.get("auth_required"))
        self.assertIn("TOBID_MCP_TOKEN", data.get("message", ""))
        self.assertNotIn("6937", TenderService._MEM_CACHE)

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_oauth_refresh_token_rotates_and_retries(self, mock_post, mock_resolve):
        import json as _json
        import tempfile

        mock_resolve.return_value = {
            "name": "漢翔航空工業股份有限公司",
            "code": "2634",
            "market": "TWSE",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = os.path.join(tmpdir, "oauth_token.json")
            with open(token_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "access_token": "expired_tok",
                    "refresh_token": "rt_initial",
                    "client_id": "cid_123",
                    "expires_at": 1000.0,
                }, f)

            def mock_post_side_effect(url, **kwargs):
                resp = MagicMock()
                if "/oauth/token" in url:
                    resp.status_code = 200
                    resp.json.return_value = {
                        "access_token": "fresh_tok_456",
                        "refresh_token": "rt_rotated_789",
                        "expires_in": 3600,
                    }
                    return resp
                headers = kwargs.get("headers") or {}
                self.assertEqual(headers.get("Authorization"), "Bearer fresh_tok_456")
                resp.status_code = 200
                resp.json.return_value = {
                    "result": {
                        "content": [{"type": "text", "text": '{"known_award_total": 5000000, "participations": 3, "wins": 2, "win_rate_pct": 66.7, "sole_bid_rate_pct": 0.0}'}]
                    }
                }
                return resp

            mock_post.side_effect = mock_post_side_effect
            with patch.dict(os.environ, {"TOBID_TOKEN_FILE": token_path}):
                TenderService.clear_in_memory_cache()
                data = TenderService.get_company_tender("2634.TW")

            self.assertTrue(data.get("has_data"))
            with open(token_path, "r", encoding="utf-8") as f:
                persisted = _json.load(f)
            self.assertEqual(persisted["access_token"], "fresh_tok_456")
            self.assertEqual(persisted["refresh_token"], "rt_rotated_789")

    def test_sse_stream_response_parsing(self):
        class FakeSSEResp:
            text = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{}"}]}}\n\n'
            def json(self):
                raise ValueError("Not raw JSON")

        parsed = TenderService._parse_mcp_json(FakeSSEResp())
        self.assertEqual(parsed["id"], 1)
        self.assertIn("result", parsed)

    def test_pkce_helper_generation(self):
        from scripts.tobid_oauth_login import generate_pkce_pair
        verifier, challenge = generate_pkce_pair()
        self.assertGreaterEqual(len(verifier), 43)
        self.assertNotIn("=", verifier)
        self.assertNotIn("=", challenge)
        self.assertNotEqual(verifier, challenge)

    @patch("data.company_network.CompanyNetworkService.resolve_company_info")
    @patch("requests.post")
    def test_sequential_short_circuit_on_not_found(self, mock_post, mock_resolve):
        mock_resolve.return_value = {
            "name": "天虹科技股份有限公司",
            "code": "6937",
            "market": "TWSE",
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "Error executing tool vendor_report: 查無此廠商：'天虹科技股份有限公司'"}],
            }
        }
        mock_post.return_value = mock_resp

        data = TenderService.get_company_tender("6937.TW")
        self.assertFalse(data.get("has_data"))
        # 查無廠商時只打 vendor_report 並立即短路返回，不再多送 vendor_awards / vendor_compliance
        self.assertEqual(mock_post.call_count, 1)


class TestTenderAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @patch("data.tender_fetcher.TenderService.get_company_tender")
    def test_api_get_company_tender(self, mock_service):
        mock_service.return_value = {
            "ticker": "2634.TW",
            "name": "漢翔航空工業股份有限公司",
            "has_data": True,
            "wins": 380,
            "known_award_total_formatted": "864.20 億元"
        }

        # Test path param
        resp = self.client.get("/api/v1/market/company-tender/2634.TW")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["data"]["ticker"], "2634.TW")
        self.assertEqual(data["data"]["wins"], 380)

        # Test query param
        resp2 = self.client.get("/api/v1/market/company-tender?ticker=2634.TW")
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertTrue(data2.get("success"))
        self.assertEqual(data2["data"]["name"], "漢翔航空工業股份有限公司")

    @patch("data.tender_fetcher.TenderService.get_company_tender")
    def test_api_get_company_tender_not_found(self, mock_service):
        mock_service.return_value = {
            "ticker": "UNKNOWN",
            "name": "UNKNOWN",
            "has_data": False,
            "message": "查無政府採購標案"
        }

        resp = self.client.get("/api/v1/market/company-tender/UNKNOWN")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertFalse(data["data"]["has_data"])


if __name__ == "__main__":
    unittest.main()
