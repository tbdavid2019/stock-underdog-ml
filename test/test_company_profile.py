"""
test/test_company_profile.py - Test CompanyProfileService and fallback mechanisms
"""

import unittest
from unittest.mock import patch, MagicMock
from data.company_profile import CompanyProfileService


class TestCompanyProfile(unittest.TestCase):
    @patch("data.company_profile.CompanyProfileService._fetch_profile_from_2md")
    @patch("data.company_profile.CompanyProfileService._fetch_news_from_2md")
    def test_normalize_and_resolve_aliases(self, mock_news, mock_profile):
        profile = CompanyProfileService.get_company_profile("2330")
        self.assertEqual(profile["ticker"], "2330.TW")
        self.assertEqual(profile["raw_code"], "2330")
        self.assertIn("links", profile)
        self.assertIn("yahoo", profile["links"])

    def test_cache_mechanism(self):
        CompanyProfileService._CACHE.clear()
        fake_profile = {
            "ticker": "9945.TW",
            "raw_code": "9945",
            "name": "潤泰新",
            "industry": "建材營造",
            "business_summary": "營建與商場營運",
            "recent_news": [],
            "links": {}
        }
        CompanyProfileService._CACHE["9945.TW"] = (9999999999, fake_profile)
        res = CompanyProfileService.get_company_profile("9945.TW")
        self.assertEqual(res["name"], "潤泰新")
        self.assertEqual(res["industry"], "建材營造")

    @patch("requests.get")
    def test_fetch_from_2md_parsing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
Title: 潤泰新(9945.TW) 基本資料 - Yahoo股市
URL Source: https://tw.stock.yahoo.com/quote/9945.TW/profile
公司名稱
潤泰新
產業類別
建材營造
董事長
簡滄圳
成立時間
1977/09/12
主要經營業務
委託營造廠興建國民住宅．商業大樓出租出售業務 辦理國內外工業區及區內之社區廠房及大樓之開發規劃及租售
"""
        mock_get.return_value = mock_resp
        profile = {
            "ticker": "9945.TW",
            "raw_code": "9945",
            "name": "",
            "industry": "綜合產業",
            "business_summary": "",
            "links": {}
        }
        CompanyProfileService._fetch_profile_from_2md(profile, is_tw=True)
        self.assertEqual(profile["name"], "潤泰新")
        self.assertEqual(profile["industry"], "建材營造")
        self.assertEqual(profile["chairman"], "簡滄圳")
        self.assertEqual(profile["established"], "1977/09/12")
        self.assertIn("委託營造廠興建國民住宅", profile["business_summary"])

    def test_api_company_profile_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        CompanyProfileService._CACHE["2330.TW"] = (9999999999, {
            "ticker": "2330.TW",
            "raw_code": "2330",
            "name": "台積電",
            "industry": "半導體",
            "business_summary": "全球領先的晶圓代工製造商",
            "chairman": "魏哲家",
            "recent_news": [],
            "links": {}
        })
        resp = client.get("/api/v1/market/company-profile?ticker=2330.TW")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["ticker"], "2330.TW")
        self.assertEqual(data["name"], "台積電")

    def test_batch_company_profiles_endpoint(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        CompanyProfileService._CACHE["2330.TW"] = (9999999999, {
            "ticker": "2330.TW",
            "raw_code": "2330",
            "name": "台積電",
            "industry": "半導體",
            "business_summary": "全球領先的晶圓代工製造商",
            "recent_news": [],
            "links": {}
        })
        resp = client.post("/api/v1/market/company-profiles/batch", json={"tickers": ["2330.TW"]})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["count"], 1)
        self.assertIn("2330.TW", data["data"])


if __name__ == "__main__":
    unittest.main()
