"""
test/test_company_profile.py - Test CompanyProfileService and fallback mechanisms
"""

import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from data.company_profile import CompanyProfileService


class TestCompanyProfile(unittest.TestCase):
    def setUp(self):
        CompanyProfileService._CACHE.clear()
        if os.path.exists(CompanyProfileService.CACHE_DIR):
            shutil.rmtree(CompanyProfileService.CACHE_DIR, ignore_errors=True)

    def tearDown(self):
        CompanyProfileService._CACHE.clear()
        if os.path.exists(CompanyProfileService.CACHE_DIR):
            shutil.rmtree(CompanyProfileService.CACHE_DIR, ignore_errors=True)

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

    def test_20_day_persistent_disk_cache(self):
        import os
        # 1. 驗證快取效期設為 20 天 (20 * 86400 秒)
        self.assertEqual(CompanyProfileService.CACHE_TTL, 86400 * 20)

        # 2. 寫入磁碟快取
        test_ticker = "2330.TW"
        fake_profile = {
            "ticker": test_ticker,
            "raw_code": "2330",
            "name": "台積電",
            "industry": "半導體業",
            "business_summary": "依客戶需求客製製造晶圓代工與先進封裝",
            "chairman": "魏哲家",
            "market_cap": "$61.8T",
            "recent_news": [],
            "links": {}
        }
        CompanyProfileService._write_cache(test_ticker, fake_profile)

        # 3. 驗證磁碟檔案確實產生
        cache_file = CompanyProfileService._get_cache_filepath(test_ticker)
        self.assertTrue(os.path.exists(cache_file))

        # 4. 清空 L1 記憶體快取 (模擬服務重啟)
        CompanyProfileService._CACHE.clear()
        self.assertNotIn(test_ticker, CompanyProfileService._CACHE)

        # 5. 再次讀取：驗證微秒級從磁碟命中，並自動回填 L1 快取，無需發出任何網路請求
        restored = CompanyProfileService._read_cache(test_ticker)
        self.assertIsNotNone(restored)
        self.assertEqual(restored["name"], "台積電")
        self.assertEqual(restored["chairman"], "魏哲家")
        self.assertIn(test_ticker, CompanyProfileService._CACHE)

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

    @patch("requests.post")
    def test_fetch_batch_from_2md_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "code": 200,
            "status": 20000,
            "data": {
                "data": [
                    {
                        "url": "https://tw.stock.yahoo.com/quote/2330.TW/profile",
                        "content": "公司名稱\n台積電\n產業類別\n半導體\n董事長\n魏哲家\n主要經營業務\n晶圓代工"
                    },
                    {
                        "url": "https://tw.stock.yahoo.com/quote/2303.TW/profile",
                        "content": "公司名稱\n聯電\n產業類別\n半導體\n董事長\n洪嘉聰\n主要經營業務\n晶圓製造"
                    }
                ]
            }
        }
        mock_post.return_value = mock_resp

        urls = [
            "https://tw.stock.yahoo.com/quote/2330.TW/profile",
            "https://tw.stock.yahoo.com/quote/2303.TW/profile"
        ]
        items = CompanyProfileService._fetch_batch_from_2md(urls)
        self.assertEqual(len(items), 2)
        self.assertIn("台積電", items[0]["content"])
        self.assertIn("聯電", items[1]["content"])

    @patch("requests.post")
    def test_fetch_batch_from_2md_fallback_on_first_host_connection_error(self, mock_post):
        import requests
        resp_success = MagicMock()
        resp_success.status_code = 200
        resp_success.json.return_value = {
            "code": 200,
            "data": {
                "data": [
                    {
                        "url": "https://tw.stock.yahoo.com/quote/2330.TW/profile",
                        "content": "公司名稱\n台積電\n主要經營業務\n晶圓代工"
                    }
                ]
            }
        }
        # First host raises ConnectionError (server down), second host succeeds
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("Connection refused"),
            resp_success
        ]

        items = CompanyProfileService._fetch_batch_from_2md(["https://tw.stock.yahoo.com/quote/2330.TW/profile"])
        self.assertEqual(len(items), 1)
        self.assertIn("台積電", items[0]["content"])
        # Verified mock_post was called twice (first host aiurl.tw -> fallback host glsoft.ai)
        self.assertEqual(mock_post.call_count, 2)

    @patch("requests.post")
    def test_fetch_batch_from_2md_fallback_on_timeout(self, mock_post):
        import requests
        resp_success = MagicMock()
        resp_success.status_code = 200
        resp_success.json.return_value = {
            "code": 200,
            "data": {
                "data": [
                    {
                        "url": "https://tw.stock.yahoo.com/quote/2330.TW/profile",
                        "content": "公司名稱\n台積電\n主要經營業務\n晶圓代工"
                    }
                ]
            }
        }
        # First host raises Timeout, second host succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout("Read timeout"),
            resp_success
        ]

        items = CompanyProfileService._fetch_batch_from_2md(["https://tw.stock.yahoo.com/quote/2330.TW/profile"])
        self.assertEqual(len(items), 1)
        self.assertIn("台積電", items[0]["content"])
        self.assertEqual(mock_post.call_count, 2)

    @patch("data.company_profile.CompanyProfileService._fetch_batch_from_2md")
    def test_get_company_profiles_batch_uses_native_batch(self, mock_batch):
        CompanyProfileService._CACHE.clear()
        mock_batch.return_value = [
            {
                "url": "https://tw.stock.yahoo.com/quote/2330.TW/profile",
                "content": "公司名稱\n台積電\n產業類別\n半導體\n董事長\n魏哲家\n市值 (百萬)\n61848702.45\n主要經營業務\n積體電路製造"
            },
            {
                "url": "https://tw.stock.yahoo.com/quote/2303.TW/profile",
                "content": "公司名稱\n聯電\n產業類別\n半導體\n董事長\n洪嘉聰\n市值 (百萬)\n650000.00\n主要經營業務\n晶圓製造代工"
            }
        ]

        results = CompanyProfileService.get_company_profiles_batch(["2330.TW", "2303.TW"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results["2330.TW"]["name"], "台積電")
        self.assertEqual(results["2330.TW"]["chairman"], "魏哲家")
        self.assertIn("積體電路製造", results["2330.TW"]["business_summary"])
        self.assertEqual(results["2303.TW"]["name"], "聯電")
        self.assertEqual(results["2303.TW"]["chairman"], "洪嘉聰")
        self.assertIn("晶圓製造代工", results["2303.TW"]["business_summary"])
        self.assertEqual(mock_batch.call_count, 1)

    @patch("data.company_profile.CompanyProfileService._fetch_profile_from_2md")
    @patch("data.company_profile.CompanyProfileService._fetch_news_from_2md")
    def test_get_company_profile_persists_and_reads_from_duckdb(self, mock_news, mock_profile):
        """驗證當使用者點擊查到公司資料後，自動存入 DuckDB，再次查詢直接命中 DuckDB 無需重抓"""
        def _fake_fetch_profile(prof, is_tw):
            prof["name"] = "大同"
            prof["industry"] = "綜合產業"
            prof["business_summary"] = "家電與重電設備"
            prof["chairman"] = "張_華"
            prof["market_cap"] = "1000億"
        mock_profile.side_effect = _fake_fetch_profile
        mock_news.return_value = []

        mock_db = MagicMock()
        mock_db.get_company_profile.return_value = None  # 初次查無快取

        # 1. 第一次呼叫：查無快取，觸發抓取並寫入 DuckDB
        res1 = CompanyProfileService.get_company_profile("2371.TW", db=mock_db)
        self.assertEqual(res1["name"], "大同")
        self.assertEqual(mock_profile.call_count, 1)
        mock_db.save_company_profile.assert_called_once()

        # 2. 第二次呼叫：DuckDB 命中，清除記憶體與磁碟快取，驗證直接回傳 DuckDB 快取且不重複抓取
        CompanyProfileService._CACHE.clear()
        mock_db.get_company_profile.return_value = {
            "ticker": "2371.TW",
            "raw_code": "2371",
            "name": "大同",
            "business_summary": "家電與重電設備 (來自 DuckDB)",
            "chairman": "張_華"
        }
        res2 = CompanyProfileService.get_company_profile("2371.TW", db=mock_db)
        self.assertEqual(res2["business_summary"], "家電與重電設備 (來自 DuckDB)")
        self.assertEqual(mock_profile.call_count, 1)  # 抓取次數依然為 1，未重抓！

    def test_foreign_stock_ticker_collision_prevented(self):
        """驗證日股 6770.T 絕不與台股 6770 (力積電) 代號碰撞"""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db._get_connection.return_value.__enter__.return_value = mock_conn

        # 模擬 market_universe 查得 ALPS ALPINE CO.,LTD.
        mock_conn.execute.return_value.fetchone.return_value = ("ALPS ALPINE CO.,LTD.",)

        name = CompanyProfileService._resolve_stock_name("6770.T", db=mock_db)
        self.assertEqual(name, "ALPS ALPINE CO.,LTD.")
        self.assertNotEqual(name, "力積電")

    def test_non_tw_stock_does_not_query_tw_raw_code(self):
        """驗證非台股代碼在查 company_profiles 時僅用 ticker 匹配，不用 raw_code 截斷比對"""
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_db._get_connection.return_value.__enter__.return_value = mock_conn

        # market_universe 查無資料
        def fake_execute(query, params=None):
            m = MagicMock()
            if "market_universe" in query:
                m.fetchone.return_value = None
            elif "company_profiles" in query:
                # 確保查詢條件中不含有 raw_code
                self.assertNotIn("raw_code", query)
                m.fetchone.return_value = ("ALPS ALPINE",)
            else:
                m.fetchone.return_value = None
            return m

        mock_conn.execute.side_effect = fake_execute

        name = CompanyProfileService._resolve_stock_name("6770.T", db=mock_db)
        self.assertEqual(name, "ALPS ALPINE")
        self.assertNotEqual(name, "力積電")

    def test_foreign_stock_contaminated_cache_filtered_and_refreshed(self):
        """驗證非台股 (6770.T) 快取若遭力積電/台股污染，會被自動識別為損壞快取並重抓/清理"""
        import time
        contaminated_cache = {
            "ticker": "6770.T",
            "raw_code": "6770.T",
            "name": "ALPS ALPINE CO LTD",
            "industry": "電子零組件",
            "business_summary": "力積電(6770)為台灣晶圓代工大廠，業務涵蓋記憶體代工...",
            "chairman": "黃崇仁",
            "market_cap": "$100B",
            "recent_news": [
                {
                    "title": "力積電(6770.TW) 走勢圖 - Yahoo股市",
                    "url": "https://tw.stock.yahoo.com/quote/6770.TW",
                    "snippet": "最新力積電即時行情"
                },
                {
                    "title": "【股市爆料同學會】力積電大漲分析",
                    "url": "https://www.cmoney.tw/forum/stock/6770",
                    "snippet": "同學會熱門討論"
                }
            ],
            "links": {}
        }

        # 寫入磁碟快取與記憶體快取
        CompanyProfileService._write_cache("6770.T", contaminated_cache)
        CompanyProfileService._CACHE["6770.T"] = (time.time(), contaminated_cache)

        # 模擬 yfinance 備援行為：回傳日股 Alps Alpine 的真實資料
        with patch("yfinance.Ticker") as mock_yf:
            mock_inst = MagicMock()
            mock_inst.info = {
                "shortName": "ALPS ALPINE CO LTD",
                "longName": "ALPS ALPINE CO., LTD.",
                "longBusinessSummary": "Alps Alpine Co., Ltd. manufactures and sells electronic components and automotive infotainment systems worldwide.",
                "industry": "Electronic Components",
                "sector": "Technology",
                "marketCap": 250000000000
            }
            mock_yf.return_value = mock_inst

            # 呼叫 get_company_profile，此時應拋棄被污染快取，並透過 yfinance 重抓
            profile = CompanyProfileService.get_company_profile("6770.T")

            # 驗證簡介絕不可包含力積電，且已被 yfinance 的正確英文簡介覆蓋
            self.assertNotIn("力積電", profile.get("business_summary", ""))
            self.assertIn("Alps Alpine", profile.get("business_summary", ""))

            # 驗證新聞列表絕不可包含任何力積電、.TW 或台灣財經網址
            for news in profile.get("recent_news", []):
                self.assertNotIn("力積電", news.get("title", ""))
                self.assertNotIn(".TW", news.get("title", ""))
                self.assertNotIn("tw.stock.yahoo.com", news.get("url", ""))
                self.assertNotIn("cmoney.tw", news.get("url", ""))

    def test_parse_news_markdown_foreign_stock_rejects_taiwan_sources(self):
        """驗證 _parse_news_markdown 對於非台股標的拒絕收錄台灣新聞與污染簡介"""
        profile = {
            "ticker": "6770.T",
            "raw_code": "6770.T",
            "name": "ALPS ALPINE CO LTD",
            "business_summary": "",
            "recent_news": []
        }
        text = """
[1] Title: 力積電(6770.TW) 走勢圖 - Yahoo股市
[1] URL Source: https://tw.stock.yahoo.com/quote/6770.TW
[1] Description: 力積電(6770.TW) 主要從事晶圓代工業務與特殊製程提供

[2] Title: ALPS ALPINE announces new sensors for automotive
[2] URL Source: https://www.bloomberg.com/news/articles/2026-11-01/alps-alpine-sensors
[2] Description: Alps Alpine Co Ltd unveiled new sensors designed for modern electric vehicles.
"""
        CompanyProfileService._parse_news_markdown(profile, text)

        # 簡介不得被力積電代工業務佔據
        self.assertNotIn("力積電", profile.get("business_summary", ""))

        # 新聞列表中應只有第二則 Bloomberg 新聞，第一則台灣力積電新聞必須被丟棄
        self.assertEqual(len(profile["recent_news"]), 1)
        self.assertEqual(profile["recent_news"][0]["title"], "ALPS ALPINE announces new sensors for automotive")

    def test_international_stock_legitimate_quote_news_not_contaminated(self):
        """驗證國際股票 (如 AAPL) 包含 /quote/ 的合法英文新聞 (如 Yahoo Finance / Bloomberg) 絕不被判定為污染"""
        legit_profile = {
            "ticker": "AAPL",
            "raw_code": "AAPL",
            "name": "Apple Inc.",
            "business_summary": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories.",
            "recent_news": [
                {
                    "title": "Apple Inc. (AAPL) Stock Price, News, Quote & History - Yahoo Finance",
                    "url": "https://finance.yahoo.com/quote/AAPL/news/apple-earnings-preview",
                    "snippet": "Apple reports quarterly earnings next week with Wall Street watching iPhone sales."
                },
                {
                    "title": "Apple Suppliers Boost Capacity in Southeast Asia",
                    "url": "https://www.bloomberg.com/quote/AAPL:US",
                    "snippet": "Apple supply chain partners continue expanding operations across Vietnam and India."
                }
            ],
            "links": {}
        }

        # 1. 驗證 _is_contaminated_for_foreign 對合法的 Yahoo Finance / Bloomberg quote 網址絕不誤報
        is_contaminated = CompanyProfileService._is_contaminated_for_foreign(legit_profile)
        self.assertFalse(is_contaminated, "國際股票合法新聞包含 finance.yahoo.com/quote/ 或 bloomberg.com/quote/ 不應被判定為污染")

        # 2. 驗證 _clean_foreign_profile_if_needed 絕不誤刪該合法新聞
        CompanyProfileService._clean_foreign_profile_if_needed("AAPL", legit_profile)
        self.assertEqual(len(legit_profile["recent_news"]), 2)


if __name__ == "__main__":
    unittest.main()
