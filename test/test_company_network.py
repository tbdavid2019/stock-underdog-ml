"""
test/test_company_network.py - Tests for Taiwan Company Network service, caching, and resolution
"""

import os
import shutil
import unittest
from unittest.mock import patch, MagicMock
from data.company_network import CompanyNetworkService
from data.duckdb_manager import DuckDBManager


class TestCompanyNetworkService(unittest.TestCase):
    def setUp(self):
        CompanyNetworkService.clear_in_memory_cache()
        if os.path.exists(CompanyNetworkService.CACHE_DIR):
            shutil.rmtree(CompanyNetworkService.CACHE_DIR, ignore_errors=True)

    def tearDown(self):
        CompanyNetworkService.clear_in_memory_cache()
        if os.path.exists(CompanyNetworkService.CACHE_DIR):
            shutil.rmtree(CompanyNetworkService.CACHE_DIR, ignore_errors=True)

    @patch("requests.get")
    def test_alias_resolution(self, mock_get):
        # Mock aliases.json response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "aliases": {
                "2330": [{"name": "台灣積體電路製造股份有限公司", "code": "2330", "market": "TWSE"}],
                "台積電": [{"name": "台灣積體電路製造股份有限公司", "code": "2330", "market": "TWSE"}],
                "2454": [{"name": "聯發科技股份有限公司", "code": "2454", "market": "TWSE"}],
                "3443": [{"name": "創意電子股份有限公司", "code": "3443", "market": "TWSE"}],
                "3374": [{"name": "精材科技股份有限公司", "code": "3374", "market": "TPEx"}],
            }
        }
        mock_get.return_value = mock_resp

        # 1. Test ticker code resolution
        info = CompanyNetworkService.resolve_company_info("2330")
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "台灣積體電路製造股份有限公司")
        self.assertEqual(info["code"], "2330")

        # 2. Test ticker with suffix
        info_tw = CompanyNetworkService.resolve_company_info("2330.TW")
        self.assertIsNotNone(info_tw)
        self.assertEqual(info_tw["name"], "台灣積體電路製造股份有限公司")

        # 3. Test alias name
        info_alias = CompanyNetworkService.resolve_company_info("台積電")
        self.assertIsNotNone(info_alias)
        self.assertEqual(info_alias["code"], "2330")

    @patch("requests.get")
    def test_get_company_network_enrichment(self, mock_get):
        # Setup mock responses for aliases and shard
        mock_aliases_resp = MagicMock()
        mock_aliases_resp.status_code = 200
        mock_aliases_resp.json.return_value = {
            "aliases": {
                "2330": [{"name": "台灣積體電路製造股份有限公司", "code": "2330", "market": "TWSE"}],
                "3443": [{"name": "創意電子股份有限公司", "code": "3443", "market": "TWSE"}],
                "3374": [{"name": "精材科技股份有限公司", "code": "3374", "market": "TPEx"}],
            }
        }

        mock_shard_resp = MagicMock()
        mock_shard_resp.status_code = 200
        mock_shard_resp.json.return_value = {
            "shard": "台",
            "count": 1,
            "data": {
                "台灣積體電路製造股份有限公司": {
                    "name": "台灣積體電路製造股份有限公司",
                    "id": "22099131",
                    "capital": 280500000000,
                    "representative": "魏_家",
                    "address": "新竹科學園區新竹市力行六路8號",
                    "aliases": [{"code": "2330", "market": "TWSE"}],
                    "relationships": {
                        "upstream": ["行政院國家發展基金管理會"],
                        "downstream": [
                            "創意電子股份有限公司",
                            "精材科技股份有限公司",
                            "未上市公司科技"
                        ]
                    },
                    "webUrl": "https://taiwan-company-network.david888.com/graph?company=test",
                    "imageUrl": "https://taiwan-company-network.david888.com/api/v1/images/test.svg"
                }
            }
        }

        def side_effect(url, *args, **kwargs):
            if "aliases.json" in url:
                return mock_aliases_resp
            return mock_shard_resp

        mock_get.side_effect = side_effect

        network = CompanyNetworkService.get_company_network("2330.TW")
        self.assertIsNotNone(network)
        self.assertEqual(network["name"], "台灣積體電路製造股份有限公司")
        self.assertEqual(network["unified_id"], "22099131")
        self.assertEqual(network["representative"], "魏_家")
        self.assertEqual(network["capital"], 280500000000)
        self.assertIn("億元", network["capital_formatted"])
        self.assertEqual(network["web_url"], "https://taiwan-company-network.david888.com/graph?company=test")
        self.assertEqual(network["image_url"], "https://taiwan-company-network.david888.com/api/v1/images/test.svg")

        # Verify downstream enrichment with tickers
        downstream = network["relationships"]["downstream"]
        self.assertEqual(len(downstream), 3)

        # 創意電子
        creative = next(item for item in downstream if item["name"] == "創意電子股份有限公司")
        self.assertTrue(creative["is_listed"])
        self.assertEqual(creative["code"], "3443")
        self.assertEqual(creative["ticker"], "3443.TW")
        self.assertEqual(creative["market"], "TWSE")

        # 精材科技 (TPEx)
        xintec = next(item for item in downstream if item["name"] == "精材科技股份有限公司")
        self.assertTrue(xintec["is_listed"])
        self.assertEqual(xintec["code"], "3374")
        self.assertEqual(xintec["ticker"], "3374.TWO")
        self.assertEqual(xintec["market"], "TPEx")

        # 未上市公司
        unlisted = next(item for item in downstream if item["name"] == "未上市公司科技")
        self.assertFalse(unlisted["is_listed"])
        self.assertIsNone(unlisted["code"])

        # Upstream
        upstream = network["relationships"]["upstream"]
        self.assertEqual(len(upstream), 1)
        self.assertEqual(upstream[0]["name"], "行政院國家發展基金管理會")
        self.assertFalse(upstream[0]["is_listed"])

    def test_non_tw_stock_graceful_handling(self):
        network = CompanyNetworkService.get_company_network("AAPL")
        self.assertIsNone(network)

    @patch("requests.get")
    def test_network_failure_when_no_cache(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        network = CompanyNetworkService.get_company_network("2330")
        self.assertIsNone(network)

    def test_duckdb_company_network_persistence(self):
        db = DuckDBManager()
        test_data = {
            "ticker": "TEST9999.TW",
            "raw_code": "TEST9999",
            "name": "測試積體電路製造股份有限公司",
            "unified_id": "22099131",
            "capital": 280500000000,
            "capital_formatted": "2805.00 億元",
            "representative": "測試代表",
            "address": "新竹科學園區新竹市力行六路8號",
            "relationships": {
                "upstream": [],
                "downstream": [{"name": "創意電子股份有限公司", "code": "3443", "is_listed": True}]
            },
            "web_url": "https://taiwan-company-network.david888.com/graph?company=%E6%B8%AC%E8%A9%A6%E7%A9%8D%E9%AB%94%E9%9B%BB%E8%B7%AF%E8%A3%BD%E9%80%A0%E8%82%A1%E4%BB%BD%E6%9C%89%E9%99%90%E5%85%AC%E5%8F%B8",
            "source": "taiwan-company-network"
        }
        # 1. 寫入 DuckDB (使用專用測試代號，避免污染正式股票記錄)
        ok = db.save_company_network("TEST9999", test_data)
        self.assertTrue(ok)

        # 2. 從 DuckDB 讀回
        loaded = db.get_company_network("TEST9999")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["unified_id"], "22099131")
        self.assertEqual(loaded["name"], "測試積體電路製造股份有限公司")
        self.assertEqual(loaded["capital"], 280500000000)
        self.assertEqual(len(loaded["relationships"]["downstream"]), 1)


if __name__ == "__main__":
    unittest.main()
