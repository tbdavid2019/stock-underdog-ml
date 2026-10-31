"""
test/test_industry_chain.py - 產業價值鏈與實體關係圖譜單元測試

涵蓋：
1. IndustryChainFetcher 離線備援、主題分類、拓撲邊過濾與快取機制
2. DuckDBManager 產業鏈圖譜資料表建立、批次 Upsert 與多維行情補強查詢
3. FastAPI REST 端點 (/api/v1/industry-chain/themes 與 /api/v1/industry-chain/graph)
"""

import os
import shutil
import tempfile
import unittest
from fastapi.testclient import TestClient

from api.main import app
from data.industry_chain_fetcher import IndustryChainFetcher, THEME_DEFINITIONS
from data.duckdb_manager import DuckDBManager


class TestIndustryChainFetcher(unittest.TestCase):
    """測試產業鏈擷取與圖譜拓撲邏輯"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fetcher = IndustryChainFetcher(cache_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_available_themes(self):
        themes = self.fetcher.get_available_themes()
        self.assertGreaterEqual(len(themes), 8)
        theme_ids = [t["id"] for t in themes]
        expected_themes = [
            "ALL", "HBM_STORAGE", "CPO_OPTICAL", "POWER_COOLING",
            "ADV_PACKAGING", "AI_CHIPS", "SERVER_ODM", "ROBOTICS"
        ]
        for expected in expected_themes:
            self.assertIn(expected, theme_ids)

        all_theme = next(t for t in themes if t["id"] == "ALL")
        self.assertGreater(all_theme["count"], 50)
        self.assertGreater(all_theme["upstream_count"], 0)
        self.assertGreater(all_theme["midstream_count"], 0)
        self.assertGreater(all_theme["downstream_count"], 0)

    def test_get_theme_graph_all(self):
        graph = self.fetcher.get_theme_graph("ALL", "ALL")
        self.assertEqual(graph["theme"], "ALL")
        self.assertEqual(graph["relation_filter"], "ALL")
        self.assertGreater(graph["total_nodes"], 50)
        self.assertGreater(len(graph["upstream_nodes"]), 0)
        self.assertGreater(len(graph["midstream_nodes"]), 0)
        self.assertGreater(len(graph["downstream_nodes"]), 0)
        self.assertGreater(len(graph["edges"]), 20)

        sample_node = graph["upstream_nodes"][0]
        self.assertIn("ticker", sample_node)
        self.assertIn("company", sample_node)
        self.assertIn("market", sample_node)
        self.assertIn("layer", sample_node)
        self.assertIn("role", sample_node)

        sample_edge = graph["edges"][0]
        self.assertIn("source", sample_edge)
        self.assertIn("target", sample_edge)
        self.assertIn("relation_type", sample_edge)
        self.assertIn("weight", sample_edge)

    def test_get_theme_graph_filtered(self):
        cpo_graph = self.fetcher.get_theme_graph("CPO_OPTICAL", "ALL")
        self.assertEqual(cpo_graph["theme"], "CPO_OPTICAL")
        for node in cpo_graph["upstream_nodes"] + cpo_graph["midstream_nodes"] + cpo_graph["downstream_nodes"]:
            self.assertIn("CPO_OPTICAL", node["theme_tags"])

        # 測試關係邊篩選
        dep_graph = self.fetcher.get_theme_graph("ALL", "大客戶依賴")
        for edge in dep_graph["edges"]:
            self.assertEqual(edge["relation_type"], "大客戶依賴")

    def test_tpex_fetch_and_fallback(self):
        res_d = self.fetcher.fetch_tpex_chain("D000")
        self.assertEqual(res_d["chain_id"], "D000")
        self.assertIn("nodes", res_d)
        self.assertGreater(len(res_d["nodes"]), 0)

        # 驗證快取檔案已寫入
        cache_file = os.path.join(self.temp_dir, "tpex_D000.json")
        self.assertTrue(os.path.exists(cache_file))

        # 驗證讀取快取
        res_cached = self.fetcher.fetch_tpex_chain("D000")
        self.assertEqual(len(res_cached["nodes"]), len(res_d["nodes"]))

    def test_sec_major_customers(self):
        sec_data = self.fetcher.fetch_sec_major_customers()
        self.assertIn("customers", sec_data)
        customers = sec_data["customers"]
        self.assertGreaterEqual(len(customers), 5)
        suppliers = [c["supplier"] for c in customers]
        self.assertIn("2330", suppliers)
        self.assertIn("NVDA", suppliers)


class TestDuckDBIndustryPersistence(unittest.TestCase):
    """測試 DuckDB 產業鏈表格結構、讀寫與量化注入"""

    def setUp(self):
        self.temp_db = tempfile.mktemp(suffix=".duckdb")
        self.db = DuckDBManager(db_path=self.temp_db)

    def tearDown(self):
        if os.path.exists(self.temp_db):
            try:
                os.remove(self.temp_db)
            except Exception:
                pass

    def test_init_and_save_nodes(self):
        self.db.init_industry_chain_tables()

        nodes = [
            {
                "ticker": "2330",
                "company": "台積電",
                "market": "TW",
                "chain_id": "D000",
                "layer": "midstream",
                "role": "晶圓代工與先進封裝",
                "key_focus": "3nm/2nm 製程",
                "theme_tags": ["AI_CHIPS", "ADV_PACKAGING"]
            },
            {
                "ticker": "3017",
                "company": "奇鋐",
                "market": "TW",
                "chain_id": "E000",
                "layer": "upstream",
                "role": "散熱水冷板",
                "key_focus": "3D VC",
                "theme_tags": ["POWER_COOLING"]
            }
        ]
        saved = self.db.save_industry_nodes_batch(nodes)
        self.assertEqual(saved, 2)

        # 查詢上游
        up_nodes = self.db.get_industry_nodes(layer="upstream")
        self.assertEqual(len(up_nodes), 1)
        self.assertEqual(up_nodes[0]["ticker"], "3017")
        self.assertEqual(up_nodes[0]["theme_tags"], ["POWER_COOLING"])

        # 查詢所有
        all_nodes = self.db.get_industry_nodes()
        self.assertEqual(len(all_nodes), 2)

    def test_save_and_get_relations(self):
        self.db.init_industry_chain_tables()

        rels = [
            {
                "source_ticker": "2330",
                "target_ticker": "NVDA",
                "relation_type": "大客戶依賴",
                "description": "NVDA 為台積電第一大 AI 客戶",
                "weight": 0.95
            },
            {
                "source_ticker": "3017",
                "target_ticker": "NVDA",
                "relation_type": "產業鏈",
                "description": "奇鋐供應 GB200 水冷板",
                "weight": 0.92
            }
        ]
        saved = self.db.save_industry_relations_batch(rels)
        self.assertEqual(saved, 2)

        queried = self.db.get_industry_relations(source_ticker="2330")
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried[0]["target_ticker"], "NVDA")

        queried_target = self.db.get_industry_relations(target_ticker="NVDA")
        self.assertEqual(len(queried_target), 2)

    def test_sync_to_duckdb(self):
        fetcher = IndustryChainFetcher()
        sync_stats = fetcher.sync_to_duckdb(self.db)
        self.assertGreater(sync_stats["nodes"], 50)
        self.assertGreater(sync_stats["relations"], 100)
        self.assertEqual(sync_stats["chains"], len(THEME_DEFINITIONS))

        # 驗證 tw_industry_chains 資料表確實已成功寫入持久化
        with self.db._get_connection(read_only=True) as con:
            count = con.execute("SELECT COUNT(*) FROM tw_industry_chains").fetchone()[0]
            self.assertEqual(count, len(THEME_DEFINITIONS))

    def test_atomic_replacement_and_dedup(self):
        """測試 DuckDB 節點與邊之記憶體去重、原子事務及平滑升級"""
        self.db.init_industry_chain_tables()

        # 包含重複鍵之節點列表 (ticker, chain_id, layer)
        dup_nodes = [
            {"ticker": "2330", "company": "台積電", "market": "TW", "chain_id": "D000", "layer": "midstream", "role": "代工", "theme_tags": ["AI"]},
            {"ticker": "2330", "company": "台積電", "market": "TW", "chain_id": "D000", "layer": "midstream", "role": "代工新版", "theme_tags": ["AI"]},
        ]
        saved = self.db.save_industry_nodes_batch(dup_nodes, clear_existing=True)
        self.assertEqual(saved, 1)

        # 驗證只有 1 筆被保存且為最新
        queried = self.db.get_industry_nodes()
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried[0]["role"], "代工新版")

        # 測試自動升級：當 DuckDB 中資料小於內嵌版本時，get_theme_graph 自動補齊同步
        fetcher = IndustryChainFetcher()
        graph_data = fetcher.get_theme_graph(theme="ALL", duckdb_mgr=self.db)
        self.assertGreaterEqual(graph_data["total_nodes"], 88)
        self.assertGreaterEqual(len(graph_data["edges"]), 150)

    def test_enriched_industry_quotes(self):
        self.db.init_industry_chain_tables()
        # 模擬分開儲存 Timestral 與 TimesFM 兩筆同時間預測記錄
        pred_records = [
            {
                "ticker": "2330.TW",
                "model_name": "Timestral",
                "current_price": 1050.0,
                "potential": 4.5,
                "timestral_potential": 4.5,
                "trust_net_5d": 12500,
                "timestamp": "2026-10-31 10:00:00"
            },
            {
                "ticker": "2330.TW",
                "model_name": "TimesFM",
                "current_price": 1050.0,
                "potential": 3.8,
                "timesfm_potential": 3.8,
                "dual_model_consensus": True,
                "timestamp": "2026-10-31 10:00:00"
            }
        ]
        self.db.save_predictions_batch(pred_records)

        enriched = self.db.get_enriched_industry_quotes(["2330", "NVDA"])
        self.assertIn("2330", enriched)
        self.assertEqual(enriched["2330"]["current_price"], 1050.0)
        # 驗證兩模型預測已聚合合併，而非任意丟棄某一模型數值
        self.assertEqual(enriched["2330"]["timestral_potential"], 4.5)
        self.assertEqual(enriched["2330"]["timesfm_potential"], 3.8)
        self.assertTrue(enriched["2330"]["dual_model_consensus"])
        self.assertIn("12,500", enriched["2330"]["institutional_streak"])


class TestIndustryChainAPI(unittest.TestCase):
    """測試 FastAPI REST 端點"""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_themes_endpoint(self):
        resp = self.client.get("/api/v1/industry-chain/themes")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertGreaterEqual(data["total_themes"], 8)
        theme_names = [t["id"] for t in data["data"]]
        self.assertIn("HBM_STORAGE", theme_names)
        self.assertIn("CPO_OPTICAL", theme_names)

    def test_get_graph_endpoint_default(self):
        resp = self.client.get("/api/v1/industry-chain/graph")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["theme"], "ALL")
        self.assertGreater(data["total_nodes"], 50)
        self.assertGreater(len(data["upstream_nodes"]), 0)
        self.assertGreater(len(data["midstream_nodes"]), 0)
        self.assertGreater(len(data["downstream_nodes"]), 0)
        self.assertGreater(len(data["edges"]), 20)

    def test_get_graph_endpoint_with_filters(self):
        resp = self.client.get("/api/v1/industry-chain/graph?theme=POWER_COOLING&relation_type=大客戶依賴")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["theme"], "POWER_COOLING")
        self.assertEqual(data["relation_filter"], "大客戶依賴")
        for edge in data["edges"]:
            self.assertEqual(edge["relation_type"], "大客戶依賴")


if __name__ == "__main__":
    unittest.main()
