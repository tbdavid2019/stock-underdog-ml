import unittest
from fastapi.testclient import TestClient
from api.main import app


class TestQuantScreenerAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_get_active_models_endpoint(self):
        """驗證 GET /api/v1/models 動態取得啟用的時序模型外掛列表"""
        resp = self.client.get("/api/v1/models")
        self.assertEqual(resp.status_code, 200)
        models = resp.json()
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 3)

        model_ids = [m.get("id") for m in models]
        self.assertIn("timesfm", model_ids)
        self.assertIn("timestral", model_ids)
        self.assertIn("moirai", model_ids)

        for m in models:
            self.assertIn("id", m)
            self.assertIn("name", m)
            self.assertIn("display_name", m)
            self.assertIn("author", m)

    def test_screen_predictions_default_bullish(self):
        """驗證 GET /api/v1/predictions/screen 預設多方篩選結構"""
        resp = self.client.get("/api/v1/predictions/screen?direction=bullish&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("success", data)
        self.assertTrue(data["success"])
        self.assertIn("count", data)
        self.assertIn("data", data)
        self.assertIsInstance(data["data"], list)

    def test_screen_predictions_bearish(self):
        """驗證 GET /api/v1/predictions/screen 空方避險篩選"""
        resp = self.client.get("/api/v1/predictions/screen?direction=bearish&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["data"], list)

    def test_screen_predictions_with_models_and_factors(self):
        """驗證帶有時序模型與因子條件之篩選"""
        resp = self.client.get(
            "/api/v1/predictions/screen?direction=bullish&models=timesfm,timestral&pe_range=reasonable&min_potential=3.0&limit=10"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIsInstance(data["data"], list)


    def test_screen_predictions_with_index_name(self):
        """驗證帶有 index_name 參數時不引發 IndexError，正常返回 200"""
        resp = self.client.get("/api/v1/predictions/screen?direction=bullish&index_name=台灣50&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_screen_predictions_with_multi_market_index_name(self):
        """驗證支援多選市場 (逗號分隔，例如 '台灣50,SP500') 查詢"""
        resp = self.client.get("/api/v1/predictions/screen?direction=bullish&index_name=台灣50,SP500&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_latest_predictions_with_multi_market_index_name(self):
        """驗證 /latest 支援逗號分隔多選市場指數"""
        resp = self.client.get("/api/v1/predictions/latest?index_name=台灣50,日經225&limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_duckdb_manager_parse_index_names(self):
        """驗證 DuckDBManager._parse_index_names 多市場解析與安全過濾"""
        from data.duckdb_manager import DuckDBManager
        self.assertEqual(DuckDBManager._parse_index_names(None), [])
        self.assertEqual(DuckDBManager._parse_index_names("ALL"), [])
        self.assertEqual(DuckDBManager._parse_index_names("全部市場"), [])
        self.assertEqual(DuckDBManager._parse_index_names("台灣50"), ["台灣50"])
        self.assertEqual(DuckDBManager._parse_index_names("台灣50,SP500, ALL"), ["台灣50", "SP500"])
        clause, params = DuckDBManager._build_index_filter_clause(["台灣50", "SP500"])
        self.assertEqual(clause, "index_name IN (?, ?)")
        self.assertEqual(params, ["台灣50", "SP500"])

    def test_screen_predictions_with_moirai_alias_and_invalid_model(self):
        """驗證帶有 moirai-2 別名或未知模型時不引發 BinderException，安全過濾"""
        resp = self.client.get("/api/v1/predictions/screen?direction=bullish&models=moirai-2,unknown_xyz&limit=5")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])

    def test_screen_predictions_with_core_quant_factors(self):
        """驗證核心量化因子 (盈虧比、季線支撐、法人合買)"""
        resp = self.client.get(
            "/api/v1/predictions/screen?direction=bullish&risk_reward=true&ma60=true&institutional=true&limit=5"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])


    def test_all_strategy_endpoints_with_multi_market(self):
        """驗證所有策略端點皆支援逗號分隔多選市場指數"""
        endpoints = [
            "/api/v1/predictions/resonance?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/tri-bullish?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/dual-bullish?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/dual-bearish?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/xuantie?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/timestral/top-bullish?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/timesfm/top-bullish?index_name=台灣50,SP500&limit=5",
            "/api/v1/predictions/moirai/top-bullish?index_name=台灣50,SP500&limit=5"
        ]
        for ep in endpoints:
            with self.subTest(endpoint=ep):
                resp = self.client.get(ep)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(data.get("success", True))

    def test_screen_predictions_with_lstm_model(self):
        """驗證當傳入 models=lstm 時映射到 potential 欄位，不引發 BinderException"""
        resp = self.client.get("/api/v1/predictions/screen?models=lstm")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success", True))


if __name__ == "__main__":
    unittest.main()
