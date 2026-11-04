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


if __name__ == "__main__":
    unittest.main()
