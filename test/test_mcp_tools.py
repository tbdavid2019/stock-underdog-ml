import unittest
from mcp_server import (
    screen_quant_stocks,
    get_industry_chain,
    get_discovery_signals,
    get_model_performance_scorecard,
)


class TestMCPTools(unittest.TestCase):
    def test_screen_quant_stocks_tool(self):
        """驗證 screen_quant_stocks FastMCP 工具執行正常"""
        res = screen_quant_stocks(direction="bullish", limit=5)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertEqual(res.get("direction"), "bullish")
        self.assertIn("data", res)
        self.assertIsInstance(res["data"], list)

        # 測試帶模型與因子參數
        res_models = screen_quant_stocks(
            direction="bullish",
            models="timesfm,timestral",
            pe_range="low",
            risk_reward=True,
            limit=5
        )
        self.assertTrue(res_models.get("success"))

    def test_get_industry_chain_tool(self):
        """驗證 get_industry_chain FastMCP 工具執行正常"""
        res = get_industry_chain(theme="ALL")
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertIn("data", res)
        self.assertIn("upstream_nodes", res["data"])
        self.assertIn("edges", res["data"])

    def test_get_discovery_signals_tool(self):
        """驗證 get_discovery_signals FastMCP 工具執行正常"""
        res = get_discovery_signals(limit=5)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertIn("data", res)
        self.assertIsInstance(res["data"], list)

    def test_get_model_performance_scorecard_tool(self):
        """驗證 get_model_performance_scorecard FastMCP 工具執行正常"""
        res = get_model_performance_scorecard(window=60)
        self.assertIsInstance(res, dict)
        self.assertTrue(res.get("success"))
        self.assertIn("window_days", res)
        self.assertIn("data", res)


if __name__ == "__main__":
    unittest.main()
