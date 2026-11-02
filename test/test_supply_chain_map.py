"""
test/test_supply_chain_map.py - 測試前沿技術瓶頸與供應鏈映射庫
"""

import unittest
from core.supply_chain_map import SupplyChainMap


class TestSupplyChainMap(unittest.TestCase):
    def test_supply_chain_map_loads_themes(self):
        scm = SupplyChainMap()
        themes = scm.get_all_themes()
        self.assertGreaterEqual(len(themes), 5)

        theme_ids = [t["id"] for t in themes]
        self.assertIn("cpo_optical_interconnect", theme_ids)
        self.assertIn("hbm_kv_cache", theme_ids)
        self.assertIn("liquid_cooling", theme_ids)
        self.assertIn("embodied_ai_robotics", theme_ids)
        self.assertIn("advanced_packaging_cowos", theme_ids)

    def test_theme_query_and_anchors(self):
        scm = SupplyChainMap()
        cpo = scm.get_theme("cpo_optical_interconnect")
        self.assertIsNotNone(cpo)
        self.assertIn("CPO", cpo["name_zh"])

        anchors = scm.get_anchors_for_theme("cpo_optical_interconnect")
        self.assertIn("NVDA", anchors)
        self.assertIn("AVGO", anchors)

        tw_chain = scm.get_supply_chain_for_theme("cpo_optical_interconnect")
        tickers = [item["ticker"] for item in tw_chain]
        self.assertIn("2330", tickers)
        self.assertIn("2345", tickers)
        self.assertIn("6442", tickers)
        self.assertIn("3450", tickers)

    def test_reverse_lookup_ticker(self):
        scm = SupplyChainMap()
        # 測試 2330 台積電
        tsmc_themes = scm.get_themes_for_ticker("2330.TW")
        self.assertGreaterEqual(len(tsmc_themes), 2)  # CPO, HBM, CoWoS 等
        theme_ids = [t["theme_id"] for t in tsmc_themes]
        self.assertIn("advanced_packaging_cowos", theme_ids)

        anchors = scm.get_anchors_for_ticker("2330")
        self.assertIn("NVDA", anchors)

        # 測試 3017 奇鋐 (液冷)
        cooling_themes = scm.get_themes_for_ticker("3017")
        self.assertTrue(any(t["theme_id"] == "liquid_cooling" for t in cooling_themes))
        cooling_anchors = scm.get_anchors_for_ticker("3017")
        self.assertIn("VRT", cooling_anchors)

    def test_paper_to_theme_mapping(self):
        scm = SupplyChainMap()
        title = "Co-Packaged Optics for Exascale AI Clusters"
        abstract = "We demonstrate silicon photonics transceiver integration with high bandwidth memory..."
        matched = scm.map_paper_to_themes(title, abstract)
        self.assertIn("cpo_optical_interconnect", matched)
        self.assertIn("hbm_kv_cache", matched)

        # 驗證詞界比對，防範 "input", "default", "vlan" 產生短詞誤判
        false_title = "Input Data Normalization with Default Settings on VLAN Switch"
        false_abstract = "We evaluate simple neural networks on standard benchmarks without hardware bottlenecks."
        false_matched = scm.map_paper_to_themes(false_title, false_abstract)
        self.assertNotIn("cpo_optical_interconnect", false_matched)
        self.assertNotIn("embodied_ai_robotics", false_matched)

    def test_all_supply_chain_tickers_set(self):
        scm = SupplyChainMap()
        all_tickers = scm.get_all_supply_chain_tickers()
        self.assertIn("2330", all_tickers)
        self.assertIn("2454", all_tickers)
        self.assertIn("3017", all_tickers)
        self.assertIn("3324", all_tickers)
        self.assertIn("5274", all_tickers)
        self.assertIn("2345", all_tickers)
        self.assertIn("6442", all_tickers)
        self.assertIn("3450", all_tickers)
        self.assertIn("NVDA", all_tickers)

    def test_us_supply_chain_and_combined_lookup(self):
        scm = SupplyChainMap()
        # 測試液冷主題之美股供應鏈
        us_cooling = scm.get_us_supply_chain_for_theme("liquid_cooling")
        us_tickers = [item["ticker"] for item in us_cooling]
        self.assertIn("VRT", us_tickers)
        self.assertIn("SMCI", us_tickers)

        # 測試台美合併供應鏈
        combined_cooling = scm.get_combined_supply_chain_for_theme("liquid_cooling")
        comb_tickers = [item["ticker"] for item in combined_cooling]
        self.assertIn("3017", comb_tickers)
        self.assertIn("3324", comb_tickers)
        self.assertIn("VRT", comb_tickers)

        # 測試美股反向查詢
        vrt_themes = scm.get_themes_for_ticker("VRT")
        self.assertTrue(any(t["theme_id"] == "liquid_cooling" for t in vrt_themes))
        self.assertTrue(any(t["market"] == "US" for t in vrt_themes))

        tsm_themes = scm.get_themes_for_ticker("TSM")
        self.assertTrue(any(t["theme_id"] == "advanced_packaging_cowos" for t in tsm_themes))


if __name__ == "__main__":
    unittest.main()

