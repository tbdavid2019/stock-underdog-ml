"""
test/test_ticker_canonicalizer.py - 測試標的代號正規化與 Universe 白名單校驗
"""

import unittest
from core.ticker_canonicalizer import (
    canonicalize_ticker,
    detect_market,
    TickerUniverseValidator,
    default_validator,
)


class TestTickerCanonicalizer(unittest.TestCase):
    def test_canonicalize_taiwan_tickers(self):
        self.assertEqual(canonicalize_ticker("2330.TW"), "2330")
        self.assertEqual(canonicalize_ticker("2330.two"), "2330")
        self.assertEqual(canonicalize_ticker("2330.TWO"), "2330")
        self.assertEqual(canonicalize_ticker("2330"), "2330")
        self.assertEqual(canonicalize_ticker("  2454.tw  "), "2454")
        self.assertEqual(canonicalize_ticker("6442.TW"), "6442")
        self.assertEqual(canonicalize_ticker("0050.TW"), "0050")

    def test_canonicalize_us_tickers(self):
        self.assertEqual(canonicalize_ticker("nvda"), "NVDA")
        self.assertEqual(canonicalize_ticker("AAPL"), "AAPL")
        self.assertEqual(canonicalize_ticker("  msft  "), "MSFT")
        self.assertEqual(canonicalize_ticker("brk.b"), "BRK.B")
        self.assertEqual(canonicalize_ticker("BRK-B"), "BRK.B")
        self.assertEqual(canonicalize_ticker("  brk-b  "), "BRK.B")
        self.assertEqual(canonicalize_ticker("GOOGL"), "GOOGL")

    def test_canonicalize_invalid_tickers(self):
        self.assertIsNone(canonicalize_ticker(""))
        self.assertIsNone(canonicalize_ticker("   "))
        self.assertIsNone(canonicalize_ticker(None))
        self.assertIsNone(canonicalize_ticker("INVALID_TICKER_12345"))
        self.assertIsNone(canonicalize_ticker("!@#$%"))
        self.assertIsNone(canonicalize_ticker("2330.XYZ"))

    def test_detect_market(self):
        self.assertEqual(detect_market("2330.TW"), "TW")
        self.assertEqual(detect_market("2454"), "TW")
        self.assertEqual(detect_market("NVDA"), "US")
        self.assertEqual(detect_market("aapl"), "US")
        self.assertEqual(detect_market("invalid_code_xyz"), "UNKNOWN")
        self.assertEqual(detect_market(""), "UNKNOWN")

    def test_whitelist_filtering(self):
        custom_whitelist = {"2330", "2454", "NVDA", "TSLA"}
        validator = TickerUniverseValidator()

        self.assertTrue(validator.is_valid("2330.TW", whitelist=custom_whitelist))
        self.assertTrue(validator.is_valid("nvda", whitelist=custom_whitelist))
        self.assertFalse(validator.is_valid("9999.TW", whitelist=custom_whitelist))
        self.assertFalse(validator.is_valid("UNKNOWN", whitelist=custom_whitelist))
        self.assertFalse(validator.is_valid("2330.TW", whitelist=set()))
        self.assertFalse(validator.is_valid("NVDA", whitelist=set()))

        raw_candidates = ["2330.TW", "2454", "nvda", "9999.TW", "FAKE_STOCK", ""]
        valid, rejected = validator.filter_and_canonicalize(raw_candidates, whitelist=custom_whitelist)

        self.assertEqual(valid, ["2330", "2454", "NVDA"])
        self.assertIn("9999.TW", rejected)
        self.assertIn("FAKE_STOCK", rejected)
        self.assertIn("", rejected)

    def test_discovery_supply_chain_whitelist(self):
        validator = TickerUniverseValidator()
        wl = validator.get_whitelist(force_refresh=True)

        # 驗證前沿技術鏈與美股錨點均已自動納入系統白名單
        self.assertIn("6442", wl)  # 光聖
        self.assertIn("3450", wl)  # 聯鈞
        self.assertIn("3017", wl)  # 奇鋐
        self.assertIn("3324", wl)  # 雙鴻
        self.assertIn("VRT", wl)   # Vertiv
        self.assertIn("NVDA", wl)  # NVIDIA
        self.assertIn("COHR", wl)  # Coherent

        self.assertTrue(validator.is_valid("6442.TW"))
        self.assertTrue(validator.is_valid("VRT"))
        self.assertTrue(validator.is_valid("COHR"))


if __name__ == "__main__":
    unittest.main()
