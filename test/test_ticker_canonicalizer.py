"""
test/test_ticker_canonicalizer.py - 測試標的代號正規化與 Universe 白名單校驗
"""

import pytest
from core.ticker_canonicalizer import (
    canonicalize_ticker,
    detect_market,
    TickerUniverseValidator,
    default_validator,
)


def test_canonicalize_taiwan_tickers():
    assert canonicalize_ticker("2330.TW") == "2330"
    assert canonicalize_ticker("2330.two") == "2330"
    assert canonicalize_ticker("2330.TWO") == "2330"
    assert canonicalize_ticker("2330") == "2330"
    assert canonicalize_ticker("  2454.tw  ") == "2454"
    assert canonicalize_ticker("6442.TW") == "6442"
    assert canonicalize_ticker("0050.TW") == "0050"


def test_canonicalize_us_tickers():
    assert canonicalize_ticker("nvda") == "NVDA"
    assert canonicalize_ticker("AAPL") == "AAPL"
    assert canonicalize_ticker("  msft  ") == "MSFT"
    assert canonicalize_ticker("brk.b") == "BRK.B"
    assert canonicalize_ticker("BRK-B") == "BRK.B"
    assert canonicalize_ticker("  brk-b  ") == "BRK.B"
    assert canonicalize_ticker("GOOGL") == "GOOGL"


def test_canonicalize_invalid_tickers():
    assert canonicalize_ticker("") is None
    assert canonicalize_ticker("   ") is None
    assert canonicalize_ticker(None) is None
    assert canonicalize_ticker("INVALID_TICKER_12345") is None
    assert canonicalize_ticker("!@#$%") is None
    assert canonicalize_ticker("2330.XYZ") is None


def test_detect_market():
    assert detect_market("2330.TW") == "TW"
    assert detect_market("2454") == "TW"
    assert detect_market("NVDA") == "US"
    assert detect_market("aapl") == "US"
    assert detect_market("invalid_code_xyz") == "UNKNOWN"
    assert detect_market("") == "UNKNOWN"


def test_whitelist_filtering():
    custom_whitelist = {"2330", "2454", "NVDA", "TSLA"}
    validator = TickerUniverseValidator()

    assert validator.is_valid("2330.TW", whitelist=custom_whitelist) is True
    assert validator.is_valid("nvda", whitelist=custom_whitelist) is True
    assert validator.is_valid("9999.TW", whitelist=custom_whitelist) is False
    assert validator.is_valid("UNKNOWN", whitelist=custom_whitelist) is False
    assert validator.is_valid("2330.TW", whitelist=set()) is False
    assert validator.is_valid("NVDA", whitelist=set()) is False

    raw_candidates = ["2330.TW", "2454", "nvda", "9999.TW", "FAKE_STOCK", ""]
    valid, rejected = validator.filter_and_canonicalize(raw_candidates, whitelist=custom_whitelist)

    assert valid == ["2330", "2454", "NVDA"]
    assert "9999.TW" in rejected
    assert "FAKE_STOCK" in rejected
    assert "" in rejected


def test_discovery_supply_chain_whitelist():
    validator = TickerUniverseValidator()
    wl = validator.get_whitelist(force_refresh=True)

    # 驗證前沿技術鏈與美股錨點均已自動納入系統白名單
    assert "6442" in wl  # 光聖
    assert "3450" in wl  # 聯鈞
    assert "3017" in wl  # 奇鋐
    assert "3324" in wl  # 雙鴻
    assert "VRT" in wl   # Vertiv
    assert "NVDA" in wl  # NVIDIA
    assert "COHR" in wl  # Coherent

    assert validator.is_valid("6442.TW") is True
    assert validator.is_valid("VRT") is True
    assert validator.is_valid("COHR") is True
