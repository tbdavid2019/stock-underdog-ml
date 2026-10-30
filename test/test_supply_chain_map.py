"""
test/test_supply_chain_map.py - 測試前沿技術瓶頸與供應鏈映射庫
"""

import pytest
from core.supply_chain_map import SupplyChainMap


def test_supply_chain_map_loads_themes():
    scm = SupplyChainMap()
    themes = scm.get_all_themes()
    assert len(themes) >= 5

    theme_ids = [t["id"] for t in themes]
    assert "cpo_optical_interconnect" in theme_ids
    assert "hbm_kv_cache" in theme_ids
    assert "liquid_cooling" in theme_ids
    assert "embodied_ai_robotics" in theme_ids
    assert "advanced_packaging_cowos" in theme_ids


def test_theme_query_and_anchors():
    scm = SupplyChainMap()
    cpo = scm.get_theme("cpo_optical_interconnect")
    assert cpo is not None
    assert "CPO" in cpo["name_zh"]

    anchors = scm.get_anchors_for_theme("cpo_optical_interconnect")
    assert "NVDA" in anchors
    assert "AVGO" in anchors

    tw_chain = scm.get_supply_chain_for_theme("cpo_optical_interconnect")
    tickers = [item["ticker"] for item in tw_chain]
    assert "2330" in tickers
    assert "2345" in tickers
    assert "6442" in tickers
    assert "3450" in tickers


def test_reverse_lookup_ticker():
    scm = SupplyChainMap()
    # 測試 2330 台積電
    tsmc_themes = scm.get_themes_for_ticker("2330.TW")
    assert len(tsmc_themes) >= 2  # CPO, HBM, CoWoS 等
    theme_ids = [t["theme_id"] for t in tsmc_themes]
    assert "advanced_packaging_cowos" in theme_ids

    anchors = scm.get_anchors_for_ticker("2330")
    assert "NVDA" in anchors

    # 測試 3017 奇鋐 (液冷)
    cooling_themes = scm.get_themes_for_ticker("3017")
    assert any(t["theme_id"] == "liquid_cooling" for t in cooling_themes)
    cooling_anchors = scm.get_anchors_for_ticker("3017")
    assert "VRT" in cooling_anchors


def test_paper_to_theme_mapping():
    scm = SupplyChainMap()
    title = "Co-Packaged Optics for Exascale AI Clusters"
    abstract = "We demonstrate silicon photonics transceiver integration with high bandwidth memory..."
    matched = scm.map_paper_to_themes(title, abstract)
    assert "cpo_optical_interconnect" in matched
    assert "hbm_kv_cache" in matched


def test_all_supply_chain_tickers_set():
    scm = SupplyChainMap()
    all_tickers = scm.get_all_supply_chain_tickers()
    assert "2330" in all_tickers
    assert "2454" in all_tickers
    assert "3017" in all_tickers
    assert "3324" in all_tickers
    assert "5274" in all_tickers
    assert "2345" in all_tickers
    assert "6442" in all_tickers
    assert "3450" in all_tickers
    assert "NVDA" in all_tickers
