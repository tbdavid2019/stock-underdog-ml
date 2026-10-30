"""
test/test_discovery_pipeline.py - 測試 Discovery 資料管線、原子快取與 DuckDB 持久化
"""

import os
import json
import tempfile
import pytest
from data.duckdb_manager import DuckDBManager
from pipeline.discovery_pipeline import DiscoveryPipeline


@pytest.fixture
def temp_env():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_stocks.duckdb")
    cache_dir = os.path.join(temp_dir, "cache", "discovery")
    db_mgr = DuckDBManager(db_path=db_path)
    pipeline = DiscoveryPipeline(db_manager=db_mgr, cache_dir=cache_dir)
    yield pipeline, db_mgr, cache_dir


def test_parse_arxiv_atom(temp_env):
    pipeline, _, _ = temp_env
    sample_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2410.99999v1</id>
        <title>Co-Packaged Optics and Silicon Photonics for Next-Gen Interconnect</title>
        <summary>This paper investigates CPO optical transceiver efficiency and laser diode thermal management.</summary>
        <published>2026-10-30T00:00:00Z</published>
        <author><name>Dr. Alice</name></author>
        <author><name>Dr. Bob</name></author>
      </entry>
    </feed>
    """
    papers = pipeline._parse_arxiv_atom(sample_xml, "cs.AI")
    assert len(papers) == 1
    p = papers[0]
    assert p["paper_id"] == "2410.99999v1"
    assert "Co-Packaged Optics" in p["title"]
    assert p["bottleneck_tag"] == "cpo_optical_interconnect"
    assert "2330" in p["related_tickers"]


def test_pipeline_run_and_atomic_persistence(temp_env):
    pipeline, db_mgr, cache_dir = temp_env

    mock_papers = [
        {
            "paper_id": "2410.11111",
            "title": "Extreme Liquid Cooling for AI Datacenters",
            "authors": ["Engineer X"],
            "abstract": "We analyze cold plate CDU liquid cooling thermal barriers in 1000W GPUs.",
            "category": "cs.RO",
            "published_date": "2026-10-30",
            "bottleneck_tag": "liquid_cooling",
            "related_tickers": ["3017", "3324", "VRT"],
            "url": "https://arxiv.org/abs/2410.11111",
            "significance_score": 90.0
        }
    ]

    status = pipeline.run(as_of="2026-10-30", mock_papers=mock_papers)

    assert status["status"] == "success"
    assert status["stale"] is False
    assert status["candidate_count"] > 0
    assert status["paper_count"] == 1

    # 驗證 DuckDB 資料庫持久化
    candidates = db_mgr.get_discovery_candidates(as_of="2026-10-30")
    assert len(candidates) > 0
    # 奇鋐 3017 或 雙鴻 3324 應存在且評分正確
    tickers = [c["ticker"] for c in candidates]
    assert "3017" in tickers or "3324" in tickers

    papers = db_mgr.get_discovery_papers(as_of="2026-10-30")
    assert len(papers) == 1
    assert papers[0]["paper_id"] == "2410.11111"

    signals = db_mgr.get_discovery_signals()
    assert len(signals) > 0

    history = db_mgr.get_discovery_history()
    assert len(history) == 1
    assert history[0]["as_of"] == "2026-10-30"

    # 同日重跑應完全冪等，history 不應重複累積
    pipeline.run(as_of="2026-10-30", mock_papers=mock_papers)
    history_after = db_mgr.get_discovery_history()
    assert len(history_after) == 1
    assert history_after[0]["as_of"] == "2026-10-30"

    # 驗證原子快取檔案生成
    assert os.path.exists(os.path.join(cache_dir, "candidates.json"))
    assert os.path.exists(os.path.join(cache_dir, "papers.json"))
    assert os.path.exists(os.path.join(cache_dir, "signals.json"))
    assert os.path.exists(os.path.join(cache_dir, "history.json"))
    assert os.path.exists(os.path.join(cache_dir, "status.json"))

    with open(os.path.join(cache_dir, "candidates.json"), "r") as f:
        cache_data = json.load(f)
        assert cache_data["success"] is True
        assert cache_data["total_count"] == len(candidates)


def test_pipeline_fallback_on_network_error(temp_env):
    pipeline, db_mgr, cache_dir = temp_env

    # 1. 先執行一次正常快照
    mock_papers = [
        {
            "paper_id": "2410.22222",
            "title": "HBM4 Memory Wall Breakthrough",
            "authors": ["Researcher Y"],
            "abstract": "Die-to-die interconnect and high bandwidth memory solutions.",
            "category": "cs.LG",
            "published_date": "2026-10-29",
            "bottleneck_tag": "hbm_kv_cache",
            "related_tickers": ["2330", "5274", "NVDA"],
            "url": "https://arxiv.org/abs/2410.22222",
            "significance_score": 92.0
        }
    ]
    pipeline.run(as_of="2026-10-29", mock_papers=mock_papers)

    # 2. 模擬異常引發 Fallback 降級
    fallback_status = pipeline.run(force_fallback=True)
    assert fallback_status["status"] == "fallback"
    assert fallback_status["stale"] is True

    # 檢查 status.json 與 candidates.json 是否更新為 fallback 與 stale 標記
    with open(os.path.join(cache_dir, "status.json"), "r") as f:
        st_data = json.load(f)["data"]
        assert st_data["status"] == "fallback"
        assert st_data["stale"] is True

    with open(os.path.join(cache_dir, "candidates.json"), "r") as f:
        cand_cache = json.load(f)
        assert cand_cache["stale"] is True


def test_ttl_pruning(temp_env):
    pipeline, db_mgr, _ = temp_env

    # 寫入包含過去很舊日期的快照
    old_paper = [
        {
            "paper_id": "old_paper_001",
            "title": "Ancient Paper",
            "authors": ["Old Guy"],
            "abstract": "Old tech",
            "category": "cs.AI",
            "published_date": "2024-01-01",
            "bottleneck_tag": "cpo_optical_interconnect",
            "related_tickers": ["2330"],
            "url": "http://old",
            "significance_score": 50.0
        }
    ]
    db_mgr.save_discovery_snapshot(
        run_id="old_run",
        as_of="2024-01-01",
        candidates=[],
        papers=old_paper,
        signals=[{
            "signal_id": "old_sig_001",
            "ticker": "2330",
            "theme_id": "cpo",
            "source_type": "arxiv",
            "title": "Old Signal",
            "detected_at": "2024-01-01 00:00:00",
            "sentiment_score": 50.0
        }],
    )

    # 執行 TTL 清理
    prune_res = db_mgr.prune_discovery_expired_records(paper_ttl_days=180, signal_ttl_days=90)
    assert prune_res["pruned_papers"] >= 1
    assert prune_res["pruned_signals"] >= 1
