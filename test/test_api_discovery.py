"""
test/test_api_discovery.py - 測試 Discovery 前沿挖掘機 REST API 端點
"""

import os
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from api.main import app
from data.duckdb_manager import DuckDBManager


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def seeded_db(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_api_discovery.duckdb")
    db_mgr = DuckDBManager(db_path=db_path)

    # 寫入測試用 Discovery 快照
    db_mgr.save_discovery_snapshot(
        run_id="test_run_123",
        as_of="2026-10-30",
        candidates=[
            {
                "ticker": "2330",
                "market": "TW",
                "name": "台積電",
                "theme_id": "cpo_optical_interconnect",
                "theme_name": "CPO 光互連與矽光子",
                "total_score": 88.5,
                "recommendation": "propose_add",
                "review_status": "pending",
                "in_pool": False,
                "source_score": 90.0,
                "purity_score": 95.0,
                "paper_score": 90.0,
                "novelty_score": 80.0,
                "sentiment_score": 85.0,
                "setup_score": 75.0,
                "risk_score": 15.0,
                "catalyst_summary": "矽光子 COUPE 封裝突破",
                "us_anchor": "NVDA",
                "paper_count": 3,
                "signal_count": 2,
            },
            {
                "ticker": "3017",
                "market": "TW",
                "name": "奇鋐",
                "theme_id": "liquid_cooling",
                "theme_name": "液冷散熱與熱管理架構",
                "total_score": 72.0,
                "recommendation": "observe",
                "review_status": "pending",
                "in_pool": False,
                "source_score": 80.0,
                "purity_score": 95.0,
                "paper_score": 70.0,
                "novelty_score": 60.0,
                "sentiment_score": 80.0,
                "setup_score": 70.0,
                "risk_score": 25.0,
                "catalyst_summary": "冷板與分歧管需求增長",
                "us_anchor": "VRT",
                "paper_count": 1,
                "signal_count": 1,
            }
        ],
        papers=[
            {
                "paper_id": "2410.99999",
                "title": "Silicon Photonics for AI Clusters",
                "authors": ["Alice", "Bob"],
                "abstract": "Breakthrough in low latency optical interconnect.",
                "category": "cs.AI",
                "published_date": "2026-10-30",
                "bottleneck_tag": "cpo_optical_interconnect",
                "related_tickers": ["2330", "NVDA"],
                "url": "https://arxiv.org/abs/2410.99999",
                "significance_score": 95.0
            }
        ],
        signals=[
            {
                "signal_id": "sig_001",
                "ticker": "2330",
                "theme_id": "cpo_optical_interconnect",
                "source_type": "arxiv",
                "title": "台積電 CPO 驗證通過",
                "content": "次世代光引擎封裝突破。",
                "url": "https://test.url",
                "sentiment_score": 90.0,
                "detected_at": "2026-10-30 08:00:00"
            }
        ],
        history_summary={
            "total_candidates": 2,
            "propose_add_count": 1,
            "observe_count": 1,
            "already_in_pool_count": 0,
            "paper_count": 1,
            "signal_count": 1,
            "avg_total_score": 80.25,
        }
    )

    from api.routes import discovery
    monkeypatch.setattr(discovery, "duckdb_manager", db_mgr)
    return db_mgr


def test_get_candidates_endpoint(client, seeded_db):
    resp = client.get("/api/v1/discovery/candidates")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_count"] == 2
    assert len(data["data"]) == 2
    assert data["data"][0]["ticker"] == "2330"

    # 測試篩選條件
    resp_filter = client.get("/api/v1/discovery/candidates?recommendation=propose_add")
    assert resp_filter.status_code == 200
    filter_data = resp_filter.json()
    assert filter_data["total_count"] == 1
    assert filter_data["data"][0]["recommendation"] == "propose_add"

    resp_theme = client.get("/api/v1/discovery/candidates?theme_id=liquid_cooling")
    assert resp_theme.status_code == 200
    theme_data = resp_theme.json()
    assert theme_data["total_count"] == 1
    assert theme_data["data"][0]["ticker"] == "3017"


def test_get_papers_endpoint(client, seeded_db):
    resp = client.get("/api/v1/discovery/papers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_count"] == 1
    p = data["data"][0]
    assert p["paper_id"] == "2410.99999"
    assert "Silicon Photonics" in p["title"]


def test_get_signals_endpoint(client, seeded_db):
    resp = client.get("/api/v1/discovery/signals?ticker=2330")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_count"] == 1
    assert data["data"][0]["ticker"] == "2330"


def test_get_history_endpoint(client, seeded_db):
    resp = client.get("/api/v1/discovery/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["total_count"] >= 1
    assert data["data"][0]["as_of"] == "2026-10-30"


def test_get_status_endpoint(client, seeded_db):
    resp = client.get("/api/v1/discovery/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    st = data["data"]
    assert st["status"] == "success"
    assert st["stale"] is False
    assert st["as_of"] == "2026-10-30"
    assert st["candidate_count"] == 2
