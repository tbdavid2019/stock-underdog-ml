"""
api/routes/discovery.py - AI 產業鏈前沿挖掘機 (AI Tech Radar & Discovery Engine) REST API

提供 Discovery 候選標的、前沿學術論文、產業訊號、歷史趨勢與管線健康度端點，
全面採用 read_only DuckDB 連線與本地快取優先策略，保障併發安全與低延遲。
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Query

from api.schemas import (
    DiscoveryCandidatesResponse,
    DiscoveryCandidate,
    DiscoveryPapersResponse,
    DiscoveryPaper,
    DiscoverySignalsResponse,
    DiscoverySignal,
    DiscoveryHistoryResponse,
    DiscoveryHistory,
    DiscoveryStatusResponse,
    DiscoveryStatus,
)
from data.duckdb_manager import DuckDBManager
from core.ticker_canonicalizer import canonicalize_ticker

logger = logging.getLogger("stock_app.discovery.routes")

router = APIRouter(prefix="/discovery", tags=["Discovery Tech Radar 前沿挖掘機"])
duckdb_manager = DuckDBManager()
CACHE_DIR = "cache/discovery"


def _read_cache_file(filename: str) -> Optional[Dict[str, Any]]:
    """讀取本地快照快取檔案"""
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"讀取快取 {filename} 失敗: {e}")
    return None


@router.get("/candidates", response_model=DiscoveryCandidatesResponse, summary="取得前沿挖掘標的候選清單")
def get_candidates(
    recommendation: Optional[str] = Query(None, description="推薦分類: propose_add / observe / already_in_pool / reject"),
    theme_id: Optional[str] = Query(None, description="依前沿技術瓶頸主題 ID 篩選"),
    as_of: Optional[str] = Query(None, description="指定資料基準日 (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=200, description="每頁筆數"),
    offset: int = Query(0, ge=0, description="分頁偏移量"),
):
    """
    查詢前沿技術瓶頸映射至台美股核心供應鏈之候選標的，包含 7 維多因子評分拆解。
    """
    # 1. 優先嘗試 DuckDB 唯讀查詢
    records, total_count = duckdb_manager.get_discovery_candidates(
        recommendation=recommendation,
        theme_id=theme_id,
        as_of=as_of,
        limit=limit,
        offset=offset,
        return_total=True
    )

    if records:
        status_info = duckdb_manager.get_discovery_status()
        return DiscoveryCandidatesResponse(
            success=True,
            total_count=total_count,
            as_of=records[0].get("as_of"),
            stale=status_info.get("stale", False),
            data=records
        )

    # 2. 若 DuckDB 無記錄，降級讀取本地靜態快照
    cache = _read_cache_file("candidates.json")
    if cache and "data" in cache:
        cached_data = cache["data"]
        if recommendation:
            cached_data = [c for c in cached_data if c.get("recommendation") == recommendation]
        if theme_id:
            cached_data = [c for c in cached_data if c.get("theme_id") == theme_id]
        if as_of:
            cached_data = [c for c in cached_data if c.get("as_of") == as_of]

        sliced = cached_data[offset:offset + limit]
        return DiscoveryCandidatesResponse(
            success=True,
            total_count=len(cached_data),
            as_of=cache.get("as_of"),
            stale=cache.get("stale", False),
            data=sliced
        )

    return DiscoveryCandidatesResponse(success=True, total_count=0, stale=False, data=[])


@router.get("/papers", response_model=DiscoveryPapersResponse, summary="取得前沿學術論文與瓶頸標籤")
def get_papers(
    theme_id: Optional[str] = Query(None, description="依主題瓶頸標籤篩選 (如 cpo_optical_interconnect)"),
    category: Optional[str] = Query(None, description="依 arXiv 分類篩選 (如 cs.AI, cs.RO, cs.LG, cs.CV)"),
    as_of: Optional[str] = Query(None, description="指定資料基準日 (YYYY-MM-DD)"),
    limit: int = Query(20, ge=1, le=100, description="每頁筆數"),
    offset: int = Query(0, ge=0, description="分頁偏移量"),
):
    """
    查詢 arXiv 前沿突破論文，包含結構化摘要、瓶頸特徵與關聯股票池標的。
    """
    records, total_count = duckdb_manager.get_discovery_papers(
        theme_id=theme_id,
        category=category,
        as_of=as_of,
        limit=limit,
        offset=offset,
        return_total=True
    )

    if records:
        status_info = duckdb_manager.get_discovery_status()
        return DiscoveryPapersResponse(
            success=True,
            total_count=total_count,
            as_of=records[0].get("as_of"),
            stale=status_info.get("stale", False),
            data=records
        )

    cache = _read_cache_file("papers.json")
    if cache and "data" in cache:
        cached_data = cache["data"]
        if theme_id:
            cached_data = [p for p in cached_data if p.get("bottleneck_tag") == theme_id]
        if category:
            cached_data = [p for p in cached_data if p.get("category") == category]
        if as_of:
            cached_data = [p for p in cached_data if p.get("as_of") == as_of]

        sliced = cached_data[offset:offset + limit]
        return DiscoveryPapersResponse(
            success=True,
            total_count=len(cached_data),
            as_of=cache.get("as_of"),
            stale=cache.get("stale", False),
            data=sliced
        )

    return DiscoveryPapersResponse(success=True, total_count=0, stale=False, data=[])


@router.get("/signals", response_model=DiscoverySignalsResponse, summary="取得產業前沿即時訊號流")
def get_signals(
    ticker: Optional[str] = Query(None, description="依股票代號篩選 (如 2330, NVDA)"),
    theme_id: Optional[str] = Query(None, description="依主題篩選"),
    limit: int = Query(50, ge=1, le=100, description="每頁筆數"),
    offset: int = Query(0, ge=0, description="分頁偏移量"),
):
    """
    查詢前沿訊號事件流與情緒評分。
    """
    canon = canonicalize_ticker(ticker) if ticker else None
    records, total_count = duckdb_manager.get_discovery_signals(
        ticker=canon,
        theme_id=theme_id,
        limit=limit,
        offset=offset,
        return_total=True
    )

    if records:
        status_info = duckdb_manager.get_discovery_status()
        return DiscoverySignalsResponse(
            success=True,
            total_count=total_count,
            as_of=records[0].get("as_of"),
            stale=status_info.get("stale", False),
            data=records
        )

    cache = _read_cache_file("signals.json")
    if cache and "data" in cache:
        cached_data = cache["data"]
        if canon:
            cached_data = [s for s in cached_data if canonicalize_ticker(s.get("ticker", "")) == canon]
        if theme_id:
            cached_data = [s for s in cached_data if s.get("theme_id") == theme_id]

        sliced = cached_data[offset:offset + limit]
        return DiscoverySignalsResponse(
            success=True,
            total_count=len(cached_data),
            as_of=cache.get("as_of"),
            stale=cache.get("stale", False),
            data=sliced
        )

    return DiscoverySignalsResponse(success=True, total_count=0, stale=False, data=[])


@router.get("/history", response_model=DiscoveryHistoryResponse, summary="取得 Discovery 歷史時序趨勢")
def get_history(
    limit: int = Query(30, ge=1, le=100, description="天數長度")
):
    """
    查詢前沿候選標的數量、論文數量與訊號數量之歷史趨勢。
    """
    records = duckdb_manager.get_discovery_history(limit=limit)
    if records:
        return DiscoveryHistoryResponse(success=True, total_count=len(records), data=records)

    cache = _read_cache_file("history.json")
    if cache and "data" in cache:
        cached_data = cache["data"][:limit]
        return DiscoveryHistoryResponse(success=True, total_count=len(cached_data), data=cached_data)

    return DiscoveryHistoryResponse(success=True, total_count=0, data=[])


@router.get("/status", response_model=DiscoveryStatusResponse, summary="取得 Discovery 管線健康度與新鮮度資訊")
def get_status():
    """
    回傳前沿雷達資料更新狀態、最近一次批次運行 ID、資料新鮮度與降級 Stale 標記。
    """
    cache = _read_cache_file("status.json")
    if cache and "data" in cache:
        return DiscoveryStatusResponse(success=True, data=cache["data"])

    db_status = duckdb_manager.get_discovery_status()
    return DiscoveryStatusResponse(
        success=True,
        data=DiscoveryStatus(
            status=db_status.get("status", "success"),
            stale=db_status.get("stale", False),
            as_of=db_status.get("as_of", "1970-01-01"),
            run_id=db_status.get("run_id"),
            last_run_timestamp=db_status.get("last_run_timestamp"),
            candidate_count=db_status.get("candidate_count", 0),
            paper_count=db_status.get("paper_count", 0),
            signal_count=db_status.get("signal_count", 0),
            message=db_status.get("message", "DuckDB state")
        )
    )
