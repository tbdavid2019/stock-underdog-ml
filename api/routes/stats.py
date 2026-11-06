"""
api/routes/stats.py - Database Analytics & Summary Endpoints
"""

import json
import logging
import os
from fastapi import APIRouter, Depends
from api.schemas import DBStatsResponse
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stats", tags=["Database & System Stats"])
CANARY_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache", "canary_health_latest.json")


def get_duckdb():
    return DuckDBManager()


@router.get("/summary", response_model=DBStatsResponse, summary="取得本地 DuckDB 時序庫全盤統計")
def get_database_summary(db: DuckDBManager = Depends(get_duckdb)):
    """
    查詢 DuckDB 總記錄筆數、獨立標的數量、資料涵蓋之最早/最新時間戳記與指數範圍。
    """
    stats = db.get_db_stats()
    return DBStatsResponse(
        total_records=stats.get("total_records", 0),
        distinct_tickers=stats.get("distinct_tickers", 0),
        earliest_timestamp=stats.get("earliest_timestamp"),
        latest_timestamp=stats.get("latest_timestamp"),
        indices=stats.get("indices", []),
        models=stats.get("models", [])
    )


@router.get("/canary-health", summary="取得上游資料源金絲雀健康巡檢快照")
def get_canary_health_snapshot():
    """
    唯讀讀取本地金絲雀健康快照 (data/cache/canary_health_latest.json)。
    嚴禁於公開 GET 端點發起即時外部網路探測。
    """
    if not os.path.exists(CANARY_CACHE_PATH):
        return {
            "status": "not_initialized",
            "checked_at": None,
            "sources_checked_count": 0,
            "sources": {},
            "is_cached": False,
            "message": "金絲雀健康巡檢快照尚未初始化，請由排程或執行 python scripts/check_loader_health.py 產生"
        }
    try:
        with open(CANARY_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Failed to read canary health snapshot: {e}")
        return {
            "status": "read_error",
            "checked_at": None,
            "sources_checked_count": 0,
            "sources": {},
            "is_cached": False,
            "message": "Failed to read canary health snapshot"
        }
