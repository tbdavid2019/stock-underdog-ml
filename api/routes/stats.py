"""
api/routes/stats.py - Database Analytics & Summary Endpoints
"""

from fastapi import APIRouter, Depends
from api.schemas import DBStatsResponse
from data.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/stats", tags=["Database & System Stats"])


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
        models=stats.get("models", []),
        db_path=stats.get("db_path", "")
    )
