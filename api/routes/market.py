"""
api/routes/market.py - 台股全市場行情與三大法人籌碼 Endpoints
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from data.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/market", tags=["Market Data & Institutional Flows"])


def get_duckdb():
    return DuckDBManager()


class InstitutionalFlowItem(BaseModel):
    date: str
    ticker: str
    raw_code: Optional[str] = None
    name: Optional[str] = None
    foreign_net: Optional[int] = 0
    trust_net: Optional[int] = 0
    dealer_net: Optional[int] = 0
    total_net: Optional[int] = 0
    foreign_ratio: Optional[float] = 0.0
    market: Optional[str] = None


class MarketStatsResponse(BaseModel):
    total_bars: int
    total_institutional_records: int
    total_predictions: int
    latest_market_date: Optional[str] = None
    latest_institutional_date: Optional[str] = None


@router.get("/stats", response_model=MarketStatsResponse, summary="取得時序庫行情與法人籌碼整體統計")
def get_market_statistics(db: DuckDBManager = Depends(get_duckdb)):
    """
    查詢 DuckDB 中台股全市場日 K 棒、三大法人歷史進出表與預測記錄之數據量。
    """
    try:
        with db._get_connection() as con:
            bars_cnt = con.execute("SELECT COUNT(*) FROM tw_daily_bars").fetchone()[0]
            inst_cnt = con.execute("SELECT COUNT(*) FROM tw_institutional_daily").fetchone()[0]
            pred_cnt = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            
            latest_bar_date = con.execute("SELECT MAX(date) FROM tw_daily_bars").fetchone()[0]
            latest_inst_date = con.execute("SELECT MAX(date) FROM tw_institutional_daily").fetchone()[0]

            return MarketStatsResponse(
                total_bars=bars_cnt,
                total_institutional_records=inst_cnt,
                total_predictions=pred_cnt,
                latest_market_date=str(latest_bar_date) if latest_bar_date else None,
                latest_institutional_date=str(latest_inst_date) if latest_inst_date else None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢市場統計失敗: {e}")


@router.get("/institutional/top", response_model=List[InstitutionalFlowItem], summary="取得三大法人買賣超排行榜")
def get_top_institutional_flows(
    date: Optional[str] = Query(None, description="指定日期 (YYYY-MM-DD)，預設為最新交易日"),
    order_by: str = Query("total_net", description="排序依據: total_net, foreign_net, trust_net, dealer_net"),
    sort_dir: str = Query("desc", description="排序方向: desc (買超榜), asc (賣超榜)"),
    market: str = Query("ALL", description="市場篩選: ALL, TWSE, TPEX"),
    limit: int = Query(20, ge=1, le=100, description="回傳數量"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢三大法人（外資、投信、自營商）當日買超與賣超排行焦點股。
    """
    valid_cols = {"total_net", "foreign_net", "trust_net", "dealer_net"}
    if order_by not in valid_cols:
        order_by = "total_net"

    sort_order = "ASC" if sort_dir.lower() == "asc" else "DESC"

    try:
        with db._get_connection() as con:
            # 若未提供日期，自動獲取最新日期
            if not date:
                max_date = con.execute("SELECT MAX(date) FROM tw_institutional_daily").fetchone()[0]
                date = str(max_date) if max_date else None

            if not date:
                return []

            where_clauses = ["date = ?"]
            params: List[Any] = [date]

            if market.upper() in ["TWSE", "TPEX"]:
                where_clauses.append("market = ?")
                params.append(market.upper())

            where_sql = " AND ".join(where_clauses)
            sql = f"""
            SELECT 
                date,
                ticker,
                raw_code,
                name,
                foreign_net,
                trust_net,
                dealer_net,
                total_net,
                foreign_ratio,
                market
            FROM tw_institutional_daily
            WHERE {where_sql}
            ORDER BY {order_by} {sort_order}
            LIMIT ?;
            """
            params.append(limit)

            df = con.execute(sql, params).df()
            if df.empty:
                return []

            return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢法人買賣超排行失敗: {e}")


@router.get("/institutional/dates", response_model=List[str], summary="取得法人籌碼所有可用歷史交易日")
def get_available_institutional_dates(
    limit: int = Query(60, ge=1, le=500, description="回傳天數上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    回傳 DuckDB 中 tw_institutional_daily 所有歷史交易日期清單 (降冪排序)。
    """
    try:
        with db._get_connection() as con:
            df = con.execute("SELECT DISTINCT date FROM tw_institutional_daily ORDER BY date DESC LIMIT ?", [limit]).df()
            return [str(d) for d in df["date"].tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢日期清單失敗: {e}")


@router.get("/institutional/{ticker}", response_model=List[InstitutionalFlowItem], summary="取得單一個股歷史法人進出時序")
def get_ticker_institutional_history(
    ticker: str,
    limit: int = Query(60, ge=1, le=250, description="歷史天數"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢特定標的（如 2330.TW）過去 60 個交易日之三大法人連續進出時序。
    """
    df = db.get_institutional_flow_for_ticker(ticker=ticker, limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")

