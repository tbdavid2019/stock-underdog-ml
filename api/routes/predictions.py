"""
api/routes/predictions.py - Stock Predictions & Multi-Strategy Quantitative Endpoints
"""

from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException, Depends
from api.schemas import PredictionResponse, StockPredictionItem
from data.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/predictions", tags=["Predictions & Quant Signals"])


def get_duckdb():
    return DuckDBManager()


@router.get("/latest", response_model=PredictionResponse, summary="取得最新量化日報批次預測與指標")
def get_latest_predictions(
    index_name: Optional[str] = Query(None, description="指數篩選 (台灣50, 台灣中型100, S&P500)"),
    model_name: Optional[str] = Query(None, description="策略篩選 (玄鐵重劍, LSTM, 多維共振)"),
    batch_only: bool = Query(True, description="是否僅回傳最新執行批次 (預設: True；設為 False 則回傳各標的歷史最新一筆)"),
    limit: int = Query(50, ge=1, le=500, description="回傳數量"),
    offset: int = Query(0, ge=0, description="分頁偏移量"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    查詢最新量化批次運算結果。預設 (batch_only=True) 僅回傳最新成功執行批次之標的，並附帶 batch_date, data_as_of, age_hours 與 is_stale 狀態。
    """
    res = db.get_latest_predictions(
        index_name=index_name, 
        model_name=model_name, 
        limit=limit, 
        offset=offset, 
        batch_only=batch_only
    )
    records = res.get("records", [])
    return PredictionResponse(
        count=len(records),
        batch_date=res.get("batch_date"),
        latest_batch_timestamp=res.get("latest_batch_timestamp"),
        is_stale=res.get("is_stale", False),
        data=records
    )


@router.get("/resonance", response_model=PredictionResponse, summary="查詢多策略交集與 🏆 三重共振焦點股")
def get_resonance_candidates(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(50, ge=1, le=200, description="回傳數量上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    篩選同時符合「技術面均線買點 ∩ LSTM 看漲 ∩ 投信/法人鎖碼」的多策略共振股票。
    """
    records = db.get_resonance_candidates(index_name=index_name, limit=limit)
    return PredictionResponse(count=len(records), data=records)


@router.get("/xuantie", response_model=PredictionResponse, summary="查詢玄鐵重劍技術面均線回調買點")
def get_xuantie_candidates(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    pullback_type: Optional[str] = Query(None, description="均線回調篩選 (MA60, MA120)"),
    limit: int = Query(50, ge=1, le=200, description="回傳數量上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    篩選處於中長期多頭趨勢且當前股價剛好回踩 MA60 季線或 MA120 半年線之支撐買點標的。
    """
    records = db.get_xuantie_candidates(index_name=index_name, pullback_type=pullback_type, limit=limit)
    return PredictionResponse(count=len(records), data=records)


@router.get("/lstm/top-bullish", response_model=PredictionResponse, summary="查詢 LSTM 預測漲幅 TOP N 潛力股")
def get_top_bullish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 LSTM 深度學習模型輸出之潛在漲幅百分比（Potential %）由高到低排序，取得短線看漲排行。
    """
    records = db.get_top_bullish(index_name=index_name, limit=limit)
    return PredictionResponse(count=len(records), data=records)


@router.get("/lstm/top-bearish", response_model=PredictionResponse, summary="查詢 LSTM 預測跌幅 TOP N 避險/做空股")
def get_top_bearish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 LSTM 深度學習模型輸出之潛在跌幅由大到小排序，用於避險或潛在放空觀察。
    """
    records = db.get_top_bearish(index_name=index_name, limit=limit)
    return PredictionResponse(count=len(records), data=records)


@router.get("/history/{ticker}", response_model=PredictionResponse, summary="查詢單一標的歷史預測軌跡")
def get_ticker_history(
    ticker: str,
    limit: int = Query(30, ge=1, le=365, description="回溯筆數"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    取得指定股票代號（如 2330.TW、AAPL）的時間序列歷史預測與實際走勢軌跡，便於回測與可視化。
    """
    records = db.get_ticker_history(ticker=ticker.strip(), limit=limit)
    if not records:
        raise HTTPException(status_code=404, detail=f"查無標的 {ticker} 之歷史預測記錄")
    return PredictionResponse(count=len(records), data=records)
