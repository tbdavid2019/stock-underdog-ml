import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Any
from fastapi import APIRouter, Query, HTTPException, Depends, Response
from api.schemas import PredictionResponse, StockPredictionItem, TickerResolutionResponse
from data.duckdb_manager import DuckDBManager
from data.forecast_store import ForecastStore

logger = logging.getLogger("stock_app.api.predictions")

router = APIRouter(prefix="/predictions", tags=["Predictions & Quant Signals"])

_PRED_CACHE: Dict[str, Tuple[float, PredictionResponse]] = {}
_PRED_CACHE_TTL = 30.0  # 30 seconds
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_ENTRIES = 500


def _get_from_cache(key: str, ttl: float = _PRED_CACHE_TTL) -> Optional[PredictionResponse]:
    now = time.time()
    with _CACHE_LOCK:
        if key in _PRED_CACHE:
            ts, val = _PRED_CACHE[key]
            if now - ts < ttl:
                return val
    return None


def _set_in_cache(key: str, value: PredictionResponse, max_size: int = _MAX_CACHE_ENTRIES) -> None:
    now = time.time()
    with _CACHE_LOCK:
        if len(_PRED_CACHE) >= max_size:
            for k in list(_PRED_CACHE.keys())[:100]:
                _PRED_CACHE.pop(k, None)
        _PRED_CACHE[key] = (now, value)


def _enrich_records_with_name(records: List[Dict[str, Any]], db: Optional[DuckDBManager] = None) -> List[Dict[str, Any]]:
    if not records or not db or not getattr(db, "enabled", False):
        return records
    missing = [r.get("ticker") for r in records if r.get("ticker") and not r.get("name")]
    if not missing:
        return records
    unique_missing = list(set(missing))
    try:
        name_map = {}
        with db._get_connection(read_only=True) as con:
            placeholders = ",".join(["?"] * len(unique_missing))
            try:
                rows = con.execute(f"""
                    SELECT normalized_symbol, COALESCE(name_local, name_en)
                    FROM market_universe
                    WHERE normalized_symbol IN ({placeholders})
                """, unique_missing).fetchall()
                for sym, n in rows:
                    if sym and n:
                        name_map[sym] = n
            except Exception:
                pass
            still_missing = [t for t in unique_missing if t not in name_map]
            if still_missing:
                try:
                    p_placeholders = ",".join(["?"] * len(still_missing))
                    p_rows = con.execute(f"""
                        SELECT ticker, name FROM company_profiles WHERE ticker IN ({p_placeholders})
                    """, still_missing).fetchall()
                    for t, n in p_rows:
                        if t and n:
                            name_map[t] = n
                except Exception:
                    pass
        for r in records:
            t = r.get("ticker")
            if t and not r.get("name") and t in name_map:
                r["name"] = name_map[t]
    except Exception:
        pass
    return records


def _build_prediction_response(records: List[Dict[str, Any]], db: Optional[DuckDBManager] = None) -> PredictionResponse:
    if db:
        records = _enrich_records_with_name(records, db)
    latest_ts = None
    if records:
        timestamps = [
            str(r.get("timestamp") or r.get("data_as_of") or "")
            for r in records
            if r.get("timestamp") or r.get("data_as_of")
        ]
        if timestamps:
            latest_ts = max(timestamps)
    batch_date = latest_ts[:10] if latest_ts else None
    return PredictionResponse(
        count=len(records),
        batch_date=batch_date,
        latest_batch_timestamp=latest_ts,
        data=records
    )


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
    cache_key = f"latest_{index_name}_{model_name}_{batch_only}_{limit}_{offset}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    res = db.get_latest_predictions(
        index_name=index_name, 
        model_name=model_name, 
        limit=limit, 
        offset=offset, 
        batch_only=batch_only
    )
    records = _enrich_records_with_name(res.get("records", []), db)
    response_obj = PredictionResponse(
        count=len(records),
        batch_date=res.get("batch_date"),
        latest_batch_timestamp=res.get("latest_batch_timestamp"),
        is_stale=res.get("is_stale", False),
        data=records
    )
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/resonance", response_model=PredictionResponse, summary="查詢雙時序大模型共振 (TimesFM × Timestral ∩ 玄鐵 ∩ 法人鎖碼)")
def get_resonance_candidates(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(50, ge=1, le=200, description="回傳數量上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    篩選同時符合「技術面均線買點 ∩ Google TimesFM 看漲 ∩ Mistral Timestral 看漲 ∩ 投信/法人鎖碼」的雙時序大模型共振股票。
    目標價由 TimesFM 與 Timestral P50 中位數加權合成，提供極致穩健的預測視野。
    """
    cache_key = f"resonance_{index_name}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_resonance_candidates(index_name=index_name, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/tri-bullish", response_model=PredictionResponse, summary="查詢 3 模型同步看漲 (TimesFM ∩ Timestral ∩ MOIRAI-2 全數潛力 > 0)")
def get_tri_bullish_candidates(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(50, ge=1, le=200, description="回傳數量上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    篩選 Google TimesFM、Mistral Timestral-209M、Salesforce MOIRAI-2 三大時序模型同步看漲標的。
    """
    cache_key = f"tri_bullish_{index_name}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_tri_bullish_candidates(index_name=index_name, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/dual-bullish", response_model=PredictionResponse, summary="查詢雙模型以上看漲 (任 2 個時序大模型潛力 > 0)")
def get_dual_bullish_candidates(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(50, ge=1, le=200, description="回傳數量上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    篩選任 2 個時序大模型同步預測隔日漲幅大於 0 之潛力標的。
    """
    cache_key = f"dual_bullish_{index_name}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_dual_bullish_candidates(index_name=index_name, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/dual-bearish", response_model=PredictionResponse, summary="查詢綜合避險排行 (≥2 個模型看跌或共振破底)")
def get_dual_bearish_candidates(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(50, ge=1, le=200, description="回傳數量上限"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    篩選至少 2 個時序大模型同步預測下跌，或具備看跌共振訊號之避險防守標的。
    """
    cache_key = f"dual_bearish_{index_name}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_dual_bearish_candidates(index_name=index_name, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


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
    cache_key = f"xuantie_{index_name}_{pullback_type}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_xuantie_candidates(index_name=index_name, pullback_type=pullback_type, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/timestral/top-bullish", response_model=PredictionResponse, summary="查詢 Mistral Timestral-209M 預測漲幅 TOP N 潛力股")
def get_timestral_top_bullish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 Mistral Timestral-209M 時序大模型輸出之潛在漲幅百分比（P50 Potential %）由高到低排序，取得看漲排行。
    """
    records = db.get_timestral_top_bullish(index_name=index_name, limit=limit)
    return _build_prediction_response(records, db)


@router.get("/timestral/top-bearish", response_model=PredictionResponse, summary="查詢 Mistral Timestral-209M 預測跌幅 TOP N 避險股")
def get_timestral_top_bearish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 Mistral Timestral-209M 時序大模型輸出之潛在跌幅由大到小排序，用於避險或潛在破底防禦。
    """
    records = db.get_timestral_top_bearish(index_name=index_name, limit=limit)
    return _build_prediction_response(records, db)


@router.get("/lstm/top-bullish", response_model=PredictionResponse, summary="[Deprecated] 查詢 LSTM 預測漲幅 TOP N 潛力股 (已由 Timestral 替代)")
def get_top_bullish(
    response: Response,
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    [Deprecated] 舊版 LSTM 預測端點。已全面升級為 Mistral Timestral-209M 時序大模型 (/timestral/top-bullish)。
    維持向下相容：若 DuckDB 無舊 LSTM 記錄則透明回傳 Timestral 預測。
    """
    response.headers["X-Deprecation-Warning"] = "/lstm endpoints are deprecated; please use /timestral endpoints."
    records = db.get_top_bullish(index_name=index_name, limit=limit)
    if not records:
        records = db.get_timestral_top_bullish(index_name=index_name, limit=limit)
    return _build_prediction_response(records, db)


@router.get("/lstm/top-bearish", response_model=PredictionResponse, summary="[Deprecated] 查詢 LSTM 預測跌幅 TOP N 避險股 (已由 Timestral 替代)")
def get_top_bearish(
    response: Response,
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    [Deprecated] 舊版 LSTM 預測端點。已全面升級為 Mistral Timestral-209M 時序大模型 (/timestral/top-bearish)。
    維持向下相容：若 DuckDB 無舊 LSTM 記錄則透明回傳 Timestral 預測。
    """
    response.headers["X-Deprecation-Warning"] = "/lstm endpoints are deprecated; please use /timestral endpoints."
    records = db.get_top_bearish(index_name=index_name, limit=limit)
    if not records:
        records = db.get_timestral_top_bearish(index_name=index_name, limit=limit)
    return _build_prediction_response(records, db)


@router.get("/timesfm/top-bullish", response_model=PredictionResponse, summary="查詢 Google TimesFM 預測漲幅 TOP N 潛力股")
def get_timesfm_top_bullish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 Google TimesFM 時序大模型輸出之潛在漲幅百分比（Potential %）由高到低排序，取得看漲排行。
    """
    records = db.get_timesfm_top_bullish(index_name=index_name, limit=limit)
    return _build_prediction_response(records, db)


@router.get("/timesfm/top-bearish", response_model=PredictionResponse, summary="查詢 Google TimesFM 預測跌幅 TOP N 避險股")
def get_timesfm_top_bearish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 Google TimesFM 時序大模型輸出之潛在跌幅由大到小排序，用於避險或潛在破底防禦。
    """
    records = db.get_timesfm_top_bearish(index_name=index_name, limit=limit)
    return _build_prediction_response(records, db)


@router.get("/moirai/top-bullish", response_model=PredictionResponse, summary="查詢 Salesforce MOIRAI-2 預測漲幅 TOP N 潛力股")
def get_moirai_top_bullish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 Salesforce MOIRAI-2 時序大模型輸出之潛在漲幅百分比（Potential %）由高到低排序，取得看漲排行。
    """
    cache_key = f"moirai_bullish_{index_name}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_moirai_top_bullish(index_name=index_name, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/moirai/top-bearish", response_model=PredictionResponse, summary="查詢 Salesforce MOIRAI-2 預測跌幅 TOP N 避險股")
def get_moirai_top_bearish(
    index_name: Optional[str] = Query(None, description="指數篩選"),
    limit: int = Query(10, ge=1, le=100, description="取得前 N 檔"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    依據 Salesforce MOIRAI-2 時序大模型輸出之潛在跌幅由大到小排序，用於避險或潛在破底防禦。
    """
    cache_key = f"moirai_bearish_{index_name}_{limit}"
    cached = _get_from_cache(cache_key)
    if cached is not None:
        return cached

    records = db.get_moirai_top_bearish(index_name=index_name, limit=limit)
    response_obj = _build_prediction_response(records, db)
    _set_in_cache(cache_key, response_obj)
    return response_obj


@router.get("/history/{ticker}", response_model=PredictionResponse, summary="查詢單一標的歷史預測軌跡")
def get_ticker_history(
    ticker: str,
    limit: int = Query(30, ge=1, le=365, description="回溯筆數"),
    db: DuckDBManager = Depends(get_duckdb)
):
    """
    取得指定股票代號（如 2330.TW、AAPL）的時間序列歷史預測與實際走勢軌跡，便於回測與可視化。
    """
    resolved = db.resolve_ticker(ticker)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"查無法辨識的股票或公司名稱：{ticker}")

    stock_name = resolved.get("name")
    records = []
    try:
        raw_history = db.get_ticker_history(ticker=resolved["ticker"], limit=limit) or []
        records = [{**r, "verification_status": "legacy_unverified", "name": stock_name} for r in raw_history]
    except Exception as exc:
        logger.warning("查詢標的 %s 歷史預測失敗: %s", resolved.get("ticker"), exc)
        records = []
    try:
        store = ForecastStore(db.db_path, read_only=True)
        evidence = store.query_records(as_of=datetime.now(timezone.utc).isoformat())
    except Exception:
        # Older databases do not yet contain the verification schema.
        evidence = []
    by_id = {}
    for entry in evidence:
        forecast = entry.get("forecast") or {}
        if forecast.get("ticker") != resolved["ticker"] or int(forecast.get("horizon", 1)) != 1:
            continue
        forecast_id = forecast.get("forecast_id")
        if not forecast_id:
            continue
        actual = entry.get("actual") or {}
        status = entry.get("status") or "unverified"
        if status == "settled" and str(actual.get("target_date")) != str(forecast.get("target_date")):
            status = "invalid"
        item = {"ticker": forecast["ticker"], "name": stock_name, "forecast_id": forecast_id,
            "model_name": forecast.get("model_name"), "model_revision": forecast.get("model_revision"),
            "forecast_session": forecast.get("last_complete_bar"), "target_date": forecast.get("target_date"),
            "timestamp": forecast.get("issued_at"), "current_price": forecast.get("current_price"),
            "predicted_price": forecast.get("predicted_price", forecast.get("prediction")),
            "verification_status": status, "actual_price": actual.get("actual_price", actual.get("actual")) if status == "settled" else None,
            "evidence_type": forecast.get("evidence_type"), "role": forecast.get("role"), "horizon": 1}
        previous = by_id.get(forecast_id)
        if previous is None or (status == "settled" and previous["verification_status"] != "settled"):
            by_id[forecast_id] = item
    records.extend(by_id.values())
    records.sort(key=lambda row: str(row.get("forecast_session") or row.get("timestamp") or ""), reverse=True)
    records = records[:limit]
    return PredictionResponse(count=len(records), data=records)


@router.get("/resolve/{query}", response_model=TickerResolutionResponse, summary="解析股票代號或公司名稱")
def resolve_stock_ticker(query: str, db: DuckDBManager = Depends(get_duckdb)):
    """將股票代號、英文名稱或中文公司名稱解析為可查詢的標準代號。"""
    resolved = db.resolve_ticker(query)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"查無法辨識的股票或公司名稱：{query}")
    return resolved
