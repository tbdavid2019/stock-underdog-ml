"""Read-only queries for verified historical forecast performance."""
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from core.config import config
from data.forecast_store import ForecastStore
from evaluators.forecast_performance import PerformanceQuery, rank_models, rank_stocks

router = APIRouter(prefix="/performance", tags=["Forecast performance"])


def get_forecast_store():
    return ForecastStore(config.db.DUCKDB_PATH, read_only=True)


def _query(as_of, window, market, model_name, model_revision, feature_profile, horizon, target_type):
    if target_type != "price":
        raise HTTPException(status_code=422, detail="此版本的績效榜僅支援價格目標")
    try:
        return PerformanceQuery(as_of=as_of or datetime.now(timezone.utc).isoformat(),
            window=window, market=market, model_name=model_name, model_revision=model_revision,
            feature_profile=feature_profile, horizon=horizon, target_type=target_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _fetch(store, query):
    try:
        return store.query_records(as_of=query.as_of)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="預測驗證資料目前無法讀取") from exc


def _result(rows, query, ranker, ticker=None, limit=50, offset=0):
    if ticker is not None:
        rows = [r for r in rows if (r.get("forecast", r).get("ticker") or "").upper() == ticker.upper()]
    result = ranker(rows, query)
    result["status"] = "ready" if result["data"] else "insufficient_data"
    result["count"] = len(result["data"])
    result["data"] = result["data"][offset:offset + limit]
    result["limit"] = limit
    result["offset"] = offset
    return result


def performance_params(
    as_of: Optional[str] = Query(None), window: int = Query(60),
    market: Optional[str] = Query(None, pattern="^(TW|US|JP)$"),
    model_name: Optional[str] = Query(None), model_revision: Optional[str] = Query(None),
    feature_profile: Optional[str] = Query(None), horizon: int = Query(1, ge=1, le=30),
    target_type: str = Query("price", pattern="^(price|return|volatility)$"),
):
    return _query(as_of, window, market, model_name, model_revision, feature_profile, horizon, target_type)


@router.get("/stocks")
def get_stock_performance(query: PerformanceQuery = Depends(performance_params),
                          store: ForecastStore = Depends(get_forecast_store),
                          limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return _result(_fetch(store, query), query, rank_stocks, limit=limit, offset=offset)


@router.get("/models")
def get_model_performance(query: PerformanceQuery = Depends(performance_params),
                          store: ForecastStore = Depends(get_forecast_store),
                          limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return _result(_fetch(store, query), query, rank_models, limit=limit, offset=offset)


@router.get("/tickers/{ticker}")
def get_ticker_performance(ticker: str, query: PerformanceQuery = Depends(performance_params),
                           store: ForecastStore = Depends(get_forecast_store),
                           limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    return _result(_fetch(store, query), query, rank_stocks, ticker=ticker, limit=limit, offset=offset)


models_router = APIRouter(prefix="/models", tags=["Model health"])


def _eval_health(as_of: Optional[str], market: Optional[str], model_name: Optional[str],
                 policy_mode: str, store: ForecastStore):
    from evaluators.model_health import evaluate_model_health
    iso = as_of or datetime.now(timezone.utc).isoformat()
    query_obj = PerformanceQuery(as_of=iso, market=market, model_name=model_name)
    rows = _fetch(store, query_obj)
    return evaluate_model_health(rows, as_of=query_obj.as_of, policy_mode=policy_mode)


@router.get("/health")
def get_performance_health(as_of: Optional[str] = Query(None),
                           market: Optional[str] = Query(None, pattern="^(TW|US|JP)$"),
                           model_name: Optional[str] = Query(None),
                           policy_mode: str = Query("observe_only", pattern="^(observe_only|enforce)$"),
                           store: ForecastStore = Depends(get_forecast_store)):
    return _eval_health(as_of, market, model_name, policy_mode, store)


@models_router.get("/health")
def get_models_health(as_of: Optional[str] = Query(None),
                      market: Optional[str] = Query(None, pattern="^(TW|US|JP)$"),
                      model_name: Optional[str] = Query(None),
                      policy_mode: str = Query("observe_only", pattern="^(observe_only|enforce)$"),
                      store: ForecastStore = Depends(get_forecast_store)):
    return _eval_health(as_of, market, model_name, policy_mode, store)
