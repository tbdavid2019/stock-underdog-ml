"""
api/routes/watchlist.py - Watchlist and Quantitative Alerts REST API Routes

Provides CRUD endpoints for DuckDB persisted watchlist items and price/regime alerts
with Fail-Closed API Key write protection and 503 database busy handling.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.config import config
from core.duckdb_lock import DuckDBBusyError
from core.netlimiter import default_limiter, verify_watchlist_write_access, verify_profile_read_access
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.watchlist_routes")


def check_rate_limit(request: Request):
    default_limiter.check(request)


router = APIRouter(prefix="/watchlist", tags=["Watchlist & Quant Alerts"], dependencies=[Depends(check_rate_limit)])
db = DuckDBManager()


def get_db() -> DuckDBManager:
    return db


class AddWatchlistItemRequest(BaseModel):
    profile_id: str = Field("default", description="Profile namespace ID")
    ticker: str = Field(..., description="Stock ticker (e.g. '2330.TW', 'NVDA')")
    note: Optional[str] = Field("", max_length=500, description="Optional note (<= 500 chars)")
    sort_order: Optional[int] = Field(None, description="Display order")


class BatchImportWatchlistRequest(BaseModel):
    profile_id: str = Field("default", description="Profile namespace ID")
    tickers: List[str] = Field(..., max_length=100, description="List of stock tickers (max 100)")


@router.get("", summary="Get persisted watchlist items enriched with latest predictions")
def get_watchlist(
    request: Request,
    profile_id: str = Query("default", description="Profile namespace ID"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Public read-only query for persisted watchlist (profile_id == 'default').
    Non-default profiles require valid WATCHLIST_API_KEY.
    """
    verify_profile_read_access(profile_id, request, x_api_key=x_api_key, authorization=authorization)
    try:
        items = db_mgr.get_watchlist_items(profile_id=profile_id)
        return {
            "success": True,
            "profile_id": profile_id,
            "count": len(items),
            "items": items
        }
    except DuckDBBusyError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database is currently busy ({e}). Please retry shortly."
        )
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("", summary="Add or update a watchlist item", dependencies=[Depends(verify_watchlist_write_access)])
def add_watchlist_item(
    req: AddWatchlistItemRequest,
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Add or update a stock in the DuckDB watchlist.
    Protected by Fail-Closed WATCHLIST_API_KEY in production.
    """
    try:
        item = db_mgr.add_watchlist_item(
            profile_id=req.profile_id,
            ticker=req.ticker,
            note=req.note or "",
            sort_order=req.sort_order
        )
        return {
            "success": True,
            "item": item
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DuckDBBusyError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database write lock is busy ({e}). Please retry shortly."
        )
    except Exception as e:
        logger.error(f"Error adding watchlist item: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{ticker}", summary="Remove a stock from watchlist", dependencies=[Depends(verify_watchlist_write_access)])
def remove_watchlist_item(
    ticker: str,
    profile_id: str = Query("default"),
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Remove a stock from the DuckDB watchlist.
    Protected by Fail-Closed WATCHLIST_API_KEY in production.
    """
    try:
        deleted = db_mgr.remove_watchlist_item(profile_id=profile_id, ticker=ticker)
        return {
            "success": True,
            "profile_id": profile_id,
            "ticker": ticker,
            "deleted": deleted
        }
    except DuckDBBusyError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database write lock is busy ({e}). Please retry shortly."
        )
    except Exception as e:
        logger.error(f"Error removing watchlist item: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", summary="Batch import stocks into watchlist", dependencies=[Depends(verify_watchlist_write_access)])
def batch_import_watchlist(
    req: BatchImportWatchlistRequest,
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Batch import stocks with regex validation, deduplication, and max 100 items cap.
    """
    try:
        result = db_mgr.batch_import_watchlist(profile_id=req.profile_id, tickers=req.tickers)
        return {
            "success": True,
            "result": result
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DuckDBBusyError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database write lock is busy ({e}). Please retry shortly."
        )
    except Exception as e:
        logger.error(f"Error batch importing watchlist: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ==================== QUANT PRICE ALERTS ROUTER ====================

alerts_router = APIRouter(prefix="/alerts", tags=["Quant Price Alerts"], dependencies=[Depends(check_rate_limit)])


class CreateQuantAlertRequest(BaseModel):
    profile_id: str = Field("default", description="Profile namespace ID")
    ticker: str = Field(..., description="Stock ticker (e.g. '2330.TW', 'NVDA')")
    rule_type: str = Field(..., description="Rule type: 'above_price', 'below_price', 'below_p10', 'above_p50', 'pullback_ma60', 'pullback_ma120'")
    threshold_value: Optional[float] = Field(None, description="Static target price (required for above_price/below_price)")
    reference_model: Optional[str] = Field("auto", description="Reference model: 'timestral', 'moirai', 'timesfm', 'auto'")
    cooldown_minutes: Optional[int] = Field(720, description="Cooldown window in minutes (default 720 / 12h)")


@alerts_router.get("", summary="Get persisted quant price alerts")
def get_quant_alerts(
    request: Request,
    profile_id: str = Query("default", description="Profile namespace ID"),
    enabled_only: bool = Query(False, description="Filter only enabled alerts"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Query persisted quant price alerts.
    In production mode or for non-default profiles, requires valid WATCHLIST_API_KEY.
    """
    deployment_mode = (os.getenv("DEPLOYMENT_MODE") or config.security.DEPLOYMENT_MODE or "production").strip().lower()
    if deployment_mode == "production" or profile_id.strip() != "default":
        verify_watchlist_write_access(request, x_api_key=x_api_key, authorization=authorization)

    try:
        alerts = db_mgr.get_quant_alerts(profile_id=profile_id, enabled_only=enabled_only)
        return {
            "success": True,
            "profile_id": profile_id,
            "count": len(alerts),
            "alerts": alerts
        }
    except DuckDBBusyError as e:
        raise HTTPException(status_code=503, detail=f"Database is busy ({e}).")
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@alerts_router.post("", summary="Create a quant price alert", dependencies=[Depends(verify_watchlist_write_access)])
def create_quant_alert(
    req: CreateQuantAlertRequest,
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a quant price alert rule.
    Guarded by Fail-Closed WATCHLIST_API_KEY in production.
    """
    try:
        alert = db_mgr.add_quant_alert(
            canonical_ticker=req.ticker,
            rule_type=req.rule_type,
            threshold_value=req.threshold_value,
            reference_model=req.reference_model or "auto",
            cooldown_minutes=req.cooldown_minutes or 720,
            profile_id=req.profile_id
        )
        return {
            "success": True,
            "alert": alert
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except DuckDBBusyError as e:
        raise HTTPException(status_code=503, detail=f"Database write lock is busy ({e}).")
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@alerts_router.delete("/{alert_id}", summary="Delete a quant price alert", dependencies=[Depends(verify_watchlist_write_access)])
def delete_quant_alert(
    alert_id: str,
    profile_id: str = Query("default"),
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete a quant price alert by ID.
    Guarded by Fail-Closed WATCHLIST_API_KEY in production.
    """
    try:
        deleted = db_mgr.delete_quant_alert(alert_id=alert_id, profile_id=profile_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {
            "success": True,
            "alert_id": alert_id,
            "deleted": deleted
        }
    except HTTPException:
        raise
    except DuckDBBusyError as e:
        raise HTTPException(status_code=503, detail=f"Database write lock is busy ({e}).")
    except Exception as e:
        logger.error(f"Error deleting alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@alerts_router.post("/check", summary="Evaluate and trigger pending quant price alerts", dependencies=[Depends(verify_watchlist_write_access)])
def check_quant_alerts(
    profile_id: str = Query("default"),
    db_mgr: DuckDBManager = Depends(get_db)
) -> Dict[str, Any]:
    """
    Trigger evaluation of active quant price alerts and dispatch notifications upon condition match.
    Guarded by Fail-Closed WATCHLIST_API_KEY in production.
    """
    from data.alert_evaluator import evaluate_and_dispatch_alerts
    try:
        res = evaluate_and_dispatch_alerts(profile_id=profile_id, db=db_mgr)
        return {
            "success": True,
            "result": res
        }
    except DuckDBBusyError as e:
        raise HTTPException(status_code=503, detail=f"Database is busy ({e}). Please retry shortly.")
    except Exception as e:
        logger.error(f"Error evaluating alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

