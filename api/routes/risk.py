"""
api/routes/risk.py - 組合風險 X 光片 (Risk X-Ray) 與部位優化端點

提供受每 IP 滑動視窗限流與 BoundedSemaphore(2) 併發閘門保護之量化風險診斷端點：
1. POST /api/v1/risk/xray (上限 <= 100 檔標的、window 30..504，零網路 I/O)
2. GET  /api/v1/watchlist/risk-xray (唯讀自選股風險診斷，表未就緒優雅降級)
3. POST /api/v1/risk/optimize-weights (上限 <= 50 檔，純 NumPy ADMM 非對稱換手優化)
4. GET  /api/v1/paper/accounts/{account_id}/risk-xray (正則驗證 + Fail-Closed 認證防帳戶枚舉)
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from typing import Any, Dict, List, Optional
import duckdb
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from backtest.paper import PaperLedger
from backtest.risk_xray import calculate_portfolio_risk_xray
from backtest.turnover_optimizer import optimize_portfolio_weights_admm
from core.config import config
from core.netlimiter import default_limiter, verify_watchlist_write_access
from data.duckdb_manager import DuckDBManager

logger = logging.getLogger("stock_app.routes.risk")

router = APIRouter(tags=["Portfolio Risk & Optimization"])

_ACCOUNT_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
_MATRIX_SEMAPHORE = threading.BoundedSemaphore(2)


# ==================== Pydantic Request Schemas ====================

class RiskXRayPayload(BaseModel):
    tickers: List[str] = Field(..., description="標的代號清單 (上限 <= 100 檔)")
    window: int = Field(252, ge=30, le=504, description="歷史回看日數 (30..504)")
    weights: Optional[Dict[str, float]] = Field(None, description="各標的自訂權重 (0..1)")
    as_of_date: Optional[str] = Field(None, description="截止基準日期 (YYYY-MM-DD)")


class OptimizeWeightsPayload(BaseModel):
    tickers: List[str] = Field(..., description="標的代號清單 (上限 <= 50 檔)")
    current_weights: Optional[Dict[str, float]] = Field(None, description="當前持倉權重")
    expected_returns: Optional[Dict[str, float]] = Field(None, description="預期日報酬率")
    gamma_buy: float = Field(0.0015, ge=0.0, description="買進摩擦係數 (手續費 + 滑價)")
    gamma_sell: float = Field(0.0045, ge=0.0, description="賣出摩擦係數 (手續費 + 滑價 + 證交稅 0.3%)")
    risk_aversion: float = Field(1.0, ge=0.0, description="風險規避係數 (支援 0.0)")
    max_single_weight: float = Field(0.10, ge=0.01, le=1.0, description="單一標的上限")
    window: int = Field(252, ge=30, le=504, description="歷史回看日數 (30..504)")
    as_of_date: Optional[str] = Field(None, description="截止基準日期 (YYYY-MM-DD)")


# ==================== Endpoints ====================

@router.post("/risk/xray", summary="計算投資組合風險 X 光片 (零網路 I/O)")
def get_portfolio_risk_xray(payload: RiskXRayPayload, request: Request):
    """
    輸入自選或持股籃子，依本地多市場離線日 K 矩陣計算：
    - 雙口徑波動率 (全組合含現金 vs 風險槽正規化)
    - 階梯式樣本門檻 95% / 99% VaR 與 CVaR
    - 支援負對沖值之歐拉波動貢獻拆解 (Euler PCV)
    - 企業行為除權息異常跳空守衛
    """
    default_limiter.check(request, limit=60)

    clean_tickers = list(dict.fromkeys(str(t).strip().upper() for t in payload.tickers if str(t).strip()))
    if not clean_tickers:
        raise HTTPException(status_code=422, detail="tickers list cannot be empty")
    if len(clean_tickers) > 100:
        raise HTTPException(status_code=422, detail="tickers count exceeds maximum limit of 100")

    if payload.weights:
        for tk, w in payload.weights.items():
            if w is None or not math.isfinite(w) or w < 0.0:
                raise HTTPException(status_code=422, detail=f"Invalid weight for {tk}: must be non-negative finite number")

    acquired = _MATRIX_SEMAPHORE.acquire(timeout=2.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy with matrix risk computations, please try again shortly")

    try:
        db = DuckDBManager(init_schema=False)
        bars_data = db.load_local_close_series(
            clean_tickers,
            as_of_date=payload.as_of_date,
            lookback_bars=payload.window
        )

        res = calculate_portfolio_risk_xray(
            series_map=bars_data["series"],
            weights=payload.weights,
            price_adjustment_modes=bars_data["price_adjustment_modes"],
            currencies=bars_data["currencies"],
            unavailable_tickers=bars_data["unavailable_tickers"],
        )
        return res
    finally:
        _MATRIX_SEMAPHORE.release()


@router.get("/watchlist/risk-xray", summary="取得 DuckDB 自選股投資組合風險 X 光片")
@router.get("/risk/watchlist/risk-xray", summary="取得 DuckDB 自選股投資組合風險 X 光片 [別名]")
def get_watchlist_risk_xray(request: Request, window: int = 252):
    """
    唯讀讀取 DuckDB 自選股清單並計算風險 X 光片。
    若自選股表格尚未初始化或為空，優雅回傳對應狀態碼。
    """
    default_limiter.check(request, limit=60)

    db = DuckDBManager(init_schema=False)
    if not os.path.exists(db.db_path):
        return {
            "status": "watchlist_unavailable",
            "message": "database not initialized",
            "available_tickers": [],
            "unavailable_tickers": []
        }

    tickers: List[str] = []

    try:
        with db._get_connection(read_only=True) as con:
            tables = [r[0] for r in con.execute("SHOW TABLES;").fetchall()]
            if "watchlist_items" not in tables:
                return {
                    "status": "watchlist_unavailable",
                    "message": "watchlist_items table not initialized",
                    "available_tickers": [],
                    "unavailable_tickers": []
                }
            rows = con.execute("SELECT canonical_ticker FROM watchlist_items WHERE profile_id = 'default' ORDER BY sort_order ASC, created_at DESC;").fetchall()
            tickers = [str(r[0]).strip().upper() for r in rows if r[0]]
    except Exception as e:
        logger.debug(f"讀取自選股表格失敗: {e}")
        return {
            "status": "watchlist_unavailable",
            "message": f"Failed reading watchlist: {e}",
            "available_tickers": [],
            "unavailable_tickers": []
        }

    if not tickers:
        return {
            "status": "empty_watchlist",
            "message": "當前自選股清單為空，請先加入標的至自選股",
            "available_tickers": [],
            "unavailable_tickers": []
        }

    # 上限 100 檔
    clamped_tickers = tickers[:100]
    clamped_window = max(30, min(504, window))

    acquired = _MATRIX_SEMAPHORE.acquire(timeout=2.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy with matrix computations, please try again shortly")

    try:
        bars_data = db.load_local_close_series(clamped_tickers, lookback_bars=clamped_window)
        return calculate_portfolio_risk_xray(
            series_map=bars_data["series"],
            price_adjustment_modes=bars_data["price_adjustment_modes"],
            currencies=bars_data["currencies"],
            unavailable_tickers=bars_data["unavailable_tickers"],
        )
    finally:
        _MATRIX_SEMAPHORE.release()


@router.post("/risk/optimize-weights", summary="計算非對稱換手率權重配置 (純 NumPy ADMM)")
@router.post("/optimize-weights", summary="計算非對稱換手率權重配置 (純 NumPy ADMM) [別名]")
def optimize_weights(payload: OptimizeWeightsPayload, request: Request):
    """
    在多頭無槓桿、單股上限與台股證交稅 (gamma_sell > gamma_buy) 約束下，
    計算目標權重配置與換手率。
    """
    default_limiter.check(request, limit=60)

    clean_tickers = list(dict.fromkeys(str(t).strip().upper() for t in payload.tickers if str(t).strip()))
    if not clean_tickers:
        raise HTTPException(status_code=422, detail="tickers list cannot be empty")
    if len(clean_tickers) > 50:
        raise HTTPException(status_code=422, detail="tickers count exceeds maximum limit of 50")

    acquired = _MATRIX_SEMAPHORE.acquire(timeout=2.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy with optimization computations, please try again shortly")

    try:
        db = DuckDBManager(init_schema=False)
        bars_data = db.load_local_close_series(
            clean_tickers,
            as_of_date=payload.as_of_date,
            lookback_bars=payload.window
        )

        # 整理歷史報酬矩陣
        series_map = bars_data["series"]
        valid_tks = [tk for tk in clean_tickers if tk in series_map and len(series_map[tk]) >= 30]

        if not valid_tks:
            zero_weights = {tk: 0.0 for tk in clean_tickers}
            return {
                "status": "insufficient_history",
                "solver_status": "fallback_hold_previous",
                "weights": zero_weights,
                "optimal_weights": zero_weights,
                "one_way_turnover": 0.0,
                "turnover": 0.0,
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
                "cash_weight": 1.0,
                "iterations": 0,
            }

        # 共同交易日交集
        date_sets = [set(series_map[tk].index) for tk in valid_tks]
        common_dates = sorted(list(set.intersection(*date_sets)))

        if len(common_dates) < 30:
            eq_weights = {tk: round(1.0 / len(valid_tks), 6) for tk in valid_tks}
            return {
                "status": "insufficient_history_fallback",
                "solver_status": "fallback_hold_previous",
                "weights": eq_weights,
                "optimal_weights": eq_weights,
                "one_way_turnover": 0.0,
                "turnover": 0.0,
                "buy_turnover": 0.0,
                "sell_turnover": 0.0,
                "cash_weight": 0.0,
                "iterations": 0,
            }

        ret_mat = np.zeros((len(common_dates) - 1, len(valid_tks)), dtype=np.float64)
        for c_idx, tk in enumerate(valid_tks):
            s = series_map[tk].loc[common_dates].values
            ret_mat[:, c_idx] = (s[1:] - s[:-1]) / s[:-1]

        # 預期報酬
        if payload.expected_returns:
            exp_rets = {tk: payload.expected_returns.get(tk, 0.001) for tk in valid_tks}
        else:
            # 預設最近 20 日均線動能或微幅正期望值
            exp_rets = {tk: float(np.mean(ret_mat[-20:, i])) if ret_mat.shape[0] >= 20 else float(np.mean(ret_mat[:, i])) for i, tk in enumerate(valid_tks)}

        res = optimize_portfolio_weights_admm(
            expected_returns=exp_rets,
            returns_history=ret_mat,
            current_weights=payload.current_weights,
            tickers=valid_tks,
            gamma_buy=payload.gamma_buy,
            gamma_sell=payload.gamma_sell,
            risk_aversion=payload.risk_aversion,
            max_single_weight=payload.max_single_weight,
            max_total_weight=1.0,
        )
        return res
    finally:
        _MATRIX_SEMAPHORE.release()


# ==================== Protected Paper Account Risk X-Ray ====================

@router.get("/paper/accounts/{account_id}/risk-xray", summary="診斷私有模擬帳戶持倉風險 (強制 API Key 認證防枚舉)")
def get_paper_account_risk_xray(
    account_id: str,
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
    window: int = 252,
):
    """
    強制驗證 account_id 格式，並以 Fail-Closed 模式認證 WATCHLIST_API_KEY。
    以唯讀方式 (read_only=True) 載入 PaperLedger 持倉與現金權重後計算 Risk X-Ray。
    """
    default_limiter.check(request, limit=60)

    # 1. 正則驗證 account_id 格式 (防止路徑注入與畸形字元)
    if not _ACCOUNT_ID_REGEX.match(account_id):
        raise HTTPException(status_code=422, detail="Invalid account_id format: must be 1-64 alphanumeric characters, underscores or hyphens")

    # 2. Fail-Closed 認證防帳戶枚舉 (未授權請求直接回傳 401，不洩漏該帳戶是否存在)
    verify_watchlist_write_access(request, x_api_key=x_api_key, authorization=authorization)

    # 3. 唯讀讀取 PaperLedger (嚴格 read_only=True)
    db_path = config.db.DUCKDB_PATH
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Paper account database not initialized")

    try:
        ledger = PaperLedger(db_path=db_path, read_only=True)
        acc = ledger.get_account(account_id)
        if acc is None:
            raise HTTPException(status_code=404, detail=f"Paper account '{account_id}' not found")
    except HTTPException:
        raise
    except duckdb.CatalogException:
        raise HTTPException(status_code=404, detail=f"Paper account '{account_id}' not found")
    except Exception as e:
        logger.warning(f"讀取模擬帳戶 {account_id} 失敗: {e}")
        raise HTTPException(status_code=500, detail="Error loading paper account ledger")

    positions = acc.get("positions", {})
    cash = float(acc.get("cash", 0.0))

    if not positions:
        return {
            "status": "empty_portfolio",
            "account_id": account_id,
            "currency": acc.get("currency", "TWD"),
            "cash_weight": 1.0,
            "annualized_volatility_total": 0.0,
            "var_95": 0.0,
            "message": "模擬帳戶目前為 100% 現金部位，無股票持倉",
            "available_tickers": [],
            "unavailable_tickers": [],
        }

    # 計算各持股市值與全組合權重
    pos_values: Dict[str, float] = {}
    total_val = cash
    for tk, pdata in positions.items():
        qty = float(pdata.get("quantity", 0.0))
        if qty <= 0:
            continue
        px = float(pdata.get("last_price") or pdata.get("total_cost", 0.0) / max(1.0, qty))
        mkt_val = qty * px
        pos_values[tk] = mkt_val
        total_val += mkt_val

    if total_val <= 0 or not pos_values:
        return {
            "status": "empty_portfolio",
            "account_id": account_id,
            "currency": acc.get("currency", "TWD"),
            "cash_weight": 1.0,
            "annualized_volatility_total": 0.0,
            "var_95": 0.0,
            "message": "模擬帳戶持股市值為 0",
            "available_tickers": [],
            "unavailable_tickers": [],
        }

    weights = {tk: pos_values[tk] / total_val for tk in pos_values}
    hold_tickers = list(pos_values.keys())

    acquired = _MATRIX_SEMAPHORE.acquire(timeout=2.0)
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy with matrix computations, please try again shortly")

    try:
        db = DuckDBManager(init_schema=False)
        bars_data = db.load_local_close_series(hold_tickers, lookback_bars=max(30, min(504, window)))
        res = calculate_portfolio_risk_xray(
            series_map=bars_data["series"],
            weights=weights,
            price_adjustment_modes=bars_data["price_adjustment_modes"],
            currencies=bars_data["currencies"],
            unavailable_tickers=bars_data["unavailable_tickers"],
        )
        res["account_id"] = account_id
        res["account_currency"] = acc.get("currency", "TWD")
        return res
    finally:
        _MATRIX_SEMAPHORE.release()
