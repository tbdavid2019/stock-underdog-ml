"""
api/routes/decision.py - Clef-Flash System One Decision API Routes

Provides endpoints for querying Clef System One typed probabilistic decisions
and checking health of Clef primary and fallback endpoints.
"""

import os
import time
from collections import defaultdict
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel, Field

from data.clef_client import ClefDecisionClient
from data.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/decision", tags=["Decision Model (Clef System One)"])
clef_client = ClefDecisionClient()
db = DuckDBManager()

# In-memory sliding window rate limiter per client IP
_RATE_LIMIT_STORE: Dict[str, list] = defaultdict(list)
RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_RATE_LIMIT_PER_MIN = int(os.getenv("DECISION_RATE_LIMIT_PER_MIN", "60"))
MAX_RATE_LIMIT_STORE_ENTRIES = 2000


def _prune_rate_limit_store(now: float):
    """Periodically remove expired client IP entries to prevent unbounded memory growth"""
    if len(_RATE_LIMIT_STORE) > MAX_RATE_LIMIT_STORE_ENTRIES:
        for ip in list(_RATE_LIMIT_STORE.keys()):
            timestamps = _RATE_LIMIT_STORE[ip]
            if not timestamps or now - timestamps[-1] >= RATE_LIMIT_WINDOW_SECONDS:
                _RATE_LIMIT_STORE.pop(ip, None)


def verify_decision_access(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Access control & Rate Limiting for remote decision inference.
    1. If DECISION_API_KEY or CLEF_API_KEY is configured, enforces valid bearer/header token.
    2. Enforces sliding window rate limit per client IP to protect upstream inference capacity.
    """
    # 1. API Key Access Control
    required_key = os.getenv("DECISION_API_KEY") or os.getenv("CLEF_API_KEY")
    if required_key:
        token = None
        if x_api_key:
            token = x_api_key.strip()
        elif authorization and authorization.startswith("Bearer "):
            token = authorization[7:].strip()

        if token != required_key:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or invalid API key for decision inference"
            )

    # 2. Sliding Window Rate Limiting (Bounded store)
    client_ip = request.client.host if request.client else "127.0.0.1"
    now = time.time()
    _prune_rate_limit_store(now)

    history = _RATE_LIMIT_STORE[client_ip]
    valid_history = [t for t in history if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _RATE_LIMIT_STORE[client_ip] = valid_history

    if len(valid_history) >= DEFAULT_RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: maximum {DEFAULT_RATE_LIMIT_PER_MIN} requests per minute."
        )

    _RATE_LIMIT_STORE[client_ip].append(now)


class EvaluateDecisionRequest(BaseModel):
    ticker: Optional[str] = Field(None, description="Stock ticker (e.g. '2330.TW' or 'NVDA')")
    candidate: Optional[Dict[str, Any]] = Field(None, description="Full candidate dictionary with metrics")
    state: Optional[Dict[str, Any]] = Field(None, description="Custom structured state dictionary")
    macro_regime: Optional[str] = Field("bull", description="Current market macro regime ('bull', 'pullback', 'defense', 'panic')")


@router.get("/health", summary="Check Clef Decision Model endpoints health")
def get_decision_health() -> Dict[str, Any]:
    """
    Check availability of Clef-Flash primary (create360), fallback 1 (aiurl.tw), and fallback 2 (jev).
    """
    health_status = clef_client.check_health()
    all_healthy = any(s.get("status") == "healthy" for s in health_status.values())
    return {
        "success": True,
        "any_available": all_healthy,
        "endpoints": health_status
    }


@router.post("/evaluate", summary="Evaluate stock or custom state using Clef System One", dependencies=[Depends(verify_decision_access)])
def evaluate_stock_decision(req: EvaluateDecisionRequest) -> Dict[str, Any]:
    """
    Evaluate a stock or custom state against typed decision questions.
    Returns choice probabilities, confidence, conviction score, and favorable entry probability.
    """
    candidate_data = dict(req.candidate) if req.candidate else {}
    ticker = req.ticker or candidate_data.get("ticker", "UNKNOWN")

    # If only ticker provided and no candidate dictionary, query DuckDB merged strategy snapshot
    if not candidate_data and req.ticker:
        # 1. Query complete multi-strategy snapshot directly from DuckDB
        snapshot = db.get_latest_candidate_snapshot(req.ticker)
        if snapshot:
            candidate_data = snapshot
        else:
            # 2. Check resonance candidates
            candidates = db.get_resonance_candidates(limit=100)
            found = next((c for c in candidates if c.get("ticker") == req.ticker), None)
            if found:
                candidate_data = found

    if not candidate_data and not req.state:
        raise HTTPException(
            status_code=404,
            detail=f"No quantitative records found for ticker '{ticker}'. Please supply candidate metrics in request or verify the ticker exists in the database."
        )

    if candidate_data and ticker != "UNKNOWN":
        candidate_data["ticker"] = ticker

    if req.state is not None:
        verdict = clef_client.evaluate_stock(
            raw_state=req.state,
            macro_regime=req.macro_regime
        )
        return {
            "success": verdict.success,
            "ticker": verdict.ticker,
            "verdict": verdict.to_dict()
        }

    verdict = clef_client.evaluate_stock(
        candidate=candidate_data,
        macro_regime=req.macro_regime
    )

    return {
        "success": verdict.success,
        "ticker": ticker,
        "verdict": verdict.to_dict()
    }
