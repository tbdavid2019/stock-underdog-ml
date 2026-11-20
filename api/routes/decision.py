"""
api/routes/decision.py - Clef-Flash System One Decision API Routes

Provides endpoints for querying Clef System One typed probabilistic decisions
and checking health of Clef primary and fallback endpoints.
"""

import os
import secrets
import time
from typing import Dict, Any, Optional, Literal
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from pydantic import BaseModel, Field

from core.netlimiter import resolve_client_ip, sanitize_health_data, SlidingWindowRateLimiter
from data.clef_client import ClefDecisionClient
from data.decision_service import DecisionService
from data.duckdb_manager import DuckDBManager

router = APIRouter(prefix="/decision", tags=["Decision Model (Clef System One)"])
db = DuckDBManager()
clef_client = ClefDecisionClient(db=db)
decision_service = DecisionService(clef_client=clef_client, db=db)

# Thread-safe sliding window rate limiter per client IP
RATE_LIMIT_WINDOW_SECONDS = 60
MAX_RATE_LIMIT_STORE_ENTRIES = 2000
DEFAULT_RATE_LIMIT_PER_MIN = int(os.getenv("DECISION_RATE_LIMIT_PER_MIN", "60"))

decision_limiter = SlidingWindowRateLimiter(
    default_limit=DEFAULT_RATE_LIMIT_PER_MIN,
    window_seconds=RATE_LIMIT_WINDOW_SECONDS,
    max_entries=MAX_RATE_LIMIT_STORE_ENTRIES
)
_RATE_LIMIT_STORE = decision_limiter._store


def _prune_rate_limit_store(now: float):
    decision_limiter.prune(now)


def _resolve_client_ip(request: Request) -> str:
    """Resolve real client IP prioritizing Cloudflare / Nginx edge headers."""
    return resolve_client_ip(request)


def verify_public_decision_rate_limit(request: Request):
    """
    Sliding window rate limit for public web UI decision queries (per client IP).
    Does NOT require DECISION_API_KEY.
    """
    decision_limiter.check(request, limit=DEFAULT_RATE_LIMIT_PER_MIN)


def verify_decision_access(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
):
    """
    Access control & Rate Limiting for remote decision inference.
    1. If DECISION_API_KEY or CLEF_API_KEY is configured, enforces valid bearer/header token.
    2. Enforces thread-safe sliding window rate limit per client IP via SlidingWindowRateLimiter.
    """
    # 1. API Key Access Control
    required_key = os.getenv("DECISION_API_KEY") or os.getenv("CLEF_API_KEY")
    if required_key:
        token = None
        if x_api_key:
            token = x_api_key.strip()
        elif authorization and authorization.startswith("Bearer "):
            token = authorization[7:].strip()

        if not token or not secrets.compare_digest(token, required_key.strip()):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or invalid API key for decision inference"
            )

    # 2. Thread-safe Sliding Window Rate Limiting
    decision_limiter.check(request, limit=DEFAULT_RATE_LIMIT_PER_MIN)


MacroRegimeLiteral = Literal["bull", "pullback", "defense", "panic", "bear", "neutral", "default"]


class EvaluateDecisionRequest(BaseModel):
    ticker: Optional[str] = Field(None, description="Stock ticker (e.g. '2330.TW' or 'NVDA')")
    candidate: Optional[Dict[str, Any]] = Field(None, description="Full candidate dictionary with metrics")
    state: Optional[Dict[str, Any]] = Field(None, description="Custom structured state dictionary")
    macro_regime: Optional[MacroRegimeLiteral] = Field("bull", description="Current market macro regime ('bull', 'pullback', 'defense', 'panic')")
    force_refresh: bool = Field(False, description="Whether to bypass DuckDB & memory cache and force remote LLM inference")


_HEALTH_CACHE: Dict[str, Any] = {}
_HEALTH_CACHE_TIME: float = 0.0
HEALTH_CACHE_TTL = 60.0


@router.get("/health", summary="Check 5-Tier Decision Model endpoints health", dependencies=[Depends(verify_public_decision_rate_limit)])
def get_decision_health() -> Dict[str, Any]:
    """
    Check availability across all 5 tiers with 60-second in-memory cache,
    per-IP rate limiting, and sanitized error and URL outputs.
    """
    global _HEALTH_CACHE, _HEALTH_CACHE_TIME
    now = time.time()
    if _HEALTH_CACHE and (now - _HEALTH_CACHE_TIME < HEALTH_CACHE_TTL):
        return _HEALTH_CACHE

    raw_results = decision_service.check_health()
    sanitized = sanitize_health_data(raw_results)
    _HEALTH_CACHE = sanitized
    _HEALTH_CACHE_TIME = now
    return sanitized


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
        verdict = decision_service.evaluate_stock(
            raw_state=req.state,
            macro_regime=req.macro_regime,
            force_refresh=req.force_refresh
        )
        return {
            "success": verdict.success,
            "ticker": verdict.ticker,
            "verdict": verdict.to_dict()
        }

    verdict = decision_service.evaluate_stock(
        candidate=candidate_data,
        macro_regime=req.macro_regime,
        force_refresh=req.force_refresh
    )

    return {
        "success": verdict.success,
        "ticker": ticker,
        "verdict": verdict.to_dict()
    }


@router.get("/stock/{ticker}", summary="Get Clef System One decision verdict for a single stock ticker", dependencies=[Depends(verify_public_decision_rate_limit)])
def get_stock_decision(
    ticker: str,
    macro_regime: Optional[MacroRegimeLiteral] = "bull",
    force_refresh: bool = False,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if force_refresh:
        required_key = os.getenv("DECISION_API_KEY") or os.getenv("CLEF_API_KEY")
        if required_key:
            token = x_api_key.strip() if x_api_key else (
                authorization[7:].strip() if authorization and authorization.startswith("Bearer ") else None
            )
            if not token or not secrets.compare_digest(token, required_key.strip()):
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized: force_refresh requires valid API key"
                )
    resolved = db.resolve_ticker(ticker) if hasattr(db, "resolve_ticker") else None
    target_ticker = resolved.get("ticker") if resolved and isinstance(resolved, dict) else ticker.strip().upper()

    snapshot = db.get_latest_candidate_snapshot(target_ticker) if hasattr(db, "get_latest_candidate_snapshot") else None
    candidate_data = None
    if snapshot:
        candidate_data = snapshot
    else:
        hist = db.get_ticker_history(target_ticker, limit=10) if hasattr(db, "get_ticker_history") else []
        if hist:
            latest = hist[0]
            candidate_data = {
                "ticker": target_ticker,
                "current_price": latest.get("current_price"),
                "composite_score": latest.get("composite_score") or latest.get("score"),
                "timesfm_potential": latest.get("timesfm_potential"),
                "timestral_potential": latest.get("timestral_potential"),
                "risk_reward_ratio": latest.get("risk_reward_ratio"),
                "ma60": latest.get("ma60"),
                "pullback_type": latest.get("pullback_type"),
                "pe": latest.get("pe"),
                "pb": latest.get("pb"),
                "trust_net_5d": latest.get("trust_net_5d"),
                "foreign_net_5d": latest.get("foreign_net_5d"),
                "tags": latest.get("tags")
            }

    if not candidate_data:
        raise HTTPException(
            status_code=404,
            detail=f"No quantitative records found for ticker '{target_ticker}'."
        )

    verdict = decision_service.evaluate_stock(
        candidate=candidate_data,
        macro_regime=macro_regime,
        force_refresh=force_refresh
    )

    return {
        "success": verdict.success,
        "ticker": target_ticker,
        "verdict": verdict.to_dict()
    }
