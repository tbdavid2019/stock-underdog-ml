"""
core/netlimiter.py - Unified client IP resolution, rate limiting, and fail-closed API key verification.
"""
from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import Header, HTTPException, Request

from core.config import config

logger = logging.getLogger("stock_app.netlimiter")


def resolve_client_ip(request: Request) -> str:
    """
    Resolve client IP prioritizing edge headers:
    1. CF-Connecting-IP
    2. X-Real-IP
    3. client.host
    """
    for header in ("CF-Connecting-IP", "X-Real-IP"):
        val = request.headers.get(header)
        if val and val.strip():
            # Return first IP if comma-separated proxy list
            return val.strip().split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter per client IP with bounded memory."""

    def __init__(self, default_limit: int = 60, window_seconds: int = 60, max_entries: int = 2000):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._store: Dict[str, List[float]] = defaultdict(list)

    def prune(self, now: float) -> None:
        with self._lock:
            if len(self._store) > self.max_entries:
                for ip in list(self._store.keys()):
                    history = self._store[ip]
                    if not history or (now - history[-1] >= self.window_seconds):
                        self._store.pop(ip, None)

    def check(self, request: Request, limit: Optional[int] = None) -> None:
        client_ip = resolve_client_ip(request)
        now = time.time()
        effective_limit = limit or self.default_limit

        with self._lock:
            if len(self._store) > self.max_entries:
                for ip in list(self._store.keys()):
                    history = self._store[ip]
                    if not history or (now - history[-1] >= self.window_seconds):
                        self._store.pop(ip, None)

            history = self._store[client_ip]
            valid_history = [t for t in history if now - t < self.window_seconds]
            self._store[client_ip] = valid_history

            if len(valid_history) >= effective_limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: maximum {effective_limit} requests per {self.window_seconds}s."
                )

            self._store[client_ip].append(now)


# Global instance
default_limiter = SlidingWindowRateLimiter()


def verify_watchlist_write_access(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
) -> bool:
    """
    Fail-closed authentication for watchlist / alerts write operations.
    1. In production (DEPLOYMENT_MODE == 'production'):
       - WATCHLIST_API_KEY is strictly required.
       - If WATCHLIST_API_KEY is not set or token does not match, raises HTTP 401.
    2. In non-production (DEPLOYMENT_MODE != 'production'):
       - If ALLOW_ANONYMOUS_LOCAL_WRITES=true is set, anonymous local write is allowed.
       - Otherwise, requires valid WATCHLIST_API_KEY or raises HTTP 401.
    """
    deployment_mode = (os.getenv("DEPLOYMENT_MODE") or config.security.DEPLOYMENT_MODE or "production").strip().lower()
    expected_key = os.getenv("WATCHLIST_API_KEY") or config.security.WATCHLIST_API_KEY
    allow_anonymous = (
        (os.getenv("ALLOW_ANONYMOUS_LOCAL_WRITES") or str(config.security.ALLOW_ANONYMOUS_LOCAL_WRITES)).strip().lower()
        in ("true", "1", "yes")
    )

    if deployment_mode in ("development", "local", "test") and allow_anonymous:
        return True

    # Extract provided token
    token = None
    if x_api_key and x_api_key.strip():
        token = x_api_key.strip()
    elif authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()

    if not expected_key or not expected_key.strip():
        # Fail-closed: In production or without anonymous flag, missing server key locks writes
        logger.warning("🚨 [Fail-Closed] 寫入遭拒絕：伺服器未設定 WATCHLIST_API_KEY 且處於嚴格保護模式")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: WATCHLIST_API_KEY is required and not configured on server"
        )

    if not token or not secrets.compare_digest(token, expected_key.strip()):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing or invalid API key for write operation"
        )

    return True


def verify_profile_read_access(
    profile_id: str,
    request: Request,
    x_api_key: Optional[str] = None,
    authorization: Optional[str] = None
) -> bool:
    """
    Enforces profile privacy:
    - 'default' profile is publicly readable.
    - Non-default profiles require valid WATCHLIST_API_KEY to prevent enumeration.
    """
    pid = str(profile_id or "default").strip()
    if pid == "default":
        return True
    try:
        return verify_watchlist_write_access(request, x_api_key=x_api_key, authorization=authorization)
    except HTTPException:
        raise HTTPException(
            status_code=401,
            detail=f"Unauthorized: Access to private profile '{pid}' requires valid API key"
        )


def sanitize_health_data(raw_health: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize health check output:
    1. Only output hostname / netloc instead of complete URLs (stripping query parameters, tokens, paths).
    2. Only output exception type names rather than full stack traces or strings.
    """
    sanitized: Dict[str, Any] = {}
    for slot_name, details in raw_health.items():
        if not isinstance(details, dict):
            sanitized[slot_name] = details
            continue

        item = dict(details)
        raw_url = item.get("url")
        if raw_url:
            parsed = urlparse(str(raw_url))
            item["host"] = parsed.netloc or parsed.path.split("/")[0]
            item.pop("url", None)

        if "error" in item:
            err = item["error"]
            if isinstance(err, Exception):
                item["error_type"] = type(err).__name__
            else:
                err_str = str(err)
                item["error_type"] = err_str.split(":")[0].strip() if ":" in err_str else "UnknownError"
            item.pop("error", None)

        sanitized[slot_name] = item
    return sanitized
