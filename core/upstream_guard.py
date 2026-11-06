"""
core/upstream_guard.py - Multi-key pool rotation, in-flight concurrency limiter, and failure cooldown.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, Generator, List, Optional, Tuple, Any

logger = logging.getLogger("stock_app.upstream_guard")


class RecentFailureCooldownError(Exception):
    """Raised when an upstream endpoint or key is in failure cooldown."""
    pass


def mask_key(key: Optional[str]) -> str:
    """Mask API key or URL for safe logging (never outputs full cleartext or sensitive queries)."""
    if not key:
        return "[EMPTY]"
    clean = str(key).strip()
    if "://" in clean or "?" in clean:
        try:
            from urllib.parse import urlparse
            p = urlparse(clean)
            return p.netloc or p.path.split("/")[0] or "***"
        except Exception:
            return "***"
    if len(clean) <= 6:
        return "***"
    return f"{clean[:3]}...{clean[-3:]}"


class KeyPoolGuard:
    """
    Thread-safe guard managing a pool of API keys with:
    1. Round-robin rotation across multiple keys (comma-separated).
    2. Bounded in-flight concurrency per key via BoundedSemaphore.
    3. 60-second failure cooldown circuit breaker.
    4. Zero-cleartext logging of secrets.
    """

    def __init__(
        self,
        keys: str | List[str] | None = None,
        max_in_flight_per_key: int = 5,
        cooldown_seconds: float = 60.0,
        acquire_timeout: float = 5.0,
        name: str = "upstream"
    ):
        self.name = name
        self.max_in_flight = max(1, max_in_flight_per_key)
        self.cooldown_seconds = cooldown_seconds
        self.acquire_timeout = acquire_timeout
        self._lock = threading.Lock()

        # Parse keys
        parsed: List[str] = []
        if isinstance(keys, str):
            parsed = [k.strip() for k in keys.split(",") if k.strip()]
        elif isinstance(keys, list):
            parsed = [str(k).strip() for k in keys if str(k).strip()]
        self._keys: List[str] = parsed if parsed else [""]
        self._key_index = 0

        # Per-key concurrency semaphores
        self._semaphores: Dict[str, threading.BoundedSemaphore] = {
            k: threading.BoundedSemaphore(self.max_in_flight) for k in self._keys
        }

        # Failure cooldown tracking: endpoint_or_key -> cooldown_expiry_epoch
        self._cooldowns: Dict[str, float] = {}

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._keys)

    def is_in_cooldown(self, identifier: Optional[str] = None) -> bool:
        """Check if the given identifier (or general guard) is currently in cooldown."""
        target = identifier or self.name
        with self._lock:
            expiry = self._cooldowns.get(target, 0.0)
            return time.time() < expiry

    def record_failure(self, identifier: Optional[str] = None, cooldown_seconds: Optional[float] = None) -> None:
        """Trip circuit breaker and enter cooldown for cooldown_seconds."""
        target = identifier or self.name
        cd = cooldown_seconds if cooldown_seconds is not None else self.cooldown_seconds
        expiry = time.time() + cd
        with self._lock:
            self._cooldowns[target] = expiry
        logger.warning(
            "⚠️ [%s] 上游進入 %.1fs 熔斷冷卻期 (目標: %s)",
            self.name,
            cd,
            target if target == self.name else mask_key(target)
        )

    def record_success(self, identifier: Optional[str] = None) -> None:
        """Clear cooldown on successful upstream response."""
        target = identifier or self.name
        with self._lock:
            self._cooldowns.pop(target, None)

    def get_next_key(self) -> str:
        """Get the next key via round-robin."""
        with self._lock:
            if not self._keys:
                return ""
            key = self._keys[self._key_index % len(self._keys)]
            self._key_index = (self._key_index + 1) % len(self._keys)
            return key

    @contextmanager
    def acquire(self, identifier: Optional[str] = None) -> Generator[str, None, None]:
        """
        Acquire a key and concurrency slot under safety bounds.
        Raises RecentFailureCooldownError if the endpoint is in cooldown.
        """
        target = identifier or self.name
        now = time.time()
        with self._lock:
            expiry = self._cooldowns.get(target, 0.0)
            if now < expiry:
                remaining = expiry - now
                masked_target = target if target == self.name else mask_key(target)
                raise RecentFailureCooldownError(
                    f"[{self.name}] Upstream '{masked_target}' in cooldown ({remaining:.1f}s remaining)"
                )

        key = self.get_next_key()
        sem = self._semaphores.get(key)
        if sem is None:
            with self._lock:
                if key not in self._semaphores:
                    self._semaphores[key] = threading.BoundedSemaphore(self.max_in_flight)
                sem = self._semaphores[key]

        acquired = sem.acquire(timeout=self.acquire_timeout)
        if not acquired:
            raise TimeoutError(
                f"[{self.name}] Timed out waiting for in-flight concurrency slot (max {self.max_in_flight})"
            )

        try:
            yield key
        finally:
            try:
                sem.release()
            except ValueError:
                pass
