"""
core/duckdb_lock.py - Cross-process file locking for DuckDB operations.
"""
from __future__ import annotations

import errno
import fcntl
import os
import random
import threading
import time
from contextlib import contextmanager
from typing import Generator, Optional


class DuckDBBusyError(Exception):
    """Raised when acquiring cross-process DuckDB write lock times out."""
    pass


_THREAD_LOCK_STATE = threading.local()


def get_lock_path(db_path: str) -> Optional[str]:
    """Get the write lock file path corresponding to a DuckDB database path."""
    if not db_path or db_path == ":memory:":
        return None
    return f"{db_path}.write.lock"


def is_current_thread_holding_write_lock(db_path: str) -> bool:
    """Return True if the calling thread currently holds the write lock for db_path."""
    lock_path = get_lock_path(db_path)
    if lock_path is None:
        return False
    real_path = os.path.realpath(lock_path)
    if not hasattr(_THREAD_LOCK_STATE, "depths"):
        return False
    return _THREAD_LOCK_STATE.depths.get(real_path, 0) > 0


@contextmanager
def duckdb_write_lock(db_path: str, timeout: float = 15.0) -> Generator[None, None, None]:
    """
    Acquire an exclusive cross-process write lock for the given DuckDB path using fcntl.flock.
    If db_path is ':memory:', file locking is safely bypassed.
    Re-entrant for the same thread to prevent self-deadlocks.
    Raises DuckDBBusyError if the lock cannot be acquired within timeout seconds.
    """
    lock_path = get_lock_path(db_path)
    if lock_path is None:
        yield
        return

    lock_dir = os.path.dirname(lock_path)
    if lock_dir:
        os.makedirs(lock_dir, exist_ok=True)

    real_path = os.path.realpath(lock_path)
    if not hasattr(_THREAD_LOCK_STATE, "depths"):
        _THREAD_LOCK_STATE.depths = {}

    current_depth = _THREAD_LOCK_STATE.depths.get(real_path, 0)
    if current_depth > 0:
        _THREAD_LOCK_STATE.depths[real_path] = current_depth + 1
        try:
            yield
        finally:
            _THREAD_LOCK_STATE.depths[real_path] -= 1
            if _THREAD_LOCK_STATE.depths[real_path] <= 0:
                _THREAD_LOCK_STATE.depths.pop(real_path, None)
        return

    start_time = time.monotonic()
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                _THREAD_LOCK_STATE.depths[real_path] = 1
                break
            except (BlockingIOError, OSError) as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise DuckDBBusyError(
                        f"Database write lock busy for '{db_path}' after {timeout:.1f}s"
                    ) from e
                sleep_dur = random.uniform(0.02, 0.08)
                time.sleep(sleep_dur)
        yield
    finally:
        _THREAD_LOCK_STATE.depths.pop(real_path, None)
        if acquired:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(fd)
        except OSError:
            pass
