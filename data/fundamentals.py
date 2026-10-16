import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from data.duckdb_manager import DuckDBManager
except ImportError:
    DuckDBManager = None

from data.cache import CacheManager

logger = logging.getLogger("stock_app.fundamentals")

_MEM_CACHE_LOCK = threading.Lock()
_MEM_CACHE_MAX_SIZE = 1000


def sanitize_metric(val: Any) -> Optional[float]:
    """Safely converts a metric value to float or None"""
    if val is None:
        return None
    try:
        f_val = float(val)
        if f_val != f_val or f_val == float("inf") or f_val == float("-inf"):  # Check NaN / Inf
            return None
        return f_val
    except (ValueError, TypeError):
        return None


class FundamentalProvider:
    """Provider for stock fundamental and valuation metrics with 3-tier caching (L1 Mem -> L2 DuckDB -> L3 Disk)"""

    _MEM_CACHE: Dict[str, Tuple[float, Dict[str, Optional[float]]]] = {}

    @classmethod
    def _store_mem_cache(cls, ticker: str, val: Dict[str, Optional[float]], ts: Optional[float] = None) -> None:
        now = ts or time.time()
        with _MEM_CACHE_LOCK:
            if len(cls._MEM_CACHE) >= _MEM_CACHE_MAX_SIZE:
                for k in list(cls._MEM_CACHE.keys())[:200]:
                    cls._MEM_CACHE.pop(k, None)
            cls._MEM_CACHE[ticker] = (now, val)

    @classmethod
    def get_fundamentals(
        cls, 
        ticker: str, 
        use_cache: bool = True, 
        max_age_hours: int = 24,
        force_refresh: bool = False
    ) -> Dict[str, Optional[float]]:
        """
        Fetch fundamental metrics for a single stock ticker with 3-tier caching.
        
        Returns:
            dict containing 'pe', 'forward_pe', 'pb', 'ev_ebitda' (values are float or None)
        """
        now = time.time()

        # 1. L1 Memory Cache
        if use_cache and not force_refresh:
            with _MEM_CACHE_LOCK:
                if ticker in cls._MEM_CACHE:
                    ts, mem_data = cls._MEM_CACHE[ticker]
                    if now - ts < max_age_hours * 3600:
                        return mem_data

        # 2. L2 DuckDB Cache
        if use_cache and not force_refresh:
            try:
                if DuckDBManager is not None:
                    db = DuckDBManager()
                    db_data = db.get_company_fundamentals(ticker, max_age_hours=max_age_hours)
                    if db_data is not None:
                        res = {
                            "pe": sanitize_metric(db_data.get("pe")),
                            "forward_pe": sanitize_metric(db_data.get("forward_pe")),
                            "pb": sanitize_metric(db_data.get("pb")),
                            "ev_ebitda": sanitize_metric(db_data.get("ev_ebitda"))
                        }
                        cls._store_mem_cache(ticker, res, now)
                        return res
            except Exception as dbe:
                logger.debug(f"DuckDB 基本面讀取跳過 ({ticker}): {dbe}")

        # 3. L3 Disk JSON Cache
        if use_cache and not force_refresh:
            cached = CacheManager.get_fundamentals(ticker, max_age_hours=max_age_hours)
            if cached is not None:
                res = {
                    "pe": sanitize_metric(cached.get("pe")),
                    "forward_pe": sanitize_metric(cached.get("forward_pe")),
                    "pb": sanitize_metric(cached.get("pb")),
                    "ev_ebitda": sanitize_metric(cached.get("ev_ebitda"))
                }
                cls._store_mem_cache(ticker, res, now)
                # Sync back to DuckDB L2
                try:
                    if DuckDBManager is not None:
                        DuckDBManager().save_company_fundamentals(ticker, res)
                except Exception:
                    pass
                return res

        # 4. Fetch from yfinance
        result = {
            "pe": None,
            "forward_pe": None,
            "pb": None,
            "ev_ebitda": None
        }

        try:
            target_yf = yf
            if target_yf is None:
                import yfinance as target_yf
            stock = target_yf.Ticker(ticker)
            info = stock.info

            if isinstance(info, dict):
                result = {
                    "pe": sanitize_metric(info.get("trailingPE")),
                    "forward_pe": sanitize_metric(info.get("forwardPE")),
                    "pb": sanitize_metric(info.get("priceToBook")),
                    "ev_ebitda": sanitize_metric(info.get("enterpriseToEbitda"))
                }

            # Save to L1 memory, L2 DuckDB, and L3 disk
            if use_cache:
                cls._store_mem_cache(ticker, result, now)
                try:
                    if DuckDBManager is not None:
                        DuckDBManager().save_company_fundamentals(ticker, result)
                except Exception as dbe:
                    logger.debug(f"寫入 DuckDB 基本面失敗 ({ticker}): {dbe}")
                CacheManager.set_fundamentals(ticker, result)

        except Exception as e:
            logger.debug(f"獲取 {ticker} 基本面資料失敗: {e}")

        return result

    @classmethod
    def get_fundamentals_batch(
        cls, 
        tickers: List[str], 
        max_workers: int = 4, 
        use_cache: bool = True,
        max_age_hours: int = 24
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Concurrently fetch fundamental metrics for multiple stock tickers.
        """
        results: Dict[str, Dict[str, Optional[float]]] = {}
        now = time.time()
        
        # 1. Check cache first in main thread to avoid thread overhead
        pending_tickers = []
        for ticker in tickers:
            if use_cache:
                # L1 Memory Cache
                hit = False
                with _MEM_CACHE_LOCK:
                    if ticker in cls._MEM_CACHE:
                        ts, mem_data = cls._MEM_CACHE[ticker]
                        if now - ts < max_age_hours * 3600:
                            results[ticker] = mem_data
                            hit = True
                if hit:
                    continue

                # L2 DuckDB Cache
                try:
                    if DuckDBManager is not None:
                        db_data = DuckDBManager().get_company_fundamentals(ticker, max_age_hours=max_age_hours)
                        if db_data is not None:
                            val = {
                                "pe": sanitize_metric(db_data.get("pe")),
                                "forward_pe": sanitize_metric(db_data.get("forward_pe")),
                                "pb": sanitize_metric(db_data.get("pb")),
                                "ev_ebitda": sanitize_metric(db_data.get("ev_ebitda"))
                            }
                            results[ticker] = val
                            cls._store_mem_cache(ticker, val, now)
                            continue
                except Exception:
                    pass

                # L3 Disk Cache
                cached = CacheManager.get_fundamentals(ticker, max_age_hours=max_age_hours)
                if cached is not None:
                    val = {
                        "pe": sanitize_metric(cached.get("pe")),
                        "forward_pe": sanitize_metric(cached.get("forward_pe")),
                        "pb": sanitize_metric(cached.get("pb")),
                        "ev_ebitda": sanitize_metric(cached.get("ev_ebitda"))
                    }
                    results[ticker] = val
                    cls._store_mem_cache(ticker, val, now)
                    continue
            pending_tickers.append(ticker)

        if not pending_tickers:
            return results

        logger.info(f"📊 正在下載 {len(pending_tickers)} 支股票之基本面數據...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(cls.get_fundamentals, ticker, use_cache, max_age_hours): ticker 
                for ticker in pending_tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    res = future.result(timeout=30)
                    results[ticker] = res
                except Exception as e:
                    logger.debug(f"批次獲取 {ticker} 基本面超時或錯誤: {e}")
                    results[ticker] = {"pe": None, "forward_pe": None, "pb": None, "ev_ebitda": None}

        return results
