"""
Fundamental Data Provider.
Retrieves and caches valuation metrics (PE, Forward PE, PB, EV/EBITDA) with rate limiting and error resiliency.
"""
import logging
import time
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.cache import CacheManager

logger = logging.getLogger("stock_app.fundamentals")


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
    """Provider for stock fundamental and valuation metrics"""

    @classmethod
    def get_fundamentals(cls, ticker: str, use_cache: bool = True, max_age_hours: int = 24) -> Dict[str, Optional[float]]:
        """
        Fetch fundamental metrics for a single stock ticker.
        
        Returns:
            dict containing 'pe', 'forward_pe', 'pb', 'ev_ebitda' (values are float or None)
        """
        # 1. Try cache
        if use_cache:
            cached = CacheManager.get_fundamentals(ticker, max_age_hours=max_age_hours)
            if cached is not None:
                return {
                    "pe": sanitize_metric(cached.get("pe")),
                    "forward_pe": sanitize_metric(cached.get("forward_pe")),
                    "pb": sanitize_metric(cached.get("pb")),
                    "ev_ebitda": sanitize_metric(cached.get("ev_ebitda"))
                }

        # 2. Fetch from yfinance
        result = {
            "pe": None,
            "forward_pe": None,
            "pb": None,
            "ev_ebitda": None
        }

        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info

            if isinstance(info, dict):
                result = {
                    "pe": sanitize_metric(info.get("trailingPE")),
                    "forward_pe": sanitize_metric(info.get("forwardPE")),
                    "pb": sanitize_metric(info.get("priceToBook")),
                    "ev_ebitda": sanitize_metric(info.get("enterpriseToEbitda"))
                }

            # Save to cache if at least one metric is valid or info was fetched
            if use_cache:
                CacheManager.set_fundamentals(ticker, result)

        except Exception as e:
            logger.debug(f"獲取 {ticker} 基本面資料失敗: {e}")

        return result

    @classmethod
    def get_fundamentals_batch(
        cls, 
        tickers: List[str], 
        max_workers: int = 4, 
        use_cache: bool = True
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Concurrently fetch fundamental metrics for multiple stock tickers.
        """
        results: Dict[str, Dict[str, Optional[float]]] = {}
        
        # 1. Check cache first in main thread to avoid thread overhead
        pending_tickers = []
        for ticker in tickers:
            if use_cache:
                cached = CacheManager.get_fundamentals(ticker)
                if cached is not None:
                    results[ticker] = {
                        "pe": sanitize_metric(cached.get("pe")),
                        "forward_pe": sanitize_metric(cached.get("forward_pe")),
                        "pb": sanitize_metric(cached.get("pb")),
                        "ev_ebitda": sanitize_metric(cached.get("ev_ebitda"))
                    }
                    continue
            pending_tickers.append(ticker)

        if not pending_tickers:
            return results

        logger.info(f"📊 正在下載 {len(pending_tickers)} 支股票之基本面數據...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(cls.get_fundamentals, ticker, use_cache): ticker 
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
