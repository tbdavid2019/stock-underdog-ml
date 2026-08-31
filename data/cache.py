"""
Unified Cache Manager for Stock Price Data, Index Lists, and Fundamentals.
Supports multi-tiered disk caching with expiration policies (TTL).
"""
import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd

logger = logging.getLogger("stock_app.cache")

# Base directory for cache storage
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.path.join(BASE_DIR, "cache")
STOCK_DATA_CACHE_DIR = os.path.join(CACHE_ROOT, "stock_data")
FUNDAMENTALS_CACHE_DIR = os.path.join(CACHE_ROOT, "fundamentals")
INDEX_CACHE_FILE = os.path.join(CACHE_ROOT, "stock_lists.json")

os.makedirs(STOCK_DATA_CACHE_DIR, exist_ok=True)
os.makedirs(FUNDAMENTALS_CACHE_DIR, exist_ok=True)


class CacheManager:
    """Unified manager for all disk caches in the application"""

    # ========== 1. OHLCV Stock Price DataFrame Cache (Pickle) ==========

    @classmethod
    def _get_stock_data_path(cls, ticker: str, period: str) -> str:
        safe_ticker = ticker.replace("/", "_")
        return os.path.join(STOCK_DATA_CACHE_DIR, f"{safe_ticker}_{period}.pkl")

    @classmethod
    def get_stock_data(cls, ticker: str, period: str, max_age_hours: int = 12) -> Optional[pd.DataFrame]:
        path = cls._get_stock_data_path(ticker, period)
        if not os.path.exists(path):
            return None

        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.now() - mtime > timedelta(hours=max_age_hours):
                return None

            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, pd.DataFrame) and not data.empty:
                logger.debug(f"📦 快取命中 (股價): {ticker} ({period})")
                return data
        except Exception as e:
            logger.warning(f"讀取股價快取失敗 {ticker}: {e}")
        return None

    @classmethod
    def set_stock_data(cls, ticker: str, period: str, data: pd.DataFrame):
        if data is None or data.empty:
            return
        path = cls._get_stock_data_path(ticker, period)
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"💾 股價快取已儲存: {ticker} ({period})")
        except Exception as e:
            logger.warning(f"儲存股價快取失敗 {ticker}: {e}")

    # ========== 2. Stock Index Components Cache (JSON) ==========

    @classmethod
    def get_index_cache(cls, index_name: str, max_age_days: int = 90) -> Optional[Dict[str, Any]]:
        if not os.path.exists(INDEX_CACHE_FILE):
            return None

        try:
            with open(INDEX_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)

            if index_name not in cache:
                return None

            entry = cache[index_name]
            updated_at = datetime.fromisoformat(entry.get("updated_at", "2000-01-01T00:00:00"))
            age = datetime.now() - updated_at

            if age > timedelta(days=max_age_days):
                logger.warning(f"⚠️ 指數快取已過期 ({age.days} 天 > {max_age_days} 天): {index_name}")
                return None

            return entry
        except Exception as e:
            logger.warning(f"讀取指數快取失敗 {index_name}: {e}")
            return None

    @classmethod
    def set_index_cache(cls, index_name: str, stocks: List[str], name_map: Optional[Dict[str, str]] = None):
        try:
            cache = {}
            if os.path.exists(INDEX_CACHE_FILE):
                with open(INDEX_CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)

            cache[index_name] = {
                "stocks": stocks,
                "name_map": name_map or {},
                "updated_at": datetime.now().isoformat()
            }

            with open(INDEX_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            logger.info(f"💾 指數快取已更新: {index_name} ({len(stocks)} 支)")
        except Exception as e:
            logger.warning(f"儲存指數快取失敗 {index_name}: {e}")

    # ========== 3. Fundamental Metrics Cache (JSON) ==========

    @classmethod
    def _get_fundamentals_path(cls, ticker: str) -> str:
        safe_ticker = ticker.replace("/", "_")
        return os.path.join(FUNDAMENTALS_CACHE_DIR, f"{safe_ticker}.json")

    @classmethod
    def get_fundamentals(cls, ticker: str, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
        path = cls._get_fundamentals_path(ticker)
        if not os.path.exists(path):
            return None

        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if datetime.now() - mtime > timedelta(hours=max_age_hours):
                return None

            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"讀取基本面快取失敗 {ticker}: {e}")
            return None

    @classmethod
    def set_fundamentals(cls, ticker: str, data: Dict[str, Any]):
        if not data:
            return
        path = cls._get_fundamentals_path(ticker)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"儲存基本面快取失敗 {ticker}: {e}")

    # ========== 4. Cache Purging ==========

    @classmethod
    def clear_all_stock_data(cls):
        for fname in os.listdir(STOCK_DATA_CACHE_DIR):
            if fname.endswith(".pkl"):
                os.remove(os.path.join(STOCK_DATA_CACHE_DIR, fname))
        logger.info("🗑️ 已清除所有股價快取")

    @classmethod
    def clear_all_fundamentals(cls):
        for fname in os.listdir(FUNDAMENTALS_CACHE_DIR):
            if fname.endswith(".json"):
                os.remove(os.path.join(FUNDAMENTALS_CACHE_DIR, fname))
        logger.info("🗑️ 已清除所有基本面快取")
