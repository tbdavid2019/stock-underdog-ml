"""
Stock Data & Index Fetcher.
Retrieves stock lists from various indices (with answerbook upstream and cache fallback)
and downloads market price series with indicator preparation.
"""
import os
import logging
import requests
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from core.config import config
from data.cache import CacheManager

logger = logging.getLogger("stock_app.fetcher")


def add_base_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorially add technical indicators to stock DataFrame:
    MA5, MA10, MA60, MA120, MA250, RSI14.
    """
    if data.empty or "Close" not in data.columns:
        return data

    df = data.copy()
    close = df["Close"]

    # Moving Averages
    df["ma5"] = close.rolling(window=5, min_periods=1).mean()
    df["ma10"] = close.rolling(window=10, min_periods=1).mean()
    df["ma60"] = close.rolling(window=60, min_periods=1).mean()
    df["ma120"] = close.rolling(window=120, min_periods=1).mean()
    df["ma250"] = close.rolling(window=250, min_periods=1).mean()

    # Uppercase aliases for strategy compatibility
    df["MA5"] = df["ma5"]
    df["MA10"] = df["ma10"]
    df["MA60"] = df["ma60"]
    df["MA120"] = df["ma120"]
    df["MA250"] = df["ma250"]

    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=1).mean()
    
    rs = gain / loss.replace(0, 1e-9)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi14"] = df["rsi14"].fillna(50.0)

    return df


class StockFetcher:
    """Fetcher for Index Constituents and OHLCV Market Data"""

    # ========== 1. Index Constituents Fetchers ==========

    @classmethod
    def _fetch_index_with_cache_fallback(
        cls, 
        index_key: str, 
        url: str, 
        json_key: str, 
        suffix: str = "", 
        limit: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Fetch index constituents from upstream API with automatic local cache fallback.
        """
        # Try fetching from upstream API first
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get(json_key, {})
                name_map = {f"{code}{suffix}": name for code, name in items.items()}
                stocks = list(name_map.keys())
                if limit:
                    stocks = stocks[:limit]
                    name_map = {k: v for k, v in name_map.items() if k in stocks}

                # Update local cache
                CacheManager.set_index_cache(index_key, stocks, name_map)
                return stocks, name_map
            else:
                logger.warning(f"⚠️ 上游 API 回應異常 ({index_key}): HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ 抓取 {index_key} 上游失敗: {e}")

        # Fallback to local cache
        cached = CacheManager.get_index_cache(index_key, max_age_days=config.pipeline.INDEX_CACHE_MAX_AGE_DAYS)
        if cached:
            logger.info(f"📦 使用本機快取 fallback ({index_key}): {len(cached.get('stocks', []))} 支股票")
            return cached.get("stocks", []), cached.get("name_map", {})

        logger.error(f"❌ 無法獲取 {index_key} 的成分股 (API 失敗且無有效快取)")
        return [], {}

    @classmethod
    def get_tw0050_stocks(cls) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("TW0050", config.api.TW0050_URL, "TW0050", suffix=".TW")
        return stocks

    @classmethod
    def get_tw0051_stocks(cls) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("TW0051", config.api.TW0051_URL, "TW0051", suffix=".TW")
        return stocks

    @classmethod
    def get_sp500_stocks(cls, limit: int = 110) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("SP500", config.api.SP500_URL, "SP500", limit=limit)
        # Format tickers like BRK.B -> BRK-B for yfinance
        return [s.replace("BRK.B", "BRK-B") for s in stocks]

    @classmethod
    def get_nasdaq_stocks(cls) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("NASDAQ", config.api.NASDAQ100_URL, "nasdaq100")
        return stocks

    @classmethod
    def get_sox_stocks(cls) -> List[str]:
        return [
            "NVDA", "AVGO", "GFS", "CRUS", "ON", "ASML", "QCOM", "SWKS", "MPWR", "ADI",
            "TSM", "AMD", "TXN", "QRVO", "AMKR", "MU", "ARM", "NXPI", "TER", "ENTG",
            "LSCC", "COHR", "ONTO", "MTSI", "KLAC", "LRCX", "MRVL", "AMAT", "INTC", "MCHP"
        ]

    @classmethod
    def get_dji_stocks(cls) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("DJI", config.api.DOWJONES_URL, "dowjones")
        return stocks

    @classmethod
    def get_index_name_map(cls, index_name: str) -> Dict[str, str]:
        index_map = {
            "台灣50": ("TW0050", config.api.TW0050_URL, "TW0050", ".TW"),
            "台灣中型100": ("TW0051", config.api.TW0051_URL, "TW0051", ".TW")
        }
        if index_name in index_map:
            key, url, jkey, sfx = index_map[index_name]
            _, name_map = cls._fetch_index_with_cache_fallback(key, url, jkey, suffix=sfx)
            return name_map
        return {}

    @classmethod
    def get_stocks_by_index(cls, index_name: str) -> List[str]:
        mapping = {
            "台灣50": cls.get_tw0050_stocks,
            "台灣中型100": cls.get_tw0051_stocks,
            "SP500": cls.get_sp500_stocks,
            "NASDAQ": cls.get_nasdaq_stocks,
            "費城半導體": cls.get_sox_stocks,
            "道瓊": cls.get_dji_stocks,
        }
        fetcher = mapping.get(index_name)
        return fetcher() if fetcher else []

    # ========== 2. OHLCV Market Data Fetchers ==========

    @classmethod
    def get_stock_data(
        cls, 
        ticker: str, 
        period: str = "6mo", 
        use_cache: bool = True, 
        max_age_hours: int = 12
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a single stock with indicators.
        """
        # 1. Try cache
        if use_cache:
            cached = CacheManager.get_stock_data(ticker, period, max_age_hours=max_age_hours)
            if cached is not None and not cached.empty:
                return add_base_indicators(cached)

        # 2. Download from yfinance
        try:
            import yfinance as yf
            logger.info(f"🌐 下載 {ticker} 行情數據 ({period})...")
            data = yf.download(ticker, period=period, progress=False)

            if data.empty:
                logger.warning(f"⚠️ {ticker} 獲取數據為空 (Empty)")
                return pd.DataFrame()

            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure basic columns exist
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in data.columns:
                    data[col] = 0.0

            # Save raw data to cache
            if use_cache:
                CacheManager.set_stock_data(ticker, period, data)

            return add_base_indicators(data)

        except Exception as e:
            logger.error(f"❌ 下載 {ticker} 行情時發生錯誤: {e}")
            return pd.DataFrame()

    @classmethod
    def get_stock_data_batch(
        cls, 
        tickers: List[str], 
        period: str = "6mo", 
        use_cache: bool = True, 
        max_workers: int = 8
    ) -> Dict[str, pd.DataFrame]:
        """
        Concurrently retrieve historical price data for a batch of tickers.
        """
        results: Dict[str, pd.DataFrame] = {}
        pending_tickers = []

        # 1. Check cache first
        for ticker in tickers:
            if use_cache:
                cached = CacheManager.get_stock_data(ticker, period)
                if cached is not None and not cached.empty:
                    results[ticker] = add_base_indicators(cached)
                    continue
            pending_tickers.append(ticker)

        if not pending_tickers:
            return results

        logger.info(f"🌐 正在並發下載 {len(pending_tickers)} 支股票之行情數據...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(cls.get_stock_data, ticker, period, use_cache): ticker
                for ticker in pending_tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result(timeout=60)
                    if not df.empty:
                        results[ticker] = df
                except Exception as e:
                    logger.warning(f"下載 {ticker} 失敗或超時: {e}")

        return results
