"""
Stock Data & Index Fetcher.
Retrieves stock lists from various indices (with answerbook upstream and cache fallback)
and downloads market price series with native batch support and indicator preparation.
"""
import os
import logging
import requests
import time
from typing import List, Optional, Dict, Any, Tuple
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
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure numeric Close
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
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

                CacheManager.set_index_cache(index_key, stocks, name_map)
                return stocks, name_map
            else:
                logger.warning(f"⚠️ 上游 API 回應異常 ({index_key}): HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ 抓取 {index_key} 上游失敗: {e}")

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
        stocks = [s.replace("BRK.B", "BRK-B") for s in stocks]
        # 強制加入 SpaceX 與太空概念股 (SPCX, DXYZ, ASTS, RKLB)
        extra_space_stocks = ["SPCX", "DXYZ", "ASTS", "RKLB"]
        for st in extra_space_stocks:
            if st not in stocks:
                stocks.append(st)
        return stocks

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
            "台灣中型100": ("TW0051", config.api.TW0051_URL, "TW0051", ".TW"),
            "SP500": ("SP500", config.api.SP500_URL, "SP500", "")
        }
        if index_name in index_map:
            key, url, jkey, sfx = index_map[index_name]
            _, name_map = cls._fetch_index_with_cache_fallback(key, url, jkey, suffix=sfx)
            if index_name == "SP500":
                name_map.setdefault("SPCX", "SpaceX (SPCX)")
                name_map.setdefault("DXYZ", "Destiny Tech100 / SpaceX (DXYZ)")
                name_map.setdefault("ASTS", "AST SpaceMobile (ASTS)")
                name_map.setdefault("RKLB", "Rocket Lab (RKLB)")
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
            # Use Ticker.history for reliable single ticker fetch
            t = yf.Ticker(ticker)
            data = t.history(period=period, auto_adjust=True)

            if data.empty:
                logger.warning(f"⚠️ {ticker} yfinance 獲取數據為空，嘗試從本地 DuckDB 備援讀取...")
                from data.duckdb_manager import DuckDBManager
                db_data = DuckDBManager().get_daily_bars_for_ticker(ticker)
                if not db_data.empty:
                    logger.info(f"🔄 成功從 DuckDB 備援載入 {ticker} 歷史日 K ({len(db_data)} 筆)")
                    return add_base_indicators(db_data)
                return pd.DataFrame()

            # Handle MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure standard OHLCV columns exist
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in data.columns:
                    data[col] = 0.0

            # Save raw data to cache
            if use_cache:
                CacheManager.set_stock_data(ticker, period, data)

            return add_base_indicators(data)

        except Exception as e:
            logger.error(f"❌ 下載 {ticker} 行情時發生錯誤: {e}，嘗試從本地 DuckDB 備援讀取...")
            try:
                from data.duckdb_manager import DuckDBManager
                db_data = DuckDBManager().get_daily_bars_for_ticker(ticker)
                if not db_data.empty:
                    logger.info(f"🔄 成功從 DuckDB 備援載入 {ticker} 歷史日 K ({len(db_data)} 筆)")
                    return add_base_indicators(db_data)
            except Exception as db_e:
                logger.debug(f"DuckDB fallback error: {db_e}")
            return pd.DataFrame()

    @classmethod
    def get_stock_data_batch(
        cls, 
        tickers: List[str], 
        period: str = "6mo", 
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical price data for a batch of tickers.
        Uses native yfinance batch download to prevent concurrency crumb race conditions.
        """
        results: Dict[str, pd.DataFrame] = {}
        pending_tickers: List[str] = []

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

        logger.info(f"🌐 正在批次下載 {len(pending_tickers)} 支股票之行情數據 ({period})...")

        try:
            import yfinance as yf
            
            # Native yfinance batch download handles internal threading safely
            batch_data = yf.download(
                pending_tickers,
                period=period,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )

            if not batch_data.empty:
                if len(pending_tickers) == 1:
                    # Single ticker batch returns 1D or 2D dataframe
                    tk = pending_tickers[0]
                    df = batch_data.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.dropna(how="all")
                    if not df.empty:
                        if use_cache:
                            CacheManager.set_stock_data(tk, period, df)
                        results[tk] = add_base_indicators(df)
                else:
                    # Multi-ticker batch
                    for tk in pending_tickers:
                        try:
                            if tk in batch_data.columns:
                                sub_df = batch_data[tk].dropna(how="all").copy()
                                if not sub_df.empty:
                                    for col in ["Open", "High", "Low", "Close", "Volume"]:
                                        if col not in sub_df.columns:
                                            sub_df[col] = 0.0
                                    if use_cache:
                                        CacheManager.set_stock_data(tk, period, sub_df)
                                    results[tk] = add_base_indicators(sub_df)
                        except Exception as e:
                            logger.debug(f"處理批次股票 {tk} 時出錯: {e}")

        except Exception as e:
            logger.warning(f"⚠️ yfinance 批次下載失敗: {e}，切換為單支回退下載")

        # 2. Fallback for any missing tickers in batch
        missing = [tk for tk in pending_tickers if tk not in results or results[tk].empty]
        if missing:
            logger.info(f"🔄 正在回退下載 {len(missing)} 支缺失的股票...")
            for tk in missing:
                df = cls.get_stock_data(tk, period=period, use_cache=use_cache)
                if not df.empty:
                    results[tk] = df

        logger.info(f"✅ 行情數據下載完成: 成功 {len(results)}/{len(tickers)} 支")
        return results
