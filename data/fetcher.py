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
    def get_tw0051_stocks(cls, exclude_tw0050: bool = True) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("TW0051", config.api.TW0051_URL, "TW0051", suffix=".TW")
        if exclude_tw0050:
            tw50 = set(cls.get_tw0050_stocks())
            stocks = [s for s in stocks if s not in tw50]
        return stocks

    @classmethod
    def get_tw_universe(cls, limit: int = 1000) -> List[str]:
        """
        動態獲取台股全市場核心與高流動性活躍標的 (預設 1,000 檔)。
        保障核心指數 (台灣50、台灣中型100、AI 題材供應鏈圖譜節點) 必選，
        其餘依據 DuckDB 全球主檔 (market_universe) 與日 K (tw_daily_bars) 之近期成交金額自動遞補，
        嚴格過濾非 ACTIVE、下市櫃與零成交殭屍股。
        """
        seen = set()
        selected: List[str] = []

        # 1. 優先加入核心指數 (台灣50、中型100)
        core_tw50 = cls.get_tw0050_stocks()
        for s in core_tw50:
            if s and s not in seen:
                seen.add(s)
                selected.append(s)

        core_tw51 = cls.get_tw0051_stocks(exclude_tw0050=False)
        for s in core_tw51:
            if s and s not in seen:
                seen.add(s)
                selected.append(s)

        # 2. 優先加入產業價值鏈圖譜重點節點
        try:
            from data.industry_chain_fetcher import EMBEDDED_NODES
            from data.duckdb_manager import DuckDBManager
            db_mgr = DuckDBManager()
            symbol_map = db_mgr.get_tw_symbol_map() if db_mgr.enabled else {}
            for node in EMBEDDED_NODES:
                raw_ticker = str(node.get("ticker", "")).strip()
                market_type = node.get("market", "")
                if market_type == "TW" and raw_ticker:
                    norm_ticker = symbol_map.get(raw_ticker, f"{raw_ticker}.TW")
                    if norm_ticker not in seen:
                        seen.add(norm_ticker)
                        selected.append(norm_ticker)
        except Exception as e:
            logger.debug(f"載入產業節點補充池略過: {e}")

        # 3. 透過 DuckDB 依近期成交金額由大至小動態遞補
        try:
            from data.duckdb_manager import DuckDBManager
            db_mgr = DuckDBManager()
            if db_mgr.enabled:
                ranked = db_mgr.get_tw_ranked_active_stocks(limit=limit + 300)
                for item in ranked:
                    tk = item.get("ticker")
                    if tk and tk not in seen:
                        seen.add(tk)
                        selected.append(tk)
                        if len(selected) >= limit:
                            break
        except Exception as e:
            logger.warning(f"自 DuckDB 遞補台股活躍標的失敗: {e}")

        logger.info(f"🇹🇼 成功構建台股量化宇宙清冊: 共 {len(selected[:limit])} 檔有效標的 (目標: {limit} 檔)")
        return selected[:limit]

    @classmethod
    def get_tw1000_stocks(cls) -> List[str]:
        """獲取台股 1,000 檔高流動性預測池"""
        return cls.get_tw_universe(limit=1000)

    @classmethod
    def get_tw_active_stocks(cls, limit: int = 850, exclude_indices: bool = True) -> List[str]:
        """
        獲取台股活躍中小型與櫃買成長股精選 (排除台灣50與中型100重疊標的，確保三層不重複)。
        """
        full_pool = cls.get_tw_universe(limit=limit + 250)
        if not exclude_indices:
            return full_pool[:limit]

        tw50_set = set(cls.get_tw0050_stocks())
        tw51_set = set(cls.get_tw0051_stocks(exclude_tw0050=False))
        excluded = tw50_set.union(tw51_set)

        active = [s for s in full_pool if s not in excluded]
        return active[:limit]

    @classmethod
    def get_sp500_stocks(cls, limit: Optional[int] = None) -> List[str]:
        stocks, _ = cls._fetch_index_with_cache_fallback("SP500", config.api.SP500_URL, "SP500", limit=limit)
        stocks = [s.replace("BRK.B", "BRK-B") for s in stocks]
        # 強制加入太空概念與美股 AI/算力前沿純度標的 (SPCX, DXYZ, ASTS, RKLB, CRWV, FN, CLS)
        extra_space_stocks = ["SPCX", "DXYZ", "ASTS", "RKLB", "CRWV", "FN", "CLS"]
        for st in extra_space_stocks:
            if st not in stocks:
                stocks.append(st)
        if limit is not None and len(stocks) > limit:
            return stocks[:limit]
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
                name_map.setdefault("CRWV", "CoreWeave (CRWV)")
                name_map.setdefault("FN", "Fabrinet (FN)")
                name_map.setdefault("CLS", "Celestica (CLS)")
            elif index_name in ("台灣中型100", "TW0051", "0051"):
                tw50_stocks = set(cls.get_tw0050_stocks())
                name_map = {k: v for k, v in name_map.items() if k not in tw50_stocks}
            return name_map

        if index_name in ("台股活躍精選", "台股1000", "TW1000", "台股全市場1000"):
            merged_map = {}
            merged_map.update(cls.get_index_name_map("台灣50"))
            merged_map.update(cls.get_index_name_map("台灣中型100"))
            try:
                from data.duckdb_manager import DuckDBManager
                db_names = DuckDBManager().get_tw_name_map()
                if db_names:
                    merged_map.update(db_names)
            except Exception as e:
                logger.debug(f"DuckDB 台股名稱提取失敗: {e}")
            return merged_map

        return {}

    @classmethod
    def get_stocks_by_index(cls, index_name: str) -> List[str]:
        mapping = {
            "台灣50": cls.get_tw0050_stocks,
            "台灣中型100": cls.get_tw0051_stocks,
            "台股活躍精選": lambda: cls.get_tw_active_stocks(limit=850, exclude_indices=True),
            "台股1000": cls.get_tw1000_stocks,
            "TW1000": cls.get_tw1000_stocks,
            "台股全市場1000": cls.get_tw1000_stocks,
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
        use_cache: bool = True,
        chunk_size: int = 200
    ) -> Dict[str, pd.DataFrame]:
        """
        Retrieve historical price data for a batch of tickers.
        Uses chunked yfinance batch download (default chunk_size=200) to prevent socket overload,
        crumb race conditions, and upstream rate limits.
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

        logger.info(f"🌐 正在批次下載 {len(pending_tickers)} 支股票之行情數據 ({period}，分塊大小: {chunk_size})...")

        # Split into chunks of chunk_size
        chunks = [pending_tickers[i:i + chunk_size] for i in range(0, len(pending_tickers), chunk_size)]

        for chunk_idx, chunk in enumerate(chunks, 1):
            if len(chunks) > 1:
                logger.info(f"   ▶ 下載分塊 [{chunk_idx}/{len(chunks)}]: 共 {len(chunk)} 支標的...")
            try:
                import yfinance as yf
                batch_data = yf.download(
                    chunk,
                    period=period,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False
                )

                if not batch_data.empty:
                    if len(chunk) == 1:
                        tk = chunk[0]
                        df = batch_data.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df = df.dropna(how="all")
                        if not df.empty:
                            if use_cache:
                                CacheManager.set_stock_data(tk, period, df)
                            results[tk] = add_base_indicators(df)
                    else:
                        for tk in chunk:
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
                logger.warning(f"⚠️ yfinance 批次分塊 [{chunk_idx}/{len(chunks)}] 下載失敗: {e}")

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
