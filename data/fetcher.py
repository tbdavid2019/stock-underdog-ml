"""
Stock Data & Index Fetcher.
Retrieves stock lists from various indices (with answerbook upstream and cache fallback)
and downloads market price series with native batch support and indicator preparation.
"""
import os
import re
import urllib.request
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


# 備援日經 225 成分股清單 (全數 225 檔東京證券交易所上市龍頭，支援離線與 API 降級容錯)
FALLBACK_NIKKEI225_STOCKS = [
    "1332.T", "1605.T", "1721.T", "1801.T", "1802.T", "1803.T", "1808.T", "1812.T", "1925.T", "1928.T",
    "1963.T", "2002.T", "2269.T", "2282.T", "2413.T", "2432.T", "2501.T", "2502.T", "2503.T", "2768.T",
    "2801.T", "2802.T", "285A.T", "2871.T", "2914.T", "3086.T", "3092.T", "3099.T", "3289.T", "3382.T",
    "3401.T", "3402.T", "3405.T", "3407.T", "3436.T", "3659.T", "3697.T", "3861.T", "4004.T", "4005.T",
    "4021.T", "4042.T", "4043.T", "4061.T", "4063.T", "4151.T", "4183.T", "4188.T", "4208.T", "4307.T",
    "4324.T", "4385.T", "4452.T", "4502.T", "4503.T", "4506.T", "4507.T", "4519.T", "4523.T", "4543.T",
    "4568.T", "4578.T", "4661.T", "4689.T", "4704.T", "4751.T", "4755.T", "4901.T", "4902.T", "4911.T",
    "5019.T", "5020.T", "5101.T", "5108.T", "5201.T", "5214.T", "5233.T", "5301.T", "5332.T", "5333.T",
    "5401.T", "5406.T", "5411.T", "543A.T", "5631.T", "5706.T", "5711.T", "5713.T", "5714.T", "5801.T",
    "5802.T", "5803.T", "5831.T", "6098.T", "6103.T", "6113.T", "6146.T", "6178.T", "6273.T", "6301.T",
    "6302.T", "6305.T", "6326.T", "6361.T", "6367.T", "6471.T", "6472.T", "6473.T", "6479.T", "6501.T",
    "6503.T", "6504.T", "6506.T", "6526.T", "6532.T", "6594.T", "6645.T", "6701.T", "6702.T", "6723.T",
    "6724.T", "6752.T", "6753.T", "6758.T", "6762.T", "6770.T", "6841.T", "6857.T", "6861.T", "6902.T",
    "6920.T", "6954.T", "6963.T", "6971.T", "6976.T", "6981.T", "6988.T", "7004.T", "7011.T", "7012.T",
    "7013.T", "7186.T", "7201.T", "7202.T", "7203.T", "7211.T", "7261.T", "7267.T", "7269.T", "7270.T",
    "7272.T", "7453.T", "7532.T", "7731.T", "7733.T", "7735.T", "7741.T", "7751.T", "7752.T", "7832.T",
    "7911.T", "7912.T", "7951.T", "7974.T", "8001.T", "8002.T", "8015.T", "8031.T", "8035.T", "8053.T",
    "8058.T", "8233.T", "8252.T", "8253.T", "8267.T", "8304.T", "8306.T", "8308.T", "8309.T", "8316.T",
    "8331.T", "8354.T", "8411.T", "8591.T", "8601.T", "8604.T", "8630.T", "8697.T", "8725.T", "8750.T",
    "8766.T", "8795.T", "8801.T", "8802.T", "8804.T", "8830.T", "9001.T", "9005.T", "9007.T", "9008.T",
    "9009.T", "9020.T", "9021.T", "9022.T", "9064.T", "9101.T", "9104.T", "9107.T", "9147.T", "9201.T",
    "9202.T", "9432.T", "9433.T", "9434.T", "9501.T", "9502.T", "9503.T", "9531.T", "9532.T", "9602.T",
    "9735.T", "9766.T", "9843.T", "9983.T", "9984.T"
]


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
                name_map = {
                    (f"{code}{suffix}" if suffix and not str(code).endswith(suffix) else str(code)): name
                    for code, name in items.items()
                }
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
    def get_nikkei225_stocks(cls, limit: Optional[int] = None) -> List[str]:
        """
        取得日經 225 (Nikkei 225) 成分股清單。
        具備 4 層容錯：
        1. 優先透過 DataAPI / Cloudflare KV (NIKKEI225_URL) 抓取。
        2. 其次自本機快取 cache/stock_lists.json (NIKKEI225) 讀取。
        3. 若快取過期，嘗試線上向 Wikipedia 抓取最新 225 檔清單並更新快取。
        4. 若外部抓取失敗，自動降級使用完整 225 檔內嵌備援清單 (FALLBACK_NIKKEI225_STOCKS)。
        """
        # 1. 嘗試向 DataAPI / Cloudflare KV 抓取
        nikkei_url = getattr(config.api, "NIKKEI225_URL", None)
        if nikkei_url:
            try:
                resp = requests.get(nikkei_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json().get("NIKKEI225", {})
                    if len(data) == 225:
                        stocks = [f"{c}.T" if not str(c).endswith(".T") else c for c in data.keys()]
                        return stocks[:limit] if limit else stocks
            except Exception as e:
                logger.debug(f"DataAPI NIKKEI225 抓取失敗: {e}")

        index_key = "NIKKEI225"
        cached = CacheManager.get_index_cache(index_key, max_age_days=config.pipeline.INDEX_CACHE_MAX_AGE_DAYS)
        if cached:
            stocks = cached if isinstance(cached, list) else cached.get("stocks", [])
            if stocks and len(stocks) == 225:
                return stocks[:limit] if limit else stocks

        # 嘗試向 Wikipedia 線上抓取最新日經 225 清單
        scraped_stocks = []
        try:
            req = urllib.request.Request(
                "https://en.wikipedia.org/wiki/Nikkei_225",
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8")
                comp_start = html.find('id="Components"')
                stat_start = html.find('id="Statistics"', comp_start) if comp_start != -1 else -1
                comp_html = html[comp_start:stat_start] if comp_start != -1 and stat_start != -1 else html
                codes = sorted(list(set(re.findall(r"topSearchStr=([0-9A-Z]{4})", comp_html))))
                # 日經 225 必須為完整 225 檔，若因結構變更或不全則視為抓取未竟全功，退回使用內嵌備援
                if len(codes) == 225:
                    scraped_stocks = [f"{c}.T" for c in codes]
        except Exception as e:
            logger.debug(f"線上抓取日經 225 成分股失敗: {e}")

        final_stocks = scraped_stocks if len(scraped_stocks) == 225 else list(FALLBACK_NIKKEI225_STOCKS)

        # 寫入快取
        try:
            CacheManager.set_index_cache(index_key, final_stocks, {})
        except Exception as e:
            logger.debug(f"寫入日經 225 快取失敗: {e}")

        return final_stocks[:limit] if limit else final_stocks

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

        if index_name in ("日經225", "日股225", "NIKKEI225", "N225", "JP225"):
            name_map = {}
            try:
                from data.duckdb_manager import DuckDBManager
                db_names = DuckDBManager().get_jp_name_map()
                if db_names:
                    name_map.update(db_names)
            except Exception as e:
                logger.debug(f"DuckDB 日股名稱提取失敗: {e}")
            # 確保重點龍頭名稱繁體化/知名度提升
            name_map.setdefault("8035.T", "東京威力科創 (Tokyo Electron)")
            name_map.setdefault("9984.T", "軟銀集團 (SoftBank)")
            name_map.setdefault("6857.T", "愛德萬測試 (Advantest)")
            name_map.setdefault("6146.T", "迪思科 (DISCO)")
            name_map.setdefault("4063.T", "信越化學 (Shin-Etsu)")
            name_map.setdefault("6920.T", "Lasertec (EUV檢測)")
            name_map.setdefault("6758.T", "索尼集團 (SONY)")
            name_map.setdefault("7203.T", "豐田汽車 (TOYOTA)")
            name_map.setdefault("7974.T", "任天堂 (Nintendo)")
            name_map.setdefault("6861.T", "基恩斯 (Keyence)")
            name_map.setdefault("9983.T", "迅銷 (Fast Retailing / UNIQLO)")
            name_map.setdefault("6723.T", "瑞薩電子 (Renesas)")
            name_map.setdefault("7741.T", "豪雅 (HOYA)")
            name_map.setdefault("6501.T", "日立製作所 (Hitachi)")
            name_map.setdefault("8058.T", "三菱商事 (Mitsubishi)")
            name_map.setdefault("6954.T", "發那科 (FANUC 機器人)")
            name_map.setdefault("6506.T", "安川電機 (Yaskawa 機器人)")
            return name_map

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
            "日經225": cls.get_nikkei225_stocks,
            "日股225": cls.get_nikkei225_stocks,
            "NIKKEI225": cls.get_nikkei225_stocks,
            "N225": cls.get_nikkei225_stocks,
            "JP225": cls.get_nikkei225_stocks,
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
