"""
Data loading utilities for stock prediction application.
Fetches stock lists from various indices and downloads stock data.
"""
import pandas as pd
import yfinance as yf
import requests
import os
import pickle
from datetime import datetime, timedelta

from typing import List, Optional
from config import config
from logger import logger

# 快取目錄
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "stock_data")
os.makedirs(CACHE_DIR, exist_ok=True)


# ========== Stock Index Fetchers ==========

def get_tw0050_stocks() -> List[str]:
    """
    Fetch Taiwan 50 (台灣50) stock component list
    Uses cache to reduce API dependency
    
    Returns:
        List of stock tickers with .TW suffix
    """
    from stock_cache import get_cached_stocks
    
    def _fetch():
        from config import config
        response = requests.get(config.api.TW0050_URL)
        data = response.json()
        return [f"{code}.TW" for code in data['TW0050'].keys()]
    
    return get_cached_stocks("TW0050", _fetch)


def get_tw0051_stocks() -> List[str]:
    """
    Fetch Taiwan Mid-Cap 100 (台灣中型100) stock component list
    Uses cache to reduce API dependency
    
    Returns:
        List of stock tickers with .TW suffix
    """
    from stock_cache import get_cached_stocks
    
    def _fetch():
        from config import config
        response = requests.get(config.api.TW0051_URL)
        data = response.json()
        return [f"{code}.TW" for code in data['TW0051'].keys()]
    
    return get_cached_stocks("TW0051", _fetch)


def get_sp500_stocks(limit: int = 110) -> List[str]:
    """
    Fetch S&P 500 stock component list
    Uses cache to reduce API dependency
    
    Args:
        limit: Maximum number of stocks to return (default: 110)
    
    Returns:
        List of stock tickers
    """
    from stock_cache import get_cached_stocks
    
    def _fetch():
        from config import config
        response = requests.get(config.api.SP500_URL)
        data = response.json()
        stocks = list(data['SP500'].keys())[:limit]
        
        # Fix special ticker formats (e.g., BRK.B -> BRK-B for yfinance)
        for i, stock in enumerate(stocks):
            if stock == "BRK.B":
                stocks[i] = "BRK-B"
        
        return stocks
    
    return get_cached_stocks("SP500", _fetch)


def get_nasdaq_stocks() -> List[str]:
    """
    Fetch NASDAQ 100 stock component list
    
    Returns:
        List of stock tickers
    """
    from config import config
    response = requests.get(config.api.NASDAQ100_URL)
    data = response.json()
    
    # Get stock ticker list
    stocks = list(data['nasdaq100'].keys())
    
    return stocks


def get_sox_stocks() -> List[str]:
    """
    Get Philadelphia Semiconductor Index (SOX) stock component list
    
    Returns:
        List of stock tickers
    """
    return [
        "NVDA", "AVGO", "GFS", "CRUS", "ON", "ASML", "QCOM", "SWKS", "MPWR", "ADI",
        "TSM", "AMD", "TXN", "QRVO", "AMKR", "MU", "ARM", "NXPI", "TER", "ENTG",
        "LSCC", "COHR", "ONTO", "MTSI", "KLAC", "LRCX", "MRVL", "AMAT", "INTC", "MCHP"
    ]


def get_dji_stocks() -> List[str]:
    """
    Fetch Dow Jones Industrial Average stock component list
    
    Returns:
        List of stock tickers
    """
    from config import config
    response = requests.get(config.api.DOWJONES_URL)
    data = response.json()
    
    # Get stock ticker list
    stocks = list(data['dowjones'].keys())
    
    return stocks


# ========== Stock Data Download ==========

def _get_cache_path(ticker: str, period: str) -> str:
    """Get cache file path for a ticker and period"""
    return os.path.join(CACHE_DIR, f"{ticker}_{period}.pkl")


def _is_cache_valid(cache_path: str, max_age_hours: int = 12) -> bool:
    """Check if cache file exists and is not too old"""
    if not os.path.exists(cache_path):
        return False
    
    # 檢查檔案修改時間
    mtime = os.path.getmtime(cache_path)
    age = datetime.now() - datetime.fromtimestamp(mtime)
    
    return age < timedelta(hours=max_age_hours)


def get_stock_data(ticker: str, period: str, use_cache: bool = True, max_age_hours: int = 12) -> pd.DataFrame:
    """
    Fetch stock data for a single ticker with cache support
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., "1mo", "6mo", "1y")
        use_cache: Whether to use cache (default: True)
        max_age_hours: Maximum cache age in hours (default: 12)
    
    Returns:
        DataFrame with stock price data
    """
    cache_path = _get_cache_path(ticker, period)
    
    # 嘗試從快取讀取
    if use_cache and _is_cache_valid(cache_path, max_age_hours):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"📦 使用快取 {ticker} ({len(data)} 條數據)")
            return data
        except Exception as e:
            logger.warning(f"讀取快取失敗 {ticker}: {e}")
    
    # 下載新數據
    try:
        logger.info(f"🌐 下載 {ticker} 的數據...")
        data = yf.download(ticker, period=period, progress=False)
        
        if data.empty:
             logger.warning(f"無法獲取 {ticker} 的數據 (Empty)")
             return data

        # 處理 MultiIndex 欄位（某些股票會有）
        if isinstance(data.columns, pd.MultiIndex):
            # 只取第一層欄位名稱
            data.columns = data.columns.get_level_values(0)
        
        logger.info(f"獲取到 {len(data)} 條交易日數據")
        
        # 儲存到快取
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"儲存快取失敗 {ticker}: {e}")
        
        return data
    except Exception as e:
        logger.error(f"獲取 {ticker} 數據時發生錯誤: {str(e)}")
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to stock DataFrame
    Adds: MA5 (5-day moving average), MA10 (10-day moving average), RSI14 (14-day RSI)
    
    Args:
        df: Stock price DataFrame
    
    Returns:
        DataFrame with added technical indicators
    """
    # Manual implementation of indicators to remove pandas_ta dependency
    # MA5, MA10
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma10'] = df['Close'].rolling(window=10).mean()
    
    # RSI 14 Logic
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    # Fill NaN values to avoid issues later
    df['ma5'] = df['ma5'].fillna(method='bfill')
    df['ma10'] = df['ma10'].fillna(method='bfill')
    df['rsi14'] = df['rsi14'].fillna(50)  # Neutral RSI for initial days
    
    return df


def download_many(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Download multiple stocks at once with technical indicators
    Used for cross-sectional models
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period (e.g., "6mo", "1y")
    
    Returns:
        Flattened DataFrame with all stocks and their indicators
    """
    data = yf.download(
        " ".join(tickers), 
        period=period,
        group_by='ticker', 
        auto_adjust=True, 
        threads=True
    )
    
    frames = []
    
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple stocks downloaded
        for tic in tickers:
            sub = data[tic].copy()
            sub.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'][:len(sub.columns)]
            sub = add_indicators(sub)
            sub['Ticker'] = tic
            frames.append(sub)
    else:
        # Only one stock downloaded
        data = add_indicators(data)
        data['Ticker'] = tickers[0]
        frames.append(data)
    
    df = pd.concat(frames).reset_index().rename(columns={'index': 'Date'})
    
    # Ensure all base columns exist
    base_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for c in base_cols:
        if c not in df.columns:
            df[c] = 0.0  # Fill missing columns with 0
    
    cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'ma5', 'ma10', 'rsi14']
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is datetime type
    df = df.reset_index(drop=True)  # Remove MultiIndex if present
    
    return df[cols].dropna().sort_values(['Date', 'Ticker'])


# ========== Index Mapping ==========

INDEX_STOCK_MAP = {
    "台灣50": get_tw0050_stocks,
    "台灣中型100": get_tw0051_stocks,
    "SP500": get_sp500_stocks,
    "NASDAQ": get_nasdaq_stocks,
    "費城半導體": get_sox_stocks,
    "道瓊": get_dji_stocks,
}


def get_stocks_by_index(index_name: str) -> Optional[List[str]]:
    """
    Get stock list for a given index name
    
    Args:
        index_name: Name of the index (e.g., "SP500", "台灣50")
    
    Returns:
        List of stock tickers or None if index not found
    """
    fetcher = INDEX_STOCK_MAP.get(index_name)
    if fetcher:
        return fetcher()
    return None


def clear_stock_cache(ticker: str = None, period: str = None):
    """
    Clear cached stock data
    
    Args:
        ticker: Specific ticker to clear (None = clear all)
        period: Specific period to clear (None = clear all periods for ticker)
    """
    if ticker and period:
        # 清除特定股票和時期
        cache_path = _get_cache_path(ticker, period)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"🗑️  已清除快取: {ticker} ({period})")
    elif ticker:
        # 清除特定股票的所有時期
        pattern = f"{ticker}_"
        for filename in os.listdir(CACHE_DIR):
            if filename.startswith(pattern):
                os.remove(os.path.join(CACHE_DIR, filename))
        logger.info(f"🗑️  已清除快取: {ticker} (所有時期)")
    else:
        # 清除所有快取
        for filename in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, filename))
        logger.info(f"🗑️  已清除所有快取")
