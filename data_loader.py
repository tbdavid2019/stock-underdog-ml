"""
Data loading utilities for stock prediction application.
Fetches stock lists from various indices and downloads stock data.
"""
import pandas as pd
import yfinance as yf
import requests

from typing import List, Optional
from config import config
from logger import logger


# ========== Stock Index Fetchers ==========

def get_tw0050_stocks() -> List[str]:
    """
    Fetch Taiwan 50 (台灣50) stock component list
    
    Returns:
        List of stock tickers with .TW suffix
    """
    from config import config
    response = requests.get(config.api.TW0050_URL)
    data = response.json()
    
    # Add .TW suffix to stock codes
    stocks = [f"{code}.TW" for code in data['TW0050'].keys()]
    
    return stocks


def get_tw0051_stocks() -> List[str]:
    """
    Fetch Taiwan Mid-Cap 100 (台灣中型100) stock component list
    
    Returns:
        List of stock tickers with .TW suffix
    """
    from config import config
    response = requests.get(config.api.TW0051_URL)
    data = response.json()
    
    # Add .TW suffix to stock codes
    stocks = [f"{code}.TW" for code in data['TW0051'].keys()]
    
    return stocks


def get_sp500_stocks(limit: int = 110) -> List[str]:
    """
    Fetch S&P 500 stock component list
    
    Args:
        limit: Maximum number of stocks to return (default: 110)
    
    Returns:
        List of stock tickers
    """
    from config import config
    response = requests.get(config.api.SP500_URL)
    data = response.json()
    
    # Get stock ticker list with limit
    stocks = list(data['SP500'].keys())[:limit]
    
    # Fix special ticker formats (e.g., BRK.B -> BRK-B for yfinance)
    for i, stock in enumerate(stocks):
        if stock == "BRK.B":
            stocks[i] = "BRK-B"
    
    return stocks


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

def get_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetch stock data for a single ticker
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (e.g., "1mo", "6mo", "1y")
    
    Returns:
        DataFrame with stock price data
    """
    try:
        logger.info(f"正在獲取 {ticker} 的數據...")
        data = yf.download(ticker, period=period, progress=False) # Disable yfinance progress bar
        
        if data.empty:
             logger.warning(f"無法獲取 {ticker} 的數據 (Empty)")
             return data

        logger.info(f"獲取到 {len(data)} 條交易日數據")
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
