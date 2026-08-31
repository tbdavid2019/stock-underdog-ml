"""
Legacy compatibility wrapper for xuantie_strategy.
Delegates to strategies.xuantie.
"""
import pandas as pd
from strategies.base import StockContext
from strategies.xuantie import XuanTieStrategy
from data.fetcher import StockFetcher, add_base_indicators


def check_xuantie_signal(data: pd.DataFrame, ticker: str = "", lookback: int = 10, tolerance: float = 0.05) -> dict:
    strategy = XuanTieStrategy(lookback=lookback, tolerance=tolerance)
    df_with_indicators = add_base_indicators(data)
    ctx = StockContext(ticker=ticker, df=df_with_indicators)
    res = strategy.evaluate(ctx)
    return {
        'ticker': ticker,
        'signal': res.is_hit,
        'major_trend': res.signals.get('major_trend', False),
        'pullback': res.signals.get('pullback', False),
        'pullback_type': res.signals.get('pullback_type', ''),
        'current_price': res.current_price,
        'ma5': res.metrics.get('ma5'),
        'ma10': res.metrics.get('ma10'),
        'ma60': res.metrics.get('ma60'),
        'ma120': res.metrics.get('ma120'),
        'ma250': res.metrics.get('ma250'),
    }


def filter_stocks_by_xuantie(stock_list: list, period: str = "1y", lookback: int = 10, tolerance: float = 0.05) -> pd.DataFrame:
    strategy = XuanTieStrategy(lookback=lookback, tolerance=tolerance)
    results = []
    for ticker in stock_list:
        df = StockFetcher.get_stock_data(ticker, period)
        if not df.empty:
            res = check_xuantie_signal(df, ticker, lookback=lookback, tolerance=tolerance)
            if res['signal']:
                results.append(res)
    return pd.DataFrame(results) if results else pd.DataFrame()


__all__ = ["check_xuantie_signal", "filter_stocks_by_xuantie", "XuanTieStrategy"]
