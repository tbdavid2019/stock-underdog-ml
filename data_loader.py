"""
Legacy compatibility wrapper for data_loader.
Delegates to data.fetcher and data.cache.
"""
from data.fetcher import (
    StockFetcher,
    add_base_indicators as add_indicators
)
from data.cache import CacheManager

get_tw0050_stocks = StockFetcher.get_tw0050_stocks
get_tw0051_stocks = StockFetcher.get_tw0051_stocks
get_sp500_stocks = StockFetcher.get_sp500_stocks
get_nasdaq_stocks = StockFetcher.get_nasdaq_stocks
get_sox_stocks = StockFetcher.get_sox_stocks
get_dji_stocks = StockFetcher.get_dji_stocks
get_index_name_map = StockFetcher.get_index_name_map
get_stocks_by_index = StockFetcher.get_stocks_by_index
get_stock_data = StockFetcher.get_stock_data
clear_stock_cache = CacheManager.clear_all_stock_data

__all__ = [
    "get_tw0050_stocks",
    "get_tw0051_stocks",
    "get_sp500_stocks",
    "get_nasdaq_stocks",
    "get_sox_stocks",
    "get_dji_stocks",
    "get_index_name_map",
    "get_stocks_by_index",
    "get_stock_data",
    "add_indicators",
    "clear_stock_cache"
]
