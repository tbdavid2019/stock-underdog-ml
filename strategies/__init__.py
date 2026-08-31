"""
Strategies module for stock-underdog-ml.
Contains pluggable strategies, strategy registry, and standardized contracts.
"""
from strategies.base import StockContext, StrategyResult, BaseStrategy

__all__ = ["StockContext", "StrategyResult", "BaseStrategy"]
