"""
Strategies module for stock-underdog-ml.
Contains pluggable strategies, strategy registry, and standardized contracts.
"""
from strategies.base import StockContext, StrategyResult, BaseStrategy
import strategies.xuantie
import strategies.sector_rotation
import strategies.institutional

try:
    import strategies.timestral
except Exception as e:
    import logging
    logging.getLogger("stock_app.strategies").warning(f"⚠️ timestral strategy import skipped: {e}")

try:
    import strategies.timesfm
except Exception as e:
    import logging
    logging.getLogger("stock_app.strategies").warning(f"⚠️ timesfm strategy import skipped: {e}")

try:
    import strategies.lstm
except Exception as e:
    import logging
    logging.getLogger("stock_app.strategies").warning(f"⚠️ legacy lstm strategy import skipped: {e}")

__all__ = ["StockContext", "StrategyResult", "BaseStrategy"]
