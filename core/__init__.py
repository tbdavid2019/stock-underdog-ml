"""
Core module for stock-underdog-ml.
Contains device management, centralized configuration, and core utilities.
"""
from core.device import DeviceManager
from core.market_calendar import MarketCalendar

__all__ = ["DeviceManager", "MarketCalendar"]
