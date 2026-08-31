"""
XuanTie Heavy Sword Strategy (玄鐵重劍平均線策略).
Core concept: Major Trend Upwards (顺大势) + Minor Pullback Buy Point (逆小势).
"""
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy

logger = logging.getLogger("stock_app.strategies.xuantie")


@register_strategy("xuantie")
class XuanTieStrategy(BaseStrategy):
    """
    玄鐵重劍波段選股策略
    Filter 1 (大勢向上): MA60 過去 lookback 天內向上傾斜且當前價格在 MA60 之上
    Filter 2 (小勢回調): 當前價格回調接近 MA120 或 MA60 (±tolerance)
    """

    name: str = "玄鐵重劍"
    category: str = "technical"
    required_lookback: int = 70

    def __init__(self, lookback: int = 10, tolerance: float = 0.05):
        self.lookback = lookback
        self.tolerance = tolerance

    def _check_major_trend_up(self, df: pd.DataFrame) -> bool:
        if len(df) < self.lookback + 1 or "MA60" not in df.columns:
            return False
        
        recent = df.tail(self.lookback + 1)
        ma60_curr = recent["MA60"].iloc[-1]
        ma60_past = recent["MA60"].iloc[0]
        curr_price = recent["Close"].iloc[-1]

        if pd.isna(ma60_curr) or pd.isna(ma60_past) or pd.isna(curr_price):
            return False

        return float(ma60_curr) > float(ma60_past) and float(curr_price) > float(ma60_curr)

    def _check_minor_pullback(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if df.empty:
            return False, ""

        latest = df.iloc[-1]
        curr_price = float(latest["Close"]) if not pd.isna(latest["Close"]) else None
        if curr_price is None:
            return False, ""

        # 優先檢查 MA120
        if "MA120" in df.columns and not pd.isna(latest["MA120"]):
            ma120 = float(latest["MA120"])
            if ma120 > 0:
                diff = (curr_price - ma120) / ma120
                if -self.tolerance <= diff <= self.tolerance:
                    return True, f"MA120 ({diff * 100:+.1f}%)"

        # 檢查 MA60
        if "MA60" in df.columns and not pd.isna(latest["MA60"]):
            ma60 = float(latest["MA60"])
            if ma60 > 0:
                diff = (curr_price - ma60) / ma60
                if -self.tolerance <= diff <= self.tolerance:
                    return True, f"MA60 ({diff * 100:+.1f}%)"

        return False, ""

    def evaluate(self, context: StockContext) -> StrategyResult:
        df = context.df
        ticker = context.ticker

        if df.empty or len(df) < self.required_lookback:
            return StrategyResult(
                ticker=ticker,
                strategy_name=self.name,
                is_hit=False,
                score=0.0,
                current_price=float(df["Close"].iloc[-1]) if not df.empty else 0.0,
                signals={"reason": "數據不足"}
            )

        latest = df.iloc[-1]
        curr_price = float(latest["Close"])

        major_up = self._check_major_trend_up(df)
        pullback, pullback_type = self._check_minor_pullback(df)
        is_hit = major_up and pullback

        # Extract MA values safely
        def safe_val(col):
            return float(latest[col]) if col in df.columns and not pd.isna(latest[col]) else None

        metrics = {
            "ma5": safe_val("MA5"),
            "ma10": safe_val("MA10"),
            "ma60": safe_val("MA60"),
            "ma120": safe_val("MA120"),
            "ma250": safe_val("MA250"),
            "pe": context.fundamentals.get("pe"),
            "pb": context.fundamentals.get("pb"),
            "forward_pe": context.fundamentals.get("forward_pe"),
            "ev_ebitda": context.fundamentals.get("ev_ebitda")
        }

        signals = {
            "major_trend": major_up,
            "pullback": pullback,
            "pullback_type": pullback_type
        }

        tags = []
        if is_hit:
            tags.append("玄鐵買點")
            tags.append(pullback_type)

        score = 80.0 if is_hit else (40.0 if major_up else 0.0)

        return StrategyResult(
            ticker=ticker,
            strategy_name=self.name,
            is_hit=is_hit,
            score=score,
            current_price=curr_price,
            signals=signals,
            metrics=metrics,
            tags=tags
        )
