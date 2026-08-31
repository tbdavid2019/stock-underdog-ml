"""
LSTM Next-Day Price Prediction Strategy.
Encapsulates LSTM model data preparation, training, and next-day price inference.
"""
import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy
from models.lstm import prepare_data, train_lstm_model, predict_next_day
from core.device import DeviceManager

logger = logging.getLogger("stock_app.strategies.lstm")


@register_strategy("lstm")
class LSTMStrategy(BaseStrategy):
    """
    LSTM 深度學習短線預測策略
    利用過去 90 日 OHLCV 序列預測下一交易日收盤價，計算預期漲跌幅潛力
    """

    name: str = "LSTM"
    category: str = "ml"
    required_lookback: int = 30

    def __init__(self, time_step: int = 90, min_potential_hit: float = 1.0):
        self.time_step = time_step
        self.min_potential_hit = min_potential_hit

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

        curr_price = float(df["Close"].iloc[-1])

        try:
            # 準備數據 (防止 data leakage)
            time_step = min(self.time_step, max(10, len(df) // 3))
            X, y, scaler = prepare_data(df, time_step=time_step)

            if len(X) < 5:
                return StrategyResult(
                    ticker=ticker,
                    strategy_name=self.name,
                    is_hit=False,
                    score=0.0,
                    current_price=curr_price,
                    signals={"reason": "有效訓練樣本不足"}
                )

            # 訓練模型
            model = train_lstm_model(X, y)

            # 預測次日價格
            predicted_price = predict_next_day(model, df, scaler, time_step=time_step)

            if predicted_price is None or predicted_price <= 0 or pd.isna(predicted_price):
                return StrategyResult(
                    ticker=ticker,
                    strategy_name=self.name,
                    is_hit=False,
                    score=0.0,
                    current_price=curr_price,
                    signals={"reason": "模型預測無效"}
                )

            # 計算潛力
            potential = ((predicted_price - curr_price) / curr_price) * 100.0
            is_hit = potential >= self.min_potential_hit

            # 計算量化分數 (0 ~ 100)
            score = max(0.0, min(100.0, 50.0 + potential * 10.0))

            tags = []
            if potential > 3.0:
                tags.append("LSTM強")
            elif potential > 0:
                tags.append("LSTM看漲")

            signals = {
                "predicted_price": float(predicted_price),
                "potential": float(potential),
                "time_step": time_step
            }

            metrics = {
                "pe": context.fundamentals.get("pe"),
                "pb": context.fundamentals.get("pb"),
                "forward_pe": context.fundamentals.get("forward_pe"),
                "ev_ebitda": context.fundamentals.get("ev_ebitda")
            }

            return StrategyResult(
                ticker=ticker,
                strategy_name=self.name,
                is_hit=is_hit,
                score=score,
                current_price=curr_price,
                predicted_price=float(predicted_price),
                potential=float(potential),
                signals=signals,
                metrics=metrics,
                tags=tags
            )

        except Exception as e:
            logger.debug(f"LSTM 預測 {ticker} 發生異常: {e}")
            return StrategyResult(
                ticker=ticker,
                strategy_name=self.name,
                is_hit=False,
                score=0.0,
                current_price=curr_price,
                signals={"error": str(e)}
            )
