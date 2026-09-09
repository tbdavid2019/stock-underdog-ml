"""
strategies/timesfm.py - Google Research TimesFM Time Series Strategy.

Encapsulates zero-shot foundation model time series forecasting, vectorized batch inference,
and probabilistic risk/reward evaluation.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy
from models.timesfm_model import get_timesfm_model

logger = logging.getLogger("stock_app.strategies.timesfm")


@register_strategy("timesfm")
class TimesFMStrategy(BaseStrategy):
    """
    TimesFM 策略 (Google Research Time Series Foundation Model)
    使用 Google 百億參數時序大模型進行 Zero-Shot 預測，提供 5 日價格軌跡與 10%~90% 分位數風控區間。
    """

    name: str = "TimesFM"
    category: str = "ml"
    required_lookback: int = 30

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        horizon: int = 5,
        context_len: int = 128,
        min_potential_hit: float = 1.0,
        min_risk_reward_hit: float = 1.2,
    ):
        self.model_id = model_id
        self.horizon = horizon
        self.context_len = context_len
        self.min_potential_hit = min_potential_hit
        self.min_risk_reward_hit = min_risk_reward_hit
        self._model_wrapper = None

    def _get_model(self):
        if self._model_wrapper is None:
            self._model_wrapper = get_timesfm_model(
                model_id=self.model_id,
                context_len=self.context_len,
            )
        return self._model_wrapper

    def evaluate(self, context: StockContext) -> StrategyResult:
        """Evaluate a single stock context."""
        batch_results = self.evaluate_batch({context.ticker: context})
        if batch_results:
            return batch_results[0]

        curr_price = float(context.df["Close"].iloc[-1]) if not context.df.empty else 0.0
        return StrategyResult(
            ticker=context.ticker,
            strategy_name=self.name,
            is_hit=False,
            score=0.0,
            current_price=curr_price,
            signals={"reason": "評估失敗或無預測結果"}
        )

    def evaluate_batch(self, contexts: Dict[str, StockContext]) -> List[StrategyResult]:
        """
        Execute high-performance vectorized batch inference across all stocks.
        """
        results: List[StrategyResult] = []
        series_dict: Dict[str, np.ndarray] = {}
        valid_contexts: Dict[str, StockContext] = {}

        # 1. Pre-filter and collect time series
        for ticker, ctx in contexts.items():
            df = ctx.df
            if df.empty or len(df) < self.required_lookback or "Close" not in df.columns:
                curr_price = float(df["Close"].iloc[-1]) if not df.empty and "Close" in df.columns else 0.0
                results.append(
                    StrategyResult(
                        ticker=ticker,
                        strategy_name=self.name,
                        is_hit=False,
                        score=0.0,
                        current_price=curr_price,
                        signals={"reason": "數據不足"}
                    )
                )
                continue

            close_series = df["Close"].dropna().values
            series_dict[ticker] = close_series
            valid_contexts[ticker] = ctx

        if not series_dict:
            return results

        # 2. Batch forecast with TimesFM Foundation Model
        model = self._get_model()
        forecasts = model.forecast_batch(series_dict, horizon=self.horizon)

        # 3. Assemble StrategyResults with probabilistic risk/reward
        for ticker, ctx in valid_contexts.items():
            f = forecasts.get(ticker)
            curr_price = float(ctx.df["Close"].iloc[-1])

            if not f:
                results.append(
                    StrategyResult(
                        ticker=ticker,
                        strategy_name=self.name,
                        is_hit=False,
                        score=0.0,
                        current_price=curr_price,
                        signals={"reason": "TimesFM 推論未產出有效結果"}
                    )
                )
                continue

            potential = float(f["potential"])
            horizon_potential = float(f.get("horizon_potential", potential))
            risk_reward = float(f.get("risk_reward_ratio", 1.0))
            predicted_price = float(f["predicted_price"])

            # Hit criteria: positive next-day/horizon potential AND viable risk/reward
            is_hit = potential >= self.min_potential_hit and risk_reward >= self.min_risk_reward_hit

            # Multi-factor score (0 ~ 100) combining potential and risk/reward
            base_score = 50.0 + (potential * 8.0)
            rr_bonus = min(20.0, max(-10.0, (risk_reward - 1.0) * 10.0))
            score = max(0.0, min(100.0, base_score + rr_bonus))

            tags = []
            if potential >= 3.0:
                tags.append("TimesFM強")
            elif potential > 0:
                tags.append("TimesFM看漲")
            elif potential <= -2.0:
                tags.append("TimesFM看跌")

            if risk_reward >= 2.0:
                tags.append("高盈虧比")

            signals = {
                "predicted_price": predicted_price,
                "horizon_predicted_price": f.get("horizon_predicted_price"),
                "potential": potential,
                "horizon_potential": horizon_potential,
                "risk_reward_ratio": risk_reward,
                "downside_risk": f.get("downside_risk"),
                "upside_potential": f.get("upside_potential"),
                "quantiles": f.get("quantiles", {}),
                "trajectory": f.get("trajectory", []),
                "horizon": self.horizon
            }

            metrics = {
                "pe": ctx.fundamentals.get("pe"),
                "pb": ctx.fundamentals.get("pb"),
                "forward_pe": ctx.fundamentals.get("forward_pe"),
                "ev_ebitda": ctx.fundamentals.get("ev_ebitda"),
                "risk_reward_ratio": risk_reward,
                "downside_risk": f.get("downside_risk"),
                "upside_potential": f.get("upside_potential"),
            }

            results.append(
                StrategyResult(
                    ticker=ticker,
                    strategy_name=self.name,
                    is_hit=is_hit,
                    score=round(score, 2),
                    current_price=curr_price,
                    predicted_price=predicted_price,
                    potential=round(potential, 2),
                    signals=signals,
                    metrics=metrics,
                    tags=tags
                )
            )

        return results
