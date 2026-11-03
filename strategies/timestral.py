"""
strategies/timestral.py - Mistral AI Timestral-209M Time Series Strategy.

Encapsulates zero-shot foundation model time series forecasting, vectorized batch inference,
probabilistic quantile intervals (P10, P50, P90), and asymmetric risk/reward evaluation.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy
from models.timestral_model import get_timestral_model
from core.config import config

logger = logging.getLogger("stock_app.strategies.timestral")


@register_strategy("timestral")
class TimestralStrategy(BaseStrategy):
    """
    Timestral 策略 (Mistral AI Time Series Foundation Model 209M)
    使用 Mistral AI 時序基礎大模型進行 Zero-Shot 多分位數預測，提供 5 日價格軌跡與 P10/P50/P90 風報比分析。
    """

    name: str = "Timestral"
    category: str = "ml"
    required_lookback: int = 60

    def __init__(
        self,
        model_id: Optional[str] = None,
        horizon: Optional[int] = None,
        context_len: Optional[int] = None,
        min_potential_hit: Optional[float] = None,
        min_risk_reward_hit: Optional[float] = None,
    ):
        self.model_id = model_id or config.pipeline.TIMESTRAL_MODEL_ID
        self.horizon = horizon if horizon is not None else config.pipeline.TIMESTRAL_HORIZON
        self.context_len = context_len if context_len is not None else config.pipeline.TIMESTRAL_CONTEXT_LEN
        self.min_potential_hit = min_potential_hit if min_potential_hit is not None else getattr(config.pipeline, "TIMESTRAL_MIN_POTENTIAL", 0.5)
        self.min_risk_reward_hit = min_risk_reward_hit if min_risk_reward_hit is not None else getattr(config.pipeline, "TIMESTRAL_MIN_RISK_REWARD", 1.2)
        self._model_wrapper = None

    def _get_model(self):
        if self._model_wrapper is None:
            self._model_wrapper = get_timestral_model(
                model_id=self.model_id,
                context_len=self.context_len,
                horizon=self.horizon,
                checkpoint_revision=config.pipeline.TIMESTRAL_CHECKPOINT_REVISION,
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
                        signals={"reason": f"數據不足 (需要至少 {self.required_lookback} 根 K 棒)"}
                    )
                )
                continue

            close_series = df["Close"].dropna().values
            series_dict[ticker] = close_series
            valid_contexts[ticker] = ctx

        if not series_dict:
            return results

        # 2. Batch forecast with Timestral Foundation Model
        model = self._get_model()
        covariates = {}
        for ticker, ctx in valid_contexts.items():
            cov = ctx.metadata.get("past_covariates")
            if ctx.metadata.get("feature_profile") == "full_features" and (
                    cov is None or ctx.metadata.get("feature_status") not in (None, "ready", "full_features")):
                results.append(StrategyResult(ticker, self.name, False,
                    current_price=float(ctx.df["Close"].iloc[-1]),
                    signals={"reason": "features_unavailable"},
                    metadata={"model": "timestral", "status": "features_unavailable",
                              "feature_profile": "full_features", "feature_status": ctx.metadata.get("feature_status"),
                              "feature_snapshot": ctx.metadata.get("feature_snapshot")}))
                series_dict.pop(ticker, None)
                continue
            if cov is not None:
                cov = np.asarray(cov, dtype=np.float32)
                if cov.ndim != 2 or cov.shape[1] != len(series_dict[ticker]) or not np.isfinite(cov).all():
                    results.append(StrategyResult(ticker, self.name, False,
                        current_price=float(ctx.df["Close"].iloc[-1]), signals={"reason": "features_unavailable"},
                        metadata={"model": "timestral", "status": "features_unavailable"}))
                    series_dict.pop(ticker, None)
                    continue
                covariates[ticker] = cov

        if not series_dict:
            return results
        valid_contexts = {ticker: ctx for ticker, ctx in valid_contexts.items() if ticker in series_dict}
        forecasts = model.forecast_batch(series_dict, horizon=self.horizon,
                                         past_covariates=covariates or None)

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
                    signals={"reason": "Timestral 推論未產出有效結果"},
                    metadata={"model": "timestral", "model_version": "timestral-209m",
                                    "role": "active", "status": getattr(model, "last_status", "inference_failed"),
                                    "profile": ctx.metadata.get("feature_profile", "market_features" if ctx.metadata.get("past_covariates") is not None else "price_only"),
                                    "feature_profile": ctx.metadata.get("feature_profile", "market_features" if ctx.metadata.get("past_covariates") is not None else "price_only"),
                                    "feature_status": ctx.metadata.get("feature_status", "ready" if ctx.metadata.get("past_covariates") is not None else "price_only"),
                                    "feature_snapshot": ctx.metadata.get("feature_snapshot")}
                    )
                )
                continue

            potential = float(f["potential"])
            horizon_potential = float(f.get("horizon_potential", potential))
            risk_reward = float(f.get("risk_reward_ratio", 1.0))
            predicted_price = float(f["predicted_price"])
            quantiles = f.get("quantiles", {})

            # Hit criteria: positive day 1 (T+1) potential AND viable asymmetric risk/reward
            is_hit = potential >= self.min_potential_hit and risk_reward >= self.min_risk_reward_hit

            # Multi-factor score (0 ~ 100) combining potential and risk/reward
            base_score = 50.0 + (potential * 8.0)
            rr_bonus = min(20.0, max(-10.0, (risk_reward - 1.0) * 10.0))
            score = max(0.0, min(100.0, base_score + rr_bonus))

            tags = []
            if potential >= 3.0:
                tags.append("Timestral強")
            elif potential > 0:
                tags.append("Timestral看漲")
            elif potential <= -2.0:
                tags.append("Timestral看跌")

            if risk_reward >= 1.5:
                tags.append("高風報比")

            signals = {
                "predicted_price": predicted_price,
                "horizon_predicted_price": f.get("horizon_predicted_price"),
                "potential": potential,
                "horizon_potential": horizon_potential,
                "day1_potential": f.get("day1_potential"),
                "risk_reward_ratio": risk_reward,
                "downside_risk": f.get("downside_risk"),
                "upside_potential": f.get("upside_potential"),
                "quantiles": quantiles,
                "trajectory": f.get("trajectory", []),
                "horizon": self.horizon,
            }

            metrics = {
                "pe": ctx.fundamentals.get("pe"),
                "pb": ctx.fundamentals.get("pb"),
                "forward_pe": ctx.fundamentals.get("forward_pe"),
                "ev_ebitda": ctx.fundamentals.get("ev_ebitda"),
                "risk_reward_ratio": risk_reward,
                "downside_risk": f.get("downside_risk"),
                "upside_potential": f.get("upside_potential"),
                "p10": quantiles.get("p10", quantiles.get("p10_horizon")),
                "p50": quantiles.get("p50", quantiles.get("p50_horizon")),
                "p90": quantiles.get("p90", quantiles.get("p90_horizon")),
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
                    metadata={"model": "timestral", "model_version": f.get("model_version", "timestral-209m"),
                              "model_id": f.get("model_id", self.model_id),
                              "checkpoint_revision": f.get("checkpoint_revision"),
                              "package_revision": f.get("package_revision"), "role": "active",
                              "profile": ctx.metadata.get("feature_profile", "market_features" if ctx.metadata.get("past_covariates") is not None else "price_only"),
                              "feature_profile": ctx.metadata.get("feature_profile", "market_features" if ctx.metadata.get("past_covariates") is not None else "price_only"),
                              "feature_status": ctx.metadata.get("feature_status", "ready" if ctx.metadata.get("past_covariates") is not None else "price_only"),
                              "feature_snapshot": ctx.metadata.get("feature_snapshot"),
                              "quantile_source": f.get("quantile_source"), "status": "ready"},
                    tags=tags
                )
            )

        return results
