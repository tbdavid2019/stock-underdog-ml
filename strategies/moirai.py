"""MOIRAI-2 research-shadow strategy."""
from typing import Dict, List, Optional

import numpy as np

from core.config import config, usage_eligibility
from models.moirai_model import get_moirai_model
from strategies.base import BaseStrategy, StockContext, StrategyResult
from strategies.registry import register_strategy


@register_strategy("moirai")
class MoiraiStrategy(BaseStrategy):
    name = "MOIRAI-2"
    category = "ml"
    required_lookback = 30

    def __init__(self, model_id: Optional[str] = None, horizon: Optional[int] = None,
                 context_len: Optional[int] = None, min_potential_hit: Optional[float] = None,
                 min_risk_reward_hit: Optional[float] = None):
        self.model_id = model_id or config.pipeline.MOIRAI_MODEL_ID
        self.horizon = horizon if horizon is not None else config.pipeline.MOIRAI_HORIZON
        self.context_len = context_len if context_len is not None else config.pipeline.MOIRAI_CONTEXT_LEN
        self.min_potential_hit = min_potential_hit if min_potential_hit is not None else config.pipeline.MOIRAI_MIN_POTENTIAL
        self.min_risk_reward_hit = min_risk_reward_hit if min_risk_reward_hit is not None else config.pipeline.MOIRAI_MIN_RISK_REWARD
        self._model_wrapper = None

    def _get_model(self):
        if self._model_wrapper is None:
            self._model_wrapper = get_moirai_model(model_id=self.model_id, context_len=self.context_len)
        return self._model_wrapper

    @staticmethod
    def _metadata(context: StockContext, status: str, forecast=None) -> dict:
        forecast = forecast or {}
        return {
            "model": "moirai", "model_version": forecast.get("model_version", "moirai-2.0-R-small"),
            "model_id": forecast.get("model_id", config.pipeline.MOIRAI_MODEL_ID),
            "checkpoint_revision": forecast.get("checkpoint_revision", config.pipeline.MOIRAI_CHECKPOINT_REVISION),
            "package_revision": forecast.get("package_revision", "cfd46d4510ed8896f263116f32928eede05b0a75"),
            "role": config.pipeline.MODEL_ROLES.get("moirai", "active"),
            "profile": context.metadata.get("feature_profile", "market_features" if context.metadata.get("past_covariates") is not None else "price_only"),
            "feature_profile": context.metadata.get("feature_profile", "market_features" if context.metadata.get("past_covariates") is not None else "price_only"),
            "feature_status": context.metadata.get("feature_status", "price_only"),
            "feature_snapshot": context.metadata.get("feature_snapshot"),
            "quantile_source": forecast.get("quantile_source"), "status": status,
        }

    def evaluate(self, context: StockContext) -> StrategyResult:
        return self.evaluate_batch({context.ticker: context})[0]

    def evaluate_batch(self, contexts: Dict[str, StockContext]) -> List[StrategyResult]:
        results: List[StrategyResult] = []
        series_by_ticker, covariates, valid_contexts = {}, {}, {}
        eligibility = usage_eligibility("moirai")
        for ticker, ctx in contexts.items():
            df = ctx.df
            current = float(df["Close"].iloc[-1]) if not df.empty and "Close" in df else 0.0
            cov = ctx.metadata.get("past_covariates")
            profile = ctx.metadata.get("feature_profile", "market_features" if cov is not None else "price_only")
            if profile == "full_features" and (cov is None or ctx.metadata.get("feature_status") not in (None, "ready", "full_features")):
                results.append(StrategyResult(ticker, self.name, False, current_price=current,
                    signals={"reason": "features_unavailable"}, metadata=self._metadata(ctx, "features_unavailable")))
                continue
            if not eligibility["eligible"]:
                results.append(StrategyResult(ticker, self.name, False, current_price=current,
                    signals={"reason": eligibility["reason"]},
                    metadata={**self._metadata(ctx, "disabled"), "eligibility": eligibility}))
                continue
            if df.empty or "Close" not in df or len(df) < self.required_lookback:
                results.append(StrategyResult(ticker, self.name, False, current_price=current,
                    signals={"reason": "insufficient_history"}, metadata=self._metadata(ctx, "insufficient_history")))
                continue
            if cov is not None:
                cov = np.asarray(cov, dtype=np.float32)
                if cov.ndim != 2 or cov.shape[1] != len(df["Close"].dropna()) or not np.isfinite(cov).all():
                    results.append(StrategyResult(ticker, self.name, False, current_price=current,
                        signals={"reason": "features_unavailable"}, metadata=self._metadata(ctx, "features_unavailable")))
                    continue
                covariates[ticker] = cov
            series_by_ticker[ticker] = df["Close"].dropna().to_numpy(dtype=float)
            valid_contexts[ticker] = ctx
        if not series_by_ticker:
            return results

        wrapper = self._get_model()
        forecasts = wrapper.forecast_batch(series_by_ticker, horizon=self.horizon,
                                           past_covariates=covariates or None)
        for ticker, ctx in valid_contexts.items():
            forecast = forecasts.get(ticker)
            current = float(ctx.df["Close"].iloc[-1])
            if not forecast:
                status = getattr(wrapper, "last_status", "inference_failed")
                results.append(StrategyResult(ticker, self.name, False, current_price=current,
                    signals={"reason": "MOIRAI-2 inference produced no valid forecast"},
                    metadata=self._metadata(ctx, status)))
                continue
            potential = float(forecast["potential"])
            rr = float(forecast.get("risk_reward_ratio", 0.0))
            candidate_hit = potential >= self.min_potential_hit and rr >= self.min_risk_reward_hit
            role = config.pipeline.MODEL_ROLES.get("moirai", "active")
            candidate_score = round(max(0., min(100., 50. + potential * 8. + min(20., (rr - 1.) * 10.))), 2)
            quantiles = forecast.get("quantiles", {})
            model_tag = "MOIRAI看漲" if potential > 0 else "MOIRAI看跌"
            if role == "shadow":
                model_tag = "MOIRAI-2 shadow"
            results.append(StrategyResult(
                ticker=ticker, strategy_name=self.name, is_hit=candidate_hit and role != "shadow",
                score=candidate_score if role != "shadow" else 0.0,
                current_price=current, predicted_price=float(forecast["predicted_price"]),
                potential=round(potential, 2),
                signals={"predicted_price": forecast["predicted_price"],
                         "horizon_predicted_price": forecast.get("horizon_predicted_price"),
                         "potential": potential, "horizon_potential": forecast.get("horizon_potential"),
                         "quantiles": quantiles, "trajectory": forecast.get("trajectory", []),
                         "risk_reward_ratio": rr, "horizon": self.horizon,
                         "candidate_hit": candidate_hit, "candidate_score": candidate_score},
                metrics={"p10": quantiles.get("p10"), "p50": quantiles.get("p50"),
                         "p90": quantiles.get("p90"), "risk_reward_ratio": rr},
                metadata=self._metadata(ctx, "ready", forecast),
                tags=[model_tag]
            ))
        return results
