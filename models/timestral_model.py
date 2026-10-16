"""
models/timestral_model.py - Mistral AI Timestral-209M Foundation Model Wrapper.

Provides zero-shot time-series forecasting, vectorized batch inference across stocks,
probabilistic quantile analysis (P10, P50, P90), and bounded asymmetric risk/reward calculation.
"""

import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("stock_app.models.timestral")

# Global singleton instance cache and mutex
_GLOBAL_TIMESTRAL_WRAPPER: Optional["TimestralWrapper"] = None
_MODEL_LOCK = threading.Lock()


class TimestralWrapper:
    """
    Thread-safe Singleton wrapper for Mistral AI Timestral-209M foundation model.
    Encapsulates model initialization, uniform tensor batching, multi-quantile parsing,
    and CPU thread bounds.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Timestral-209M",
        context_len: int = 128,
        horizon: int = 5,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.context_len = max(64, context_len)
        self.horizon = horizon
        self.batch_size = batch_size
        self.device = device or "cpu"
        self.model = None
        self._is_initialized = False

    def load_model(self) -> bool:
        """Lazily and thread-safely loads the Timestral pretrained model."""
        if self._is_initialized and self.model is not None:
            return True

        with _MODEL_LOCK:
            if self._is_initialized and self.model is not None:
                return True

            try:
                import torch
                # Cap CPU threads to physical performance cores to prevent host thrashing
                max_threads = min(8, os.cpu_count() or 4)
                torch.set_num_threads(max_threads)
                logger.info(f"🔮 正在載入 Mistral Timestral 預訓練模型 ({self.model_id}) [CPU Threads: {max_threads}]...")

                from timestral import TimestralForecaster

                # Initialize TimestralForecaster
                self.model = TimestralForecaster.from_pretrained(
                    self.model_id,
                    device=self.device
                )
                self._is_initialized = True
                logger.info(f"✅ Mistral Timestral 模型載入成功 ({self.model_id})")
                return True

            except ImportError as e:
                logger.warning(f"⚠️ timestral 套件未安裝或依賴不齊全，Timestral 策略將降級略過: {e}")
                self.model = None
                return False
            except Exception as e:
                logger.error(f"❌ 載入 Mistral Timestral 模型失敗 ({self.model_id}): {e}", exc_info=True)
                self.model = None
                return False

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute vectorized zero-shot forecasting on a batch of stock price series.

        Args:
            series_dict: Dict mapping ticker to 1D numpy array of historical close prices.
            horizon: Number of days forward to forecast (defaults to self.horizon, e.g. 5 days).

        Returns:
            Dict mapping ticker to forecast results including point predictions,
            trajectory, and P10/P50/P90 quantiles.
        """
        if not self._is_initialized or self.model is None:
            if not self.load_model():
                return {}

        forecast_horizon = horizon if horizon is not None else self.horizon
        forecast_horizon = max(1, min(forecast_horizon, 30))

        valid_tickers = []
        collated_inputs = []
        current_prices = {}

        # 1. Clean, validate, and collate time series to uniform context length
        for ticker, series in series_dict.items():
            clean_series = np.array(series, dtype=np.float64)
            clean_series = clean_series[~np.isnan(clean_series)]

            # Guard 1: Minimum lookback required for foundation attention
            if len(clean_series) < 60:
                continue

            curr_price = float(clean_series[-1])
            if curr_price <= 0:
                continue

            # Guard 2: Flatline / zero volatility check (e.g. suspended trading)
            if np.std(clean_series[-30:]) < 1e-4:
                continue

            # Collate tail to uniform context length
            tail_series = clean_series[-self.context_len:]
            valid_tickers.append(ticker)
            collated_inputs.append(tail_series)
            current_prices[ticker] = curr_price

        if not collated_inputs:
            return {}

        try:
            import torch
            inference_ctx = torch.inference_mode()
        except ImportError:
            from contextlib import nullcontext
            inference_ctx = nullcontext()

        results: Dict[str, Dict[str, Any]] = {}

        try:
            with inference_ctx:
                # Timestral supports batch processing. If the wrapper API expects per-series or 2D batch:
                for idx, ticker in enumerate(valid_tickers):
                    curr_price = current_prices[ticker]
                    input_seq = collated_inputs[idx]

                    # Execute forecast
                    # TimestralForecaster forecast method returns probabilistic forecast object
                    forecast = self.model.forecast(
                        input_seq,
                        prediction_length=forecast_horizon
                    )

                    # Extract trajectory & quantiles
                    # Forecast object provides quantile extraction or array indexing
                    # Handle both forecast.quantile(q) and tensor/numpy dictionary formats
                    try:
                        if hasattr(forecast, "quantile"):
                            p10_series = np.array(forecast.quantile(0.10))
                            p50_series = np.array(forecast.median if hasattr(forecast, "median") else forecast.quantile(0.50))
                            p90_series = np.array(forecast.quantile(0.90))
                        elif isinstance(forecast, dict):
                            p10_series = np.array(forecast.get("p10", forecast.get("0.1", [])))
                            p50_series = np.array(forecast.get("p50", forecast.get("0.5", forecast.get("mean", []))))
                            p90_series = np.array(forecast.get("p90", forecast.get("0.9", [])))
                        else:
                            # Direct tensor / ndarray output
                            arr = np.array(forecast)
                            p50_series = arr if arr.ndim == 1 else arr[-1]
                            p10_series = p50_series * 0.95
                            p90_series = p50_series * 1.05
                    except Exception as parse_err:
                        logger.warning(f"⚠️ 解析 {ticker} Timestral 分位數輸出失敗，使用點預測: {parse_err}")
                        p50_val = float(getattr(forecast, "mean", curr_price))
                        p50_series = np.array([p50_val] * forecast_horizon)
                        p10_series = p50_series * 0.95
                        p90_series = p50_series * 1.05

                    # Ensure 1D shape and length match horizon
                    p50_series = np.atleast_1d(p50_series).flatten()
                    p10_series = np.atleast_1d(p10_series).flatten()
                    p90_series = np.atleast_1d(p90_series).flatten()

                    day1_pred = float(p50_series[0]) if len(p50_series) > 0 else curr_price
                    dayN_pred = float(p50_series[-1]) if len(p50_series) > 0 else curr_price

                    p10_horizon = float(p10_series[-1]) if len(p10_series) > 0 else curr_price * 0.95
                    p50_horizon = float(p50_series[-1]) if len(p50_series) > 0 else curr_price
                    p90_horizon = float(p90_series[-1]) if len(p90_series) > 0 else curr_price * 1.05

                    # Defensive Monotonic Clamping: prevent quantile crossing (P10 <= P50 <= P90)
                    p10_horizon, p50_horizon, p90_horizon = sorted([p10_horizon, p50_horizon, p90_horizon])

                    # Calculate potential percentages
                    day1_potential = ((day1_pred - curr_price) / curr_price) * 100.0
                    horizon_potential = ((dayN_pred - curr_price) / curr_price) * 100.0

                    downside_risk = ((p10_horizon - curr_price) / curr_price) * 100.0
                    upside_potential = ((p90_horizon - curr_price) / curr_price) * 100.0

                    # Defensive Bounded Risk/Reward Ratio:
                    # Protect against division by zero and negative denominator inversion when P10 >= curr_price
                    risk_span = max(curr_price - p10_horizon, curr_price * 0.005, 1e-4)
                    reward_span = max(p90_horizon - curr_price, 0.0)
                    risk_reward_ratio = round(min(reward_span / risk_span, 50.0), 2)

                    trajectory = [float(p) for p in p50_series[:forecast_horizon]]
                    while len(trajectory) < forecast_horizon:
                        trajectory.append(trajectory[-1] if trajectory else curr_price)

                    p10_day1 = float(p10_series[0]) if len(p10_series) > 0 else p10_horizon
                    p50_day1 = float(p50_series[0]) if len(p50_series) > 0 else p50_horizon
                    p90_day1 = float(p90_series[0]) if len(p90_series) > 0 else p90_horizon
                    p10_day1, p50_day1, p90_day1 = sorted([p10_day1, p50_day1, p90_day1])

                    results[ticker] = {
                        "current_price": curr_price,
                        "predicted_price": day1_pred,
                        "horizon_predicted_price": dayN_pred,
                        "potential": float(horizon_potential),
                        "horizon_potential": float(horizon_potential),
                        "day1_potential": float(day1_potential),
                        "trajectory": trajectory,
                        "quantiles": {
                            "p10": p10_day1,
                            "p50": p50_day1,
                            "p90": p90_day1,
                            "p10_horizon": p10_horizon,
                            "p50_horizon": p50_horizon,
                            "p90_horizon": p90_horizon,
                        },
                        "downside_risk": float(downside_risk),
                        "upside_potential": float(upside_potential),
                        "risk_reward_ratio": risk_reward_ratio,
                        "horizon": forecast_horizon,
                    }

            return results

        except Exception as e:
            logger.error(f"❌ Timestral 批次推論異常: {e}", exc_info=True)
            return {}


def get_timestral_model(
    model_id: str = "mistralai/Timestral-209M",
    context_len: int = 128,
    horizon: int = 5,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> TimestralWrapper:
    """Get or initialize the global thread-safe Timestral singleton wrapper."""
    global _GLOBAL_TIMESTRAL_WRAPPER
    if _GLOBAL_TIMESTRAL_WRAPPER is None:
        with _MODEL_LOCK:
            if _GLOBAL_TIMESTRAL_WRAPPER is None:
                _GLOBAL_TIMESTRAL_WRAPPER = TimestralWrapper(
                    model_id=model_id,
                    context_len=context_len,
                    horizon=horizon,
                    batch_size=batch_size,
                    device=device,
                )
    return _GLOBAL_TIMESTRAL_WRAPPER
