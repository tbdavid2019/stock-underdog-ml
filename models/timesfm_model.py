"""
models/timesfm_model.py - Google Research TimesFM (Time Series Foundation Model) Wrapper.

Provides zero-shot time series forecasting, batch inference across stocks,
and probabilistic quantile analysis (P10, P50, P90) for downside risk and upside reward.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger("stock_app.models.timesfm")

# Global singleton instance cache
_GLOBAL_TIMESFM_WRAPPER: Optional["TimesFMWrapper"] = None


class TimesFMWrapper:
    """
    Singleton wrapper for Google Research TimesFM model.
    Encapsulates model initialization, batch inference, and quantile parsing.
    """

    def __init__(
        self,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = 128,
        max_horizon: int = 128,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.context_len = max(32, (context_len // 32) * 32)  # Must be multiple of 32
        self.max_horizon = max(128, (max_horizon // 128) * 128)  # Must be multiple of 128
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self._is_initialized = False

    def load_model(self) -> bool:
        """Lazily loads the TimesFM pretrained weights and compiles the forecasting config."""
        if self._is_initialized and self.model is not None:
            return True

        try:
            import timesfm
            import torch
            from core.device import DeviceManager

            logger.info(f"🔮 正在載入 Google TimesFM 預訓練模型 ({self.model_id})...")
            
            # 1. Load model checkpoint from Hugging Face
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id,
                torch_compile=False  # Disable torch.compile to maximize compatibility
            )

            # 2. Compile with forecast configuration
            fc = timesfm.ForecastConfig(
                max_context=self.context_len,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                per_core_batch_size=self.batch_size,
                fix_quantile_crossing=True,
                infer_is_positive=True
            )
            self.model.compile(fc)
            self._is_initialized = True
            logger.info(f"✅ Google TimesFM 模型載入與編譯成功 (Context: {self.context_len}, Horizon: {self.max_horizon})")
            return True

        except ImportError as e:
            logger.warning(f"⚠️ timesfm 套件未正確安裝，TimesFM 策略將降級略過: {e}")
            self.model = None
            return False
        except Exception as e:
            logger.error(f"❌ 載入 TimesFM 模型失敗 ({self.model_id}): {e}", exc_info=True)
            self.model = None
            return False

    def forecast_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        horizon: int = 5,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute vectorized zero-shot forecasting on a batch of price series.

        Args:
            series_dict: Dict mapping ticker to 1D numpy array of historical close prices.
            horizon: Number of days forward to forecast (e.g. 5 days).

        Returns:
            Dict mapping ticker to forecast results including point predictions,
            trajectory, and P10/P50/P90 quantiles.
        """
        if not self._is_initialized or self.model is None:
            if not self.load_model():
                return {}

        valid_tickers = []
        inputs = []
        current_prices = {}

        for ticker, series in series_dict.items():
            clean_series = np.array(series, dtype=np.float64)
            clean_series = clean_series[~np.isnan(clean_series)]
            if len(clean_series) < 10:
                continue

            curr_price = float(clean_series[-1])
            if curr_price <= 0:
                continue

            valid_tickers.append(ticker)
            inputs.append(clean_series)
            current_prices[ticker] = curr_price

        if not inputs:
            return {}

        horizon = max(1, min(horizon, self.max_horizon))

        try:
            # Execute batch forecast through TimesFM
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=inputs
            )
            # point_forecast: (batch, horizon)
            # quantile_forecast: (batch, horizon, 10)

            results: Dict[str, Dict[str, Any]] = {}

            for idx, ticker in enumerate(valid_tickers):
                curr_price = current_prices[ticker]
                points = point_forecast[idx]  # shape (horizon,)
                quantiles = quantile_forecast[idx]  # shape (horizon, 10)

                # Day 1 prediction & Day N (horizon) prediction
                day1_pred = float(points[0])
                dayN_pred = float(points[-1])

                # Quantiles at horizon step (index 1: P10, index 5: P50, index 9: P90)
                p10_day1 = float(quantiles[0, 1])
                p50_day1 = float(quantiles[0, 5])
                p90_day1 = float(quantiles[0, 9])

                p10_horizon = float(quantiles[-1, 1])
                p50_horizon = float(quantiles[-1, 5])
                p90_horizon = float(quantiles[-1, 9])

                # Potential calculations (%)
                potential = ((day1_pred - curr_price) / curr_price) * 100.0
                horizon_potential = ((dayN_pred - curr_price) / curr_price) * 100.0

                # Risk / Reward ratio calculation based on P10 (downside) and P90 (upside)
                downside_risk = ((p10_day1 - curr_price) / curr_price) * 100.0
                upside_potential = ((p90_day1 - curr_price) / curr_price) * 100.0

                risk_span = max(abs(curr_price - p10_day1), 1e-4)
                reward_span = max(p90_day1 - curr_price, 0.0)
                risk_reward_ratio = float(reward_span / risk_span)

                results[ticker] = {
                    "current_price": curr_price,
                    "predicted_price": day1_pred,
                    "horizon_predicted_price": dayN_pred,
                    "potential": float(potential),
                    "horizon_potential": float(horizon_potential),
                    "trajectory": [float(p) for p in points],
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
                    "risk_reward_ratio": round(risk_reward_ratio, 2),
                    "horizon": horizon
                }

            return results

        except Exception as e:
            logger.error(f"❌ TimesFM 批次推論異常: {e}", exc_info=True)
            return {}


def get_timesfm_model(
    model_id: str = "google/timesfm-2.5-200m-pytorch",
    context_len: int = 128,
    max_horizon: int = 128,
    batch_size: int = 32,
) -> TimesFMWrapper:
    """Get or initialize the global TimesFM singleton wrapper."""
    global _GLOBAL_TIMESFM_WRAPPER
    if _GLOBAL_TIMESFM_WRAPPER is None:
        _GLOBAL_TIMESFM_WRAPPER = TimesFMWrapper(
            model_id=model_id,
            context_len=context_len,
            max_horizon=max_horizon,
            batch_size=batch_size,
        )
    return _GLOBAL_TIMESFM_WRAPPER
