"""Google TimesFM 3.0 adapter with native quantile and covariate support."""
import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

from core.config import config, usage_eligibility

logger = logging.getLogger("stock_app.models.timesfm")
_GLOBAL_TIMESFM_WRAPPERS: Dict[tuple, "TimesFMWrapper"] = {}
_MODEL_LOCK = threading.Lock()


class TimesFMWrapper:
    """Thin adapter for the official ``timesfm3`` PyTorch evaluator."""
    PACKAGE_REVISION = "e51928e27119cb17bebc005be2696b75e0a9e688"
    MODEL_VERSION = "timesfm-3.0"

    def __init__(self, model_id: str = config.pipeline.TIMESFM_MODEL_ID,
                 context_len: int = 128, max_horizon: int = 128,
                 batch_size: int = 8, device: Optional[str] = None,
                 checkpoint_revision: str = config.pipeline.TIMESFM_CHECKPOINT_REVISION):
        self.model_id = model_id
        self.checkpoint_revision = checkpoint_revision
        self.context_len = max(32, int(context_len))
        self.max_horizon = max(1, int(max_horizon))
        self.batch_size = max(1, int(batch_size))
        self.device = device or config.pipeline.DEFAULT_DEVICE
        self.model = None
        self._is_initialized = False
        self.last_status = "uninitialized"

    def load_model(self) -> bool:
        if self._is_initialized and self.model is not None:
            return True
        eligibility = usage_eligibility("timesfm")
        if not eligibility["eligible"]:
            self.last_status = "disabled"
            logger.warning("TimesFM3 disabled by deployment eligibility: %s", eligibility["reason"])
            return False
        if "timesfm-2.5" in self.model_id.lower():
            self.last_status = "legacy_model_retired"
            logger.error("TimesFM 2.5 checkpoints are historical-only; refusing model load")
            return False
        try:
            from timesfm3 import ModelConfig, TimesFM3Evaluator
            model_config = ModelConfig(
                checkpoint_path=self.model_id,
                revision=self.checkpoint_revision,
                per_core_batch_size=self.batch_size,
                device=None if self.device == "auto" else self.device,
            )
            self.model = TimesFM3Evaluator(model_config)
            self._is_initialized = True
            self.last_status = "ready"
            return True
        except ImportError as exc:
            self.last_status = "dependency_unavailable"
            logger.warning("TimesFM3 dependency unavailable: %s", exc)
        except Exception as exc:
            self.last_status = "load_failed"
            logger.error("TimesFM3 load failed (%s@%s): %s", self.model_id,
                         self.checkpoint_revision, exc)
        self.model = None
        return False

    def forecast_batch(self, series_dict: Dict[str, np.ndarray], horizon: int = 5,
                       past_covariates: Optional[Dict[str, np.ndarray]] = None
                       ) -> Dict[str, Dict[str, Any]]:
        if not self._is_initialized and not self.load_model():
            return {}
        tickers, contexts, covariates, current_prices = [], [], [], {}
        for ticker, series in series_dict.items():
            values = np.asarray(series, dtype=np.float32).reshape(-1)
            values = values[np.isfinite(values)]
            if values.size < 10 or values[-1] <= 0:
                continue
            cov = (past_covariates or {}).get(ticker)
            if cov is not None:
                cov = np.asarray(cov, dtype=np.float32)
                if cov.ndim != 2 or cov.shape[1] != len(values) or not np.isfinite(cov).all():
                    logger.warning("TimesFM3 covariates invalid for ticker %s", ticker)
                    continue
            tickers.append(ticker)
            contexts.append(values[-self.context_len:])
            covariates.append(cov[:, -self.context_len:] if cov is not None else None)
            current_prices[ticker] = float(values[-1])
        if not contexts:
            return {}

        horizon = max(1, min(int(horizon), self.max_horizon))
        try:
            forecasts = list(self.model.predict_batch(
                contexts=contexts, horizon=horizon,
                past_only_covariates=covariates, return_quantiles=True,
                use_symmetric_averaging=False, make_positive=False,
            ))
        except Exception as exc:
            self.last_status = "inference_failed"
            logger.error("TimesFM3 batch inference failed: %s", exc)
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        for ticker, output in zip(tickers, forecasts):
            try:
                point = np.asarray(output.forecast, dtype=float).reshape(-1)
                quantiles = np.asarray(output.quantiles, dtype=float)
                if quantiles.ndim == 3:
                    quantiles = quantiles[0]
                if quantiles.ndim != 2 or quantiles.shape[1] != 9:
                    raise ValueError("expected (horizon, 9) native quantiles")
                usable = min(horizon, len(point), len(quantiles))
                if usable < horizon or not np.isfinite(point[:usable]).all() or not np.isfinite(quantiles[:usable]).all():
                    raise ValueError("nonfinite or incomplete forecast")
                # Official 3.0 quantiles are [0.1, ..., 0.9], P10/P50/P90 at 0/4/8 with defensive monotonic sort
                raw_q1 = [float(quantiles[0, idx]) for idx in (0, 4, 8)]
                raw_qh = [float(quantiles[horizon - 1, idx]) for idx in (0, 4, 8)]
                q1 = sorted(raw_q1)
                qh = sorted(raw_qh)
                current, day1, dayn = current_prices[ticker], float(point[0]), float(point[horizon - 1])
                risk = max(abs(current - q1[0]), current * 0.005, 1e-8)
                reward = max(q1[1] - current, 0.0)
                results[ticker] = {
                    "current_price": current, "predicted_price": day1,
                    "horizon_predicted_price": dayn,
                    "potential": (day1 / current - 1) * 100,
                    "day1_potential": (day1 / current - 1) * 100,
                    "horizon_potential": (dayn / current - 1) * 100,
                    "trajectory": point[:horizon].tolist(),
                    "quantiles": {"p10": q1[0], "p50": q1[1], "p90": q1[2],
                                  "p10_horizon": qh[0], "p50_horizon": qh[1], "p90_horizon": qh[2]},
                    "quantile_source": "native", "quantile_levels": [round(q, 1) for q in np.arange(0.1, 1.0, 0.1)],
                    "quantile_trajectory": quantiles[:horizon].tolist(),
                    "downside_risk": (q1[0] / current - 1) * 100,
                    "upside_potential": (q1[2] / current - 1) * 100,
                    "risk_reward_ratio": round(reward / risk, 2), "horizon": horizon,
                    "model_version": self.MODEL_VERSION, "model_id": self.model_id,
                    "checkpoint_revision": self.checkpoint_revision,
                    "package_revision": self.PACKAGE_REVISION,
                }
            except Exception as exc:
                logger.warning("TimesFM3 output rejected for ticker %s: %s", ticker, exc)
        self.last_status = "ready" if results else "invalid_output"
        return results


def get_timesfm_model(model_id: str = config.pipeline.TIMESFM_MODEL_ID,
                      context_len: int = config.pipeline.TIMESFM_CONTEXT_LEN,
                      max_horizon: int = 128, batch_size: int = 8,
                      checkpoint_revision: str = config.pipeline.TIMESFM_CHECKPOINT_REVISION) -> TimesFMWrapper:
    key = (model_id, checkpoint_revision, context_len, max_horizon, batch_size)
    with _MODEL_LOCK:
        if key not in _GLOBAL_TIMESFM_WRAPPERS:
            _GLOBAL_TIMESFM_WRAPPERS[key] = TimesFMWrapper(
                model_id, context_len, max_horizon, batch_size,
                checkpoint_revision=checkpoint_revision)
        return _GLOBAL_TIMESFM_WRAPPERS[key]
