"""Salesforce MOIRAI-2 R-small adapter using native Uni2TS forecasts."""
import logging
import threading
from typing import Any, Dict, Optional

import numpy as np

from core.config import config, usage_eligibility

logger = logging.getLogger("stock_app.models.moirai")
_GLOBAL_MOIRAI_WRAPPERS: Dict[tuple, "MoiraiWrapper"] = {}
_MODEL_LOCK = threading.Lock()


class MoiraiWrapper:
    PACKAGE_REVISION = "cfd46d4510ed8896f263116f32928eede05b0a75"
    MODEL_VERSION = "moirai-2.0-R-small"

    def __init__(self, model_id: str = config.pipeline.MOIRAI_MODEL_ID,
                 checkpoint_revision: str = config.pipeline.MOIRAI_CHECKPOINT_REVISION,
                 context_len: int = config.pipeline.MOIRAI_CONTEXT_LEN,
                 batch_size: int = config.pipeline.MOIRAI_BATCH_SIZE,
                 device: Optional[str] = None):
        self.model_id = model_id
        self.checkpoint_revision = checkpoint_revision
        # MOIRAI-2's released checkpoint was demonstrated with a 1680-step
        # context. Keep the configured lookback for lineage, and left-pad short
        # market series so a 128-bar caller does not create an empty patch.
        self.context_len = max(16, int(context_len))
        self.model_context_len = max(1680, self.context_len)
        self.batch_size = max(1, int(batch_size))
        self.device = device or config.pipeline.DEFAULT_DEVICE
        self.module = None
        self.model = None
        self.quantile_levels = tuple(float(x) for x in np.arange(0.1, 1.0, 0.1))
        self._is_initialized = False
        self.last_status = "uninitialized"

    def load_model(self) -> bool:
        if self._is_initialized and self.module is not None:
            return True
        eligibility = usage_eligibility("moirai")
        if not eligibility["eligible"]:
            self.last_status = "disabled"
            logger.warning("MOIRAI-2 disabled by deployment eligibility: %s", eligibility["reason"])
            return False
        try:
            from uni2ts.model.moirai2 import Moirai2Module
            self.module = Moirai2Module.from_pretrained(
                self.model_id, revision=self.checkpoint_revision)
            levels = getattr(self.module, "quantile_levels", self.quantile_levels)
            self.quantile_levels = tuple(float(x) for x in levels)
            if len(self.quantile_levels) != 9 or not np.allclose(
                    self.quantile_levels, np.arange(0.1, 1.0, 0.1)):
                raise ValueError("checkpoint must expose native quantile levels 0.1 through 0.9")
            self._is_initialized = True
            self.last_status = "ready"
            return True
        except ImportError as exc:
            self.last_status = "dependency_unavailable"
            logger.warning("Uni2TS dependency unavailable: %s", exc)
        except Exception as exc:
            self.last_status = "load_failed"
            logger.error("MOIRAI-2 load failed (%s@%s): %s", self.model_id,
                         self.checkpoint_revision, exc)
        self.module = None
        return False

    def forecast_batch(self, series_dict: Dict[str, np.ndarray], horizon: int = 5,
                       past_covariates: Optional[Dict[str, np.ndarray]] = None
                       ) -> Dict[str, Dict[str, Any]]:
        if not self._is_initialized and not self.load_model():
            return {}

        # Uni2TS uses one covariate schema per predictor, so split mixed profiles explicitly before looping.
        if past_covariates:
            has_featured = any(past_covariates.get(t) is not None for t in series_dict)
            has_price_only = any(past_covariates.get(t) is None for t in series_dict)
            if has_featured and has_price_only:
                featured = {t: s for t, s in series_dict.items() if past_covariates.get(t) is not None}
                price_only = {t: s for t, s in series_dict.items() if past_covariates.get(t) is None}
                featured_res = self.forecast_batch(featured, horizon, {t: past_covariates[t] for t in featured}) if featured else {}
                price_res = self.forecast_batch(price_only, horizon) if price_only else {}
                return {**featured_res, **price_res}

        tickers, targets, covs, current_prices = [], [], [], {}
        channel_count = None
        covariate_projection = None
        for ticker, series in series_dict.items():
            values = np.asarray(series, dtype=np.float32).reshape(-1)
            if values.size < 10 or not np.isfinite(values).all() or values[-1] <= 0:
                continue
            values = values[-self.context_len:]
            original_len = values.size
            if values.size < self.model_context_len:
                pad = np.full(self.model_context_len - values.size, values[0], dtype=np.float32)
                values = np.concatenate([pad, values])
            cov = (past_covariates or {}).get(ticker)
            if cov is not None:
                cov = np.asarray(cov, dtype=np.float32)
                if cov.ndim != 2 or cov.shape[1] < original_len or not np.isfinite(cov).all():
                    logger.warning("MOIRAI-2 covariates invalid for ticker %s", ticker)
                    continue
                cov = cov[:, -original_len:].T  # Uni2TS expects [time, channel].
                # Uni2TS 2.0's native numpy padding path creates a (0, 1)
                # block even for multi-channel past features. Project the
                # supplied channels explicitly so the adapter remains usable
                # on the pinned release and records the loss of dimensions.
                if cov.shape[1] > 1:
                    cov = cov.mean(axis=1, keepdims=True)
                    covariate_projection = "mean_channels_for_uni2ts_v2_padding"
                if cov.shape[0] < self.model_context_len:
                    cov = np.vstack([np.repeat(cov[:1], self.model_context_len - cov.shape[0], axis=0), cov])
                if channel_count is None:
                    channel_count = cov.shape[1]
                if cov.shape[1] != channel_count:
                    logger.warning("MOIRAI-2 covariate channel mismatch for ticker %s", ticker)
                    continue
            elif channel_count not in (None, 0):
                logger.warning("MOIRAI-2 covariate channels missing for ticker %s", ticker)
                continue
            tickers.append(ticker)
            targets.append(values[:, None])
            covs.append(cov)
            current_prices[ticker] = float(values[-1])
        if not targets:
            return {}

        horizon = max(1, min(int(horizon), 30))
        # The MOIRAI-2 patch decoder needs at least one full prediction patch
        # when dynamic past features are present. Decode a padded horizon and
        # expose only the requested T+N steps to callers.
        model_horizon = max(horizon, 16)
        try:
            import torch
            from uni2ts.model.moirai2 import Moirai2Forecast
            signature = (model_horizon, channel_count or 0)
            if not hasattr(self, "_predictors"):
                self._predictors = {}
            if signature not in self._predictors:
                self._predictors[signature] = Moirai2Forecast(
                    module=self.module, prediction_length=model_horizon, context_length=self.model_context_len,
                    target_dim=1, feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=channel_count or 0,
                ).to("cpu" if self.device == "auto" else self.device)
            self.model = self._predictors[signature]
            prediction_chunks = []
            with torch.inference_mode():
                for start in range(0, len(targets), self.batch_size):
                    end = start + self.batch_size
                    chunk = self.model.predict(
                        past_target=targets[start:end],
                        past_feat_dynamic_real=covs[start:end] if channel_count else None,
                    )
                    chunk = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
                    prediction_chunks.append(chunk)
            prediction = np.concatenate(prediction_chunks, axis=0)
        except ImportError as exc:
            self.last_status = "dependency_unavailable"
            logger.warning("MOIRAI-2 inference dependencies unavailable: %s", exc)
            return {}
        except Exception as exc:
            self.last_status = "inference_failed"
            logger.exception("MOIRAI-2 batch inference failed: %s", exc)
            return {}

        results = {}
        for index, ticker in enumerate(tickers):
            try:
                qarr = np.asarray(prediction[index], dtype=float)
                if qarr.ndim == 3:
                    qarr = qarr[..., 0]
                if qarr.shape[0] != len(self.quantile_levels) or qarr.shape[1] < horizon:
                    raise ValueError("unexpected native quantile forecast shape")
                qidx = {round(q, 6): i for i, q in enumerate(self.quantile_levels)}
                q10 = qarr[qidx[0.1], :horizon]
                q50 = qarr[qidx[0.5], :horizon]
                q90 = qarr[qidx[0.9], :horizon]
                if not all(np.isfinite(q).all() for q in (q10, q50, q90)):
                    raise ValueError("nonfinite native quantile")
                current = current_prices[ticker]
                day1, dayn = float(q50[0]), float(q50[-1])
                p10, p50, p90 = sorted([float(q10[0]), float(q50[0]), float(q90[0])])
                p10_h, p50_h, p90_h = sorted([float(q10[-1]), float(q50[-1]), float(q90[-1])])
                risk = max(abs(current - p10), current * 0.005, 1e-8)
                results[ticker] = {
                    "current_price": current, "predicted_price": day1,
                    "horizon_predicted_price": dayn,
                    "potential": (day1 / current - 1) * 100,
                    "day1_potential": (day1 / current - 1) * 100,
                    "horizon_potential": (dayn / current - 1) * 100,
                    "trajectory": q50.tolist(),
                    "quantiles": {"p10": p10, "p50": p50, "p90": p90,
                                  "p10_horizon": p10_h, "p50_horizon": p50_h, "p90_horizon": p90_h},
                    "quantile_source": "native", "quantile_levels": list(self.quantile_levels),
                    "covariate_projection": covariate_projection,
                    "quantile_trajectory": qarr[:, :horizon].tolist(),
                    "risk_reward_ratio": round(max(float(q50[0]) - current, 0.0) / risk, 2),
                    "downside_risk": (p10 / current - 1) * 100,
                    "upside_potential": (p90 / current - 1) * 100,
                    "horizon": horizon, "model_version": self.MODEL_VERSION,
                    "model_id": self.model_id, "checkpoint_revision": self.checkpoint_revision,
                    "package_revision": self.PACKAGE_REVISION,
                }
            except Exception as exc:
                logger.warning("MOIRAI-2 output rejected for ticker %s: %s", ticker, exc)
        self.last_status = "ready" if results else "invalid_output"
        return results


def get_moirai_model(model_id: str = config.pipeline.MOIRAI_MODEL_ID,
                     checkpoint_revision: str = config.pipeline.MOIRAI_CHECKPOINT_REVISION,
                     context_len: int = config.pipeline.MOIRAI_CONTEXT_LEN,
                     batch_size: int = config.pipeline.MOIRAI_BATCH_SIZE,
                     device: Optional[str] = None) -> MoiraiWrapper:
    key = (model_id, checkpoint_revision, context_len, batch_size, device)
    with _MODEL_LOCK:
        if key not in _GLOBAL_MOIRAI_WRAPPERS:
            _GLOBAL_MOIRAI_WRAPPERS[key] = MoiraiWrapper(
                model_id, checkpoint_revision, context_len, batch_size, device)
        return _GLOBAL_MOIRAI_WRAPPERS[key]
