"""
models/plugins.py - Standard Hot-Pluggable Adapters for Foundation Time-Series Models.

Adapts TimesFM, Timestral, and MOIRAI-2 to the ModelPlugin Protocol.
Ensures zero-code-change extensibility, market constraint enforcement, and dynamic registration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from models.protocol import BaseModelPlugin, ModelPlugin, model_registry

logger = logging.getLogger("stock_app.models.plugins")


def _format_prediction_output(
    raw_res: Dict[str, Any],
    current_price: float,
    market_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    統一格式化推論輸出，並施加跨市場防呆截斷（如台股 ±10% 漲跌幅限制）。
    """
    predicted_price = float(raw_res.get("predicted_price", current_price))
    quantiles = raw_res.get("quantiles", {})
    p10 = float(quantiles.get("p10", predicted_price * 0.95))
    p50 = float(quantiles.get("p50", predicted_price))
    p90 = float(quantiles.get("p90", predicted_price * 1.05))
    raw_trajectory = raw_res.get("trajectory", [predicted_price])
    trajectory = [float(p) for p in raw_trajectory]
    risk_reward = float(raw_res.get("risk_reward_ratio", 1.0))

    # 跨市場約束：台股 ±10% 剛性截斷
    if market_params and market_params.get("max_daily_change_pct") is not None:
        max_pct = float(market_params["max_daily_change_pct"])
        lower_cap = current_price * (1.0 - max_pct / 100.0)
        upper_cap = current_price * (1.0 + max_pct / 100.0)
        predicted_price = max(lower_cap, min(upper_cap, predicted_price))
        p50 = max(lower_cap, min(upper_cap, p50))
        p10 = max(lower_cap, min(upper_cap, p10))
        p90 = max(lower_cap, min(upper_cap, p90))
        trajectory = [max(lower_cap, min(upper_cap, p)) for p in trajectory]

    potential = ((predicted_price / current_price) - 1.0) * 100.0 if current_price > 0 else 0.0

    return {
        "predicted_price": round(predicted_price, 4),
        "potential": round(potential, 2),
        "quantiles": {
            "p10": round(p10, 4),
            "p50": round(p50, 4),
            "p90": round(p90, 4),
        },
        "trajectory": [round(p, 4) for p in trajectory],
        "risk_reward_ratio": round(risk_reward, 2),
    }


def _extract_past_covariates(
    covariates_dict: Optional[Dict[str, Any]]
) -> Optional[Dict[str, np.ndarray]]:
    """
    將通用字典格式或矩陣格式的側參數轉換為 (Channels, Time) 即 (K, T) 陣列格式供模型適配器消費。
    """
    if not covariates_dict:
        return None
    res = {}
    for ticker, cov_map in covariates_dict.items():
        if isinstance(cov_map, np.ndarray):
            # 自適應轉置：若為 (Time, Channels) 且 Time > Channels，轉置為 (Channels, Time)
            if cov_map.ndim == 2 and cov_map.shape[0] > cov_map.shape[1]:
                res[ticker] = cov_map.T
            else:
                res[ticker] = cov_map
        elif isinstance(cov_map, dict) and cov_map:
            arrs = [np.asarray(v, dtype=np.float32).reshape(-1) for v in cov_map.values() if len(v) > 0]
            if arrs:
                min_len = min(len(a) for a in arrs)
                aligned = [a[-min_len:] for a in arrs]
                res[ticker] = np.vstack(aligned)
    return res


class TimesFMPlugin(BaseModelPlugin):
    """Google TimesFM 3.0 標準插件適配器"""

    model_id: str = "google/timesfm-3.0-pytorch"
    plugin_id: str = "timesfm"
    tag_name: str = "TimesFM"
    display_name: str = "TimesFM (Google)"
    version: str = "3.0"
    author: str = "Google Research"
    supported_markets: List[str] = ["tw", "us", "jp"]
    default_role: str = "active"
    supports_covariates: bool = True
    required_covariates: List[str] = ["vix", "usdtwd"]

    def __init__(self, wrapper: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wrapper = wrapper

    def initialize(self, config: Dict[str, Any]) -> bool:
        if "wrapper" in config:
            self.wrapper = config["wrapper"]
            return True
        try:
            from models.timesfm_model import get_timesfm_model
            model_id = config.get("model_id", self.model_id)
            context_len = config.get("context_len", 128)
            checkpoint_revision = config.get("checkpoint_revision", "43046b85ec22d584a13f8098c2ed39c889e129c2")
            self.wrapper = get_timesfm_model(
                model_id=model_id,
                context_len=context_len,
                checkpoint_revision=checkpoint_revision,
            )
            return self.wrapper.load_model()
        except Exception as exc:
            logger.warning(f"TimesFMPlugin initialization failed: {exc}")
            return False

    def health_check(self) -> Dict[str, Any]:
        status = "uninitialized"
        device = "unknown"
        if self.wrapper is not None:
            last_st = getattr(self.wrapper, "last_status", "uninitialized")
            status = "healthy" if last_st == "ready" else last_st
            device = getattr(self.wrapper, "device", "unknown")
        return {
            "status": status,
            "device": device,
            "model_id": self.model_id,
            "version": self.version,
            "supports_covariates": self.supports_covariates,
        }

    def predict_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        covariates_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        horizon: int = 5,
        market: str = "tw",
        market_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if self.wrapper is None:
            if not self.initialize({}):
                return {}

        past_covs = _extract_past_covariates(covariates_dict)
        raw_outputs = self.wrapper.forecast_batch(
            series_dict=series_dict,
            horizon=horizon,
            past_covariates=past_covs,
        )

        results = {}
        for ticker, raw_res in raw_outputs.items():
            curr_price = float(raw_res.get("current_price", series_dict[ticker][-1] if len(series_dict[ticker]) else 1.0))
            results[ticker] = _format_prediction_output(raw_res, curr_price, market_params=market_params)
        return results

    def teardown(self) -> None:
        self.wrapper = None


class TimestralPlugin(BaseModelPlugin):
    """Mistral AI Timestral-209M 標準插件適配器"""

    model_id: str = "mistralai/Timestral-209M"
    plugin_id: str = "timestral"
    tag_name: str = "Timestral"
    display_name: str = "Timestral (Mistral)"
    version: str = "209M"
    author: str = "Mistral AI"
    supported_markets: List[str] = ["tw", "us", "jp"]
    default_role: str = "active"
    supports_covariates: bool = False
    required_covariates: List[str] = []

    def __init__(self, wrapper: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wrapper = wrapper

    def initialize(self, config: Dict[str, Any]) -> bool:
        if "wrapper" in config:
            self.wrapper = config["wrapper"]
            return True
        try:
            from models.timestral_model import get_timestral_model
            model_id = config.get("model_id", self.model_id)
            context_len = config.get("context_len", 128)
            checkpoint_revision = config.get("checkpoint_revision", "9874d252ab0aae16f7f7c999f98f98d6ad7d8365")
            self.wrapper = get_timestral_model(
                model_id=model_id,
                context_len=context_len,
                checkpoint_revision=checkpoint_revision,
            )
            return self.wrapper.load_model()
        except Exception as exc:
            logger.warning(f"TimestralPlugin initialization failed: {exc}")
            return False

    def health_check(self) -> Dict[str, Any]:
        status = "uninitialized"
        device = "unknown"
        if self.wrapper is not None:
            last_st = getattr(self.wrapper, "last_status", "uninitialized")
            status = "healthy" if last_st == "ready" else last_st
            device = getattr(self.wrapper, "device", "unknown")
        return {
            "status": status,
            "device": device,
            "model_id": self.model_id,
            "version": self.version,
            "supports_covariates": self.supports_covariates,
        }

    def predict_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        covariates_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        horizon: int = 5,
        market: str = "tw",
        market_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if self.wrapper is None:
            if not self.initialize({}):
                return {}

        raw_outputs = self.wrapper.forecast_batch(
            series_dict=series_dict,
            horizon=horizon,
        )

        results = {}
        for ticker, raw_res in raw_outputs.items():
            curr_price = float(raw_res.get("current_price", series_dict[ticker][-1] if len(series_dict[ticker]) else 1.0))
            results[ticker] = _format_prediction_output(raw_res, curr_price, market_params=market_params)
        return results

    def teardown(self) -> None:
        self.wrapper = None


class Moirai2Plugin(BaseModelPlugin):
    """Salesforce MOIRAI-2 R-small 標準插件適配器"""

    model_id: str = "Salesforce/moirai-2.0-R-small"
    plugin_id: str = "moirai"
    tag_name: str = "MOIRAI-2"
    display_name: str = "MOIRAI-2 (Salesforce)"
    version: str = "2.0-R-small"
    author: str = "Salesforce"
    supported_markets: List[str] = ["tw", "us", "jp"]
    default_role: str = "active"
    supports_covariates: bool = True
    required_covariates: List[str] = ["vix", "usdtwd", "sox", "tnx"]

    def __init__(self, wrapper: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.wrapper = wrapper

    def initialize(self, config: Dict[str, Any]) -> bool:
        if "wrapper" in config:
            self.wrapper = config["wrapper"]
            return True
        try:
            from models.moirai_model import get_moirai_model
            model_id = config.get("model_id", self.model_id)
            context_len = config.get("context_len", 128)
            checkpoint_revision = config.get("checkpoint_revision", "cfd46d4510ed8896f263116f32928eede05b0a75")
            self.wrapper = get_moirai_model(
                model_id=model_id,
                context_len=context_len,
                checkpoint_revision=checkpoint_revision,
            )
            return self.wrapper.load_model()
        except Exception as exc:
            logger.warning(f"Moirai2Plugin initialization failed: {exc}")
            return False

    def health_check(self) -> Dict[str, Any]:
        status = "uninitialized"
        device = "unknown"
        if self.wrapper is not None:
            last_st = getattr(self.wrapper, "last_status", "uninitialized")
            status = "healthy" if last_st == "ready" else last_st
            device = getattr(self.wrapper, "device", "unknown")
        return {
            "status": status,
            "device": device,
            "model_id": self.model_id,
            "version": self.version,
            "supports_covariates": self.supports_covariates,
        }

    def predict_batch(
        self,
        series_dict: Dict[str, np.ndarray],
        covariates_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        horizon: int = 5,
        market: str = "tw",
        market_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if self.wrapper is None:
            if not self.initialize({}):
                return {}

        past_covs = _extract_past_covariates(covariates_dict)
        raw_outputs = self.wrapper.forecast_batch(
            series_dict=series_dict,
            horizon=horizon,
            past_covariates=past_covs,
        )

        results = {}
        for ticker, raw_res in raw_outputs.items():
            curr_price = float(raw_res.get("current_price", series_dict[ticker][-1] if len(series_dict[ticker]) else 1.0))
            results[ticker] = _format_prediction_output(raw_res, curr_price, market_params=market_params)
        return results

    def teardown(self) -> None:
        self.wrapper = None


# Auto-register standard foundation model plugins into ModelPluginRegistry
model_registry.register(TimesFMPlugin)
model_registry.register(TimestralPlugin)
model_registry.register(Moirai2Plugin)
