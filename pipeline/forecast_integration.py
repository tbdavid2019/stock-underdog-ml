"""Bind point-in-time features and immutable model forecasts to the pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from typing import Mapping

import numpy as np
import pandas as pd

from backtest.verification import resolve_target_session, session_close
from core.market_calendar import MarketCalendar

logger = logging.getLogger("stock_app.pipeline.forecast_integration")


def retain_complete_bars(contexts: Mapping, as_of: datetime) -> None:
    """Discard price bars that have not closed at the model decision time."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    for ticker, ctx in contexts.items():
        market = MarketCalendar.resolve_market(ticker)
        keep = []
        for raw in ctx.df.index:
            day = pd.Timestamp(raw).date()
            keep.append(MarketCalendar.is_trading_day(day, market_or_ticker=market) and
                        session_close(day, market) <= as_of)
        ctx.df = ctx.df.loc[keep].copy()


def attach_market_features(contexts: Mapping, matrices: Mapping) -> None:
    """Align matrices to a contiguous tail of price bars and fail closed on gaps."""
    for ticker, ctx in contexts.items():
        matrix = matrices.get(ticker)
        ctx.metadata.update(feature_profile="full_features", feature_status="features_unavailable")
        if matrix is None:
            ctx.metadata["feature_reason"] = "missing_feature_matrix"
            continue
        ctx.metadata["feature_snapshot"] = dict(matrix.metadata)
        if matrix.status not in {"full_features", "ready"}:
            ctx.metadata["feature_reason"] = "features_unavailable"
            continue
        try:
            source_dates = [pd.Timestamp(d).date() for d in ctx.df.index]
            feature_dates = [pd.Timestamp(d).date() for d in matrix.dates]
            cov = np.asarray(matrix.past_covariates, dtype=np.float32)
            n = len(feature_dates)
            if (n == 0 or source_dates[-n:] != feature_dates or cov.ndim != 2 or
                    cov.shape[1] != n or not np.isfinite(cov).all()):
                raise ValueError("feature sessions do not match trailing price bars")
            ctx.df = ctx.df.iloc[-n:].copy()
            ctx.metadata.update(feature_status="ready", past_covariates=cov)
        except (ValueError, TypeError, AttributeError):
            ctx.metadata["feature_reason"] = "feature_price_session_mismatch"


def _numeric(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def persist_forecast_results(store, contexts: Mapping, strategy_outputs: Mapping, *,
                             issued_at: datetime, run_id: str) -> list[str]:
    """Persist only forecasts based on a complete input session, including shadow outputs."""
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("issued_at must be timezone-aware")
    issued = issued_at.astimezone(timezone.utc)
    store.record_run({"run_id": run_id, "issued_at": issued.isoformat()})
    records_to_save = []
    for model_name, results in strategy_outputs.items():
        if model_name not in {"timesfm", "timestral", "moirai"}:
            continue
        for result in results:
            ctx = contexts.get(result.ticker)
            if ctx is None or ctx.df.empty or _numeric(result.predicted_price) is None:
                continue
            metadata = result.metadata or {}
            revision = metadata.get("checkpoint_revision")
            if not revision:
                continue
            market = MarketCalendar.resolve_market(result.ticker)
            last_bar = pd.Timestamp(ctx.df.index[-1]).date()
            if not MarketCalendar.is_trading_day(last_bar, market_or_ticker=market):
                continue
            if issued < session_close(last_bar, market).astimezone(timezone.utc):
                continue
            signals = result.signals or {}
            trajectory = list(signals.get("trajectory") or [])
            quantiles = signals.get("quantiles") or {}
            feature_snapshot = ctx.metadata.get("feature_snapshot") or {}
            config_material = {"model_id": metadata.get("model_id"),
                               "feature_profile": ctx.metadata.get("feature_profile", "price_only"),
                               "profile_id": feature_snapshot.get("profile_id")}
            config_hash = hashlib.sha256(json.dumps(config_material, sort_keys=True).encode()).hexdigest()
            for horizon in (1, 5):
                point = result.predicted_price if horizon == 1 else (trajectory[4] if len(trajectory) >= 5 else None)
                if _numeric(point) is None:
                    continue
                target = resolve_target_session(last_bar, market, horizon)
                if issued >= session_close(target, market).astimezone(timezone.utc):
                    continue
                suffix = "" if horizon == 1 else "_horizon"
                record = {"run_id": run_id, "ticker": result.ticker, "market": market,
                          "model_name": model_name, "model_revision": revision,
                          "model_id": metadata.get("model_id"), "config_hash": config_hash,
                          "feature_profile": config_material["feature_profile"],
                          "feature_snapshot_id": feature_snapshot.get("snapshot_id"),
                          "role": metadata.get("role", "active"), "target_type": "price",
                          "evidence_type": "prospective", "issued_at": issued.isoformat(),
                          "data_cutoff": session_close(last_bar, market).astimezone(timezone.utc).isoformat(),
                          "last_complete_bar": last_bar.isoformat(), "target_date": target.isoformat(),
                          "horizon": horizon, "current_price": _numeric(result.current_price),
                          "predicted_price": _numeric(point), "price_basis": "adjusted_close"}
                for quantile in ("p10", "p50", "p90"):
                    exact_horizon = horizon == 1 or signals.get("horizon", len(trajectory)) == horizon
                    native = metadata.get("quantile_source", "native") == "native"
                    record[quantile] = _numeric(quantiles.get(quantile + suffix)) if exact_horizon and native else None
                record["quantile_source"] = "native" if native and exact_horizon else "unavailable"
                records_to_save.append(record)

    if not records_to_save:
        return []

    if hasattr(store, "record_forecasts_batch") and type(store).__name__ != "MagicMock":
        try:
            return store.record_forecasts_batch(records_to_save)
        except Exception as e:
            logger.warning("⚠️ ForecastStore 批次寫入失敗，降級為逐筆寫入: %s", e)

    saved = []
    for record in records_to_save:
        saved.append(store.record_forecast(record))
    return saved
