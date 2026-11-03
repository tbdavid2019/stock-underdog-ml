from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from pipeline.forecast_integration import attach_market_features, persist_forecast_results, retain_complete_bars
from strategies.base import StockContext, StrategyResult


def test_attach_features_aligns_prices_and_rejects_missing():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    ctx = StockContext("2330.TW", pd.DataFrame({"Close": [10, 11, 12, 13, 14]}, index=dates))
    matrix = SimpleNamespace(dates=list(dates[-3:].date), past_covariates=np.ones((2, 3)),
        status="full_features", metadata={"snapshot_id": "snap", "profile_id": "TW-v1"})
    attach_market_features({ctx.ticker: ctx}, {ctx.ticker: matrix})
    assert len(ctx.df) == 3
    assert ctx.metadata["past_covariates"].shape == (2, 3)
    assert ctx.metadata["feature_profile"] == "full_features"
    assert ctx.metadata["feature_snapshot"]["snapshot_id"] == "snap"

    missing = StockContext("AAPL", pd.DataFrame({"Close": [1, 2]}, index=dates[:2]))
    attach_market_features({missing.ticker: missing}, {})
    assert missing.metadata["feature_status"] == "features_unavailable"
    assert missing.metadata["feature_profile"] == "full_features"


def test_missing_matrix_session_cannot_be_mislabeled_full_features():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    ctx = StockContext("2330.TW", pd.DataFrame({"Close": [10, 11, 12, 13, 14]}, index=dates))
    matrix = SimpleNamespace(dates=[dates[0].date(), dates[-1].date()],
        past_covariates=np.ones((2, 2)), status="full_features", metadata={})
    attach_market_features({ctx.ticker: ctx}, {ctx.ticker: matrix})
    assert ctx.metadata["feature_status"] == "features_unavailable"


def test_persist_shadow_and_multiple_horizons_without_future_backfill():
    store = MagicMock()
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    ctx = StockContext("AAPL", pd.DataFrame({"Close": [99, 100]}, index=dates),
        metadata={"feature_profile": "price_only"})
    result = StrategyResult("AAPL", "MOIRAI-2", False, current_price=100,
        predicted_price=101, potential=1,
        signals={"trajectory": [101, 102, 103, 104, 105],
                 "quantiles": {"p10": 98, "p50": 101, "p90": 104,
                               "p10_horizon": 95, "p50_horizon": 105, "p90_horizon": 110}},
        metadata={"checkpoint_revision": "rev", "role": "shadow", "model_id": "model"})
    now = datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc)
    with patch("pipeline.forecast_integration.MarketCalendar.is_trading_day", return_value=True):
        saved = persist_forecast_results(store, {"AAPL": ctx}, {"moirai": [result]},
            issued_at=now, run_id="run")
    assert len(saved) == 2
    saved_rows = [call.args[0] for call in store.record_forecast.call_args_list]
    assert [r["horizon"] for r in saved_rows] == [1, 5]
    assert [r["predicted_price"] for r in saved_rows] == [101, 105]
    assert all(r["role"] == "shadow" for r in saved_rows)
    assert all(r["last_complete_bar"] == "2026-01-05" for r in saved_rows)


def test_config_hash_is_stable_across_feature_snapshots():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    ctx = StockContext("AAPL", pd.DataFrame({"Close": [99, 100]}, index=dates),
        metadata={"feature_profile": "full_features", "feature_snapshot": {"snapshot_id": "first"}})
    result = StrategyResult("AAPL", "TimesFM", True, current_price=100, predicted_price=101,
        metadata={"checkpoint_revision": "rev", "model_id": "model"})
    now = datetime(2026, 1, 5, 22, 0, tzinfo=timezone.utc)
    store = MagicMock()
    persist_forecast_results(store, {"AAPL": ctx}, {"timesfm": [result]}, issued_at=now, run_id="one")
    ctx.metadata["feature_snapshot"]["snapshot_id"] = "second"
    persist_forecast_results(store, {"AAPL": ctx}, {"timesfm": [result]}, issued_at=now, run_id="two")
    first, second = [c.args[0] for c in store.record_forecast.call_args_list]
    assert first["config_hash"] == second["config_hash"]
    assert first["feature_snapshot_id"] != second["feature_snapshot_id"]


def test_five_day_forecast_never_uses_ten_day_quantiles():
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    ctx = StockContext("AAPL", pd.DataFrame({"Close": [99, 100]}, index=dates))
    result = StrategyResult("AAPL", "TimesFM", True, current_price=100, predicted_price=101,
        signals={"trajectory": list(range(101, 111)), "horizon": 10,
                 "quantiles": {"p10": 98, "p50": 101, "p90": 104,
                               "p10_horizon": 90, "p50_horizon": 110, "p90_horizon": 120}},
        metadata={"checkpoint_revision": "rev", "model_id": "model", "quantile_source": "native"})
    store = MagicMock()
    persist_forecast_results(store, {"AAPL": ctx}, {"timesfm": [result]},
        issued_at=datetime(2026, 1, 5, 22, tzinfo=timezone.utc), run_id="run")
    day5 = [c.args[0] for c in store.record_forecast.call_args_list if c.args[0]["horizon"] == 5][0]
    assert day5["predicted_price"] == 105
    assert day5["p10"] is None and day5["p50"] is None and day5["p90"] is None


def test_later_timestamp_does_not_accept_current_incomplete_bar():
    store = MagicMock()
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    ctx = StockContext("AAPL", pd.DataFrame({"Close": [100, 102]}, index=dates))
    result = StrategyResult("AAPL", "TimesFM", True, current_price=102, predicted_price=103,
        metadata={"checkpoint_revision": "rev"})
    now = datetime(2026, 1, 6, 15, tzinfo=timezone.utc)
    saved = persist_forecast_results(store, {"AAPL": ctx}, {"timesfm": [result]},
                                     issued_at=now, run_id="run")
    assert saved == []
    store.record_forecast.assert_not_called()


def test_retain_complete_bars_uses_exchange_close_before_forecast():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    ctx = StockContext("AAPL", pd.DataFrame({"Close": [100, 102]}, index=dates))
    retain_complete_bars({"AAPL": ctx}, datetime(2026, 1, 6, 15, tzinfo=timezone.utc))
    assert list(ctx.df["Close"]) == [100]
