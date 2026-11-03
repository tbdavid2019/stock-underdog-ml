from datetime import datetime, timezone
from unittest.mock import MagicMock

from scripts.verify_forecasts import verify_due_forecasts


def test_scheduled_verification_uses_only_target_date():
    store = MagicMock()
    store.list_forecasts.return_value = [{"forecast_id": "f", "ticker": "AAPL", "market": "US",
        "target_date": "2026-01-05", "last_complete_bar": "2026-01-02", "horizon": 1,
        "issued_at": "2026-01-02T22:00:00Z", "price_basis": "adjusted_close"}]
    store.get_forecast.return_value = store.list_forecasts.return_value[0]
    store.list_actuals.return_value = []
    provider = MagicMock(return_value=[{"date": "2026-01-06", "close": 99}])
    results = verify_due_forecasts(store, provider, now=datetime(2026, 1, 7, tzinfo=timezone.utc))
    assert results[0]["status"] == "pending"
    assert results[0]["reason"] == "target_price_missing"
    store.settle_forecast.assert_not_called()


def test_scheduled_verification_skips_settled_and_records_provider_failure():
    store = MagicMock()
    store.list_forecasts.return_value = [
        {"forecast_id": "done", "ticker": "AAPL", "target_date": "2026-01-05"},
        {"forecast_id": "failed", "ticker": "MSFT", "market": "US", "target_date": "2026-01-05"}]
    store.list_actuals.side_effect = [[{"status": "settled"}], []]
    provider = MagicMock(side_effect=RuntimeError("secret in provider error"))
    results = verify_due_forecasts(store, provider, now=datetime(2026, 1, 7, tzinfo=timezone.utc))
    assert len(results) == 1
    assert results[0]["reason"] == "provider_unavailable"
    store.mark_pending.assert_called_once_with("failed", "provider_unavailable")
    assert "secret" not in str(results)
