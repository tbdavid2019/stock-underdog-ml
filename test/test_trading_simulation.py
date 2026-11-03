from datetime import date, datetime, timedelta, timezone

import pytest

from backtest.simulation import CostSchedule, execute_open, walk_forward_windows


def costs():
    return CostSchedule(market="US", instrument="equity", effective_from=date(2020, 1, 1),
        commission_rate=0.001, minimum_commission=1, tax_rate=0, slippage_bps=10,
        impact_bps=0, lot_size=1, version="test")


def test_signal_after_close_fills_only_next_open_and_charges_costs():
    signal = {"issued_at": "2026-01-05T22:00:00+00:00", "ticker": "AAPL", "side": "buy", "quantity": 10}
    same_day = {"date": "2026-01-05", "open": 100, "open_at": "2026-01-05T14:30:00+00:00"}
    later = {"date": "2026-01-06", "open": 101, "open_at": "2026-01-06T14:30:00+00:00"}
    hist = [{"date": (date(2025, 12, 1) + timedelta(days=i)).isoformat(), "open_volume": 20000, "volume": 30000} for i in range(20)]
    denied = execute_open(signal, same_day, hist, costs())
    assert denied["status"] == "rejected" and denied["reason"] == "signal_not_available_before_open"
    fill = execute_open(signal, later, hist, costs())
    assert fill["status"] == "filled"
    assert fill["fill_price"] == pytest.approx(101.101)
    assert fill["commission"] == pytest.approx(1.01101)
    assert fill["net_cash_change"] == pytest.approx(-1012.02101)


def test_opening_capacity_uses_only_available_history_and_partial_fills():
    signal = {"issued_at": "2026-01-05T22:00:00+00:00", "ticker": "AAPL", "side": "buy", "quantity": 20}
    bar = {"date": "2026-01-06", "open": 100, "volume": 10**9, "open_at": "2026-01-06T14:30:00+00:00"}
    hist = [{"date": (date(2025, 12, 1) + timedelta(days=i)).isoformat(), "open_volume": 1000, "volume": 5000} for i in range(20)]
    result = execute_open(signal, bar, hist, costs())
    assert result["status"] == "partially_filled"
    assert result["filled_quantity"] == 10 and result["cancelled_quantity"] == 10
    assert result["remaining_quantity"] == 0
    assert execute_open(signal, bar, hist[:9], costs())["reason"] == "capacity_unavailable"


def test_rejects_limit_open_and_missing_cost_schedule():
    signal = {"issued_at": "2026-01-05T22:00:00+00:00", "ticker": "AAPL", "side": "buy", "quantity": 10}
    bar = {"date": "2026-01-06", "open": 100, "open_at": "2026-01-06T14:30:00+00:00", "limit_locked": True}
    assert execute_open(signal, bar, [], costs())["reason"] == "limit_locked"
    with pytest.raises(ValueError, match="cost"):
        execute_open(signal, bar, [], None)


def test_walk_forward_purges_overlapping_labels():
    sessions = [(date(2024, 1, 1) + timedelta(days=i)).isoformat() for i in range(150)]
    folds = walk_forward_windows(sessions, calibration=120, test=20, horizon=5)
    assert len(folds) == 1
    assert folds[0]["train"][-1] == sessions[114]
    assert folds[0]["purged"] == sessions[115:120]
    assert folds[0]["test"] == sessions[120:140]
