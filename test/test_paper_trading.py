import pytest

from backtest.paper import PaperLedger

def execute(ledger, order, fill, **kwargs):
    order = {"decision_session": "2026-01-05", **order}
    risk = {"ticker": order["ticker"], "side": order["side"], "allowed_quantity": order["quantity"],
            "policy_version": "test-v1", "reasons": []}
    return ledger.record_execution(order, fill, risk_decision=risk, **kwargs)


def test_same_order_retry_is_idempotent_and_partial_order_reconciles(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.duckdb")
    ledger.create_account("a", market="US", currency="USD", cash=10000.)
    order = {"order_id": "o1", "account_id": "a", "signal_id": "s1", "ticker": "AAPL", "side": "buy", "quantity": 20}
    fill = {"fill_id": "f1", "filled_quantity": 10, "cancelled_quantity": 10, "fill_price": 100., "commission": 1., "tax": 0.}
    execute(ledger, order, fill)
    execute(ledger, order, fill)
    state = ledger.get_account("a")
    assert state["cash"] == pytest.approx(8999.)
    assert state["positions"]["AAPL"]["quantity"] == 10
    assert ledger.get_orders("a")[0]["status"] == "partially_filled"
    assert ledger.get_orders("a")[0]["remaining_quantity"] == 0


def test_failed_transaction_rolls_back_order_and_cash(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.duckdb")
    ledger.create_account("a", market="US", currency="USD", cash=10000.)
    order = {"order_id": "o1", "account_id": "a", "signal_id": "s1", "ticker": "AAPL", "side": "buy", "quantity": 10}
    fill = {"fill_id": "f1", "filled_quantity": 10, "cancelled_quantity": 0, "fill_price": 100., "commission": 1., "tax": 0.}
    with pytest.raises(RuntimeError, match="injected"):
        execute(ledger, order, fill, fail_after_order=True)
    assert ledger.get_orders("a") == []
    assert ledger.get_account("a")["cash"] == pytest.approx(10000.)


def test_split_and_dividend_are_idempotent_and_preserve_cash(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.duckdb")
    ledger.create_account("a", market="US", currency="USD", cash=10000.)
    execute(ledger, {"order_id": "o", "account_id": "a", "signal_id": "s", "ticker": "AAPL", "side": "buy", "quantity": 10},
        {"fill_id": "f", "filled_quantity": 10, "cancelled_quantity": 0, "fill_price": 100., "commission": 0., "tax": 0.})
    ledger.apply_corporate_action("a", "AAPL", {"event_id": "split", "type": "split", "ratio": 2, "source": "fixture"})
    ledger.apply_corporate_action("a", "AAPL", {"event_id": "split", "type": "split", "ratio": 2, "source": "fixture"})
    ledger.apply_corporate_action("a", "AAPL", {"event_id": "div", "type": "dividend", "amount_per_share": 1, "source": "fixture"})
    state = ledger.get_account("a")
    assert state["positions"]["AAPL"]["quantity"] == 20
    assert state["cash"] == pytest.approx(9020.)


def test_daily_equity_reconciles_cash_and_marks_missing_price_stale(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.duckdb")
    ledger.create_account("a", market="US", currency="USD", cash=10000.)
    execute(ledger, {"order_id": "o", "account_id": "a", "signal_id": "s", "ticker": "AAPL", "side": "buy", "quantity": 10},
        {"fill_id": "f", "filled_quantity": 10, "cancelled_quantity": 0, "fill_price": 100., "commission": 1., "tax": 0.})
    first = ledger.mark_to_market("a", "2026-01-06", {"AAPL": 110.}, as_of="2026-01-06T22:00:00Z")
    assert ledger.mark_to_market("a", "2026-01-06", {"AAPL": 110.}, as_of="2026-01-06T22:00:00Z") == first
    with pytest.raises(ValueError, match="immutable"):
        ledger.mark_to_market("a", "2026-01-06", {"AAPL": 112.}, as_of="2026-01-06T22:00:00Z")
    assert first["equity"] == pytest.approx(10099.)
    assert first["unrealized_pnl"] == pytest.approx(99.)
    assert first["stale"] is False
    second = ledger.mark_to_market("a", "2026-01-07", {}, as_of="2026-01-07T22:00:00Z")
    assert second["equity"] == pytest.approx(10099.)
    assert second["stale"] is True
    assert second["missing_prices"] == ["AAPL"]
    assert second["new_exposure_allowed"] is False
    assert len(ledger.get_equity("a")) == 2


def test_pretrade_check_and_execution_share_one_transaction(tmp_path):
    ledger = PaperLedger(tmp_path / "paper.duckdb")
    ledger.create_account("a", market="US", currency="USD", cash=10000.)
    base = {"account_id": "a", "ticker": "AAPL", "side": "buy", "quantity": 10,
        "decision_session": "2026-01-05", "decision_at": "2026-01-05T22:00:00Z",
        "data_available_at": "2026-01-05T21:00:00Z", "sector": "tech", "group": "tech",
        "price": 100., "average_volume": 20000}
    fill = {"fill_id": "f1", "filled_quantity": 10, "cancelled_quantity": 0,
            "fill_price": 100., "commission": 0., "tax": 0.}
    risk_context = {"market": "US", "currency": "USD", "data_as_of": "2026-01-05T21:00:00Z",
        "data_fresh": True, "correlation_available": True, "macro_exposure": 1.,
        "peak_equity": 10000., "marks": {"AAPL": 100.},
        "sector_by_ticker": {"AAPL": "tech"}, "group_by_ticker": {"AAPL": "tech"}}
    ledger.record_execution({**base, "order_id": "o1", "signal_id": "s1"}, fill, risk_context=risk_context)
    with pytest.raises(ValueError, match="risk"):
        ledger.record_execution({**base, "order_id": "o2", "signal_id": "s2"},
            {**fill, "fill_id": "f2"}, risk_context=risk_context)
    assert ledger.get_account("a")["positions"]["AAPL"]["quantity"] == 10
    assert len(ledger.get_orders("a")) == 1
