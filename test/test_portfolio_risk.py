from datetime import date, timedelta

from backtest.risk import RiskPolicy, assess_order, build_correlation_groups, summarize_exposure


def account(**changes):
    result = {"equity": 10000., "cash": 5000., "positions": [], "peak_equity": 10000.,
              "macro_exposure": 1., "data_fresh": True, "correlation_available": True,
              "data_as_of": "2026-01-05T21:00:00Z"}
    result.update(changes)
    return result


def test_sector_limit_caps_and_records_why():
    positions = [{"ticker": "NVDA", "sector": "semiconductor", "group": "chips", "market_value": 2900.}]
    order = {"ticker": "AMD", "sector": "semiconductor", "group": "chips", "side": "buy", "quantity": 10, "price": 100., "average_volume": 20000, "decision_at": "2026-01-05T22:00:00Z", "data_available_at": "2026-01-05T21:00:00Z"}
    result = assess_order(account(positions=positions), order, RiskPolicy())
    assert result["allowed_quantity"] == 1
    assert "sector_limit" in result["reasons"]


def test_stale_macro_or_unknown_sector_blocks_buys_but_allows_reductions():
    buy = {"ticker": "AAPL", "sector": None, "group": "tech", "side": "buy", "quantity": 1, "price": 100., "average_volume": 20000, "decision_at": "2026-01-05T22:00:00Z", "data_available_at": "2026-01-05T21:00:00Z"}
    assert assess_order(account(data_fresh=False), buy, RiskPolicy())["allowed_quantity"] == 0
    sell = dict(buy, side="sell", quantity=1)
    held = [{"ticker": "AAPL", "sector": None, "group": "tech", "market_value": 100., "quantity": 1}]
    assert assess_order(account(data_fresh=False, positions=held), sell, RiskPolicy())["allowed_quantity"] == 1


def test_drawdown_and_correlation_missing_block_buys():
    buy = {"ticker": "AAPL", "sector": "tech", "group": "tech", "side": "buy", "quantity": 1, "price": 100., "average_volume": 20000, "decision_at": "2026-01-05T22:00:00Z", "data_available_at": "2026-01-05T21:00:00Z"}
    assert "drawdown_limit" in assess_order(account(peak_equity=13000), buy, RiskPolicy())["reasons"]
    assert "correlation_unavailable" in assess_order(account(correlation_available=False), buy, RiskPolicy())["reasons"]


def test_missing_average_volume_blocks_increase_and_future_data_fails_guard():
    buy = {"ticker": "AAPL", "sector": "tech", "group": "tech", "side": "buy", "quantity": 1, "price": 100., "decision_at": "2026-01-05T22:00:00Z", "data_available_at": "2026-01-05T21:00:00Z"}
    assert "volume_unavailable" in assess_order(account(), buy, RiskPolicy())["reasons"]
    buy["average_volume"] = 10000
    buy["data_available_at"] = "2026-01-06T15:00:00Z"
    buy["decision_at"] = "2026-01-05T22:00:00Z"
    assert "future_data" in assess_order(account(), buy, RiskPolicy())["reasons"]


def test_missing_or_future_account_snapshot_is_rejected():
    buy = {"ticker": "AAPL", "sector": "tech", "group": "tech", "side": "buy", "quantity": 1, "price": 100., "average_volume": 20000,
           "decision_at": "2026-01-05T22:00:00Z", "data_available_at": "2026-01-05T21:00:00Z"}
    assert "data_time_unavailable" in assess_order(account(data_as_of=None), buy, RiskPolicy())["reasons"]
    assert "future_data" in assess_order(account(data_as_of="2026-01-06T00:00:00Z"), buy, RiskPolicy())["reasons"]


def test_correlations_use_only_history_before_decision_session():
    dates = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(70)]
    sample = [(d, ((i % 7) - 3) / 100) for i, d in enumerate(dates)]
    linked = [(d, 2 * r) for d, r in sample]
    # A radically different return on the decision session cannot alter groups.
    data = {"A": sample[:60] + [(dates[60], 999.)], "B": linked[:60] + [(dates[60], -999.)]}
    result = build_correlation_groups(data, decision_session=dates[60])
    assert result["correlation_available"] is True
    assert result["groups"]["A"] == result["groups"]["B"]
    assert result["pairs"][0]["observations"] == 60
    sparse = build_correlation_groups({"A": sample[:39], "B": linked[:39]}, decision_session=dates[60])
    assert sparse["correlation_available"] is False


def test_post_trade_exposure_reports_breaches_without_changing_positions():
    positions = [{"ticker": "NVDA", "sector": "chip", "group": "chip", "market_value": 3500.},
                 {"ticker": "MSFT", "sector": "software", "group": "software", "market_value": 1500.}]
    result = summarize_exposure(account(positions=positions, macro_exposure=0.4), RiskPolicy())
    assert result["total_exposure"] == 0.5
    assert result["sector_exposures"]["chip"] == 0.35
    assert "total_exposure_limit" in result["breaches"]
    assert result["reduction_suggestions"]
