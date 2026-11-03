from datetime import datetime, timedelta, timezone

import pytest

from evaluators.forecast_performance import PerformanceQuery, rank_models, rank_stocks


def records(model="timesfm", ticker="AAPL", count=40, predicted=102., actual=103.):
    start = datetime(2026, 7, 6, 21, tzinfo=timezone.utc)
    result = []
    day = start
    for _ in range(count):
        while day.weekday() > 4:
            day += timedelta(days=1)
        result.append(dict(ticker=ticker, market="US", model_name=model,
            model_revision="r1", config_hash="c1", feature_profile="price_only",
            horizon=1, target_type="price", evidence_type="prospective",
            issued_at=(day - timedelta(days=1)).isoformat(), target_date=day.date().isoformat(),
            actual_available_at=day.isoformat(), verified_at=day.isoformat(),
            current_price=100., predicted_price=predicted, actual_price=actual,
            status="settled", forecast_id=f"{model}-{ticker}-{day.date()}"))
        day += timedelta(days=1)
    return result


def query(**kwargs):
    options = {"as_of": "2026-10-30T23:00:00+00:00", "window": 120}
    options.update(kwargs)
    return PerformanceQuery(**options)


def test_small_perfect_sample_is_observation_not_winner():
    output = rank_stocks(records(count=3) + records(ticker="MSFT"), query())
    assert [r["ticker"] for r in output["data"]] == ["MSFT"]
    assert output["observations"][0]["sample_count"] == 3
    assert output["observations"][0]["status"] == "insufficient_data"


def test_baseline_flat_and_error_metrics_are_distinct():
    output = rank_stocks(records(predicted=100., actual=100.), query())
    row = output["data"][0]
    assert row["direction_accuracy"] == 1
    assert row["mape"] == 0
    assert row["baseline_improvement"] is None
    assert row["oos_r2_zero_return"] is None


def test_future_and_legacy_and_duplicate_do_not_inflate_samples():
    data = records()
    future = dict(data[-1], forecast_id="future", verified_at="2027-01-01T00:00:00+00:00")
    legacy = dict(data[-1], forecast_id="legacy", evidence_type="legacy_unverified")
    output = rank_stocks(data + [data[0], future, legacy], query())
    assert output["data"][0]["sample_count"] == 40
    assert output["excluded_count"] >= 2


def test_pending_store_version_before_settled_version_does_not_hide_settlement():
    data = records(count=40)
    pending = dict(data[-1], status="pending", forecast_id="same-forecast", verified_at="2026-09-01T00:00:00+00:00")
    settled = dict(data[-1], forecast_id="same-forecast")
    output = rank_stocks(data[:-1] + [pending, settled], query())
    assert output["data"][0]["sample_count"] == 40


def test_as_of_uses_latest_noninvalidated_actual_correction():
    data = records(count=40)
    old = dict(data[-1], forecast_id="corrected", actual_price=103.,
               invalidated_at="2026-10-01T00:00:00+00:00")
    correction = dict(data[-1], forecast_id="corrected", actual_price=108.,
                      verified_at="2026-10-02T00:00:00+00:00",
                      actual_available_at="2026-10-02T00:00:00+00:00")
    output = rank_stocks(data[:-1] + [old, correction], query())
    assert output["data"][0]["mae"] == pytest.approx(1.125)


def test_model_comparison_uses_common_dates_and_equal_stock_weights():
    data = records(count=40) + records(model="moirai", count=35)
    data += records(model="timesfm", ticker="MSFT", count=40)
    output = rank_models(data, query())
    assert {r["sample_count"] for r in output["data"]} == {35}
    assert {r["stock_count"] for r in output["data"]} == {1}
    assert output["common_sample_count"] == 35
    assert {r["full_sample_count"] for r in output["data"]} == {80, 35}


def test_common_sample_ignores_model_specific_config_hash_but_keeps_feature_profile():
    baseline = records(count=40)
    candidate = [dict(r, model_name="moirai", config_hash="candidate-config",
                      forecast_id=r["forecast_id"].replace("timesfm", "moirai"))
                 for r in records(count=35)]
    output = rank_models(baseline + candidate, query())
    assert {r["common_sample_count"] for r in output["data"]} == {35}
    assert {r["sample_count"] for r in output["data"]} == {35}


def test_return_r2_uses_return_error_not_price_error():
    data = records(count=40, predicted=20., actual=21.)
    for row in data:
        row["current_price"] = 10.
    output = rank_stocks(data, query())
    assert output["data"][0]["oos_r2_zero_return"] == pytest.approx(1 - (0.1 ** 2 / 1.1 ** 2))


def test_return_r2_is_scale_invariant_across_stock_price_levels():
    low = records(ticker="LOW", count=40, predicted=110., actual=111.)
    high = records(ticker="HIGH", count=40, predicted=1100., actual=1110.)
    for row in high:
        row["current_price"] = 1000.
    output = rank_stocks(low + high, query())
    scores = {row["ticker"]: row["oos_r2_zero_return"] for row in output["data"]}
    assert scores["LOW"] == pytest.approx(scores["HIGH"])


def test_latest_duplicate_forecast_selected_before_scoring():
    data = records(count=40)
    old = dict(data[-1], forecast_id="older-forecast", issued_at="2026-09-01T12:00:00+00:00",
               predicted_price=150.)
    output = rank_stocks(data + [old], query())
    assert output["data"][0]["sample_count"] == 40
    assert output["data"][0]["mae"] == pytest.approx(1.)


def test_negative_return_r2_does_not_mean_direction_below_half():
    output = rank_stocks(records(predicted=140., actual=101.), query())
    row = output["data"][0]
    assert row["direction_accuracy"] == 1
    assert row["oos_r2_zero_return"] < 0
    assert row["baseline_improvement"] < 0


def test_groups_never_mix_feature_profiles_or_revisions():
    data = records()
    other = [dict(r, forecast_id=r["forecast_id"] + "-r2", model_revision="r2", feature_profile="full_features") for r in records()]
    output = rank_stocks(data + other, query())
    assert len(output["data"]) == 2
    assert all(r["sample_count"] == 40 for r in output["data"])


def test_invalid_query_and_naive_time_rejected():
    with pytest.raises(ValueError):
        query(window=7)
    with pytest.raises(ValueError):
        PerformanceQuery(as_of="2026-10-30", window=60)


def test_window_is_trading_sessions_not_number_of_predictions():
    # A stale old record remains out even when recent forecasts are missing.
    output = rank_stocks(records(count=40), PerformanceQuery(
        as_of="2026-10-30T23:00:00+00:00", window=20))
    assert output["data"] == []
    assert output["observations"] == []
