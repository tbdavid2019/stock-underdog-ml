from datetime import datetime, timedelta, timezone

from evaluators.model_health import evaluate_model_health


def health_rows(sessions=25, tickers=("A", "B"), hit=True):
    start = datetime(2026, 8, 3, 20, tzinfo=timezone.utc)
    rows = []
    for offset in range(sessions):
        day = start + timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        for ticker in tickers:
            rows.append({
                "ticker": ticker, "market": "US", "model_name": "model",
                "model_revision": "r1", "horizon": 1, "evidence_type": "prospective",
                "target_date": day.date().isoformat(), "issued_at": (day - timedelta(days=1)).isoformat(),
                "actual_available_at": day.isoformat(), "verified_at": day.isoformat(),
                "current_price": 100., "predicted_price": 101. if hit else 99.,
                "actual_price": 101., "status": "settled",
                "forecast_id": f"{ticker}-{day.date()}",
            })
    return rows


def test_health_session_weights_stocks_equally_and_deduplicates_reruns():
    rows = health_rows(sessions=30)
    # Repeated evaluation of one forecast must not add a second stock/session vote.
    rows.append(dict(rows[0]))
    result = evaluate_model_health(rows, as_of="2026-10-30T23:00:00+00:00",
        eligible_tickers_by_session={row["target_date"]: ["A", "B"] for row in rows})
    assert result["settled_session_count"] == 22
    assert result["sessions"][-1]["sample_count"] == 2
    assert result["sessions"][-1]["direction_accuracy"] == 1.


def test_health_coverage_below_threshold_does_not_advance_trigger_streak():
    rows = health_rows(sessions=5)
    dates = sorted({row["target_date"] for row in rows})
    # B is part of the eligible pool but the evaluator only has A for the final date.
    rows = [row for row in rows if not (row["target_date"] == dates[-1] and row["ticker"] == "B")]
    result = evaluate_model_health(rows, as_of="2026-10-30T23:00:00+00:00",
        eligible_tickers_by_session={day: ["A", "B"] for day in dates},
        coverage_threshold=0.8)
    assert result["sessions"][-1]["coverage"] == 0.5
    assert result["sessions"][-1]["countable"] is False


def test_health_as_of_excludes_actuals_not_yet_available():
    rows = health_rows(sessions=5)
    rows[-1]["actual_available_at"] = "2026-11-01T00:00:00+00:00"
    result = evaluate_model_health(rows, as_of="2026-10-30T23:00:00+00:00",
        eligible_tickers_by_session={row["target_date"]: ["A", "B"] for row in rows})
    assert result["settled_session_count"] == 5
    assert result["countable_session_count"] == 4


def test_observe_only_reports_candidate_without_changing_weight():
    result = evaluate_model_health(health_rows(sessions=30, hit=False),
        as_of="2026-10-30T23:00:00+00:00",
        eligible_tickers_by_session={row["target_date"]: ["A", "B"] for row in health_rows(sessions=30)},
        policy_mode="observe_only", current_weight=1.0)
    assert result["policy_mode"] == "observe_only"
    assert result["effective_weight"] == 1.0
    assert result["recommended_weight"] <= 1.0


def test_session_block_bootstrap_is_seeded_and_reports_suspension_evidence():
    # The 30-day fixture ends on September 1; evaluating in late October would
    # correctly leave the latest 20-session window without current evidence.
    kwargs = dict(as_of="2026-09-01T23:00:00+00:00",
        eligible_tickers_by_session={row["target_date"]: ["A", "B"] for row in health_rows(sessions=30)},
        policy_mode="enforce")
    first = evaluate_model_health(health_rows(sessions=30, hit=False), **kwargs)
    second = evaluate_model_health(health_rows(sessions=30, hit=False), **kwargs)
    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["block_length"] == 5
    assert first["bootstrap"]["resamples"] == 2000
    assert first["bootstrap"]["hit_rate_upper_95"] < 0.5
    assert first["state"] == "suspended"
    assert first["effective_weight"] == 0


def test_recovery_uses_only_new_sessions_after_suspension_and_has_hysteresis():
    rows = health_rows(sessions=50)
    denominator = {row["target_date"]: ["A", "B"] for row in rows}
    suspended_at = "2026-08-01T00:00:00+00:00"
    first = evaluate_model_health(rows, as_of="2026-09-21T23:00:00+00:00",
        eligible_tickers_by_session=denominator, prior_events=[{
            "from_state": "suspended", "effective_at": suspended_at,
            "effective_weight": 0.0, "event_key": "suspend-r1",
        }], policy_mode="enforce", current_weight=0.0)
    # Labor Day is not an NYSE session, even when a fixture contains a row.
    assert first["recovery_sample_count"] == 70
    assert first["state"] == "degraded"
    assert first["effective_weight"] == 0.5
    degraded_event = first["event"]
    same_snapshot = evaluate_model_health(rows, as_of="2026-09-21T23:00:00+00:00",
        eligible_tickers_by_session=denominator, prior_events=[{
            "from_state": "suspended", "effective_at": suspended_at,
            "effective_weight": 0.0, "event_key": "suspend-r1",
        }, degraded_event], policy_mode="enforce", current_weight=0.5)
    assert same_snapshot["state"] == "degraded"
    assert same_snapshot["recovery_session_count"] == 0

    from datetime import date
    added = []
    day = date(2026, 11, 2)
    added_sessions = 0
    while added_sessions < 20:
        if day.weekday() < 5 and day != date(2026, 11, 26):
            added_sessions += 1
            for ticker in ("A", "B"):
                added.append({
                    "ticker": ticker, "market": "US", "model_name": "model",
                    "model_revision": "r1", "horizon": 1, "evidence_type": "prospective",
                    "target_date": day.isoformat(), "issued_at": (day - timedelta(days=1)).isoformat() + "T12:00:00+00:00",
                    "actual_available_at": day.isoformat() + "T21:00:00+00:00",
                    "verified_at": day.isoformat() + "T21:00:00+00:00",
                    "current_price": 100., "predicted_price": 101., "actual_price": 101.,
                    "status": "settled", "forecast_id": f"{ticker}-{day.isoformat()}",
                })
        day += timedelta(days=1)
    all_rows = rows + added
    all_denominator = {row["target_date"]: ["A", "B"] for row in all_rows}
    second = evaluate_model_health(all_rows, as_of="2026-11-30T23:00:00+00:00",
        eligible_tickers_by_session=all_denominator, prior_events=[{
            "from_state": "suspended", "effective_at": suspended_at,
            "effective_weight": 0.0, "event_key": "suspend-r1",
        }, degraded_event], policy_mode="enforce", current_weight=0.5)
    assert second["state"] == "healthy"
    assert second["effective_weight"] == 1.0


def test_health_rejects_forecast_issued_after_target_session_close():
    rows = health_rows(sessions=5)
    last = sorted({row["target_date"] for row in rows})[-1]
    for row in rows:
        if row["target_date"] == last:
            row["issued_at"] = "2026-10-30T23:00:00+00:00"
    result = evaluate_model_health(rows, as_of="2026-10-30T23:30:00+00:00",
        eligible_tickers_by_session={row["target_date"]: ["A", "B"] for row in rows})
    assert result["sessions"][-1]["sample_count"] == 0


def test_future_health_events_do_not_change_historical_state():
    rows = health_rows(sessions=30)
    denominator = {row["target_date"]: ["A", "B"] for row in rows}
    kwargs = dict(as_of="2026-10-30T23:00:00+00:00", eligible_tickers_by_session=denominator,
                  policy_mode="enforce", current_weight=1.)
    baseline = evaluate_model_health(rows, **kwargs)
    future = evaluate_model_health(rows, prior_events=[{
        "to_state": "suspended", "effective_at": "2026-11-02T15:00:00+00:00",
        "event_key": "future", "effective_weight": 0.}], **kwargs)
    assert future["state"] == baseline["state"]
    assert future["effective_weight"] == baseline["effective_weight"]


def test_recent_suspension_supersedes_prior_recovery():
    rows = health_rows(sessions=30)
    denominator = {row["target_date"]: ["A", "B"] for row in rows}
    result = evaluate_model_health(rows, as_of="2026-10-30T23:00:00+00:00",
        eligible_tickers_by_session=denominator, policy_mode="enforce", current_weight=0.,
        prior_events=[
            {"to_state": "suspended", "effective_at": "2026-08-01T00:00:00+00:00", "event_key": "s1"},
            {"to_state": "degraded", "effective_at": "2026-09-15T00:00:00+00:00", "event_key": "d1"},
            {"to_state": "healthy", "effective_at": "2026-10-01T00:00:00+00:00", "event_key": "h1"},
            {"to_state": "suspended", "effective_at": "2026-10-29T00:00:00+00:00", "event_key": "s2"},
        ])
    assert result["state"] == "suspended"
    assert result["effective_weight"] == 0.
    assert result["recovery_session_count"] == 0


def test_missing_coverage_breaks_consecutive_degradation_streak():
    rows = health_rows(sessions=12, hit=False)
    dates = sorted({row["target_date"] for row in rows})
    missing = dates[-4]
    rows = [row for row in rows if not (row["target_date"] == missing and row["ticker"] == "B")]
    result = evaluate_model_health(rows, as_of="2026-08-14T23:00:00+00:00",
        eligible_tickers_by_session={day: ["A", "B"] for day in dates},
        policy_mode="enforce", min_samples=1, min_sessions=1)
    assert result["trigger_streak"] == 3
    assert result["state"] == "degraded"


def test_missing_day_cannot_pull_older_session_into_fixed_twenty_day_window():
    from datetime import date
    rows = []
    day = date(2026, 10, 1)
    while day <= date(2026, 10, 30):
        if day.weekday() < 5:
            for ticker in ("A", "B"):
                rows.append({"ticker": ticker, "market": "US", "model_name": "model",
                    "model_revision": "r1", "horizon": 1, "evidence_type": "prospective",
                    "target_date": day.isoformat(), "issued_at": (day - timedelta(days=1)).isoformat() + "T12:00:00+00:00",
                    "actual_available_at": day.isoformat() + "T21:00:00+00:00",
                    "verified_at": day.isoformat() + "T21:00:00+00:00",
                    "current_price": 100., "predicted_price": 99., "actual_price": 101.,
                    "status": "settled", "forecast_id": f"{ticker}-{day.isoformat()}"})
        day += timedelta(days=1)
    denominator = {row["target_date"]: ["A", "B"] for row in rows}
    rows = [row for row in rows if not (row["target_date"] == "2026-10-20" and row["ticker"] == "B")]
    result = evaluate_model_health(rows, as_of="2026-10-30T23:00:00+00:00",
        eligible_tickers_by_session=denominator, policy_mode="enforce")
    assert result["recent_window_complete"] is False
    assert result["bootstrap"]["hit_rate_upper_95"] is None
    assert result["state"] != "suspended"


def test_health_groups_market_model_revision_horizon_and_feature_profile_separately():
    base = [dict(row, feature_profile="market_features", config_hash="c1")
            for row in health_rows(sessions=5)]
    scopes = [
        base,
        [dict(row, market="JP", forecast_id="jp-" + row["forecast_id"]) for row in base],
        [dict(row, model_revision="r2", forecast_id="r2-" + row["forecast_id"]) for row in base],
        [dict(row, horizon=5, forecast_id="h5-" + row["forecast_id"]) for row in base],
        [dict(row, feature_profile="full_features", forecast_id="full-" + row["forecast_id"]) for row in base],
    ]
    rows = [row for scope in scopes for row in scope]
    denominator = {row["target_date"]: ["A", "B"] for row in base}
    scoped_event = {"market": "US", "model_name": "model", "model_revision": "r1",
        "horizon": 1, "config_hash": "c1", "feature_profile": "market_features",
        "evidence_type": "prospective", "target_type": "price", "to_state": "suspended",
        "effective_at": "2026-08-01T00:00:00+00:00", "event_key": "scoped-suspension"}
    result = evaluate_model_health(rows, as_of="2026-08-10T23:00:00+00:00",
        eligible_tickers_by_session=denominator, prior_events=[scoped_event],
        policy_mode="enforce", current_weight=1.)
    reports = result["reports"]
    assert len(reports) == 5
    assert len({(r["market"], r["model_revision"], r["horizon"], r["feature_profile"]) for r in reports}) == 5
    suspended = [r for r in reports if r["state"] == "suspended"]
    assert len(suspended) == 1
    assert suspended[0]["market"] == "US"
    assert suspended[0]["model_revision"] == "r1"
    assert suspended[0]["horizon"] == 1
    assert suspended[0]["feature_profile"] == "market_features"


def test_ambiguous_unscoped_prior_event_does_not_suspend_multiple_model_groups():
    base = [dict(row, feature_profile="market_features", config_hash="c1")
            for row in health_rows(sessions=5)]
    other = [dict(row, model_revision="r2", forecast_id="r2-" + row["forecast_id"]) for row in base]
    denominator = {row["target_date"]: ["A", "B"] for row in base}
    result = evaluate_model_health(base + other, as_of="2026-08-10T23:00:00+00:00",
        eligible_tickers_by_session=denominator,
        prior_events=[{"model_name": "model", "to_state": "suspended",
                       "effective_at": "2026-08-01T00:00:00+00:00", "event_key": "ambiguous"}],
        policy_mode="enforce", current_weight=1.)
    assert len(result["reports"]) == 2
    assert all(report["state"] != "suspended" for report in result["reports"])
