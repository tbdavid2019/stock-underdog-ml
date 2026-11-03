"""Session-based, as-of model quality and weight policy summaries."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import random
from typing import Any, Iterable, Mapping

from evaluators.forecast_performance import _direction, _normalize, _time
from backtest.verification import session_close
from core.market_calendar import MarketCalendar
from zoneinfo import ZoneInfo

_MARKET_TZ = {"US": "America/New_York", "TW": "Asia/Taipei", "JP": "Asia/Tokyo"}


def _as_of(value: str) -> datetime:
    parsed = _time(value)
    if parsed is None:
        raise ValueError("as_of must be an ISO-8601 timestamp with timezone")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _block_bootstrap_upper(session_hits: list[float], *, seed: int, block_length: int = 5,
                           resamples: int = 2000) -> float | None:
    """One-sided 95th percentile from circular moving blocks of market sessions."""
    n = len(session_hits)
    if n < 20:
        return None
    rng = random.Random(seed)
    draws = []
    blocks_needed = math.ceil(n / block_length)
    for _ in range(resamples):
        sample = []
        for _ in range(blocks_needed):
            start = rng.randrange(n)
            sample.extend(session_hits[(start + offset) % n] for offset in range(block_length))
        draws.append(sum(sample[:n]) / n)
    draws.sort()
    return draws[math.ceil(0.95 * resamples) - 1]


def _closed_market_sessions(market: str, cutoff: datetime, count: int = 80) -> list[str]:
    """Return the actual preceding market sessions, including days with no data."""
    day = cutoff.astimezone(ZoneInfo(_MARKET_TZ[market])).date()
    sessions = []
    while len(sessions) < count:
        if (MarketCalendar.is_trading_day(day, market_or_ticker=market) and
                session_close(day, market) <= cutoff):
            sessions.append(day.isoformat())
        day -= timedelta(days=1)
    return list(reversed(sessions))


def evaluate_model_health(
    records: Iterable[Mapping[str, Any]], *, as_of: str,
    eligible_tickers_by_session: Mapping[str, Iterable[str]],
    coverage_threshold: float = 0.8, policy_mode: str = "observe_only",
    current_weight: float = 1.0, min_samples: int = 30, min_sessions: int = 20,
    flat_threshold: float = 0.0, prior_events: Iterable[Mapping[str, Any]] = (),
    policy_version: str = "model-health-v1", bootstrap_seed: int = 17,
) -> dict[str, Any]:
    """Aggregate settled forecasts by stock then session and recommend a policy weight.

    The eligible ticker map is the versioned coverage denominator supplied by the caller.
    A session below its coverage floor remains visible but cannot extend a trigger streak.
    """
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0, 1]")
    if policy_mode not in {"observe_only", "enforce"}:
        raise ValueError("policy_mode must be observe_only or enforce")
    if not math.isfinite(current_weight) or current_weight < 0:
        raise ValueError("current_weight must be finite and non-negative")
    cutoff = _as_of(as_of)
    # Events only become applicable from the instant they were actually recorded.
    prior_events = [dict(event) for event in prior_events
                    if _time(event.get("effective_at")) is not None
                    and _time(event.get("effective_at")) <= cutoff]
    by_stock_session: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in records:
        row = _normalize(raw)
        if row.get("status") != "settled":
            continue
        market = str(row.get("market", "")).upper()
        if market not in {"US", "TW", "JP"}:
            continue
        verified = _time(row.get("verified_at"))
        available = _time(row.get("actual_available_at")) or verified
        issued = _time(row.get("issued_at"))
        if verified is None or available is None or issued is None or available > cutoff or verified > cutoff:
            continue
        target_date = str(row.get("target_date", ""))[:10]
        ticker = str(row.get("ticker", ""))
        current, predicted, actual = (_finite(row.get(key)) for key in
            ("current_price", "predicted_price", "actual_price"))
        if not target_date or not ticker or not current or current <= 0 or predicted is None or actual is None:
            continue
        try:
            close = session_close(datetime.fromisoformat(target_date).date(), market)
        except (ValueError, TypeError):
            continue
        if issued >= close:
            continue
        ident = (market, str(row.get("model_name", "")), str(row.get("model_revision", "")),
                 int(row.get("horizon", 1)), str(row.get("config_hash", "")),
                 str(row.get("feature_profile", "")), str(row.get("evidence_type", "")),
                 str(row.get("target_type", "price")))
        # One report per market/model revision/horizon; filter later groups independently.
        key = ("|".join(map(str, ident)), target_date, ticker)
        previous = by_stock_session.get(key)
        if previous is None or issued > previous["_issued"]:
            predicted_return = (predicted - current) / current
            actual_return = (actual - current) / current
            by_stock_session[key] = {
                "market": market, "model_name": ident[1], "model_revision": ident[2],
                "horizon": ident[3], "config_hash": ident[4], "feature_profile": ident[5],
                "evidence_type": ident[6], "target_type": ident[7],
                "target_date": target_date, "ticker": ticker,
                "hit": float(_direction(predicted_return, flat_threshold) ==
                             _direction(actual_return, flat_threshold)),
                "_available": available,
                "improvement": 1 - abs(predicted - actual) / abs(current - actual)
                    if current != actual else None,
                "_issued": issued,
            }
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for rows in by_stock_session.values():
        ident = (rows["market"], rows["model_name"], rows["model_revision"], rows["horizon"],
                 rows["config_hash"], rows["feature_profile"], rows["evidence_type"], rows["target_type"])
        groups.setdefault(ident, []).append(rows)
    reports = []
    multiple_groups = len(groups) > 1
    for ident, rows in groups.items():
        def event_matches(event: Mapping[str, Any]) -> bool:
            scoped_keys = ("market", "model_name", "model_revision", "horizon", "config_hash",
                           "feature_profile", "evidence_type", "target_type")
            if multiple_groups and any(event.get(key) in (None, "") for key in scoped_keys):
                return False
            return all(event.get(key) in (None, "", ident[index]) for key, index in
                (("market", 0), ("model_name", 1), ("model_revision", 2), ("horizon", 3),
                 ("config_hash", 4), ("feature_profile", 5), ("evidence_type", 6), ("target_type", 7)))
        group_events = sorted((event for event in prior_events if event_matches(event)),
                              key=lambda event: (_time(event.get("effective_at")), str(event.get("event_key", ""))))
        # Rebuild the latest state chronologically. A later suspension starts a new
        # recovery cycle; an earlier degraded/healthy event cannot override it.
        suspended_event = degraded_event = None
        prior_state = None
        for event in group_events:
            state_at_event = event.get("to_state", event.get("from_state"))
            if state_at_event == "suspended":
                suspended_event, degraded_event = event, None
            elif state_at_event == "degraded" and suspended_event:
                degraded_event = event
            elif state_at_event == "healthy":
                suspended_event = degraded_event = None
            if state_at_event:
                prior_state = state_at_event
        sessions: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            sessions.setdefault(row["target_date"], []).append(row)
        metrics = []
        # Include eligible sessions with zero settled forecasts so coverage gaps remain visible.
        all_dates = set(sessions)
        for day in eligible_tickers_by_session:
            try:
                day_date = datetime.fromisoformat(str(day)[:10]).date()
                if (MarketCalendar.is_trading_day(day_date, market_or_ticker=ident[0]) and
                        session_close(day_date, ident[0]) <= cutoff):
                    all_dates.add(day_date.isoformat())
            except (TypeError, ValueError):
                continue
        for date in sorted(all_dates):
            day_date = datetime.fromisoformat(date).date()
            if not MarketCalendar.is_trading_day(day_date, market_or_ticker=ident[0]):
                continue
            values = sessions.get(date, [])
            eligible = set(map(str, eligible_tickers_by_session.get(date, ())))
            observed = {row["ticker"] for row in values}
            coverage = len(observed & eligible) / len(eligible) if eligible else 0.0
            improvements = [v["improvement"] for v in values if v["improvement"] is not None]
            metrics.append({"session": date, "sample_count": len(values),
                "coverage": coverage, "countable": bool(eligible and coverage >= coverage_threshold),
                "direction_accuracy": sum(v["hit"] for v in values) / len(values) if values else None,
                "baseline_improvement": sum(improvements) / len(improvements) if improvements else None,
                "evidence_available_at": max((v["_available"] for v in values), default=None)})
        countable = [value for value in metrics if value["countable"]]
        all_samples = sum(value["sample_count"] for value in countable)
        metric_by_day = {value["session"]: value for value in metrics}
        # Compute window boundaries from the market calendar, not the rows which
        # happened to arrive. An absent or low-coverage session remains a gap.
        calendar_days = _closed_market_sessions(ident[0], cutoff, count=100)
        fixed = [metric_by_day.get(day, {"session": day, "countable": False,
                 "sample_count": 0, "direction_accuracy": None, "baseline_improvement": None,
                 "evidence_available_at": None}) for day in calendar_days]
        recent_size = min(20, max(1, min_sessions))
        def evaluate_window(end_index: int):
            end = end_index + 1
            recent_window = fixed[max(0, end - recent_size):end]
            prior_window = fixed[max(0, end - recent_size - 60):max(0, end - recent_size)]
            recent_ok = len(recent_window) == recent_size and all(x["countable"] for x in recent_window)
            prior_ok = len(prior_window) == 60 and all(x["countable"] for x in prior_window)
            decline = (recent_ok and prior_ok and
                sum(x["direction_accuracy"] for x in recent_window) / recent_size <=
                sum(x["direction_accuracy"] for x in prior_window) / 60 - 0.10)
            improvements = [x["baseline_improvement"] for x in recent_window
                            if x["baseline_improvement"] is not None]
            baseline = bool(recent_ok and improvements and sum(improvements) / len(improvements) <= 0)
            has_evidence = recent_ok and sum(x["sample_count"] for x in recent_window) >= min_samples
            return recent_window, recent_ok, prior_ok, decline, baseline, has_evidence

        recent, recent_complete, prior_complete, decline_trigger, baseline_trigger, enough = evaluate_window(len(fixed) - 1)
        trigger = enough and (decline_trigger or baseline_trigger)
        streak = 0
        for index in range(len(fixed) - 1, -1, -1):
            _, _, _, decline, baseline, has_evidence = evaluate_window(index)
            if not has_evidence or not (decline or baseline):
                break
            streak += 1
        hit_sessions = [value["direction_accuracy"] for value in recent if value["direction_accuracy"] is not None]
        bootstrap = _block_bootstrap_upper(hit_sessions, seed=bootstrap_seed) if recent_complete else None
        if not enough:
            state, recommended = prior_state or "insufficient_data", current_weight
        elif bootstrap is not None and bootstrap < 0.5 or (trigger and streak >= 5):
            state, recommended = "suspended", 0.0
        elif trigger and streak >= 3:
            state, recommended = "degraded", current_weight * 0.5
        elif trigger:
            state, recommended = "watch", current_weight
        else:
            state, recommended = "healthy", current_weight

        # Recovery evidence begins strictly after the most recent suspension event.
        recovery = []
        suspended_at = _time(suspended_event.get("effective_at")) if suspended_event else None
        degraded_at = _time(degraded_event.get("effective_at")) if degraded_event else None
        recovery_floor = degraded_at or suspended_at
        if recovery_floor:
            recovery = [value for value in countable if
                session_close(datetime.fromisoformat(value["session"]).date(), ident[0]) > recovery_floor and
                value["session"] > (suspended_at.date().isoformat() if suspended_at else "")]
        recovery_hits = [value["direction_accuracy"] for value in recovery if value["direction_accuracy"] is not None]
        recovery_improvements = [value["baseline_improvement"] for value in recovery if value["baseline_improvement"] is not None]
        recovery_quality = bool(recovery and len(recovery_hits) >= 20 and
            sum(value["sample_count"] for value in recovery) >= 30 and
            recovery_improvements and sum(recovery_improvements) / len(recovery_improvements) > 0 and
            sum(recovery_hits) / len(recovery_hits) >= 0.55)
        consecutive_recovery_good = 0
        recovery_by_day = {value["session"]: value for value in recovery}
        for day in reversed(calendar_days):
            value = recovery_by_day.get(day)
            if value is None:
                break
            good = (value["direction_accuracy"] is not None and value["direction_accuracy"] >= 0.55 and
                    value["baseline_improvement"] is not None and value["baseline_improvement"] > 0)
            if not good:
                break
            consecutive_recovery_good += 1
        if suspended_event:
            if degraded_event:
                if recovery_quality and consecutive_recovery_good >= 20:
                    state, recommended = "healthy", 1.0
                else:
                    state, recommended = "degraded", 0.5
            elif recovery_quality and consecutive_recovery_good >= 5:
                state, recommended = "degraded", 0.5
            else:
                state, recommended = "suspended", 0.0
        recovery_count = sum(value["sample_count"] for value in recovery)
        effective = recommended if policy_mode == "enforce" else current_weight
        event = None
        if state != prior_state and state in {"watch", "degraded", "suspended", "healthy"}:
            event_date = (recovery[4]["session"] if state == "degraded" and suspended_event and not degraded_event and len(recovery) >= 5
                          else recovery[19]["session"] if state == "healthy" and degraded_event and len(recovery) >= 20
                          else countable[-1]["session"] if countable else None)
            if event_date:
                event_metric = next((value for value in metrics if value["session"] == event_date), None)
                target_close = session_close(datetime.fromisoformat(event_date).date(), ident[0])
                evidence_time = event_metric.get("evidence_available_at") if event_metric else None
                # A state transition becomes actionable only when this as-of evaluation
                # has all of its evidence; the target session alone is not decision time.
                effective_time = max(cutoff, target_close, evidence_time) if evidence_time else max(cutoff, target_close)
                event = {"event_key": ":".join(map(str, (*ident, event_date, state, policy_version))),
                    "market": ident[0], "model_name": ident[1], "model_revision": ident[2], "horizon": ident[3],
                    "config_hash": ident[4], "feature_profile": ident[5], "evidence_type": ident[6], "target_type": ident[7],
                    "from_state": prior_state or "insufficient_data", "to_state": state,
                    "effective_at": effective_time.isoformat(),
                    "original_weight": current_weight, "effective_weight": effective,
                    "policy_version": policy_version, "evidence": {"trigger_streak": streak,
                        "bootstrap_upper_95": bootstrap, "recovery_sample_count": recovery_count}}
        reports.append({"market": ident[0], "model_name": ident[1], "model_revision": ident[2],
            "horizon": ident[3], "config_hash": ident[4], "feature_profile": ident[5],
            "evidence_type": ident[6], "target_type": ident[7],
            "state": state, "policy_mode": policy_mode,
            "sample_count": all_samples, "settled_session_count": len(metrics),
            "countable_session_count": len(countable), "sessions": metrics,
            "trigger_streak": streak, "decline_trigger": decline_trigger,
            "recent_window_complete": recent_complete, "prior_window_complete": prior_complete,
            "baseline_trigger": baseline_trigger, "effective_weight": effective,
            "recommended_weight": recommended, "bootstrap": {"method": "session_block", "block_length": 5,
                "resamples": 2000, "seed": bootstrap_seed, "hit_rate_upper_95": bootstrap},
            "recovery_sample_count": recovery_count, "recovery_session_count": len(recovery),
            "event": event, "as_of": as_of,
            "coverage_threshold": coverage_threshold})
    return reports[0] if len(reports) == 1 else {"reports": reports, "as_of": as_of}
