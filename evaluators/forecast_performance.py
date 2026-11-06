"""Deterministic forecast performance summaries for the local verification store."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from backtest.verification import session_close
from core.market_calendar import MarketCalendar

_ALLOWED_WINDOWS = {20, 60, 120}
_TZ = {"TW": "Asia/Taipei", "US": "America/New_York", "JP": "Asia/Tokyo"}


@dataclass(frozen=True)
class PerformanceQuery:
    as_of: str
    window: int = 60
    market: str | None = None
    model_name: str | None = None
    model_revision: str | None = None
    config_hash: str | None = None
    feature_profile: str | None = None
    horizon: int | None = None
    target_type: str | None = None
    evidence_type: str | None = "prospective"
    min_samples: int = 30
    flat_threshold: float = 0.0
    score_version: str = "forecast-score-v1"

    def __post_init__(self):
        if self.window not in _ALLOWED_WINDOWS:
            raise ValueError("window must be 20, 60, or 120 trading sessions")
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not math.isfinite(self.flat_threshold) or self.flat_threshold < 0:
            raise ValueError("flat_threshold must be a finite non-negative return")
        try:
            parsed = datetime.fromisoformat(self.as_of.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("as_of must be an ISO-8601 timestamp with timezone") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("as_of must include a timezone")

    @property
    def cutoff(self) -> datetime:
        return datetime.fromisoformat(self.as_of.replace("Z", "+00:00")).astimezone(timezone.utc)


def _time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            return None
        return result
    except (TypeError, ValueError):
        return None


def _date(value: Any):
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except (TypeError, ValueError):
        return None


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (TypeError, ValueError):
        return None


def _normalize(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten ForecastStore.query_records rows while accepting direct fixtures."""
    if isinstance(row.get("forecast"), Mapping):
        result = dict(row["forecast"])
        actual = row.get("actual") or {}
        if isinstance(actual, Mapping):
            result.update(actual)
        result["status"] = row.get("status", result.get("status"))
        result["verified_at"] = row.get("verified_at", result.get("verified_at"))
        return result
    return dict(row)


def _calendar_dates(query: PerformanceQuery, market: str) -> set[str]:
    market = str(market).upper()
    if market not in _TZ:
        market = MarketCalendar.resolve_market(market)
    as_of = query.cutoff.astimezone(ZoneInfo(_TZ[market]))
    days = MarketCalendar.get_recent_trading_days(query.window, reference_dt=as_of, market_or_ticker=market)
    return {f"{value[:4]}-{value[4:6]}-{value[6:8]}" for value in days}


def _eligible_records(records: Iterable[Mapping[str, Any]], query: PerformanceQuery):
    source = list(records)
    accepted: list[dict[str, Any]] = []
    excluded = 0
    selected_by_id: dict[str, dict[str, Any]] = {}
    cutoff = query.cutoff
    session_cache: dict[str, set[str]] = {}
    for raw in source:
        row = _normalize(raw)
        if row.get("status") != "settled" or (query.evidence_type is not None and row.get("evidence_type", "legacy_unverified") != query.evidence_type):
            excluded += 1
            continue
        market = str(row.get("market", "")).upper()
        if market not in _TZ:
            market = MarketCalendar.resolve_market(market)
        if query.market and market != MarketCalendar.resolve_market(query.market):
            excluded += 1
            continue
        filters = (("model_name", query.model_name), ("model_revision", query.model_revision),
                   ("config_hash", query.config_hash), ("feature_profile", query.feature_profile),
                   ("horizon", query.horizon), ("target_type", query.target_type))
        if any(expected is not None and row.get(key) != expected for key, expected in filters):
            excluded += 1
            continue
        target = _date(row.get("target_date"))
        verified = _time(row.get("verified_at"))
        available = _time(row.get("actual_available_at")) or verified
        issued = _time(row.get("issued_at"))
        if not target or not verified or not available or not issued:
            excluded += 1
            continue
        if verified.astimezone(cutoff.tzinfo) > cutoff or available.astimezone(cutoff.tzinfo) > cutoff:
            excluded += 1
            continue
        invalidated = _time(row.get("invalidated_at"))
        if invalidated is not None and invalidated.astimezone(cutoff.tzinfo) <= cutoff:
            excluded += 1
            continue
        if market not in session_cache:
            session_cache[market] = _calendar_dates(query, market)
        if target.isoformat() not in session_cache[market]:
            excluded += 1
            continue
        close = session_close(target, market)
        if issued.astimezone(close.tzinfo) >= close:
            excluded += 1
            continue
        current = _finite_positive(row.get("current_price"))
        predicted = _finite_positive(row.get("predicted_price", row.get("prediction")))
        actual = _finite_positive(row.get("actual_price", row.get("actual")))
        if current is None or predicted is None or actual is None:
            excluded += 1
            continue
        row.update({"market": market, "target_date": target.isoformat(), "current_price": current,
                    "predicted_price": predicted, "actual_price": actual})
        fid = str(row.get("forecast_id", ""))
        if not fid:
            accepted.append(row)
            continue
        old = selected_by_id.get(fid)
        if old is None:
            selected_by_id[fid] = row
            continue
        old_verified = _time(old.get("verified_at")) or datetime.min.replace(tzinfo=timezone.utc)
        if verified > old_verified:
            selected_by_id[fid] = row
        excluded += 1
    accepted.extend(selected_by_id.values())
    return accepted, excluded


def _direction(change: float, threshold: float) -> int:
    return 0 if abs(change) <= threshold else (1 if change > 0 else -1)


def _wilson_lower(successes: int, n: int, z: float = 1.959963984540054) -> float | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (center - margin) / denom


def _summarize(rows: list[dict[str, Any]], query: PerformanceQuery) -> dict[str, Any]:
    n = len(rows)
    daily: dict[str, list[tuple[int, float, float, float, float, float]]] = {}
    for row in rows:
        current, predicted, actual = row["current_price"], row["predicted_price"], row["actual_price"]
        pdir = _direction((predicted - current) / current, query.flat_threshold)
        adir = _direction((actual - current) / current, query.flat_threshold)
        # Score return errors so the metric is comparable across price levels.
        predicted_return = (predicted - current) / current
        actual_return = (actual - current) / current
        strat_ret = actual_return if pdir > 0 else (-actual_return if pdir < 0 else 0.0)
        daily.setdefault(row["target_date"], []).append((int(pdir == adir),
            abs(predicted - actual), abs(current - actual), (predicted_return - actual_return) ** 2,
            actual_return ** 2, abs(predicted - actual) / actual, strat_ret))
    # First average stocks within each session, then weight each session equally.
    day_values = []
    day_hits = []
    session_returns = []
    for _, values in sorted(daily.items()):
        day_values.append(tuple(sum(v[i] for v in values) / len(values) for i in range(1, 6)))
        day_hits.append(sum(v[0] for v in values) / len(values))
        session_returns.append(sum(v[6] for v in values) / len(values))
    sessions = len(day_values)
    mae = sum(v[0] for v in day_values) / sessions if sessions else None
    baseline_mae = sum(v[1] for v in day_values) / sessions if sessions else 0
    sq_error = sum(v[2] for v in day_values) / sessions if sessions else None
    sq_baseline = sum(v[3] for v in day_values) / sessions if sessions else 0
    mape = sum(v[4] for v in day_values) / sessions if sessions else None
    successes = sum(hit >= 0.5 for hit in day_hits)

    stat_val = {}
    try:
        from backtest.statistical_validation import validate_returns_significance
        stat_val = validate_returns_significance(daily_returns=session_returns)
    except Exception:
        stat_val = {
            "status": "insufficient_samples",
            "p_value_return": None,
            "sharpe_ci_low": None,
            "sharpe_ci_high": None,
            "is_statistically_significant": False
        }

    return {
        "sample_count": n,
        "session_count": sessions,
        "direction_accuracy": sum(day_hits) / sessions if sessions else None,
        "direction_hit_count": successes,
        "wilson_95_lower": _wilson_lower(successes, sessions),
        "mape": mape,
        "mae": mae,
        "baseline_improvement": 1 - mae / baseline_mae if mae is not None and baseline_mae else None,
        "oos_r2_zero_return": 1 - sq_error / sq_baseline if sq_error is not None and sq_baseline else None,
        "direction_threshold": query.flat_threshold,
        "flat_rule": "absolute_return_lte_threshold_is_flat",
        "p_value_return": stat_val.get("p_value_return"),
        "sharpe_ci_low": stat_val.get("sharpe_ci_low"),
        "sharpe_ci_high": stat_val.get("sharpe_ci_high"),
        "is_statistically_significant": bool(stat_val.get("is_statistically_significant", False)),
        "validation_status": stat_val.get("status", "insufficient_samples"),
    }


def _scope_key(row: Mapping[str, Any], include_model: bool = True) -> tuple[Any, ...]:
    # Config hashes identify a model's run setup, not the market/sample cohort.
    # Different models can have different valid config hashes on the same ticker/session.
    base = (row["market"], int(row.get("horizon", 1)), row.get("target_type", "price"),
            row.get("evidence_type"), row.get("feature_profile"),
            row.get("price_basis", "unknown"), row.get("label_definition_version", "v1"))
    if include_model:
        return base + (row.get("model_name"), row.get("model_revision"))
    return base


def _stock_groups(rows):
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = _scope_key(row) + (row.get("config_hash"), row.get("ticker"),)
        groups.setdefault(key, []).append(row)
    return groups


def rank_stocks(records: Iterable[Mapping[str, Any]], query: PerformanceQuery) -> dict[str, Any]:
    eligible, excluded = _eligible_records(records, query)
    data, observations = [], []
    for key, group in _stock_groups(eligible).items():
        # Keep at most one forecast per ticker/model/session, selecting the latest issued eligible forecast.
        by_date: dict[str, dict[str, Any]] = {}
        for row in group:
            old = by_date.get(row["target_date"])
            if old is None or row.get("issued_at", "") > old.get("issued_at", ""):
                by_date[row["target_date"]] = row
        selected = list(by_date.values())
        report = {"market": key[0], "horizon": key[1], "target_type": key[2], "evidence_type": key[3],
                  "feature_profile": key[4], "price_basis": key[5], "label_definition_version": key[6],
                  "model_name": key[7], "model_revision": key[8],
                  "config_hash": key[9], "ticker": key[10], **_summarize(selected, query)}
        report["status"] = "eligible" if report["sample_count"] >= query.min_samples else "insufficient_data"
        (data if report["status"] == "eligible" else observations).append(report)
    data.sort(key=lambda r: (-(r["wilson_95_lower"] if r["wilson_95_lower"] is not None else -1), -(r["baseline_improvement"] if r["baseline_improvement"] is not None else -math.inf), -r["sample_count"], r["ticker"]))
    observations.sort(key=lambda r: (r["market"], r["ticker"], r["model_name"] or "", r["model_revision"] or ""))
    return {"data": data, "observations": observations, "excluded_count": excluded,
            "as_of": query.as_of, "window": query.window, "score_version": query.score_version,
            "minimum_samples": query.min_samples}


def rank_models(records: Iterable[Mapping[str, Any]], query: PerformanceQuery) -> dict[str, Any]:
    eligible, excluded = _eligible_records(records, query)
    cohorts: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in eligible:
        cohorts.setdefault(_scope_key(row, include_model=False), []).append(row)
    data, observations = [], []
    overall_common = 0
    for cohort, rows in cohorts.items():
        models: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            ident = (str(row.get("model_name", "")), str(row.get("model_revision", "")), str(row.get("config_hash", "")))
            models.setdefault(ident, []).append(row)
        per_model_pairs = {}
        for ident, model_rows in models.items():
            pair_map = {}
            for row in model_rows:
                pair = (row["ticker"], row["target_date"])
                old = pair_map.get(pair)
                if old is None or (row.get("issued_at", "") or "") > (old.get("issued_at", "") or ""):
                    pair_map[pair] = row
            per_model_pairs[ident] = pair_map
        if not per_model_pairs:
            continue
        common = set.intersection(*(set(value) for value in per_model_pairs.values())) if per_model_pairs else set()
        overall_common = max(overall_common, len(common))
        for ident, pair_map in per_model_pairs.items():
            full = list(pair_map.values())
            shared = [pair_map[pair] for pair in sorted(common)]
            if not shared:
                continue
            report = {"market": cohort[0], "horizon": cohort[1], "target_type": cohort[2],
                      "evidence_type": cohort[3], "feature_profile": cohort[4], "price_basis": cohort[5],
                      "label_definition_version": cohort[6], "config_hash": ident[2],
                      "model_name": ident[0], "model_revision": ident[1],
                      "full_sample_count": len(full), "common_sample_count": len(shared),
                      "stock_count": len({r["ticker"] for r in shared}),
                      "session_count": len({r["target_date"] for r in shared}), **_summarize(shared, query)}
            report["status"] = "eligible" if report["sample_count"] >= query.min_samples else "insufficient_data"
            (data if report["status"] == "eligible" else observations).append(report)
    data.sort(key=lambda r: (-(r["wilson_95_lower"] if r["wilson_95_lower"] is not None else -1), -(r["baseline_improvement"] if r["baseline_improvement"] is not None else -math.inf), r["model_name"], r["model_revision"]))
    return {"data": data, "observations": observations, "excluded_count": excluded,
            "common_sample_count": overall_common, "as_of": query.as_of, "window": query.window,
            "score_version": query.score_version, "minimum_samples": query.min_samples}
