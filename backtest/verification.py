"""Session aligned forecast target resolution and exact-date settlement."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from core.market_calendar import MarketCalendar

TZ = {"TW": ZoneInfo("Asia/Taipei"), "US": ZoneInfo("America/New_York"), "JP": ZoneInfo("Asia/Tokyo")}


def _market(market: str) -> str:
    m = str(market).upper()
    if m in TZ:
        return m
    return MarketCalendar.resolve_market(m)


def _thanksgiving(year: int) -> dt.date:
    d = dt.date(year, 11, 1)
    return d + dt.timedelta(days=(3 - d.weekday()) % 7 + 21)


def session_close(day: dt.date, market: str) -> dt.datetime:
    market = _market(market)
    hour, minute = {"TW": (13, 30), "US": (16, 0), "JP": (15, 30)}[market]
    if market == "US":
        # NYSE regular early closes: Thanksgiving Friday, eligible July 3, and Christmas Eve.
        early = day == _thanksgiving(day.year) + dt.timedelta(days=1)
        early = early or (day.month == 7 and day.day == 3 and day.weekday() < 5)
        early = early or (day.month == 12 and day.day == 24 and day.weekday() < 5)
        if early:
            hour, minute = 13, 0
    return dt.datetime.combine(day, dt.time(hour, minute), TZ[market])


def resolve_target_session(last_complete_bar: str | dt.date, market: str, horizon: int = 1,
                           calendar: type[MarketCalendar] = MarketCalendar) -> dt.date:
    """Advance exactly `horizon` sessions from the last complete daily bar."""
    market = _market(market)
    day = last_complete_bar if isinstance(last_complete_bar, dt.date) else dt.date.fromisoformat(str(last_complete_bar)[:10])
    if not calendar.is_trading_day(day, market_or_ticker=market):
        raise ValueError("last_complete_bar is not a trading session")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    for _ in range(horizon):
        day = calendar.get_next_trading_day(day, market)
    return day


def verify_forecast(store: Any, forecast_id: str, bars: Iterable[Mapping[str, Any]], *,
                    now: dt.datetime | None = None, source: str = "market_data",
                    source_revision: str = "unknown", price_basis: str = "adjusted_close",
                    dry_run: bool = False) -> dict[str, Any]:
    """Settle only the forecast's target session; absence never rolls forward."""
    forecast = store.get_forecast(forecast_id)
    if not forecast:
        raise KeyError(f"unknown forecast_id: {forecast_id}")
    now = now or dt.datetime.now(dt.timezone.utc)
    market = _market(forecast["market"])
    target = dt.date.fromisoformat(str(forecast["target_date"])[:10])
    target_close = session_close(target, market)
    def pending(reason: str, status: str = "pending"):
        if not dry_run:
            if status == "invalid":
                store.mark_invalid(forecast_id, reason, target_date=target.isoformat())
            else:
                store.mark_pending(forecast_id, reason, target_date=target.isoformat())
        return {"status": status, "reason": reason, "target_date": target.isoformat()}
    expected = resolve_target_session(forecast.get("last_complete_bar", forecast.get("last_bar_date")), market,
                                      int(forecast["horizon"]))
    if target != expected:
        return pending("target_date_mismatch", "invalid")
    issued = dt.datetime.fromisoformat(str(forecast["issued_at"]).replace("Z", "+00:00"))
    if issued.tzinfo is None:
        issued = issued.replace(tzinfo=TZ[market])
    if issued >= target_close:
        return pending("issued_after_target_close", "invalid")
    if now.astimezone(dt.timezone.utc) < target_close.astimezone(dt.timezone.utc):
        return pending("target_session_not_closed")
    match = None
    for bar in bars or []:
        raw_date = bar.get("date", bar.get("session_date"))
        if raw_date is not None and str(raw_date)[:10] == target.isoformat():
            match = bar
            break
    if match is None or match.get("close", match.get("Close")) is None:
        return pending("target_price_missing")
    if match.get("complete") is False or match.get("is_complete") is False:
        return pending("target_bar_incomplete")
    declared_basis = forecast.get("price_basis")
    if declared_basis and declared_basis != price_basis:
        return pending("price_basis_mismatch", "invalid")
    try:
        price = float(match.get("close", match.get("Close")))
    except (TypeError, ValueError):
        return pending("target_price_invalid", "invalid")
    if not math.isfinite(price) or price <= 0:
        return pending("target_price_invalid", "invalid")
    available = match.get("actual_available_at") or now.isoformat()
    details = {"actual": price, "actual_price": price,
               "target_date": target.isoformat(), "source": source, "source_revision": source_revision,
               "price_basis": price_basis, "actual_available_at": available, "verified_at": now.isoformat()}
    if not dry_run:
        store.settle_forecast(forecast_id, details)
    return {"status": "settled", **details}
