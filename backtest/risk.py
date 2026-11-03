"""Versioned pre-trade constraints for simulated long-only portfolios."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class RiskPolicy:
    max_single: float = 0.10
    max_sector: float = 0.30
    max_group: float = 0.40
    max_total: float = 1.0
    max_adv_participation: float = 0.01
    max_drawdown: float = 0.20
    version: str = "risk-v1"

    def __post_init__(self):
        for k, v in asdict(self).items():
            if k != "version" and (not math.isfinite(v) or not 0 <= v <= 1):
                raise ValueError(f"invalid risk policy {k}")
        if not self.version:
            raise ValueError("risk policy requires a version")


def _num(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _time(value: Any):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("risk timestamps must contain a timezone")
    return dt.astimezone(timezone.utc)


def assess_order(account: Mapping[str, Any], order: Mapping[str, Any], policy: RiskPolicy) -> dict[str, Any]:
    """Return a conservative, auditable quantity without modifying the account.

    The caller supplies point-in-time positions, grouped exposures, macro state,
    account currency and average volume. Values with missing evidence fail closed
    for buys. Sells are limited to currently held shares and remain possible while
    valuation or macro data are stale.
    """
    side, ticker = order.get("side"), order.get("ticker")
    quantity, price = _num(order.get("quantity")), _num(order.get("price"))
    if side not in ("buy", "sell") or not ticker or quantity is None or quantity <= 0 or quantity != int(quantity):
        raise ValueError("order requires ticker, side and positive integer quantity")
    lot_size = int(order.get("lot_size", 1))
    if lot_size < 1 or int(quantity) % lot_size:
        raise ValueError("order quantity must respect lot size")
    positions = list(account.get("positions") or [])
    result = {"ticker": ticker, "side": side, "requested_quantity": int(quantity),
              "allowed_quantity": 0, "reasons": [], "limits": {}, "policy_version": policy.version,
              "decision_at": order.get("decision_at")}
    if side == "sell":
        held = sum(max(0.0, _num(p.get("quantity")) or 0) for p in positions if p.get("ticker") == ticker)
        result["allowed_quantity"] = min(int(quantity), math.floor(held / lot_size) * lot_size)
        if result["allowed_quantity"] < int(quantity):
            result["reasons"].append("insufficient_position")
        return result

    def reject(reason):
        if reason not in result["reasons"]:
            result["reasons"].append(reason)
        return result

    equity, cash, peak, macro = (_num(account.get(key)) for key in ("equity", "cash", "peak_equity", "macro_exposure"))
    if equity is None or equity <= 0 or cash is None or cash < 0 or peak is None or peak <= 0 or price is None or price <= 0:
        return reject("account_or_price_unavailable")
    if not account.get("data_fresh", False) or macro is None or not 0 <= macro <= 1:
        return reject("data_stale")
    if not order.get("sector"):
        return reject("sector_unavailable")
    if not order.get("group") or not account.get("correlation_available", False):
        return reject("correlation_unavailable")
    adv = _num(order.get("average_volume"))
    if adv is None or adv <= 0:
        return reject("volume_unavailable")
    if not order.get("decision_at") or not order.get("data_available_at") or not account.get("data_as_of"):
        return reject("data_time_unavailable")
    try:
        decision_at = _time(order["decision_at"])
        if _time(order["data_available_at"]) > decision_at or _time(account["data_as_of"]) > decision_at:
            return reject("future_data")
    except ValueError:
        return reject("future_data")
    if (peak - equity) / peak >= policy.max_drawdown:
        return reject("drawdown_limit")

    for p in positions:
        mv = _num(p.get("market_value"))
        if mv is None or mv < 0 or not p.get("sector") or not p.get("group"):
            return reject("position_risk_data_unavailable")
    total = sum(float(p["market_value"]) for p in positions)
    same = sum(float(p["market_value"]) for p in positions if p.get("ticker") == ticker)
    sector = sum(float(p["market_value"]) for p in positions if p.get("sector") == order["sector"])
    group = sum(float(p["market_value"]) for p in positions if p.get("group") == order["group"])
    monetary_limits = {
        "cash_limit": cash,
        "single_limit": max(0.0, equity * policy.max_single - same),
        "sector_limit": max(0.0, equity * policy.max_sector - sector),
        "correlation_limit": max(0.0, equity * policy.max_group - group),
        "total_exposure_limit": max(0.0, equity * min(policy.max_total, macro) - total),
    }
    qty_limits = {name: math.floor(value / price / lot_size) * lot_size for name, value in monetary_limits.items()}
    qty_limits["volume_limit"] = math.floor(adv * policy.max_adv_participation / lot_size) * lot_size
    result["limits"] = {"currency": account.get("currency"), "equity": equity, "cash": cash,
                        "macro_exposure": macro, "value_limits": monetary_limits, "quantity_limits": qty_limits}
    allowed = max(0, min(int(quantity), *qty_limits.values()))
    result["allowed_quantity"] = allowed
    if allowed < int(quantity):
        result["reasons"] = [name for name, limit in qty_limits.items() if limit == allowed]
    return result


def build_correlation_groups(return_series: Mapping[str, list], *, decision_session: str,
                             window: int = 60, min_samples: int = 40,
                             threshold: float = 0.8) -> dict[str, Any]:
    """Build connected components from aligned returns strictly before decision day.

    A return input is `(session_date, return)`; the caller is responsible for
    supplying only returns whose complete source bars were available at decision.
    Insufficient or zero-variance pairs produce an unavailable guard, not a fake
    zero correlation. The latest `window` common rows are used for each pair.
    """
    decision = date.fromisoformat(str(decision_session))
    if window < min_samples or min_samples < 2 or not -1 <= threshold <= 1:
        raise ValueError("invalid correlation policy")
    names = sorted(return_series)
    parents = {name: name for name in names}
    observed = {}
    for name, series in return_series.items():
        values = {}
        for raw_day, raw_value in series:
            day = date.fromisoformat(str(raw_day))
            value = _num(raw_value)
            if day >= decision or value is None:
                continue
            if day in values:
                raise ValueError(f"duplicate return session for {name}")
            values[day] = value
        observed[name] = values

    def find(name):
        while parents[name] != name:
            name = parents[name]
        return name

    pairs, unavailable = [], []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            days = sorted(observed[left].keys() & observed[right].keys())[-window:]
            entry = {"left": left, "right": right, "observations": len(days), "correlation": None}
            if len(days) >= min_samples:
                xs, ys = [observed[left][d] for d in days], [observed[right][d] for d in days]
                mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
                xx = sum((x - mx) ** 2 for x in xs)
                yy = sum((y - my) ** 2 for y in ys)
                if xx > 0 and yy > 0:
                    entry["correlation"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)
                    if entry["correlation"] >= threshold:
                        parents[find(right)] = find(left)
            if entry["correlation"] is None:
                unavailable.append((left, right))
            pairs.append(entry)
    return {"groups": {name: find(name) for name in names}, "pairs": pairs,
            "correlation_available": not unavailable, "insufficient_pairs": unavailable,
            "window": window, "min_samples": min_samples, "threshold": threshold,
            "decision_session": decision.isoformat()}


def summarize_exposure(account: Mapping[str, Any], policy: RiskPolicy) -> dict[str, Any]:
    """Return current sector/group exposure and recommendations for the next session."""
    equity, cash, macro = (_num(account.get(k)) for k in ("equity", "cash", "macro_exposure"))
    if equity is None or equity <= 0 or cash is None or macro is None or not 0 <= macro <= 1:
        return {"status": "data_unavailable", "breaches": [], "reduction_suggestions": []}
    positions = account.get("positions") or []
    sector, group, tickers = {}, {}, {}
    for p in positions:
        value = _num(p.get("market_value"))
        if value is None or value < 0 or not p.get("ticker") or not p.get("sector") or not p.get("group"):
            return {"status": "data_unavailable", "breaches": [], "reduction_suggestions": []}
        for book, key in ((sector, p["sector"]), (group, p["group"]), (tickers, p["ticker"])):
            book[key] = book.get(key, 0) + value / equity
    total = sum(tickers.values())
    breaches, suggestions = [], []
    total_limit = min(policy.max_total, macro)
    if total > total_limit:
        breaches.append("total_exposure_limit")
        suggestions.append({"kind": "total", "excess_value": (total - total_limit) * equity})
    for name, values, limit in (("single", tickers, policy.max_single), ("sector", sector, policy.max_sector), ("group", group, policy.max_group)):
        for key, ratio in values.items():
            if ratio > limit:
                breaches.append(f"{name}_limit:{key}")
                suggestions.append({"kind": name, "key": key, "excess_value": (ratio - limit) * equity})
    return {"status": "ready", "currency": account.get("currency"), "cash": cash,
            "total_exposure": total, "total_limit": total_limit,
            "sector_exposures": sector, "group_exposures": group,
            "ticker_exposures": tickers, "breaches": breaches,
            "reduction_suggestions": suggestions, "policy_version": policy.version}
