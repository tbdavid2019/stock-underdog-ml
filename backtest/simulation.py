"""Deterministic, conservative primitives shared by historical and paper trading.

Prices used by this module are unadjusted.  The caller supplies a versioned,
market-specific fee schedule and opening-auction observations available before
the order decision.  Missing cost or opening-volume history is never filled in.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from statistics import median
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CostSchedule:
    market: str
    instrument: str
    effective_from: date
    commission_rate: float
    minimum_commission: float
    tax_rate: float
    slippage_bps: float
    impact_bps: float
    lot_size: int
    version: str
    effective_to: date | None = None

    def __post_init__(self):
        for field in ("commission_rate", "minimum_commission", "tax_rate", "slippage_bps", "impact_bps"):
            v = getattr(self, field)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"invalid cost schedule {field}")
        if self.lot_size < 1 or not self.version or not self.market or not self.instrument:
            raise ValueError("cost schedule requires market, instrument, positive lot size and version")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("cost schedule effective_to precedes effective_from")


def _when(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _day(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def execute_open(signal: Mapping[str, Any], bar: Mapping[str, Any],
                 historical_volumes: Sequence[Mapping[str, Any]], costs: CostSchedule | None) -> dict[str, Any]:
    """Execute against the next eligible open using exclusively pre-decision liquidity.

    `historical_volumes` carries date, volume, open_volume, and optional timezone-aware
    available_at. Rows without available_at are accepted only from dates strictly before
    the signal's decision session. An order remainder is cancelled in that session.
    """
    if costs is None:
        raise ValueError("cost schedule is required")
    issued = _when(signal["issued_at"])
    opened = _when(bar["open_at"])
    session = _day(bar["date"])
    if session < costs.effective_from or (costs.effective_to is not None and session > costs.effective_to):
        raise ValueError("cost schedule is not effective for the session")
    side = signal.get("side")
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")
    quantity = _number(signal.get("quantity"))
    if quantity is None or quantity <= 0 or quantity != int(quantity):
        raise ValueError("quantity must be a positive integer")
    qty = int(quantity)
    if qty % costs.lot_size:
        raise ValueError("quantity does not satisfy lot size")
    base = {"ticker": signal.get("ticker"), "side": side, "requested_quantity": qty,
            "filled_quantity": 0, "cancelled_quantity": qty, "remaining_quantity": 0,
            "status": "rejected", "cost_version": costs.version, "session": session.isoformat()}
    def reject(reason: str):
        return {**base, "reason": reason}
    if opened <= issued:
        return reject("signal_not_available_before_open")
    if bar.get("suspended"):
        return reject("suspended")
    if bar.get("limit_locked"):
        return reject("limit_locked")
    price = _number(bar.get("open"))
    if price is None or price <= 0:
        return reject("open_unavailable")

    daily = {}
    for row in historical_volumes:
        try:
            observed = _day(row["date"])
            # A historical row from the target session can never serve as an opening capacity estimate.
            if observed >= session or observed > issued.date():
                continue
            if row.get("available_at"):
                if _when(row["available_at"]) > issued:
                    continue
            elif observed >= issued.date():
                continue
            ov, dv = _number(row.get("open_volume")), _number(row.get("volume"))
            if ov is None or dv is None or ov <= 0 or dv <= 0:
                continue
            # Duplicate dates are ambiguous rather than an extra observation.
            if observed in daily:
                daily[observed] = None
            else:
                daily[observed] = (ov, dv)
        except (KeyError, TypeError, ValueError):
            continue
    values = [x for _, x in sorted(daily.items()) if x is not None][-20:]
    if len(values) < 10:
        return reject("capacity_unavailable")
    capacity = min(median(x[0] for x in values) * 0.01,
                   sum(x[1] for x in values) / len(values) * 0.01)
    filled = min(qty, math.floor(capacity / costs.lot_size) * costs.lot_size)
    if filled < costs.lot_size:
        return reject("capacity_unavailable")
    # Impact is a configurable non-negative bps per unit of participation.
    slip_bps = costs.slippage_bps + costs.impact_bps * filled / capacity
    execution_price = price * (1 + slip_bps / 10000 * (1 if side == "buy" else -1))
    if execution_price <= 0:
        return reject("invalid_execution_price")
    notional = execution_price * filled
    commission = max(costs.minimum_commission, notional * costs.commission_rate)
    tax = notional * costs.tax_rate if side == "sell" else 0.0
    return {**base, "status": "filled" if filled == qty else "partially_filled",
            "reason": None if filled == qty else "capacity_limited",
            "filled_quantity": filled, "cancelled_quantity": qty - filled,
            "fill_price": execution_price, "gross_value": notional, "commission": commission,
            "tax": tax, "slippage_bps": slip_bps, "capacity": capacity,
            "capacity_sample_count": len(values),
            "net_cash_change": (-notional - commission - tax) if side == "buy" else (notional - commission - tax)}


def walk_forward_windows(sessions: Sequence[str], *, calibration: int = 120,
                         test: int = 20, horizon: int = 1) -> list[dict[str, list[str]]]:
    """Return ordered OOS folds with labels purged immediately before each test fold."""
    if calibration <= 0 or test <= 0 or horizon <= 0 or horizon >= calibration:
        raise ValueError("calibration, test and horizon must define a nonempty train window")
    dates = [_day(s).isoformat() for s in sessions]
    if dates != sorted(set(dates)):
        raise ValueError("sessions must be unique and strictly chronological")
    folds = []
    for start in range(calibration, len(dates) - test + 1, test):
        folds.append({"train": dates[start-calibration:start-horizon],
                      "purged": dates[start-horizon:start], "test": dates[start:start+test]})
    return folds
