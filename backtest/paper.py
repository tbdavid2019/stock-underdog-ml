"""Transactional local paper-trading ledger; never places brokerage orders."""
from __future__ import annotations

from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import threading
from typing import Any, Mapping

import duckdb

_LOCK = threading.RLock()


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("ledger payload must be strict JSON") from exc


def _positive(value: Any, name: str, *, zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if not math.isfinite(result) or (result < 0 if zero else result <= 0):
        raise ValueError(f"invalid {name}")
    return result


class PaperLedger:
    """Create isolated long-only currency accounts and record auditable executions."""

    def __init__(self, db_path, read_only: bool = False):
        self.db_path = str(db_path)
        self.read_only = read_only
        if not read_only:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with _LOCK, self._connect() as con:
                con.execute("""CREATE TABLE IF NOT EXISTS paper_accounts (
                    account_id VARCHAR PRIMARY KEY, market VARCHAR NOT NULL, currency VARCHAR NOT NULL,
                    initial_cash DOUBLE NOT NULL, cash DOUBLE NOT NULL, created_at TIMESTAMPTZ NOT NULL)""")
                con.execute("""CREATE TABLE IF NOT EXISTS paper_positions (
                    account_id VARCHAR NOT NULL, ticker VARCHAR NOT NULL, quantity DOUBLE NOT NULL,
                    total_cost DOUBLE NOT NULL, realized_pnl DOUBLE NOT NULL DEFAULT 0,
                    last_price DOUBLE, last_price_at TIMESTAMPTZ,
                    PRIMARY KEY(account_id,ticker))""")
                con.execute("""CREATE TABLE IF NOT EXISTS paper_orders (
                    order_id VARCHAR PRIMARY KEY, account_id VARCHAR NOT NULL, signal_id VARCHAR NOT NULL,
                    ticker VARCHAR NOT NULL, side VARCHAR NOT NULL, decision_session DATE NOT NULL,
                    quantity DOUBLE NOT NULL, status VARCHAR NOT NULL, payload_json VARCHAR NOT NULL,
                    risk_json VARCHAR NOT NULL,
                    UNIQUE(account_id,signal_id,decision_session,ticker,side))""")
                con.execute("""CREATE TABLE IF NOT EXISTS paper_fills (
                    fill_id VARCHAR PRIMARY KEY, order_id VARCHAR NOT NULL, account_id VARCHAR NOT NULL,
                    filled_quantity DOUBLE NOT NULL, cancelled_quantity DOUBLE NOT NULL, remaining_quantity DOUBLE NOT NULL,
                    fill_price DOUBLE NOT NULL, commission DOUBLE NOT NULL, tax DOUBLE NOT NULL,
                    payload_json VARCHAR NOT NULL)""")
                con.execute("""CREATE TABLE IF NOT EXISTS paper_account_events (
                    account_id VARCHAR NOT NULL, event_id VARCHAR NOT NULL, ticker VARCHAR NOT NULL,
                    event_type VARCHAR NOT NULL, payload_json VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY(account_id,event_id))""")
                con.execute("""CREATE TABLE IF NOT EXISTS paper_equity_snapshots (
                    account_id VARCHAR NOT NULL, session DATE NOT NULL, payload_json VARCHAR NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY(account_id,session))""")

    def _connect(self):
        return duckdb.connect(self.db_path, read_only=self.read_only)

    def create_account(self, account_id: str, *, market: str, currency: str, cash: float) -> None:
        initial = _positive(cash, "initial cash", zero=True)
        if not account_id or market not in {"TW", "US", "JP"} or currency not in {"TWD", "USD", "JPY"}:
            raise ValueError("account requires supported market and currency")
        if {"TW": "TWD", "US": "USD", "JP": "JPY"}[market] != currency:
            raise ValueError("account currency must match market")
        with _LOCK, self._connect() as con:
            row = con.execute("SELECT market,currency,initial_cash FROM paper_accounts WHERE account_id=?", [account_id]).fetchone()
            if row:
                if row != (market, currency, initial):
                    raise ValueError("account identity already exists with different setup")
                return
            con.execute("INSERT INTO paper_accounts VALUES (?,?,?,?,?,?)",
                        [account_id, market, currency, initial, initial, datetime.now(timezone.utc)])

    def record_execution(self, order: Mapping[str, Any], fill: Mapping[str, Any], *,
                         risk_decision: Mapping[str, Any] | None = None,
                         risk_context: Mapping[str, Any] | None = None,
                         fail_after_order: bool = False) -> None:
        """Atomically record an order, fill, cash and position with a pre-trade decision.

        `risk_decision` must be computed from the same account snapshot immediately
        before this call. A production coordinator must serialize decision + commit;
        this ledger validates quantity but does not itself reconstruct sector exposure.
        """
        order, fill = dict(order), dict(fill)
        if risk_decision is None:
            if risk_context is None:
                raise ValueError("matching versioned pre-trade risk decision required")
            from backtest.risk import RiskPolicy, assess_order
            account = self.get_account(order["account_id"])
            position_map = account.get("positions", {}) or {}
            positions = [{"ticker": ticker, **dict(position)} for ticker, position in position_map.items()]
            merged = {**account, **dict(risk_context), "positions": positions}
            merged.setdefault("equity", float(account.get("cash", 0.0)) + sum(
                float(position.get("total_cost", 0.0)) for position in positions))
            merged.setdefault("peak_equity", merged.get("equity", account.get("initial_cash", 0.0)))
            risk_decision = assess_order(merged, order, RiskPolicy())
        risk = dict(risk_decision)
        for key in ("order_id", "account_id", "signal_id", "ticker", "side", "quantity", "decision_session"):
            if key not in order or order[key] in (None, ""):
                raise ValueError(f"missing order {key}")
        for key in ("fill_id", "filled_quantity", "cancelled_quantity", "fill_price", "commission", "tax"):
            if key not in fill:
                raise ValueError(f"missing fill {key}")
        if order["side"] not in {"buy", "sell"}:
            raise ValueError("long-only side must be buy or sell")
        requested = _positive(order["quantity"], "requested quantity")
        filled = _positive(fill["filled_quantity"], "filled quantity", zero=True)
        cancelled = _positive(fill["cancelled_quantity"], "cancelled quantity", zero=True)
        price = _positive(fill["fill_price"], "fill price", zero=True)
        commission = _positive(fill["commission"], "commission", zero=True)
        tax = _positive(fill["tax"], "tax", zero=True)
        if not all(v.is_integer() for v in (requested, filled, cancelled)) or filled + cancelled != requested:
            raise ValueError("fill and cancelled quantities must reconcile to integer requested quantity")
        if risk.get("ticker") != order["ticker"] or risk.get("side") != order["side"] or not risk.get("policy_version"):
            raise ValueError("matching versioned pre-trade risk decision required")
        allowed = _positive(risk.get("allowed_quantity"), "risk allowed quantity", zero=True)
        if filled > allowed:
            raise ValueError("filled quantity exceeds risk decision")
        order_json, fill_json, risk_json = _json(order), _json(fill), _json(risk)
        status = "cancelled" if filled == 0 else ("filled" if cancelled == 0 else "partially_filled")
        with _LOCK, self._connect() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                previous = con.execute("""SELECT o.payload_json,f.payload_json,o.risk_json FROM paper_orders o
                    LEFT JOIN paper_fills f ON o.order_id=f.order_id WHERE o.order_id=?""", [order["order_id"]]).fetchone()
                if previous:
                    if previous != (order_json, fill_json, risk_json):
                        raise ValueError("order id already has different immutable execution")
                    con.execute("COMMIT")
                    return
                account = con.execute("SELECT cash,market,currency FROM paper_accounts WHERE account_id=?", [order["account_id"]]).fetchone()
                if account is None:
                    raise ValueError("unknown account")
                if (order.get("market") and order["market"] != account[1]) or (order.get("currency") and order["currency"] != account[2]):
                    raise ValueError("order market or currency differs from account")
                cash = float(account[0])
                position = con.execute("SELECT quantity,total_cost,realized_pnl FROM paper_positions WHERE account_id=? AND ticker=?", [order["account_id"], order["ticker"]]).fetchone()
                held, cost, realized = tuple(map(float, position)) if position else (0.0, 0.0, 0.0)
                gross = filled * price
                if order["side"] == "buy":
                    new_cash = cash - gross - commission - tax
                    if new_cash < -1e-9:
                        raise ValueError("insufficient cash after costs")
                    new_held, new_cost, new_realized = held + filled, cost + gross + commission + tax, realized
                else:
                    if filled > held:
                        raise ValueError("cannot sell more than held quantity")
                    cost_sold = cost * filled / held
                    new_cash = cash + gross - commission - tax
                    new_held, new_cost = held - filled, cost - cost_sold
                    new_realized = realized + gross - commission - tax - cost_sold
                con.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)", [order["order_id"], order["account_id"], order["signal_id"], order["ticker"], order["side"], order["decision_session"], requested, status, order_json, risk_json])
                if fail_after_order:
                    raise RuntimeError("injected failure after order")
                con.execute("INSERT INTO paper_fills VALUES (?,?,?,?,?,?,?,?,?,?)", [fill["fill_id"], order["order_id"], order["account_id"], filled, cancelled, 0, price, commission, tax, fill_json])
                con.execute("UPDATE paper_accounts SET cash=? WHERE account_id=?", [max(0., new_cash), order["account_id"]])
                con.execute("""INSERT INTO paper_positions(account_id,ticker,quantity,total_cost,realized_pnl)
                    VALUES (?,?,?,?,?) ON CONFLICT(account_id,ticker) DO UPDATE SET
                    quantity=excluded.quantity,total_cost=excluded.total_cost,realized_pnl=excluded.realized_pnl""",
                    [order["account_id"], order["ticker"], new_held, new_cost, new_realized])
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def apply_corporate_action(self, account_id: str, ticker: str, event: Mapping[str, Any]) -> None:
        event = dict(event)
        if not event.get("event_id") or not event.get("source") or event.get("type") not in {"split", "dividend"}:
            raise ValueError("corporate action requires id, source and supported type")
        event_json = _json(event)
        with _LOCK, self._connect() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                previous = con.execute("SELECT payload_json FROM paper_account_events WHERE account_id=? AND event_id=?", [account_id, event["event_id"]]).fetchone()
                if previous:
                    if previous[0] != event_json:
                        raise ValueError("event identity already exists with different content")
                    con.execute("COMMIT")
                    return
                position = con.execute("SELECT quantity,total_cost,last_price FROM paper_positions WHERE account_id=? AND ticker=?", [account_id, ticker]).fetchone()
                if position is None:
                    raise ValueError("corporate action requires a held position")
                quantity, total_cost, last_price = position
                if event["type"] == "split":
                    ratio = _positive(event.get("ratio"), "split ratio")
                    con.execute("UPDATE paper_positions SET quantity=?,last_price=? WHERE account_id=? AND ticker=?", [quantity * ratio, last_price / ratio if last_price is not None else None, account_id, ticker])
                else:
                    payment = _positive(event.get("amount_per_share"), "dividend", zero=True) * quantity
                    con.execute("UPDATE paper_accounts SET cash=cash+? WHERE account_id=?", [payment, account_id])
                con.execute("INSERT INTO paper_account_events VALUES (?,?,?,?,?,?)", [account_id, event["event_id"], ticker, event["type"], event_json, datetime.now(timezone.utc)])
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

    def get_account(self, account_id: str) -> dict[str, Any] | None:
        with _LOCK, self._connect() as con:
            row = con.execute("SELECT market,currency,initial_cash,cash FROM paper_accounts WHERE account_id=?", [account_id]).fetchone()
            if row is None:
                return None
            rows = con.execute("SELECT ticker,quantity,total_cost,realized_pnl,last_price,last_price_at FROM paper_positions WHERE account_id=? ORDER BY ticker", [account_id]).fetchall()
        return {"account_id": account_id, "market": row[0], "currency": row[1], "initial_cash": row[2], "cash": row[3],
                "simulated": True, "positions": {r[0]: {"quantity": r[1], "total_cost": r[2], "realized_pnl": r[3], "last_price": r[4], "last_price_at": r[5]} for r in rows}}

    def get_orders(self, account_id: str) -> list[dict[str, Any]]:
        with _LOCK, self._connect() as con:
            rows = con.execute("""SELECT o.payload_json,o.status,f.filled_quantity,f.cancelled_quantity,f.remaining_quantity,o.risk_json
                FROM paper_orders o LEFT JOIN paper_fills f ON o.order_id=f.order_id
                WHERE o.account_id=? ORDER BY o.decision_session,o.order_id""", [account_id]).fetchall()
        return [{**json.loads(r[0]), "status": r[1], "filled_quantity": r[2], "cancelled_quantity": r[3],
                 "remaining_quantity": r[4], "risk_decision": json.loads(r[5])} for r in rows]

    def mark_to_market(self, account_id: str, session: str, prices: Mapping[str, Any], *, as_of: str) -> dict[str, Any]:
        """Record one immutable end-of-session valuation; missing prices block new exposure."""
        from backtest.verification import session_close

        day = date.fromisoformat(session)
        timestamp = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("valuation as_of must be timezone-aware")
        timestamp = timestamp.astimezone(timezone.utc)
        with _LOCK, self._connect() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                account = con.execute("SELECT market,currency,initial_cash,cash FROM paper_accounts WHERE account_id=?", [account_id]).fetchone()
                if account is None:
                    raise ValueError("unknown account")
                if timestamp < session_close(day, account[0]).astimezone(timezone.utc):
                    raise ValueError("valuation session has not closed")
                previous = con.execute("SELECT payload_json FROM paper_equity_snapshots WHERE account_id=? AND session=?", [account_id, session]).fetchone()
                if previous:
                    payload = json.loads(previous[0])
                    if payload.get("as_of") != timestamp.isoformat() or payload.get("input_prices") != dict(prices):
                        raise ValueError("existing session snapshot is immutable")
                    con.execute("COMMIT")
                    return payload
                last_session = con.execute("SELECT MAX(session) FROM paper_equity_snapshots WHERE account_id=?", [account_id]).fetchone()[0]
                if last_session is not None and day < last_session:
                    raise ValueError("existing or earlier session snapshot is immutable")
                rows = con.execute("""SELECT ticker,quantity,total_cost,realized_pnl,last_price FROM paper_positions
                    WHERE account_id=? AND quantity>0 ORDER BY ticker""", [account_id]).fetchall()
                cash = float(account[3])
                missing, unavailable, valuations = [], [], []
                for ticker, quantity, cost, realized, last_price in rows:
                    supplied = prices.get(ticker)
                    if supplied is not None:
                        mark = _positive(supplied, f"mark price for {ticker}")
                        con.execute("UPDATE paper_positions SET last_price=?,last_price_at=? WHERE account_id=? AND ticker=?", [mark, timestamp, account_id, ticker])
                    else:
                        missing.append(ticker)
                        mark = float(last_price) if last_price is not None else None
                    if mark is None:
                        unavailable.append(ticker)
                    valuations.append({"ticker": ticker, "quantity": quantity, "total_cost": cost,
                                       "realized_pnl": realized, "price": mark,
                                       "market_value": quantity * mark if mark is not None else None,
                                       "unrealized_pnl": quantity * mark - cost if mark is not None else None})
                equity = None if unavailable else cash + sum(v["market_value"] for v in valuations)
                unrealized = None if unavailable else sum(v["unrealized_pnl"] for v in valuations)
                history = con.execute("SELECT payload_json FROM paper_equity_snapshots WHERE account_id=? ORDER BY session", [account_id]).fetchall()
                prev = [json.loads(r[0]) for r in history]
                # Stale valuations are visible but do not update the risk peak.
                valid_equities = [float(item["equity"]) for item in prev if item.get("equity") is not None and not item.get("stale")]
                peak = max([float(account[2])] + valid_equities + ([equity] if equity is not None and not missing else []))
                drawdown = (peak - equity) / peak if equity is not None and peak else None
                prior_max = max([item.get("max_drawdown") or 0 for item in prev], default=0.0)
                max_drawdown = max(prior_max, drawdown or 0) if equity is not None and not missing else prior_max
                realized_total = con.execute("SELECT COALESCE(SUM(realized_pnl),0) FROM paper_positions WHERE account_id=?", [account_id]).fetchone()[0]
                payload = {"account_id": account_id, "session": session, "as_of": timestamp.isoformat(),
                           "market": account[0], "currency": account[1], "simulated": True,
                           "cash": cash, "equity": equity, "unrealized_pnl": unrealized,
                           "realized_pnl": float(realized_total), "input_prices": dict(prices),
                           "stale": bool(missing), "missing_prices": missing,
                           "new_exposure_allowed": not missing, "peak_equity": peak,
                           "drawdown": drawdown, "max_drawdown": max_drawdown, "positions": valuations}
                con.execute("INSERT INTO paper_equity_snapshots VALUES (?,?,?,?)", [account_id, session, _json(payload), timestamp])
                con.execute("COMMIT")
                return payload
            except Exception:
                con.execute("ROLLBACK")
                raise

    def get_equity(self, account_id: str) -> list[dict[str, Any]]:
        with _LOCK, self._connect() as con:
            rows = con.execute("SELECT payload_json FROM paper_equity_snapshots WHERE account_id=? ORDER BY session", [account_id]).fetchall()
        return [json.loads(row[0]) for row in rows]
