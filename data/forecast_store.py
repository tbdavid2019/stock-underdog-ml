"""Append-only local DuckDB evidence store for forecasts and settlements."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Mapping

import duckdb

_WRITE_LOCK = threading.RLock()


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be strict JSON (finite numbers only)") from exc


def _as_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


class ForecastStore:
    def __init__(self, db_path: str | Path, read_only: bool = False):
        self.db_path = str(db_path)
        self.read_only = read_only
        if not read_only:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        if not read_only:
            with self._lock(), self._connect() as con:
                con.execute("""CREATE TABLE IF NOT EXISTS model_runs (
                run_id VARCHAR PRIMARY KEY, payload_json VARCHAR NOT NULL, created_at TIMESTAMPTZ DEFAULT current_timestamp)""")
                con.execute("""CREATE TABLE IF NOT EXISTS forecast_records (
                forecast_id VARCHAR PRIMARY KEY, run_id VARCHAR NOT NULL, ticker VARCHAR NOT NULL, market VARCHAR NOT NULL,
                model_revision VARCHAR NOT NULL, target_date DATE NOT NULL, horizon INTEGER NOT NULL,
                issued_at VARCHAR NOT NULL, last_complete_bar DATE NOT NULL, payload_json VARCHAR NOT NULL,
                created_at TIMESTAMPTZ DEFAULT current_timestamp)""")
                con.execute("""CREATE TABLE IF NOT EXISTS forecast_actuals (
                actual_version_id VARCHAR PRIMARY KEY, forecast_id VARCHAR NOT NULL, status VARCHAR NOT NULL,
                reason VARCHAR, actual DOUBLE, target_date DATE, source VARCHAR, source_revision VARCHAR,
                price_basis VARCHAR, payload_json VARCHAR NOT NULL, verified_at VARCHAR NOT NULL,
                invalidated_at VARCHAR)""")
                con.execute("""CREATE TABLE IF NOT EXISTS performance_snapshots (
                snapshot_id VARCHAR PRIMARY KEY, payload_json VARCHAR NOT NULL, invalidated_at VARCHAR)""")
                con.execute("""CREATE TABLE IF NOT EXISTS health_snapshots (
                snapshot_id VARCHAR PRIMARY KEY, payload_json VARCHAR NOT NULL, invalidated_at VARCHAR)""")
                con.execute("""CREATE TABLE IF NOT EXISTS supabase_outbox (
                event_id VARCHAR PRIMARY KEY, payload_json VARCHAR NOT NULL, status VARCHAR NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, last_error VARCHAR, created_at TIMESTAMPTZ DEFAULT current_timestamp)""")

    def _connect(self):
        return duckdb.connect(self.db_path, read_only=self.read_only)

    def _lock(self):
        if self.read_only:
            return _WRITE_LOCK
        # Reuse the repository-wide writer lock when the manager is available.
        try:
            from data.duckdb_manager import DuckDBManager
            return DuckDBManager._WRITE_LOCK
        except Exception:
            return _WRITE_LOCK

    @staticmethod
    def _identity(prefix: str, value: Mapping[str, Any]) -> str:
        return prefix + "_" + hashlib.sha256(_json(value).encode()).hexdigest()[:32]

    def record_run(self, run: Mapping[str, Any]) -> str:
        payload = dict(run)
        run_id = str(payload.get("run_id") or self._identity("run", payload))
        payload["run_id"] = run_id
        encoded = _json(payload)
        with self._lock(), self._connect() as con:
            row = con.execute("SELECT payload_json FROM model_runs WHERE run_id=?", [run_id]).fetchone()
            if row and row[0] != encoded:
                raise ValueError("run_id already exists with different immutable content")
            con.execute("INSERT OR IGNORE INTO model_runs(run_id,payload_json) VALUES (?,?)", [run_id, encoded])
        return run_id

    def record_forecast(self, forecast: Mapping[str, Any]) -> str:
        p = dict(forecast)
        if "prediction" not in p and "predicted_price" in p:
            p["prediction"] = p["predicted_price"]
        if "last_complete_bar" not in p and "last_bar_date" in p:
            p["last_complete_bar"] = p["last_bar_date"]
        if "last_complete_bar" not in p and "data_cutoff" in p:
            p["last_complete_bar"] = str(p["data_cutoff"])[:10]
        if "current_price" not in p:
            p["current_price"] = p.get("price_basis_value")
        required = ("run_id", "ticker", "market", "model_revision", "target_date", "horizon", "issued_at", "last_complete_bar", "prediction")
        missing = [key for key in required if key not in p]
        if missing:
            raise ValueError("forecast missing required fields: " + ", ".join(missing))
        if int(p["horizon"]) < 1:
            raise ValueError("horizon must be positive")
        identity_fields = ("run_id", "ticker", "model_revision", "config_hash", "target_date", "horizon", "target_type", "feature_profile")
        fid = str(p.get("forecast_id") or self._identity("fc", {k: p[k] for k in identity_fields if k in p}))
        p["forecast_id"] = fid
        encoded = _json(p)
        with self._lock(), self._connect() as con:
            if not con.execute("SELECT 1 FROM model_runs WHERE run_id=?", [str(p["run_id"])]).fetchone():
                raise ValueError("forecast run_id does not exist")
            row = con.execute("SELECT payload_json FROM forecast_records WHERE forecast_id=?", [fid]).fetchone()
            if row and row[0] != encoded:
                raise ValueError("forecast identity already exists with different immutable content")
            con.execute("""INSERT OR IGNORE INTO forecast_records
                (forecast_id,run_id,ticker,market,model_revision,target_date,horizon,issued_at,last_complete_bar,payload_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", [fid, str(p["run_id"]), str(p["ticker"]), str(p["market"]).upper(), str(p["model_revision"]),
                 p["target_date"], int(p["horizon"]), str(p["issued_at"]), p["last_complete_bar"], encoded])
        return fid

    def list_forecasts(self, filters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        allowed = {"forecast_id", "run_id", "ticker", "market", "model_revision", "target_date"}
        if set(filters) - allowed:
            raise ValueError("unsupported forecast filter")
        sql, args = "SELECT payload_json FROM forecast_records", []
        if filters:
            sql += " WHERE " + " AND ".join(f"{key}=?" for key in filters)
            args = list(filters.values())
        sql += " ORDER BY created_at, forecast_id"
        with self._lock(), self._connect() as con:
            return [json.loads(r[0]) for r in con.execute(sql, args).fetchall()]

    def get_forecast(self, forecast_id: str) -> dict[str, Any] | None:
        rows = self.list_forecasts({"forecast_id": forecast_id})
        return rows[0] if rows else None

    def _write_actual(self, forecast_id: str, status: str, actual: Mapping[str, Any]) -> str:
        p = dict(actual)
        encoded = _json(p)
        stable_keys = ("actual", "actual_price", "target_date", "source", "source_revision", "price_basis", "reason")
        stable = {key: p[key] for key in stable_keys if key in p}
        version = str(p.get("actual_version_id") or self._identity("act", {"forecast_id": forecast_id, "status": status, "data": stable}))
        verified_at = str(p.get("verified_at") or dt.datetime.now(dt.timezone.utc).isoformat())
        with self._lock(), self._connect() as con:
            con.execute("BEGIN TRANSACTION")
            try:
                if not con.execute("SELECT 1 FROM forecast_records WHERE forecast_id=?", [forecast_id]).fetchone():
                    raise KeyError(f"unknown forecast_id: {forecast_id}")
                existing = con.execute("SELECT payload_json,status FROM forecast_actuals WHERE actual_version_id=?", [version]).fetchone()
                if existing:
                    old = json.loads(existing[0])
                    old_stable = {key: old[key] for key in stable_keys if key in old}
                    if existing[1] != status or old_stable != stable:
                        raise ValueError("actual version identity already has different content")
                    con.execute("COMMIT")
                    return version
                if status == "settled":
                    con.execute("UPDATE forecast_actuals SET invalidated_at=? WHERE forecast_id=? AND status='settled' AND invalidated_at IS NULL", [verified_at, forecast_id])
                    con.execute("UPDATE performance_snapshots SET invalidated_at=? WHERE invalidated_at IS NULL", [verified_at])
                    con.execute("UPDATE health_snapshots SET invalidated_at=? WHERE invalidated_at IS NULL", [verified_at])
                con.execute("""INSERT INTO forecast_actuals(actual_version_id,forecast_id,status,reason,actual,target_date,source,source_revision,price_basis,payload_json,verified_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)""", [version, forecast_id, status, p.get("reason"), p.get("actual"), p.get("target_date"), p.get("source"), p.get("source_revision"), p.get("price_basis"), encoded, verified_at])
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        return version

    def settle_forecast(self, forecast_id: str, actual: Mapping[str, Any]) -> str:
        actual = dict(actual)
        if "actual" not in actual and "actual_price" in actual:
            actual["actual"] = actual["actual_price"]
        if "actual" not in actual or "source" not in actual or "price_basis" not in actual:
            raise ValueError("settlement requires actual, source, and price_basis")
        return self._write_actual(forecast_id, "settled", actual)

    def mark_pending(self, forecast_id: str, reason: str, **details: Any) -> str:
        return self._write_actual(forecast_id, "pending", {"reason": reason, **details})

    def mark_invalid(self, forecast_id: str, reason: str, **details: Any) -> str:
        return self._write_actual(forecast_id, "invalid", {"reason": reason, **details})

    def list_actuals(self, forecast_id: str, include_invalidated: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT payload_json,status,reason,actual_version_id,invalidated_at FROM forecast_actuals WHERE forecast_id=?"
        if not include_invalidated:
            sql += " AND invalidated_at IS NULL"
        sql += " ORDER BY verified_at"
        with self._lock(), self._connect() as con:
            result = []
            for r in con.execute(sql, [forecast_id]).fetchall():
                payload = json.loads(r[0])
                result.append({**payload, "actual_price": payload.get("actual_price", payload.get("actual")),
                               "status": r[1], "reason": r[2], "actual_version_id": r[3], "invalidated_at": r[4]})
            return result

    def record_snapshot(self, snapshot: Mapping[str, Any]) -> str:
        p = dict(snapshot)
        sid = str(p.get("snapshot_id") or self._identity("snap", p))
        with self._lock(), self._connect() as con:
            con.execute("INSERT OR IGNORE INTO performance_snapshots(snapshot_id,payload_json) VALUES (?,?)", [sid, _json(p)])
        return sid

    def record_health_snapshot(self, snapshot: Mapping[str, Any]) -> str:
        p = dict(snapshot)
        sid = str(p.get("snapshot_id") or self._identity("health", p))
        encoded = _json(p)
        with self._lock(), self._connect() as con:
            row = con.execute("SELECT payload_json FROM health_snapshots WHERE snapshot_id=?", [sid]).fetchone()
            if row and row[0] != encoded:
                raise ValueError("health snapshot_id already exists with different content")
            con.execute("INSERT OR IGNORE INTO health_snapshots(snapshot_id,payload_json) VALUES (?,?)", [sid, encoded])
        return sid

    def list_invalidated_snapshots(self) -> list[dict[str, Any]]:
        with self._lock(), self._connect() as con:
            return [{"snapshot_id": r[0], "payload": json.loads(r[1]), "invalidated_at": r[2]} for r in con.execute("SELECT snapshot_id,payload_json,invalidated_at FROM performance_snapshots WHERE invalidated_at IS NOT NULL").fetchall()]

    def list_invalidated_health_snapshots(self) -> list[dict[str, Any]]:
        with self._lock(), self._connect() as con:
            return [{"snapshot_id": r[0], "payload": json.loads(r[1]), "invalidated_at": r[2]} for r in con.execute("SELECT snapshot_id,payload_json,invalidated_at FROM health_snapshots WHERE invalidated_at IS NOT NULL").fetchall()]

    def enqueue_supabase(self, event_id: str, payload: Mapping[str, Any]) -> None:
        encoded = _json(payload)
        with self._lock(), self._connect() as con:
            row = con.execute("SELECT payload_json FROM supabase_outbox WHERE event_id=?", [event_id]).fetchone()
            if row and row[0] != encoded:
                raise ValueError("outbox event_id already exists with different payload")
            con.execute("INSERT OR IGNORE INTO supabase_outbox(event_id,payload_json) VALUES (?,?)", [event_id, encoded])

    def list_outbox(self, status: str = "pending") -> list[dict[str, Any]]:
        with self._lock(), self._connect() as con:
            return [{"event_id": r[0], "payload": json.loads(r[1]), "status": r[2], "attempts": r[3], "last_error": r[4]}
                    for r in con.execute("SELECT event_id,payload_json,status,attempts,last_error FROM supabase_outbox WHERE status=? ORDER BY created_at", [status]).fetchall()]

    def mark_outbox_result(self, event_id: str, error: str | None = None) -> None:
        with self._lock(), self._connect() as con:
            status = "pending" if error else "delivered"
            # Never persist raw remote exception text; it can contain credentials or URLs.
            safe_error = "remote_write_failed" if error else None
            con.execute("UPDATE supabase_outbox SET status=?, attempts=attempts+1, last_error=? WHERE event_id=?", [status, safe_error, event_id])

    def query_records(self, as_of: str | None = None) -> list[dict[str, Any]]:
        """Return forecast/settlement rows visible at an optional ISO as-of cutoff."""
        try:
            with self._lock(), self._connect() as con:
                rows = con.execute("""SELECT f.payload_json,a.payload_json,a.status,a.verified_at,a.invalidated_at
                    FROM forecast_records f LEFT JOIN forecast_actuals a ON a.forecast_id=f.forecast_id
                    ORDER BY f.forecast_id,a.verified_at""").fetchall()
        except (duckdb.CatalogException, duckdb.IOException):
            # A newly deployed service may precede the first forecast run. This
            # is an empty evidence set, not an application crash or fake score.
            return []
        out = []
        cutoff = _as_utc(as_of) if as_of else None
        for forecast_json, actual_json, status, verified_at, invalidated_at in rows:
            f = json.loads(forecast_json)
            if not actual_json:
                out.append({"forecast": f, "actual": None, "status": "unverified"})
                continue
            a = json.loads(actual_json)
            if cutoff and (_as_utc(verified_at) > cutoff or (invalidated_at is not None and _as_utc(invalidated_at) <= cutoff)):
                continue
            if cutoff and a.get("actual_available_at") and _as_utc(a["actual_available_at"]) > cutoff:
                continue
            out.append({"forecast": f, "actual": a, "status": status, "verified_at": verified_at, "invalidated_at": invalidated_at})
        return out
