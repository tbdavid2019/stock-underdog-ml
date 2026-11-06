"""Append-only DuckDB storage for versioned market covariate observations."""
from __future__ import annotations

import json
import math
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import duckdb


class CovariateStore:
    """Owns independent covariate tables; does not alter DuckDBManager schema."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            from core.config import config
            db_path = config.db.DUCKDB_PATH
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        return duckdb.connect(self.db_path)

    @contextmanager
    def _write_lock(self):
        """Serialize DuckDB access across threads and processes for this database."""
        from core.duckdb_lock import duckdb_write_lock
        with duckdb_write_lock(self.db_path):
            yield

    @staticmethod
    def _json(value):
        try:
            return json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"strict JSON validation failed: {exc}") from exc

    def _init_schema(self):
        with self._write_lock(), self._connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS covariate_observations (
                    series_id VARCHAR NOT NULL, observation_date DATE NOT NULL,
                    value DOUBLE, raw_value DOUBLE, unit VARCHAR, source VARCHAR NOT NULL,
                    published_at TIMESTAMPTZ, available_at TIMESTAMPTZ,
                    ingested_at TIMESTAMPTZ NOT NULL, revision VARCHAR NOT NULL,
                    quality VARCHAR DEFAULT 'ok', metadata_json VARCHAR DEFAULT '{}',
                    PRIMARY KEY(series_id, observation_date, source, revision)
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS covariate_sync_runs (
                    sync_id VARCHAR PRIMARY KEY, started_at TIMESTAMPTZ NOT NULL,
                    finished_at TIMESTAMPTZ, requested_json VARCHAR NOT NULL,
                    result_json VARCHAR, status VARCHAR NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS feature_profiles (
                    profile_id VARCHAR PRIMARY KEY, market VARCHAR NOT NULL,
                    definition_json VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    snapshot_id VARCHAR PRIMARY KEY, ticker VARCHAR NOT NULL,
                    as_of TIMESTAMPTZ NOT NULL, profile_id VARCHAR NOT NULL,
                    payload_json VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL
                )
            """)

    def upsert_observations(self, observations: Iterable) -> int:
        rows = list(observations)
        if not rows:
            return 0
        now = datetime.now(timezone.utc)
        for o in rows:
            if o.value is not None and not math.isfinite(float(o.value)):
                raise ValueError("covariate values must be finite")
            if getattr(o, 'raw_value', None) is not None and not math.isfinite(float(o.raw_value)):
                raise ValueError("raw covariate values must be finite")
            self._json(getattr(o, 'metadata', {}) or {})
        with self._write_lock():
            con = self._connect()
            try:
                con.execute("BEGIN TRANSACTION")
                inserted = 0
                for o in rows:
                    exists = con.execute("""SELECT 1 FROM covariate_observations
                    WHERE series_id=? AND observation_date=? AND source=? AND revision=?""",
                    [o.series_id, o.observation_date, o.source, o.revision]).fetchone()
                    if exists:
                        continue
                    con.execute("""INSERT INTO covariate_observations
                    (series_id, observation_date, value, raw_value, unit, source,
                     published_at, available_at, ingested_at, revision, quality, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [o.series_id, o.observation_date, o.value, getattr(o, 'raw_value', o.value),
                     o.unit, o.source, o.published_at, o.available_at, now, o.revision,
                     getattr(o, 'quality', 'ok'), self._json(getattr(o, 'metadata', {}) or {})])
                    inserted += 1
                con.execute("COMMIT")
                return inserted
            except Exception:
                con.execute("ROLLBACK")
                raise
            finally:
                con.close()

    def get_observations(self, series_id: str, *, as_of: Optional[datetime] = None, include_revisions: bool = False):
        from data.covariates import Observation
        with self._write_lock(), self._connect() as con:
            if as_of is None:
                rows = con.execute("""SELECT series_id, observation_date, value, source, unit,
                    published_at, available_at, revision, raw_value, quality, metadata_json, ingested_at
                    FROM covariate_observations WHERE series_id=? ORDER BY observation_date, ingested_at""", [series_id]).fetchall()
            else:
                rows = con.execute("""SELECT series_id, observation_date, value, source, unit,
                    published_at, available_at, revision, raw_value, quality, metadata_json, ingested_at
                    FROM covariate_observations WHERE series_id=? AND available_at <= ?
                    ORDER BY observation_date, available_at, ingested_at""", [series_id, as_of]).fetchall()
        latest = {}
        for row in rows:
            key = (row[0], row[1])
            if include_revisions:
                latest[(key, row[3], row[7])] = row
            else:
                latest[key] = row
        return [Observation(series_id=r[0], observation_date=r[1], value=r[2], source=r[3], unit=r[4],
                            published_at=r[5], available_at=r[6], revision=r[7], raw_value=r[8],
                            quality=r[9], metadata=json.loads(r[10] or '{}'), ingested_at=r[11]) for r in latest.values()]

    def latest_observation_date(self, series_id: str):
        with self._connect() as con:
            row=con.execute("SELECT MAX(observation_date) FROM covariate_observations WHERE series_id=?",[series_id]).fetchone()
        return row[0] if row else None

    def save_snapshot(self, snapshot_id, ticker, as_of, profile_id, payload):
        payload_json=self._json(payload)
        with self._write_lock(), self._connect() as con:
            definition={"profile_id":profile_id,"features":payload.get("features",[]),"observations_are_asof":True}
            definition_json=self._json(definition)
            con.execute("INSERT OR IGNORE INTO feature_profiles VALUES (?, ?, ?, ?)",
                        [profile_id, profile_id.split("-v",1)[0], definition_json, datetime.now(timezone.utc)])
            con.execute("""INSERT OR REPLACE INTO feature_snapshots VALUES (?, ?, ?, ?, ?, ?)""",
                        [snapshot_id, ticker, as_of, profile_id, payload_json, datetime.now(timezone.utc)])

    def record_sync_run(self, sync_id, series_ids, result, status):
        now=datetime.now(timezone.utc)
        requested_json=self._json(list(series_ids))
        result_json=self._json(result)
        with self._write_lock(), self._connect() as con:
            con.execute("""INSERT OR REPLACE INTO covariate_sync_runs
                (sync_id,started_at,finished_at,requested_json,result_json,status)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [sync_id,now,now,requested_json,result_json,status])
