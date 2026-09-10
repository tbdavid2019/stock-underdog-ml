"""Last-known-good cache and persistence orchestration for security universes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from data.duckdb_manager import DuckDBManager


@dataclass
class UniverseSyncResult:
    source_id: str
    records: List[Dict[str, Any]]
    stale: bool
    snapshot_date: Optional[str]
    last_success_at: Optional[str]
    error: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.records)


class MarketUniverseSync:
    def __init__(
        self,
        provider,
        cache_path: Optional[str] = None,
        db_manager: Optional[DuckDBManager] = None,
        min_records: int = 1000,
    ):
        self.provider = provider
        self.cache_path = Path(cache_path or f"cache/universe/{provider.source_id}.json")
        self.db_manager = db_manager
        self.min_records = min_records

    def refresh(self) -> UniverseSyncResult:
        try:
            records = self._validate(self.provider.fetch_records())
            previous = self._load_cache()
            if previous and len(records) < max(self.min_records, int(len(previous["records"]) * 0.5)):
                raise ValueError(
                    f"validated universe shrank from {len(previous['records'])} to {len(records)} records"
                )
            now = datetime.now(timezone.utc).isoformat()
            snapshot_date = date.today().isoformat()
            payload = {
                "source_id": self.provider.source_id,
                "snapshot_date": snapshot_date,
                "last_success_at": now,
                "records": records,
            }
            self._atomic_write(payload)
            if self.db_manager:
                self.db_manager.save_market_universe_snapshot(snapshot_date, records)
                self.db_manager.save_universe_sync_run(self.provider.source_id, snapshot_date, False, len(records), None)
            return UniverseSyncResult(self.provider.source_id, records, False, snapshot_date, now)
        except Exception as exc:
            cached = self._load_cache()
            if cached and cached.get("records"):
                error = str(exc)
                if self.db_manager:
                    self.db_manager.save_market_universe_snapshot(
                        cached.get("snapshot_date"), cached["records"]
                    )
                    self.db_manager.save_universe_sync_run(
                        self.provider.source_id,
                        cached.get("snapshot_date"),
                        True,
                        len(cached["records"]),
                        error,
                    )
                return UniverseSyncResult(
                    self.provider.source_id,
                    cached["records"],
                    True,
                    cached.get("snapshot_date"),
                    cached.get("last_success_at"),
                    error,
                )
            raise RuntimeError(f"{self.provider.source_id} unavailable and no valid cache: {exc}") from exc

    def _validate(self, records: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen = set()
        for record in records or []:
            row = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            source_id = str(row.get("source_id") or self.provider.source_id).strip()
            local_symbol = str(row.get("local_symbol") or "").strip()
            normalized_symbol = str(row.get("normalized_symbol") or "").strip()
            if not local_symbol or not normalized_symbol:
                continue
            key = (source_id, local_symbol)
            if key in seen:
                continue
            seen.add(key)
            row["source_id"] = source_id
            row["local_symbol"] = local_symbol
            row["normalized_symbol"] = normalized_symbol
            normalized.append(row)
        if len(normalized) < self.min_records:
            raise ValueError(f"validated universe has {len(normalized)} records; minimum is {self.min_records}")
        return normalized

    def _load_cache(self) -> Optional[Dict[str, Any]]:
        try:
            if not self.cache_path.exists():
                return None
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("source_id") != self.provider.source_id or not isinstance(payload.get("records"), list):
                return None
            return payload
        except (OSError, ValueError, TypeError):
            return None

    def _atomic_write(self, payload: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{self.cache_path.name}.", dir=self.cache_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.cache_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def sync_providers(providers: Iterable[Any], cache_dir: str = "cache/universe", db_manager=None) -> List[UniverseSyncResult]:
    results = []
    for provider in providers:
        sync = MarketUniverseSync(
            provider,
            cache_path=os.path.join(cache_dir, f"{provider.source_id}.json"),
            db_manager=db_manager,
        )
        try:
            results.append(sync.refresh())
        except RuntimeError as exc:
            results.append(UniverseSyncResult(provider.source_id, [], True, None, None, str(exc)))
    return results
