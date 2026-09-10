#!/usr/bin/env python3
"""Sync non-Taiwan official security directories with per-source fallback."""

import logging
import os
import sys

# Allow direct execution from the repository root, matching the Taiwan sync script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.duckdb_manager import DuckDBManager
from data.market_universe import sync_providers
from data.universe_providers import (
    EuronextProvider,
    HkexProvider,
    JpxProvider,
    LseProvider,
    NasdaqTraderProvider,
    SseProvider,
    SzseProvider,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("global_universe_sync")


def main() -> int:
    db = DuckDBManager()
    results = sync_providers(
        [
            NasdaqTraderProvider(),
            HkexProvider(),
            JpxProvider(),
            SseProvider(),
            SzseProvider(),
            EuronextProvider(),
            LseProvider(),
        ],
        db_manager=db,
    )
    available = 0
    for result in results:
        if result.records:
            available += 1
            logger.info(
                "✅ %s 清冊: %s 檔 (%s, snapshot=%s)",
                result.source_id,
                result.count,
                "stale cache" if result.stale else "fresh",
                result.snapshot_date,
            )
        else:
            logger.warning("⚠️ %s 清冊目前不可用: %s", result.source_id, result.error)
    return 0 if available else 1


if __name__ == "__main__":
    sys.exit(main())
