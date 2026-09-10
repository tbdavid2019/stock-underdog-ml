# 全球證券清冊與最後成功快照 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可擴充至全球交易所的證券清冊同步平台，接入 TWSE/TPEx、NASDAQ Trader、HKEX、JPX、SSE、SZSE、Euronext 與 LSE，並以最後成功快照 cache 確保上游某天失敗時仍可回答。

**Architecture:** 以 `market_universe` 為全球標準資料表，將 `market/exchange/local_symbol/normalized_symbol/name_local/name_en/isin/security_type/listing_status` 與來源欄位分離，避免綁死任何單一市場格式。每個 provider 擁有自己的 source id、解析器、快取與同步狀態；全球同步一次依序更新所有 provider，只有完整且通過驗證的來源快照才原子提交。上游失敗、空回應或格式異常時保留該來源最後成功快照並標記 `stale=true`；不同來源互不覆蓋。

**Tech Stack:** Python 3.12, requests, pandas, DuckDB, unittest, FastAPI, shell cron。

---

### Task 0: Define the global provider contract before implementation

**Files:**
- Create: `data/universe_providers.py`
- Create: `test/test_universe_providers.py`
- Modify: `docs/superpowers/plans/2026-09-10-tw-market-universe-cache-plan.md`

- [ ] **Step 1: Write failing provider contract tests**

Define tests for normalized records from Taiwan, NASDAQ pipe-delimited files, and HKEX tabular rows; assert each provider returns the same canonical fields while preserving local symbol, local name, exchange, security type, and source metadata.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `python -m unittest test/test_universe_providers.py -v`

Expected: import/API failures because the canonical record and providers do not exist yet.

- [ ] **Step 3: Implement the provider interface and three providers**

Add a canonical `UniverseRecord`, a provider protocol/base class, `TwseTpexProvider` using the existing official fetcher, `NasdaqTraderProvider` parsing `nasdaqlisted.txt` and `otherlisted.txt`, and `HkexProvider` accepting the official securities-list download as CSV/XLSX/tabular input. Keep source URLs/configuration in provider classes and leave future JPX/SSE/SZSE/LSE/Euronext adapters compatible with the same interface.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m unittest test/test_universe_providers.py -v`

Expected: all provider normalization tests pass.

### Task 1: Lock the global snapshot and per-source fallback contract with tests

**Files:**
- Create: `test/test_market_universe.py`
- Create: `data/market_universe.py`

- [ ] **Step 1: Write failing tests**

Cover these behaviors: complete provider responses create fresh source snapshots; an empty/partial upstream response returns that source's last successful JSON cache with `stale=True`; a valid cache is never overwritten by a failed refresh; malformed rows and duplicate `(source_id, local_symbol)` keys are rejected or normalized.

- [ ] **Step 2: Run the focused test and verify the expected failure**

Run: `python -m unittest test/test_market_universe.py -v`

Expected: import/API failures because the dedicated synchronizer does not exist yet.

- [ ] **Step 3: Implement the minimal synchronizer**

Add `MarketUniverseSync` with per-source results containing `records`, `stale`, `snapshot_date`, `last_success_at`, and `error`. Use `cache/universe/<source_id>.json` by default, write through a temporary file plus `os.replace`, and only persist complete validated snapshots. On fallback, read the previous valid snapshot without modifying it.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m unittest test/test_market_universe.py -v`

Expected: all new cache and fallback tests pass.

### Task 2: Persist the global universe and source health independently in DuckDB

**Files:**
- Modify: `data/duckdb_manager.py:90-125, 816-850`
- Modify: `test/test_market_universe.py`

- [ ] **Step 1: Add a failing DuckDB persistence test**

Assert that a source snapshot can be saved idempotently by `(source_id, snapshot_date, local_symbol)` and queried with market, exchange, local code, normalized symbol, names, security type, and source status intact.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m unittest test/test_market_universe.py -v`

Expected: missing table or persistence method failure.

- [ ] **Step 3: Add schema and persistence methods**

Create `market_universe` with `source_id`, `market`, `exchange`, `snapshot_date`, `local_symbol`, `normalized_symbol`, `name_local`, `name_en`, `isin`, `security_type`, `listing_status`, `market_category`, `financial_status`, `board_lot`, `source`, and `updated_at`; add `universe_sync_runs` for fresh/stale/error status. Add batch save and latest-source query methods; keep historical snapshots and replace only the same source/date.

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python -m unittest test/test_market_universe.py -v`

Expected: all focused tests pass.

### Task 3: Connect providers, cache-aware synchronization, and daily command

**Files:**
- Modify: `data/market_universe.py`
- Modify: `scripts/sync_twse_market.py`
- Modify: `test/test_market_universe.py`

- [ ] **Step 1: Add a failing integration-style test**

Mock the provider network calls, assert TWSE+TPEx both are required for a fresh Taiwan source snapshot, assert NASDAQ and HKEX failures fall back independently, and assert fresh Taiwan records update both `market_universe` and `tw_daily_bars`.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python -m unittest test/test_market_universe.py -v`

Expected: the command does not yet route the official fetcher through the cache-aware synchronizer.

- [ ] **Step 3: Implement the shared daily sync path**

Have the synchronization command run the enabled providers and write their source snapshots. A fresh Taiwan run also saves daily bars; a fallback run restores cached records and reports stale status while returning success. If a source has no cache on its first failed run, return an explicit unavailable error for that source. Keep the current official endpoints and ticker normalization.

- [ ] **Step 4: Run focused and existing TWSE tests**

Run: `python -m unittest test/test_market_universe.py test/test_twse_fetcher.py test/test_stock_cache.py -v`

Expected: all tests pass.

### Task 4: Make scheduled execution use the same cache-aware command

**Files:**
- Modify: `docker/crontab:9-12`
- Modify: `run_daily.sh:116-149` only if needed for consistent exit/status handling
- Modify: `.github/workflows/tw_stock_daily_sync.yml` only if needed for the new table artifact

- [ ] **Step 1: Add a testable command contract**

Use shell syntax that runs `scripts/sync_twse_market.py` before the Taiwan analysis and prevents analysis from silently proceeding without either a fresh or cached universe.

- [ ] **Step 2: Implement the schedule wiring**

Ensure Docker cron invokes the synchronizer before `main.py --market tw`; keep the US schedule from doing unnecessary TWSE/TPEx work. Preserve the existing GitHub Actions sync and DuckDB artifact behavior.

- [ ] **Step 3: Verify shell syntax and command references**

Run: `bash -n run_daily.sh docker/entrypoint.sh` and `rg -n "sync_twse_market|main.py --market tw" run_daily.sh docker/crontab .github/workflows/tw_stock_daily_sync.yml`

Expected: valid shell and an explicit pre-analysis sync in both local and container Taiwan schedules.

### Task 5: Expose freshness and document the user-visible behavior

**Files:**
- Modify: `api/routes/market.py`
- Modify: `README.md`
- Modify: `docs/CHANGELOG.md`
- Modify: `test/test_api_market.py` or `test/test_market_universe.py`

- [ ] **Step 1: Add a failing API contract test**

Assert the universe response includes count, snapshot date, stale flag, cache age/last success metadata, and normalized items.

- [ ] **Step 2: Implement the read endpoint**

Add `GET /api/v1/market/universe` backed by DuckDB/cache metadata. Return explicit stale metadata instead of presenting an old snapshot as fresh.

- [ ] **Step 3: Update documentation**

Document the new table, cache path, daily refresh order, fallback behavior, first-run failure behavior, and that the count is measured from the official response rather than hardcoded to 2,234.

- [ ] **Step 4: Run the complete verification suite**

Run: `python -m unittest test/test_market_universe.py test/test_twse_fetcher.py test/test_stock_cache.py test/test_api_market.py -v`, `python -m compileall data scripts api`, and `bash -n run_daily.sh docker/entrypoint.sh`.

Expected: zero test failures, successful compilation, and valid shell syntax.
