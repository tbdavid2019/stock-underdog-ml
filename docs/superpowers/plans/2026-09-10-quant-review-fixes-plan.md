# Quant Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the recent Polymarket/TimesFM/API/CI regressions and commit synchronized code, tests, and documentation on `main`.

**Architecture:** Keep the existing service boundaries. Normalize Polymarket data inside `PolymarketService`, expose one stable response shape to REST/MCP/WebMCP, and make the frontend consume that shape. Keep TimesFM forecasts and strategy decisions horizon-based. Use a PR-capable yfinance workflow without skipping post-update CI.

**Tech Stack:** Python 3.12, FastAPI, unittest, Vue-in-template JavaScript, GitHub Actions, GitHub CLI.

---

### Task 1: Lock the Polymarket contract with regression tests

**Files:**
- Modify: `test/test_polymarket_service.py`
- Modify: `test/test_api.py`

- [x] **Step 1: Add tests for percentage output and frontend fields**

Use a fixture whose `outcomes` and `outcomePrices` are JSON strings and assert `probability`, `top_outcome`, and percentage values are present.

- [x] **Step 2: Add a test that `category="all"` returns the same markets as no category**

Call `get_macro_sentiment(force_refresh=True, category="all")` with the existing mocked market fixture and assert the result count is non-zero.

- [x] **Step 3: Add tests for failed upstream and stale cache semantics**

Mock `_fetch_raw_markets` to return no data and assert the response is not falsely reported as a fresh successful snapshot. Add a valid previous result fixture and assert it is marked stale when reused.

- [x] **Step 4: Run only the new tests and confirm they fail**

Run `python -m unittest test/test_polymarket_service.py -v`; expected failures are the old frontend field contract, `all` behavior, and failure semantics.

### Task 2: Normalize Polymarket service output and fallback behavior

**Files:**
- Modify: `data/polymarket_service.py:80-313`

- [x] **Step 1: Add explicit cache metadata helpers**

Store the last valid result separately from a failed fetch, and return a failure envelope with `success=false`, `error`, `source`, and `stale` metadata when no valid result is available.

- [x] **Step 2: Normalize category input**

Treat `None`, an empty string, and `all` as no category filter; reject or return an explicit empty result for unknown categories rather than silently hiding all markets.

- [x] **Step 3: Normalize market payload fields defensively**

Accept list or JSON-string outcomes/prices, parse numeric fields per market, derive the highest-probability outcome, and skip malformed markets without aborting the complete snapshot.

- [x] **Step 4: Keep source and fetch route accurate**

Return `2md_reader` for a successful 2MD fetch and `doh_direct` for direct fallback. Do not hardcode a source that claims both routes were used.

- [x] **Step 5: Scope and restore the urllib3 fallback hook**

Use a session-scoped connection customization or a narrowly scoped adapter implementation so DoH fallback cannot mutate process-global connection behavior for unrelated requests.

- [x] **Step 6: Run the Polymarket tests and confirm green**

Run `python -m unittest test/test_polymarket_service.py -v` and verify all tests pass.

### Task 3: Make UI and WebMCP consume the stable Polymarket contract

**Files:**
- Modify: `api/templates/index.html:566-629,2939-2955`
- Modify: `api/routes/macro.py:89-98`

- [x] **Step 1: Render percentage values without a second conversion**

Render `fed_real_money_odds` directly as percentage values and use `m.probability`/`m.top_outcome` for market cards.

- [x] **Step 2: Stop sending the sentinel `all` as a real category**

Omit the query parameter for the default WebMCP category or map `all` to no filter.

- [x] **Step 3: Add API assertions for the public response shape**

Extend `test/test_api.py` to assert the route forwards the category and refresh flags and returns the stable envelope.

- [x] **Step 4: Run focused API tests**

Run `python -m unittest test/test_api.py -v` and verify all tests pass.

### Task 4: Align TimesFM calculation with the documented strategy horizon

**Files:**
- Modify: `models/timesfm_model.py:142-185`
- Modify: `strategies/timesfm.py:124-159`
- Modify: `test/test_timesfm_strategy.py`

- [x] **Step 1: Add a failing test distinguishing day-one and horizon decisions**

Create a forecast where day one potential is below the hit threshold but horizon P50 potential is above it; assert the strategy decision follows the documented horizon contract.

- [x] **Step 2: Add a failing test for P50-based risk/reward**

Use P10/P50/P90 values where P90 would pass but P50 would not; assert only P50 determines `risk_reward_ratio` and hit status.

- [x] **Step 3: Implement horizon P50/P10 calculations**

Calculate potential and risk/reward from horizon quantiles while retaining day-one values as separately named diagnostic fields.

- [x] **Step 4: Run TimesFM tests**

Run `python -m unittest test/test_timesfm_strategy.py -v` and verify all tests pass.

### Task 5: Centralize version metadata and synchronize docs

**Files:**
- Modify: `core/config.py` or add a small version module used by `api/main.py`, `api/schemas.py`, and the template
- Modify: `api/main.py:14-29,589-598`
- Modify: `api/schemas.py:86-90`
- Modify: `api/templates/index.html:189-192`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/CHANGELOG.md`

- [x] **Step 1: Add a single application version constant**

Expose version `2.4.0` from one module and use it in FastAPI metadata, health response, and homepage badge.

- [x] **Step 2: Document all user-visible fixes and limitations**

Record the Polymarket contract, stale/error behavior, TimesFM definition, and CI update flow in the required changelog files and README sections.

- [x] **Step 3: Run metadata/API tests**

Run `python -m unittest test/test_api.py -v` and assert all public version surfaces match.

### Task 6: Make automated dependency updates verifiable

**Files:**
- Modify: `.github/workflows/yfinance_autoupdate.yml`
- Modify: `.github/workflows/docker-ci-cd.yml`
- Add or modify: `CONTRIBUTING.md` or `README.md`

- [x] **Step 1: Add a branch-safe update workflow**

Have scheduled automation create/update a dependency branch or pull request instead of pushing directly to `main`.

- [x] **Step 2: Remove `[skip ci]` from dependency commits**

Ensure a dependency change runs the normal test and Docker workflow before merge.

- [x] **Step 3: Restrict workflow permissions**

Keep read-only permissions for test jobs and grant package write only to the image publishing job.

- [x] **Step 4: Document required branch protection settings**

Document requiring the CI check before merging `main`; do not attempt to mutate repository settings automatically.

### Task 7: Full verification and commit

**Files:**
- All files above

- [x] **Step 1: Run syntax and focused tests**

Run `python -m py_compile data/polymarket_service.py data/macro.py api/main.py api/routes/macro.py models/timesfm_model.py strategies/timesfm.py` and the focused unittest commands.

- [x] **Step 2: Run the full suite**

Run `python -m unittest discover -s test -p 'test_*.py'` and resolve every failure.

- [x] **Step 3: Review the diff and preserve unrelated user changes**

Run `git status --short`, `git diff --check`, and `git diff --stat`; stage only this implementation, tests, and documentation.

- [x] **Step 4: Commit the implementation**

Use a descriptive commit such as `fix(review): align Polymarket, TimesFM, and CI contracts`.

- [x] **Step 5: Push `main` and monitor GitHub Actions**

Run `git push origin main`, then inspect the resulting workflow run. Report exact run status and any remaining external PR Agent limitation.
