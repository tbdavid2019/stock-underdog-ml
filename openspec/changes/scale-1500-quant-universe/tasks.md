# Implementation Tasks: Scale 1,500 Stock Quant Universe

- [x] 1. Core Data Layer Implementation
  - [x] 1.1 Implement `StockFetcher.get_tw_universe(limit=1000)` and `get_tw1000_stocks()` in `data/fetcher.py` with dynamic DuckDB ranking (`market_universe` + `tw_daily_bars`).
  - [x] 1.2 Implement dynamic delisting and zero-volume filtering in `data/fetcher.py`.
  - [x] 1.3 Update `StockFetcher.get_sp500_stocks()` to default to full 503 constituents + space/AI leaders.
  - [x] 1.4 Add chunked batch fetching (chunks of 200) in `StockFetcher.get_stock_data_batch()`.
  - [x] 1.5 Add `StockFetcher.get_index_name_map()` support for `台股活躍精選` and `台股1000`.

- [x] 2. Pipeline Orchestrator & CLI Updates
  - [x] 2.1 Update `PipelineOrchestrator.run_all_indices()` to route `--market tw` to `台灣50`, `台灣中型100`, and `台股活躍精選` (850 disjoint stocks), totaling 1,000 unique stocks.
  - [x] 2.2 Support `--index 台股1000` / `TW1000` for monolithic execution.
  - [x] 2.3 Verify `--market us` pulls all 500+ US stocks.

- [x] 3. API & Web UI Integration
  - [x] 3.1 Update web dashboard dropdown in `api/templates/partials/tab_dashboard.html` to include `台股活躍精選 (TW-Active)` and `台股全市場1000 (TW1000)`.
  - [x] 3.2 Ensure API route `api/routes/predictions.py` handles new index names properly.

- [x] 4. Automated Tests & Validation
  - [x] 4.1 Create `test/test_scaled_universe.py` testing TW1000 selection, delisting filtering, US500 fetching, and chunked download.
  - [x] 4.2 Run existing regression test suite (`test_fetcher.py`, `test_orchestrator.py`, etc.).

- [ ] 5. Code Review, Commit, & Server Deployment
  - [ ] 5.1 Run dual-code-review protocol (Codex CLI / OpenCode).
  - [ ] 5.2 Update `CHANGELOG.md` and `docs/CHANGELOG.md`.
  - [ ] 5.3 Commit, push to GitHub, and deploy to production server `10.9.0.99`.
