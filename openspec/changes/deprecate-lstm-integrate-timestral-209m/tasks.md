# Tasks: Deprecate LSTM & Integrate Mistral Timestral-209M for Dual Foundation Model Resonance

## 1. Foundation Model Engine & Strategy Implementation

- [x] 1.0 Update `requirements.txt` adding `timestral` (`git+https://github.com/mistralai/timestral.git`) and decoupling heavy legacy LSTM dependencies.
- [x] 1.1 Implement `models/timestral_model.py` with `TimestralWrapper` and singleton `get_timestral_model()`, supporting thread-safe lazy-loading (`threading.Lock`), uniform context collation (128 bars), CPU thread capping (`torch.set_num_threads(8)`), monotonic quantile clamping, and P10/P50/P90 quantile extraction. Verify with unit tests.
- [x] 1.2 Implement `strategies/timestral.py` registering `TimestralStrategy` in `StrategyRegistry`, synchronizing 5-day horizon (`horizon=5`), evaluating forward return, and computing bounded asymmetric risk-reward ratio with division-by-zero protection. Verify strategy execution via `StrategyRegistry.get("timestral")`.
- [x] 1.3 Deprecate `models/lstm.py` and `strategies/lstm.py`, decoupling them from standard execution pipelines while preserving backward-compatible archive stubs.
- [x] 1.4 Decouple `strategies/__init__.py` and `models/__init__.py`, replacing top-level imports of `strategies.lstm` with `strategies.timestral`.
- [x] 1.5 Add numerical edge-case unit tests in `test/test_timestral_strategy.py` (covering flat zero-volatility series, series with < 60 bars, quantile crossing inversion, and risk-reward calculation when $P_{10} \ge P_{current}$).

## 2. Data Persistence & DuckDB Storage Evolution (Executed Before Pipeline)

- [x] 2.1 Update `data/duckdb_manager.py` table schema with non-destructive migrations adding `timestral_predicted_price`, `timestral_potential`, `timestral_p10`, `timestral_p90`, and `dual_model_consensus`. Verify schema evolution on existing database file.
- [x] 2.2 Refactor `save_predictions_batch` in `data/duckdb_manager.py` to use explicit column binding (`INSERT INTO predictions ({col_names}) SELECT {col_names} FROM temp_batch_df`) to eliminate `BinderError` on evolved schemas.
- [x] 2.3 Update `save_daily_predictions()` and snapshot insertion methods to persist Timestral forecasts, quantile values, and consensus flags, setting legacy LSTM columns to null on new runs. Verify DuckDB insertion succeeds.
- [x] 2.4 Update `get_resonance_candidates()` and `load_predictions_dataframe()` to query Timestral fields and provide backward-compatible fallback for legacy records. Verify query output via DuckDB test assertions in `test/test_duckdb.py`.

## 3. Pipeline Orchestration & Dual-FM Resonance Overhaul

- [x] 3.1 Refactor `pipeline/orchestrator.py` to execute Timestral batch inference alongside TimesFM in Stage 2 compute workloads, eliminating LSTM execution. Verify orchestrator run produces valid Timestral strategy results.
- [x] 3.2 Overhaul `evaluators/composite_evaluator.py` to establish Dual Foundation Model Resonance: requiring simultaneous bullish signals from Google TimesFM and Mistral Timestral, confirmed by XuanTie MA pullback and institutional capital accumulation.
- [x] 3.3 Synthesize Resonance target price and expected return as a defensive ensemble of TimesFM and Timestral P50 forecasts with graceful single-model fallback, eliminating single-model reliance.
- [x] 3.4 Add dual-model confluence unit tests in `test/test_composite_evaluator.py` verifying cross-model agreement, disagreement rejection, and target price synthesis under missing values.

## 4. REST API & FastMCP Contract Modernization

- [x] 4.1 Update `api/schemas.py` Pydantic models to include Timestral predictions, quantile intervals, and dual-model consensus indicators, retaining `lstm_potential` as an optional deprecated alias.
- [x] 4.2 Update `api/routes/predictions.py` adding `GET /api/v1/predictions/timestral/top-bullish` and `top-bearish` endpoints, updating `/resonance` and `/latest` responses. Verify endpoint responses with Swagger/OpenAPI.
- [x] 4.3 Implement backward-compatible route aliases for `GET /api/v1/predictions/lstm/*` mapping transparently to Timestral predictions with deprecation headers (`X-Deprecation-Warning`). Verify route aliasing via API unit tests.
- [x] 4.4 Update `mcp_server.py` to register `@mcp.tool(name="get_timestral_top_predictions")` and alias `get_lstm_top_predictions` with deprecation notice for external AI agent parity.

## 5. Notification Services & Narrative Formatting

- [x] 5.1 Update `evaluators/formatter.py` and `notifier_dual.py` to generate Telegram, Discord, and Email alerts displaying dual foundation model blocks (`🤖 TimesFM` and `🔮 Timestral`), quantile ranges, and `🏆雙模型共振` tags. Verify formatted markdown message output.
- [x] 5.2 Update `evaluators/ai_narrative.py` prompt templates to analyze cross-model consensus and quantile risk-reward for daily subscriber reports.

## 6. Financial Terminal UI & Charting

- [x] 6.1 Update `api/templates/partials/tab_history.html` to add Timestral Target Trajectory controls, legend toggles, and quantile band options.
- [x] 6.2 Update `api/static/js/chart-manager.js` to render the Timestral Target Price line (Cyan) and optional P10–P90 shaded confidence area alongside historical closing prices and TimesFM forecasts.
- [x] 6.3 Update `api/static/js/app.js` strategy dropdown filters, table columns, and badge renderers to feature Timestral forecasts and Dual Resonance indicators instead of LSTM.
- [x] 6.4 Verify terminal-grade styling, responsive mobile layouts, and seamless light/dark mode contrast adhering to Impeccable design guidelines.

## 7. Verification, Code Review & Production Deployment

- [x] 7.1 Audit and update existing unit test suite assertions (`test/test_composite_evaluator.py`, `test/test_duckdb.py`, `test/test_api_predictions.py`) to transition test fixtures from LSTM to Timestral.
- [x] 7.2 Execute full local test suite ensuring all 260+ tests pass with zero regressions (273 tests ran, 272 passed, 1 skipped).
- [x] 7.3 Run final verification reviews with subagents (`codex-reviewer` & `opencode-reviewer`) and address runtime edge cases (applied day-1 quantile sorting and trajectory padding guard).
- [x] 7.4 Update `docs/CHANGELOG.md` and `README.md` documenting the transition from LSTM to Mistral Timestral-209M and Dual Foundation Model Resonance in compliance with `AGENTS.md`.
- [x] 7.5 Deploy the updated codebase and Docker container to production host `10.9.0.99`, verifying live endpoints, web dashboard, and prediction pipeline. (Deployed v2.4.30, verified healthy on both 10.9.0.99:8088 and https://stockdata.david888.com).
