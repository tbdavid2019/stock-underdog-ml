# Design: Deprecate LSTM & Integrate Mistral Timestral-209M for Dual Foundation Model Resonance

## Context

The system currently runs two distinct predictive layers:
1. Google Research TimesFM 2.5 (200M parameter time-series foundation model in `models/timesfm_model.py` & `strategies/timesfm.py`).
2. Legacy LSTM recurrent network (`models/lstm.py` & `strategies/lstm.py`).

Historically, `evaluators/composite_evaluator.py` and `data/duckdb_manager.py` copied `lstm_res.predicted_price` and `lstm_potential` as the primary target price and potential percentage for the flagship `多維共振 (Multi-Dimensional Resonance)` signal. Because the single-layer LSTM suffers from severe autoregressive mean-regression ($\hat{y}_{t+1} \approx y_t$), this compromised the accuracy of resonance recommendations.

Mistral AI's **Timestral-209M** (`mistralai/Timestral-209M`) is a state-of-the-art decoder-only foundation model featuring a 32,768-step context window, dual Time-Variate attention, and multi-quantile outputs (P01 to P99). Host verification on our production host (`10.9.0.99`) confirms 16 CPU cores, 40GB free RAM, and sub-100ms CPU inference speed per batch.

## Goals / Non-Goals

**Goals:**
- Eliminate the legacy LSTM model from active execution, data persistence, and scoring pipelines.
- Integrate `mistralai/Timestral-209M` via `models/timestral_model.py` and `strategies/timestral.py` with multi-quantile (P10, P50, P90) probabilistic forecasting.
- Synchronize temporal horizons: both TimesFM and Timestral operate on an identical 5-trading-day forward horizon (`horizon = 5`).
- Overhaul `多維共振` into **Dual Foundation Model Resonance**: requiring unanimous bullish consensus between Google TimesFM and Mistral Timestral, confirmed by institutional capital accumulation and XuanTie technical pullback setups.
- Synthesize resonance target price and potential from an ensemble of TimesFM and Timestral P50 predictions, weighted by historical model confidence with defensive single-model fallbacks.
- Maintain backward compatibility across DuckDB schemas, REST API endpoints (`/api/v1/predictions/lstm/*`), and FastMCP tools (`mcp_server.py`).
- Implement defensive numerical and resource boundaries: thread-capping on host CPU, bounded risk-reward calculations, monotonic quantile clamping, and explicit DuckDB column binding.

**Non-Goals:**
- Training custom weights from scratch: Timestral-209M and TimesFM-200M will run in zero-shot inference mode.
- Deprecating technical indicators (XuanTie, Bollinger, KD, Institutional Flow); these remain vital complementary pillars for multi-strategy confluence.

## Decisions

### Decision 1: Foundation Model Architecture & Execution Strategy
- **Choice**: Implement `models/timestral_model.py` wrapping `mistralai/Timestral-209M` (`from timestral import TimestralForecaster`) with lazy-loading, singleton cache, and PyTorch CPU inference on `10.9.0.99`.
- **Inference Strategy**: Extract 5-day forecasts with 3 specific quantiles:
  - **P50**: Median forecast, serving as the central target price.
  - **P90**: Upper bound (90th percentile), representing breakout ceiling.
  - **P10**: Lower bound (10th percentile), representing downside floor.
- **Defensive Quantile Ordering**: Enforce monotonic clamp `p10, p50, p90 = sorted([p10, p50, p90])` to prevent quantile crossing anomalies.
- **Bounded Risk-Reward Ratio**: Protect against division by zero and negative denominator inversion ($P_{10} \ge P_{current}$):
  $$\text{risk\_span} = \max(P_{current} - P_{10},\ P_{current} \times 0.005,\ 1e-4)$$
  $$\text{reward\_span} = \max(P_{90} - P_{current},\ 0.0)$$
  $$\text{Risk-Reward Ratio} = \min\left(\frac{\text{reward\_span}}{\text{risk\_span}},\ 50.0\right)$$
- **Collation & Uniform Batching**: Collate historical series to a fixed operational context window (e.g. `tail(context_len=128)`), rejecting series with fewer than 60 trading days to prevent inhomogeneous tensor shape errors.
- **CPU Resource Allocation**: Thread-cap PyTorch CPU workers to physical performance cores (`torch.set_num_threads(min(8, os.cpu_count() or 4))`) wrapped in `torch.inference_mode()` to prevent CPU thrashing on the 16-core hybrid Intel i7-12700 host.
- **Thread-Safe Singleton**: Protect `TimestralWrapper` instantiation with a double-checked mutex lock (`threading.Lock()`).

### Decision 2: Dual Foundation Model Resonance Overhaul
- **Choice**: Multi-Dimensional Resonance (`🏆多維共振`) requires 4-way confluence:
  1. **Google TimesFM**: Bullish signal (`potential > 0%` and `risk_reward >= 1.0`).
  2. **Mistral Timestral**: Bullish signal (`P50 potential > 0%` and asymmetric risk-reward $\ge 1.0$).
  3. **Institutional Flow**: Net buying streak or positive institutional accumulation over 3–5 days.
  4. **XuanTie Setup**: Upward-sloping MA60 with price pulling back to MA60/MA120 support band.
- **Defensive Target Price Synthesis**:
  ```python
  valid_targets = []
  if timesfm_target is not None and np.isfinite(timesfm_target):
      valid_targets.append(timesfm_target)
  if timestral_target is not None and np.isfinite(timestral_target):
      valid_targets.append(timestral_target)

  if len(valid_targets) == 2:
      synthesized_target = round(0.5 * valid_targets[0] + 0.5 * valid_targets[1], 2)
  elif len(valid_targets) == 1:
      synthesized_target = round(valid_targets[0], 2)
  else:
      synthesized_target = current_price
  ```

### Decision 3: Storage & Schema Evolution (DuckDB)
- **Choice**: Non-destructive schema evolution with explicit column binding.
  - Add new columns to `predictions` table:
    - `timestral_predicted_price` (DOUBLE)
    - `timestral_potential` (DOUBLE)
    - `timestral_p10` (DOUBLE)
    - `timestral_p90` (DOUBLE)
    - `dual_model_consensus` (BOOLEAN)
  - **Explicit Binder Safety**: In `data/duckdb_manager.py#save_predictions_batch`, avoid `INSERT INTO predictions SELECT * FROM temp_df`. Explicitly bind columns: `INSERT INTO predictions ({col_names}) SELECT {col_names} FROM temp_df` to prevent DuckDB `BinderError` across schema migrations.
  - Keep legacy columns (`lstm_predicted_price`, `lstm_potential`) nullable for existing rows, populating them with `None` on new writes.
  - Update `load_predictions_dataframe()` and `get_resonance_candidates()` to dynamically read Timestral columns with fallback.

### Decision 4: REST API & FastMCP Backward Compatibility
- **Choice**:
  - Add `/api/v1/predictions/timestral/top-bullish` and `/api/v1/predictions/timestral/top-bearish`.
  - Maintain `/api/v1/predictions/lstm/top-bullish` and `/top-bearish` as backward-compatible aliases returning Timestral data with a deprecation header (`X-Deprecation-Warning: LSTM deprecated, use Timestral`).
  - Register `@mcp.tool(name="get_timestral_top_predictions")` in `mcp_server.py` and alias `get_lstm_top_predictions` for external AI agent parity.
  - Update `/api/v1/predictions/resonance` and `/api/v1/predictions/latest` schemas to include Timestral fields, keeping `lstm_potential` as an optional deprecated alias in `StockPredictionItem`.

### Decision 5: Visual Presentation (Terminal-Grade UI)
- **Choice**:
  - In `tab_history.html` and `chart-manager.js`:
    - Add Timestral Target line (Cyan `#06b6d4`) and optional P10–P90 shaded area.
    - Retain TimesFM Target line (Amber/Purple `#8b5cf6`).
    - Remove LSTM line or replace its toggle with Timestral.
  - In `app.js` and tables:
    - Replace LSTM strategy tag with `🔮 Timestral` badge.
    - Update filter dropdown: replace "LSTM 看漲/看跌" with "🔮 Timestral 潛力精選".
    - Mark Dual Resonance candidates with `🏆雙模型共振` badge.

## Risks / Trade-offs

- **[Risk] Host Memory Pressure on Concurrent Execution**:
  - *Analysis*: Host `10.9.0.99` has 40GB free RAM. TimesFM 200M uses ~1.2GB; Timestral 209M uses ~1.5GB. Both easily fit simultaneously (< 3GB combined).
  - *Mitigation*: Run TimesFM and Timestral sequentially or reuse PyTorch thread pools with `torch.inference_mode()` and explicit thread capping (`torch.set_num_threads(8)`).
- **[Risk] Missing HuggingFace Weights or Rate Limits**:
  - *Mitigation*: Pre-cache weights in `/root/.cache/huggingface/hub` inside the Docker volume or container. If offline, gracefully fall back with informative warnings without crashing the pipeline.
- **[Risk] External Sinks / Consumers Expecting LSTM fields**:
  - *Mitigation*: Provide legacy field aliases (`lstm_potential` = `timestral_potential` if requested by older external scripts).

## Migration Plan

1. **Step 1 - Core Model Implementation**: Create `models/timestral_model.py` and `strategies/timestral.py`. Add unit tests (`test/test_timestral_strategy.py`).
2. **Step 2 - Data Storage & Migrations**: Update `data/duckdb_manager.py` migrations and queries (execute before pipeline to guarantee column availability).
3. **Step 3 - Evaluator & Pipeline Refactor**: Update `evaluators/composite_evaluator.py` and `pipeline/orchestrator.py` to replace LSTM with Timestral and implement Dual-FM Resonance.
4. **Step 4 - API Routes & FastMCP Compatibility**: Update `api/routes/predictions.py`, `api/schemas.py`, and `mcp_server.py`.
5. **Step 5 - Notifier & Narrative**: Update Telegram, Discord, and AI narrative formatters.
6. **Step 6 - Frontend Dashboard & Charts**: Update `tab_history.html`, `chart-manager.js`, and `app.js`.
7. **Step 7 - Verification & Review**: Run unit tests (260+ tests). Deploy to `10.9.0.99` container. Verify live endpoints and UI.
